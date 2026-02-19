import torch
import time
import triton
import triton.language as tl
import sys


@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        # From 0 to the left of the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # Only used for non-causal attention
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # Just let the compiler know that start_n is a multiple of BLOCK_N, so the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # -- compute qk ----
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            # Compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # Compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)
        # Compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1)

        # This is the correction factor for the previous l_i
        alpha = tl.math.exp(m_i - m_ij)
        # Apply the correction factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)
        # This computes the following: O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        # Move to the next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    return O_block, l_i, m_i


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # This indicate which block in the sequence length to process
    block_index_q = tl.program_id(0)

    # This indicates which head and batch to process. Each program is associated with a single head of a single batch
    index_batch_head = tl.program_id(1)
    # This indicate which batch this program is associated with (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # This indicate the position of the head in the batch
    index_head = index_batch_head % NUM_HEADS

    # This allows to get the (N_CTX, HEAD_DIM) block in the Q, K, V by selecting indexing it by batch and head
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(
            stride_K_dim,
            stride_K_seq,
        ),  # We invert the strides w.r.t Q, so we transpose the matrix
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # offs_q: the offsets for the tokens in the Q to process
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # offs_kv: the offsets for the tokens in the K and V sequence to process
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # m_i: the running maximum. We have one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # l_i: the running sum. We have one for each query (as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    # acc: the accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptr)

    # Stage: 3 if causal, else 1

    if STAGE == 1 or STAGE == 3:
        # This step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
    # epilogue
    O_block = O_block / l_i[:, None]
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


def flash_attention_forward(Q, K, V, causal, softmax_scale):
    """
    Forward pass of flash attention
    
    Args:
        Q: Query tensor of shape (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        K: Key tensor of shape (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        V: Value tensor of shape (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        causal: Whether to apply causal masking
        softmax_scale: Scaling factor for softmax (typically 1/sqrt(HEAD_DIM))
    
    Returns:
        O: Output tensor of shape (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    """
    HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
    HEAD_DIM_V = V.shape[-1]

    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

    O = torch.empty_like(Q)
    stage = 3 if causal else 1

    grid = lambda args: (
        triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
        BATCH_SIZE * NUM_HEADS,
        1,
    )

    _attn_fwd[grid](
        Q=Q,
        K=K,
        V=V,
        softmax_scale=softmax_scale,
        O=O,
        stride_Q_batch=Q.stride(0),
        stride_Q_head=Q.stride(1),
        stride_Q_seq=Q.stride(2),
        stride_Q_dim=Q.stride(3),
        stride_K_batch=K.stride(0),
        stride_K_head=K.stride(1),
        stride_K_seq=K.stride(2),
        stride_K_dim=K.stride(3),
        stride_V_batch=V.stride(0),
        stride_V_head=V.stride(1),
        stride_V_seq=V.stride(2),
        stride_V_dim=V.stride(3),
        stride_O_batch=O.stride(0),
        stride_O_head=O.stride(1),
        stride_O_seq=O.stride(2),
        stride_O_dim=O.stride(3),
        BATCH_SIZE=Q.shape[0],
        NUM_HEADS=Q.shape[1],
        SEQ_LEN=Q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        STAGE=stage,
    )

    return O


def benchmark_attention(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16, num_warmup=10, num_runs=100):
    """Benchmark normal attention vs flash attention (forward pass only)"""
    
    print(f"\n{'='*80}")
    print(f"Benchmarking Configuration:")
    print(f"  BATCH_SIZE={BATCH_SIZE}, NUM_HEADS={NUM_HEADS}, SEQ_LEN={SEQ_LEN}, HEAD_DIM={HEAD_DIM}")
    print(f"  Causal={causal}, dtype={dtype}")
    print(f"{'='*80}\n")
    
    # Create input tensors
    Q = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    K = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    V = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    
    softmax_scale = 1 / (HEAD_DIM**0.5)
    
    # ===========================
    # Benchmark Normal Attention
    # ===========================
    print("Benchmarking Normal Attention...")
    
    # Warmup
    for _ in range(num_warmup):
        MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
        P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
        if causal:
            P[:, :, MASK == 0] = float("-inf")
        P = torch.softmax(P.float(), dim=-1).half()
        ref_O = torch.matmul(P, V)
    
    torch.cuda.synchronize()
    
    # Benchmark forward pass
    start_time = time.time()
    for _ in range(num_runs):
        MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
        P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
        if causal:
            P[:, :, MASK == 0] = float("-inf")
        P = torch.softmax(P.float(), dim=-1).half()
        ref_O = torch.matmul(P, V)
    torch.cuda.synchronize()
    normal_fwd_time = (time.time() - start_time) / num_runs * 1000  # Convert to ms
    
    # ===========================
    # Benchmark Flash Attention
    # ===========================
    print("Benchmarking Flash Attention (Triton)...")
    
    # Warmup
    for _ in range(num_warmup):
        tri_out = flash_attention_forward(Q, K, V, causal, softmax_scale)
    
    torch.cuda.synchronize()
    
    # Benchmark forward pass
    start_time = time.time()
    for _ in range(num_runs):
        tri_out = flash_attention_forward(Q, K, V, causal, softmax_scale)
    torch.cuda.synchronize()
    flash_fwd_time = (time.time() - start_time) / num_runs * 1000  # Convert to ms
    
    # ===========================
    # Print Results
    # ===========================
    print(f"\n{'='*80}")
    print(f"RESULTS (averaged over {num_runs} runs):")
    print(f"{'='*80}")
    print(f"\nNormal Attention:")
    print(f"  Forward:  {normal_fwd_time:.3f} ms")
    
    print(f"\nFlash Attention (Triton):")
    print(f"  Forward:  {flash_fwd_time:.3f} ms")
    
    print(f"\nSpeedup:")
    print(f"  Forward:  {normal_fwd_time / flash_fwd_time:.2f}x")
    print(f"{'='*80}\n")


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    """Test correctness of flash attention implementation (forward pass only)"""
    Q = torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
    ).normal_(mean=0.0, std=0.5)
    K = torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
    ).normal_(mean=0.0, std=0.5)
    V = torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
    ).normal_(mean=0.0, std=0.5)

    softmax_scale = 1 / (HEAD_DIM**0.5)

    # reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)

    # triton implementation
    tri_out = flash_attention_forward(Q, K, V, causal, softmax_scale).half()

    # compare
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    # Open a file in write mode, output.txt will be in /workspace inside the container
    with open("flash_attention_output.txt", "w") as f:
        # Redirect stdout and stderr to the file
        sys.stdout = f
        sys.stderr = f

        print("Testing correctness...")
        test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=1024, HEAD_DIM=64, causal=True)
        test_op(BATCH_SIZE=2, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=True)
        test_op(BATCH_SIZE=8, NUM_HEADS=4, SEQ_LEN=4096, HEAD_DIM=64, causal=True)

        print("PASSED\n")

        print("\n" + "="*80)
        print("STARTING BENCHMARKS")
        print("="*80)
        
        # Run benchmarks
        benchmark_attention(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=1024, HEAD_DIM=64, causal=True)
        benchmark_attention(BATCH_SIZE=2, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=True)