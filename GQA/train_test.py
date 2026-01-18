import torch
import torch.nn as nn
from model import build_gqa_transformer

def verify_gqa():
    print("--- Verifying GQA Implementation ---")
    
    src_vocab_size = 100
    tgt_vocab_size = 100
    src_seq_len = 20
    tgt_seq_len = 20
    d_model = 64
    h = 8
    
    # 1. Verify MQA Equivalence (G=1)
    print("\n1. Building GQA-1 (MQA equivalent)...")
    model_mqa = build_gqa_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, 
                                      d_model=d_model, h=h, num_groups=1)
    
    # Check weight shapes
    # Query: (D, D) -> (64, 64)
    # Key: (D, G*Dk) -> (64, 1*8) = (64, 8)
    q_params = sum(p.numel() for p in model_mqa.encoder.layers[0].self_attention_block.w_q.parameters())
    k_params = sum(p.numel() for p in model_mqa.encoder.layers[0].self_attention_block.w_k.parameters())
    print(f"   Query Params: {q_params} (Expected ~4096 + bias)")
    print(f"   Key Params: {k_params} (Expected ~512 + bias)")
    assert k_params < q_params, "MQA Key params should be significantly smaller than Query params"
    print("   [PASS] Parameter checks for GQA-1")

    # 2. Verify MHA Equivalence (G=H)
    print("\n2. Building GQA-8 (MHA equivalent)...")
    model_mha = build_gqa_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, 
                                      d_model=d_model, h=h, num_groups=h)
    
    q_params = sum(p.numel() for p in model_mha.encoder.layers[0].self_attention_block.w_q.parameters())
    k_params = sum(p.numel() for p in model_mha.encoder.layers[0].self_attention_block.w_k.parameters())
    print(f"   Query Params: {q_params}")
    print(f"   Key Params: {k_params}")
    assert q_params == k_params, "MHA Key params should be equal to Query params (approx)"
    print("   [PASS] Parameter checks for GQA-8")

    # 3. Verify Interpolation (G=4)
    print("\n3. Building GQA-4 (Interpolation)...")
    model_gqa = build_gqa_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, 
                                      d_model=d_model, h=h, num_groups=4)
    
    # Forward Pass
    src = torch.randint(0, src_vocab_size, (2, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (2, tgt_seq_len))
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, L)
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = tgt_mask & torch.tril(torch.ones((tgt_seq_len, tgt_seq_len))).bool()
    
    print("   Running Forward Pass...")
    encoder_output = model_gqa.encode(src, src_mask)
    output = model_gqa.decode(encoder_output, src_mask, tgt, tgt_mask)
    projected = model_gqa.project(output)
    
    print(f"   Output Shape: {projected.shape}")
    print("   [PASS] Forward Pass")

if __name__ == "__main__":
    verify_gqa()
