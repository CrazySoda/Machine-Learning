import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from gpu_profiler import GPUProfiler

profiler = GPUProfiler()

class input_embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)   # inputs --> index number --> vector of d_model dimension

    def forward(self, x):
        profiler.start()
        x = self.embedding(x) * math.sqrt(self.d_model) # given index number --> returns vector
        profiler.end("Input Embedding")
        return x


class positional_encoding(nn.Module):
    # seq_len = maximum length of sentence , dropout = makes model less overfit
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # calculating in log space for stability

        # apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # change dimensions to 3D (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        profiler.start()
        # adding the positional encoding to word of a sentence
        x = x + self.pe[:, :x.shape[1], :]   # buffer does not require grad
        x = self.dropout(x)
        profiler.end("Positional Encoding")
        return x


class layer_normalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 10**-6): # eps so that denominator isn't 0
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))    # multiplied
        self.bias = nn.Parameter(torch.zeros(d_model))    # added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class feed_forwardblock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        profiler.start()
        # (batch, seq_len , d_model) --> linear1 --> linear2
        x = self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

        profiler.end("FeedForward")

        return x


class nystrom_attention_block(nn.Module):
    """
    Nystrom-based self-attention (Nystromformer, Xiong et al. 2021).

    Instead of computing the full N×N attention matrix, this method:
    1. Selects m landmark points by segment-mean pooling of Q and K
    2. Computes three smaller kernel matrices:
       - kernel_1: Q vs K_landmarks  (N×m)
       - kernel_2: Q_landmarks vs K_landmarks  (m×m), then inverted
       - kernel_3: Q_landmarks vs K  (m×N)
    3. Approximates attention as: kernel_1 @ inv(kernel_2) @ kernel_3 @ V

    This reduces complexity from O(N²) to O(N·m).

    An optional depthwise convolution residual captures local patterns,
    following the original paper's design.
    """
    def __init__(self, d_model: int, h: int, num_landmarks: int, dropout: float, conv_kernel_size: int = None):
        super().__init__()
        self.d_model = d_model
        self.num_head = h
        self.num_landmarks = num_landmarks

        assert d_model % h == 0, "d_model is not divisible by h"
        self.head_dim = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Optional depthwise conv residual (from the original Nystromformer paper)
        self.use_conv = conv_kernel_size is not None
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels=self.num_head, out_channels=self.num_head,
                kernel_size=(conv_kernel_size, 1), padding=(conv_kernel_size // 2, 0),
                bias=False,
                groups=self.num_head)

    def iterative_inv(self, mat, n_iter=6):
        """
        Newton-Schulz iterative matrix inversion.
        Approximates mat^{-1} using 6 iterations of cubic convergence.
        Used to invert the m×m landmark kernel matrix.
        """
        I = torch.eye(mat.size(-1), device=mat.device)
        K = mat

        # Initialization: V0 = (1 / max row-sum of K) * K^T
        # This ensures ||KV0||_inf <= 1 for convergence
        V = 1 / torch.max(torch.sum(K, dim=-2), dim=-1).values[:, :, None, None] * K.transpose(-1, -2)

        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def forward(self, q, k, v, mask=None):
        profiler.start()
        batch_size, seq_len, _ = q.shape

        # Projections: (B, L, D) -> (B, H, L, Dk)
        Q = self.w_q(q).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        K = self.w_k(k).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        V = self.w_v(v).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)

        # Mask handling (matching the original Nystromformer repo):
        # mask input is (B, 1, 1, L) from the classifier/encoder
        # mask[:, None, :, None] -> (B, 1, L, 1) for zeroing padded rows of Q/K
        # mask[:, None, None, :] -> (B, 1, 1, L) for masking attention scores
        if mask is not None:
            if mask.dim() == 2:
                # (B, L) -> expand for both uses
                mask_qk = mask[:, None, :, None].float()       # (B, 1, L, 1)
                mask_score = mask[:, None, None, :].float()     # (B, 1, 1, L)
            elif mask.dim() == 4:
                # Already (B, 1, 1, L)
                mask_score = mask.float()                                     # (B, 1, 1, L)
                mask_qk = mask.transpose(-1, -2).float()                     # (B, 1, L, 1)
            else:
                mask_qk = None
                mask_score = None
        else:
            mask_qk = None
            mask_score = None

        # Apply mask to Q and K, and scale (matching original repo exactly)
        # Q = Q * mask[:, None, :, None] / sqrt(sqrt(head_dim))
        if mask_qk is not None:
            Q = Q * mask_qk / math.sqrt(math.sqrt(self.head_dim))
            K = K * mask_qk / math.sqrt(math.sqrt(self.head_dim))
        else:
            Q = Q / math.sqrt(math.sqrt(self.head_dim))
            K = K / math.sqrt(math.sqrt(self.head_dim))

        if self.num_landmarks >= seq_len:
            # Fallback to standard softmax attention when seq_len <= num_landmarks
            scores = torch.matmul(Q, K.transpose(-1, -2))
            if mask_score is not None:
                scores = scores - 1e9 * (1 - mask_score)
            attn = F.softmax(scores, dim=-1)
            X = torch.matmul(attn, V)
        else:
            # Nystrom approximation
            # Pad seq_len to be divisible by num_landmarks if needed
            remainder = seq_len % self.num_landmarks
            if remainder != 0:
                pad_len = self.num_landmarks - remainder
                Q = F.pad(Q, (0, 0, 0, pad_len))   # pad sequence dim
                K = F.pad(K, (0, 0, 0, pad_len))
                V = F.pad(V, (0, 0, 0, pad_len))
                padded_len = seq_len + pad_len
            else:
                pad_len = 0
                padded_len = seq_len

            # Landmark selection via segment-mean pooling
            seg_size = padded_len // self.num_landmarks
            Q_landmarks = Q.reshape(batch_size, self.num_head, self.num_landmarks, seg_size, self.head_dim).mean(dim=-2)
            K_landmarks = K.reshape(batch_size, self.num_head, self.num_landmarks, seg_size, self.head_dim).mean(dim=-2)

            # Three Nystrom kernel matrices
            kernel_1 = F.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim=-1)                # (B, H, L, m)
            kernel_2 = F.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim=-1)      # (B, H, m, m)

            # kernel_3 needs score masking to avoid attending to padding
            scores_3 = torch.matmul(Q_landmarks, K.transpose(-1, -2))        # (B, H, m, L)
            if mask_score is not None and pad_len == 0:
                scores_3 = scores_3 - 1e9 * (1 - mask_score)
            kernel_3 = F.softmax(scores_3, dim=-1)

            # Nystrom approximation: kernel_1 @ inv(kernel_2) @ kernel_3 @ V
            X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, V))

            # Remove padding if we added any
            if pad_len > 0:
                X = X[:, :, :seq_len, :]
                V = V[:, :, :seq_len, :]

        # Optional conv residual (captures local patterns)
        if self.use_conv:
            if mask_qk is not None:
                X += self.conv(V * mask_qk)
            else:
                X += self.conv(V)

        X = self.dropout(X)
        # (B, H, L, Dk) -> (B, L, H, Dk) -> (B, L, D)
        X = X.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        X = self.w_o(X)

        profiler.end("NystromAttention")
        return X


class residual_connection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = layer_normalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class encoder_block(nn.Module):
    def __init__(self, self_attention_block, feed_forward_block, d_model, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [residual_connection(d_model, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int):
        super().__init__()
        self.layers = layers
        self.norm = layer_normalization(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class decoder_block(nn.Module):
    def __init__(self,
                 self_attention_block,
                 cross_attention_block,
                 feed_forward_block,
                 d_model,
                 dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block
        self.residual_connections = nn.ModuleList(
            [residual_connection(d_model, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int):
        super().__init__()
        self.layers = layers
        self.norm = layer_normalization(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


# linear layer at the end
class projection_layer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        profiler.start()
        out = torch.log_softmax(self.proj(x), dim=-1)
        profiler.end("Projection")
        return out


class nystromformer_model(nn.Module):
    def __init__(self, encoder: encoder, decoder: decoder,
                 src_embed: input_embeddings, tgt_embed: input_embeddings,
                 src_pos: positional_encoding, tgt_pos: positional_encoding,
                 projection_layer: projection_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_nystromformer(src_vocab_size: int, tgt_vocab_size: int,
                        src_seq_len: int, tgt_seq_len: int,
                        d_model: int = 512, N: int = 6,
                        h: int = 8, dropout: float = 0.1, d_ff: int = 2048,
                        num_landmarks: int = 64, conv_kernel_size: int = None):

    # create embedding layers
    src_embed = input_embeddings(d_model, src_vocab_size)
    tgt_embed = input_embeddings(d_model, tgt_vocab_size)

    # positional encoding layers
    src_pos = positional_encoding(d_model, src_seq_len, dropout)
    tgt_pos = positional_encoding(d_model, tgt_seq_len, dropout)

    # create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = nystrom_attention_block(
            d_model, h, num_landmarks, dropout, conv_kernel_size
        )
        feed_forward_block_ = feed_forwardblock(d_model, d_ff, dropout)
        temp_encoder_block = encoder_block(
            encoder_self_attention_block, feed_forward_block_, d_model, dropout
        )
        encoder_blocks.append(temp_encoder_block)

    # create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = nystrom_attention_block(
            d_model, h, num_landmarks, dropout, conv_kernel_size
        )
        decoder_cross_attention_block = nystrom_attention_block(
            d_model, h, num_landmarks, dropout, conv_kernel_size
        )
        feed_forward_block_ = feed_forwardblock(d_model, d_ff, dropout)
        temp_decoder_block = decoder_block(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block_,
            d_model,
            dropout
        )
        decoder_blocks.append(temp_decoder_block)

    # create encoder and decoder
    main_encoder = encoder(nn.ModuleList(encoder_blocks), d_model)
    main_decoder = decoder(nn.ModuleList(decoder_blocks), d_model)
    main_projection_layer = projection_layer(d_model, tgt_vocab_size)

    # create transformer
    main_transformer = nystromformer_model(
        main_encoder, main_decoder,
        src_embed, tgt_embed,
        src_pos, tgt_pos,
        main_projection_layer
    )

    # initialize the parameters
    for p in main_transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return main_transformer
