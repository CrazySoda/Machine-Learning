
#flash attention

import torch
import torch.nn as nn
import math
from gpu_profiler import GPUProfiler
from program import TritonAttention

profiler = GPUProfiler()


class input_embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        profiler.start()
        x = self.embedding(x) * math.sqrt(self.d_model)
        profiler.end("Input Embedding")
        return x


class positional_encoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        profiler.start()
        x = x + self.pe[:, :x.shape[1], :]
        x = self.dropout(x)
        profiler.end("Positional Encoding")
        return x


class layer_normalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

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
        x = self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        profiler.end("FeedForward")
        return x


class multihead_attentionblock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float, use_flash: bool = True, causal: bool = False):
        """
        Multi-head attention with optional Triton Flash Attention.
        Args:
            d_model: embedding dimension
            h: number of heads
            dropout: dropout probability
            use_flash: if True, use Triton flash attention; else standard attention
            causal: if True, apply causal masking in flash attention
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.use_flash = use_flash
        self.causal = causal
        self.attention_scores = None

    @staticmethod
    def standard_attention(query, key, value, mask=None, dropout=None):
        """
        Compute scaled dot-product attention (standard, non-flash).
        query: (batch, heads, seq_len_q, d_k)
        key, value: (batch, heads, seq_len_k, d_k)
        mask: (batch, 1, seq_len_q, seq_len_k) or (batch, 1, 1, seq_len_k)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e9"))

        attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        output = torch.matmul(attn, value)
        return output, attn

    def forward(self, q, k, v, mask=None):
        """
        Forward pass.
        q, k, v: (batch, seq_len, d_model)
        mask: (batch, 1, seq_len, seq_len) or (batch, 1, 1, seq_len)
        """
        profiler.start()
        batch_size, seq_len, _ = q.shape

        # Linear projections
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split heads: (batch, seq_len, d_model) -> (batch, h, seq_len, d_k)
        query = query.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)

        if self.use_flash and query.is_cuda:
            # Flash attention requires contiguous fp16 tensors
            # and expects shape (batch, heads, seq_len, d_k)
            softmax_scale = 1.0 / math.sqrt(self.d_k)

            # Flash attention needs fp16 input
            orig_dtype = query.dtype
            if orig_dtype != torch.float16:
                query = query.half()
                key = key.half()
                value = value.half()

            # Make sure tensors are contiguous
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            # Apply Triton flash attention
            x = TritonAttention.apply(query, key, value, self.causal, softmax_scale)

            # Cast back to original dtype if needed
            if orig_dtype != torch.float16:
                x = x.to(orig_dtype)

            self.attention_scores = None  # flash attention doesn't return scores
        else:
            # Fallback to standard attention
            x, self.attention_scores = multihead_attentionblock.standard_attention(
                query, key, value, mask, self.dropout
            )

        # Merge heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        x = self.w_o(x)

        profiler.end("Multi-Head Attention")
        return x


class residual_connection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = layer_normalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class encoder_block(nn.Module):
    def __init__(self, self_attention_block: multihead_attentionblock,
                 feed_forward_block: feed_forwardblock,
                 d_model: int,
                 dropout: float):
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
                 self_attention_block: multihead_attentionblock,
                 cross_attention_block: multihead_attentionblock,
                 feed_forward_block: feed_forwardblock,
                 d_model: int,
                 dropout: float):
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


class projection_layer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        profiler.start()
        out = torch.log_softmax(self.proj(x), dim=-1)
        profiler.end("Projection")
        return out


class transformer(nn.Module):
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


def build_transformer(src_vocab_size: int, tgt_vocab_size: int,
                      src_seq_len: int, tgt_seq_len: int,
                      d_model: int = 512, N: int = 6,
                      h: int = 8, dropout: float = 0.1, d_ff: int = 2048,
                      use_flash: bool = True, causal: bool = False):
    """
    Build a transformer with optional Triton Flash Attention.
    Args:
        use_flash: if True, use flash attention in multi-head attention blocks
        causal: if True, use causal masking in flash attention
    """
    # create embedding layers
    src_embed = input_embeddings(d_model, src_vocab_size)
    tgt_embed = input_embeddings(d_model, tgt_vocab_size)

    # positional encoding layers
    src_pos = positional_encoding(d_model, src_seq_len, dropout)
    tgt_pos = positional_encoding(d_model, tgt_seq_len, dropout)

    # create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = multihead_attentionblock(
            d_model, h, dropout, use_flash=use_flash, causal=causal
        )
        feed_forward_block_ = feed_forwardblock(d_model, d_ff, dropout)
        temp_encoder_block = encoder_block(
            encoder_self_attention_block, feed_forward_block_, d_model, dropout
        )
        encoder_blocks.append(temp_encoder_block)

    # create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = multihead_attentionblock(
            d_model, h, dropout, use_flash=use_flash, causal=True  # decoder self-attn is always causal
        )
        decoder_cross_attention_block = multihead_attentionblock(
            d_model, h, dropout, use_flash=use_flash, causal=False  # cross-attn is not causal
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
    main_transformer = transformer(
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
