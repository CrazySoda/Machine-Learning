import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from gpu_profiler import GPUProfiler

profiler = GPUProfiler()

class InputEmbeddings(nn.Module):
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
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
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

class GroupedQueryAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, num_groups: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.num_groups = num_groups
        
        assert h % num_groups == 0, "Number of heads must be divisible by number of groups"
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        
        # Key and Value heads are shared per group
        # Total KV heads = num_groups
        # Dimension of KV projection = num_groups * d_k
        self.w_k = nn.Linear(d_model, num_groups * self.d_k)
        self.w_v = nn.Linear(d_model, num_groups * self.d_k)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def repeat_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        # x: (Batch, Seq_Len, Num_Groups, D_k)
        # return: (Batch, Seq_Len, Num_Groups * Num_Repeats, D_k)
        # We need to repeat each group's KV head `num_repeats` times to match Q heads
        
        batch, seq_len, n_kv_heads, d_k = x.shape
        if num_repeats == 1:
            return x
            
        # (B, L, G, 1, D) -> (B, L, G, R, D) -> (B, L, G*R, D)
        return (
            x[:, :, :, None, :]
            .expand(batch, seq_len, n_kv_heads, num_repeats, d_k)
            .reshape(batch, seq_len, n_kv_heads * num_repeats, d_k)
        )

    def forward(self, q, k, v, mask=None):
        profiler.start()
        batch_size, seq_len, _ = q.shape
        
        # Query: (B, L, H, Dk)
        query = self.w_q(q).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        
        # Key/Value: (B, L, G, Dk)
        key = self.w_k(k).view(batch_size, -1, self.num_groups, self.d_k) # Note: src len might differ from seq_len if Cross Attn
        value = self.w_v(v).view(batch_size, -1, self.num_groups, self.d_k)
        
        # We need to upscale K and V to match H heads for dot product Attention
        # Or we can do the dot product with the grouped form and expand later? 
        # Easier to expand K, V to match Q for parallel implementation
        
        num_repeats = self.h // self.num_groups
        key = self.repeat_kv(key, num_repeats).transpose(1, 2) # (B, H, L_src, Dk)
        value = self.repeat_kv(value, num_repeats).transpose(1, 2) # (B, H, L_src, Dk)
        
        # Scaled Dot-Product Attention
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ value).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        x = self.w_o(x)
        
        profiler.end("GQA")
        return x

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
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

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block, feed_forward_block, d_model, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block, cross_attention_block, feed_forward_block, d_model, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        profiler.start()
        out = torch.log_softmax(self.proj(x), dim=-1)
        profiler.end("Projection")
        return out

class GQATransformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, 
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
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

def build_gqa_transformer(src_vocab_size: int, tgt_vocab_size: int,
                          src_seq_len: int, tgt_seq_len: int,
                          d_model: int = 512, N: int = 6, 
                          h: int = 8, num_groups: int = None,
                          dropout: float = 0.1, d_ff: int = 2048):
    
    if num_groups is None:
        num_groups = h # Default to MHA (Groups = Heads)
        
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    encoder_blocks = []
    for _ in range(N):
        self_attn = GroupedQueryAttentionBlock(d_model, h, num_groups, dropout)
        ff = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(self_attn, ff, d_model, dropout))
        
    decoder_blocks = []
    for _ in range(N):
        self_attn = GroupedQueryAttentionBlock(d_model, h, num_groups, dropout)
        cross_attn = GroupedQueryAttentionBlock(d_model, h, num_groups, dropout)
        ff = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(self_attn, cross_attn, ff, d_model, dropout))
        
    encoder = Encoder(nn.ModuleList(encoder_blocks), d_model)
    decoder = Decoder(nn.ModuleList(decoder_blocks), d_model)
    projection = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = GQATransformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer
