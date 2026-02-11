import torch
import torch.nn as nn 
import math
from gpu_profiler import GPUProfiler

# Import attention mechanisms
from bigbird_attention import bigbird_attentionblock

profiler = GPUProfiler()


class input_embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)   # inputs --> index number --> vector of 512 dimension
            
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
        return self.alpha * (x - mean) / (std + self.eps) + self.bias  # just the formula for layer normalization
       

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


class multihead_attentionblock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):  # h = heads and d_model should be divisible by h
        super().__init__()
        self.d_model = d_model
        self.h = h 
        assert d_model % h == 0, "d_model is not divisible by h" 
        
        # d_model / h = dk 
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # Wq 
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv
        
        self.w_o = nn.Linear(d_model, d_model)  # Wo
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # part of masked attention 
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return (attention_scores @ value), attention_scores
        
    def forward(self, q, k, v, mask):
        profiler.start()
        
        query = self.w_q(q)  # Q'
        key = self.w_k(k)    # K'
        value = self.w_v(v)  # V'
        
        # split into small matrices
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = multihead_attentionblock.attention(
            query, key, value, mask, self.dropout
        )
        
        # (batch, h, seq_len, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        x = self.w_o(x)
        
        profiler.end("MultiHeadAttention")
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


# ============================================================================
# ATTENTION MECHANISM FACTORY
# ============================================================================
def get_attention_mechanism(attention_type: str, d_model: int, h: int, dropout: float, **kwargs):
    """
    Factory function to create different attention mechanisms.
    
    Args:
        attention_type: Type of attention mechanism
            - 'standard': Standard multi-head attention
            - 'bigbird': BigBird sparse attention
            - [Add more attention types here in future]
        d_model: Model dimension
        h: Number of attention heads
        dropout: Dropout rate
        **kwargs: Additional parameters specific to each attention type
        
    Returns:
        Attention block instance
        
    Example:
        # Standard attention
        attn = get_attention_mechanism('standard', d_model=512, h=8, dropout=0.1)
        
        # BigBird attention
        attn = get_attention_mechanism('bigbird', d_model=512, h=8, dropout=0.1,
                                       window_size=3, num_global_tokens=2, num_random_tokens=3)
    """
    
    if attention_type == 'standard':
        return multihead_attentionblock(d_model, h, dropout)
    
    elif attention_type == 'bigbird':
        # Extract BigBird-specific parameters
        window_size = kwargs.get('window_size', 3)
        num_global_tokens = kwargs.get('num_global_tokens', 2)
        num_random_tokens = kwargs.get('num_random_tokens', 3)
        
        return bigbird_attentionblock(
            d_model, h, dropout,
            window_size=window_size,
            num_global_tokens=num_global_tokens,
            num_random_tokens=num_random_tokens
        )
    
    # ========================================================================
    # ADD NEW ATTENTION MECHANISMS HERE
    # ========================================================================
    # elif attention_type == 'linformer':
    #     from linformer_attention import linformer_attentionblock
    #     projection_dim = kwargs.get('projection_dim', 256)
    #     return linformer_attentionblock(d_model, h, dropout, projection_dim=projection_dim)
    #
    # elif attention_type == 'performer':
    #     from performer_attention import performer_attentionblock
    #     num_features = kwargs.get('num_features', 256)
    #     return performer_attentionblock(d_model, h, dropout, num_features=num_features)
    #
    # elif attention_type == 'reformer':
    #     from reformer_attention import reformer_attentionblock
    #     num_buckets = kwargs.get('num_buckets', 64)
    #     return reformer_attentionblock(d_model, h, dropout, num_buckets=num_buckets)
    # ========================================================================
    
    else:
        raise ValueError(f"Unknown attention type: {attention_type}. "
                        f"Available types: 'standard', 'bigbird'")


def build_transformer(src_vocab_size: int, tgt_vocab_size: int,
                      src_seq_len: int, tgt_seq_len: int,
                      d_model: int = 512, N: int = 6,
                      h: int = 8, dropout: float = 0.1, d_ff: int = 2048,
                      attention_type: str = 'standard',
                      attention_config: dict = None):
    """
    Build transformer with configurable attention mechanism.
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        src_seq_len: Maximum source sequence length
        tgt_seq_len: Maximum target sequence length
        d_model: Model dimension (default: 512)
        N: Number of encoder/decoder layers (default: 6)
        h: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
        d_ff: Feed-forward dimension (default: 2048)
        attention_type: Type of attention mechanism (default: 'standard')
            Options: 'standard', 'bigbird', [add more in future]
        attention_config: Dictionary of attention-specific configuration
            For 'bigbird': {
                'window_size': 3,
                'num_global_tokens': 2,
                'num_random_tokens': 3
            }
            
    Returns:
        Transformer model
        
    Examples:
        # Standard attention
        model = build_transformer(
            src_vocab_size=10000, tgt_vocab_size=10000,
            src_seq_len=512, tgt_seq_len=512,
            attention_type='standard'
        )
        
        # BigBird attention
        model = build_transformer(
            src_vocab_size=10000, tgt_vocab_size=10000,
            src_seq_len=512, tgt_seq_len=512,
            attention_type='bigbird',
            attention_config={
                'window_size': 5,
                'num_global_tokens': 4,
                'num_random_tokens': 5
            }
        )
    """
    
    # Default attention config if not provided
    if attention_config is None:
        attention_config = {}

    # create embedding layers 
    src_embed = input_embeddings(d_model, src_vocab_size)
    tgt_embed = input_embeddings(d_model, tgt_vocab_size)
    
    # positional encoding layers 
    src_pos = positional_encoding(d_model, src_seq_len, dropout) 
    tgt_pos = positional_encoding(d_model, tgt_seq_len, dropout)
    
    # create encoder blocks 
    encoder_blocks = []
    for _ in range(N):
        # Create attention mechanism using factory
        encoder_self_attention_block = get_attention_mechanism(
            attention_type, d_model, h, dropout, **attention_config
        )
        feed_forward_block_ = feed_forwardblock(d_model, d_ff, dropout)
        temp_encoder_block = encoder_block(
            encoder_self_attention_block, feed_forward_block_, d_model, dropout
        )
        encoder_blocks.append(temp_encoder_block)
        
    # create decoder blocks 
    decoder_blocks = []
    for _ in range(N):
        # Decoder self-attention uses the specified attention type
        decoder_self_attention_block = get_attention_mechanism(
            attention_type, d_model, h, dropout, **attention_config
        )
        # Cross-attention typically uses standard attention (can be customized if needed)
        decoder_cross_attention_block = multihead_attentionblock(d_model, h, dropout)
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