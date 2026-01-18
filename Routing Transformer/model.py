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


class local_attention_block(nn.Module):
    def __init__(self, d_model: int, h: int, window_size: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.window_size = window_size
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        profiler.start()
        batch_size, seq_len, _ = q.shape
        
        query = self.w_q(q).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        
        # Local Attention Mask
        # We want to mask out keys that are further than window_size from the query
        # Create a local mask
        local_mask = (torch.ones(seq_len, seq_len, device=q.device).tril(0).bool() & 
                     torch.ones(seq_len, seq_len, device=q.device).triu(-self.window_size).bool())
        local_mask = local_mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)
        
        if mask is not None:
             local_mask = local_mask & mask
             
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(local_mask == 0, -1e9)
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ value).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        x = self.w_o(x)
        profiler.end("LocalAttention")
        return x


class routing_attention_block(nn.Module):
    def __init__(self, d_model: int, h: int, num_clusters: int, window_size: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.num_clusters = num_clusters # k
        self.window_size = window_size   # For gathering top-k items
        
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Centroids for K-Means (shared across batch, but per head?)
        # Paper says "Centroid parameters are model parameters and are shared across sequences"
        # We need centroids for each head or shared? 
        # "Each attention module considers a clustering of the space"
        # Usually distinct per layer/head.
        self.centroids = nn.Parameter(torch.randn(h, num_clusters, self.d_k))
        # Initialize centroids?
        nn.init.orthogonal_(self.centroids)
        
        self.decay = 0.999 # Exponential Moving Average decay
        
    def forward(self, q, k, v, mask=None):
        profiler.start()
        batch_size, seq_len, _ = q.shape
        
        # Projections
        query = self.w_q(q).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2) # (B, H, L, Dk)
        key = self.w_k(k).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        
        # Normalize for spherical k-means (LayerNorm with no affine)
        # We can just normalize manually L2 norm
        query_norm = F.normalize(query, p=2, dim=-1)
        key_norm = F.normalize(key, p=2, dim=-1)
        # centroids should also be normalized usually, or at least used for comparison
        centroids_norm = F.normalize(self.centroids, p=2, dim=-1) # (H, K, Dk)
        
        # Assign clusters
        # Compute distances (dot product closest to 1)
        # Query: (B, H, L, Dk) @ (H, Dk, K) -> (B, H, L, K)
        centroids_t = centroids_norm.transpose(1, 2) # (H, Dk, K)
        
        # We need to broadcast matrix mult over batch
        # einsum is easy: b h l d, h d k -> b h l k
        sim_q = torch.einsum('bhld,hdk->bhlk', query_norm, centroids_t)
        sim_k = torch.einsum('bhld,hdk->bhlk', key_norm, centroids_t)
        
        # Helper to get cluster assignments
        # In paper, they use top-k or just argmax. "current time-step only attends to context belonging to the same cluster"
        # "Specifically, for every position i, self-attention computes weights for its whole context"
        # "Our strategy first assigns queries and keys to clusters. Then only queries and keys from the same cluster are considered"
        
        # Simple hard routing: Argmax
        cluster_q = sim_q.argmax(dim=-1) # (B, H, L)
        cluster_k = sim_k.argmax(dim=-1) # (B, H, L)
        
        # Update Centroids (EMA) if training
        if self.training:
             with torch.no_grad():
                # We need to mean vectors assigned to each cluster
                # This is efficient with scatter_add or similar
                # Simple implementation Loop over clusters (slow) or use one-hot
                
                # one_hot Q: (B, H, L, K)
                one_hot_q = F.one_hot(cluster_q, num_classes=self.num_clusters).float()
                one_hot_k = F.one_hot(cluster_k, num_classes=self.num_clusters).float()
                
                # Sum vectors: (B, H, L, K) * (B, H, L, Dk) -> aggregate
                # einsum: b h l k, b h l d -> h k d
                sum_q = torch.einsum('bhlk,bhld->hkd', one_hot_q, query.detach())
                sum_k = torch.einsum('bhlk,bhld->hkd', one_hot_k, key.detach())
                
                count_q = one_hot_q.sum(dim=(0, 2)).unsqueeze(-1) + 1e-9 # (H, K, 1)
                count_k = one_hot_k.sum(dim=(0, 2)).unsqueeze(-1) + 1e-9
                
                mean_q = sum_q / count_q
                mean_k = sum_k / count_k
                
                new_centers = (mean_q + mean_k) / 2.0
                
                self.centroids.data = self.decay * self.centroids.data + (1 - self.decay) * new_centers
                
        # Attention Computation
        # We need to gather keys/values that match the query cluster
        # Optimizing this in PyTorch without custom kernels is tricky (padding/masking)
        # Approach:
        # Sort queries and keys by cluster index?
        # Or masked attention with a "Cluster Mask".
        
        # Creating a cluster mask: match if cluster_q[i] == cluster_k[j]
        # shape (B, H, L, L)
        # cluster_q: (B, H, L, 1)
        # cluster_k: (B, H, 1, L)
        cluster_mask = cluster_q.unsqueeze(-1) == cluster_k.unsqueeze(-2) # (B, H, L, L)
        
        if mask is not None:
            cluster_mask = cluster_mask & mask
            
        # Standard Attention with this mask
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply cluster mask
        scores = scores.masked_fill(~cluster_mask, -1e9)
        
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ value).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        x = self.w_o(x)
        
        profiler.end("RoutingAttention")
        return x


class sparse_multihead_attention_block(nn.Module):
    def __init__(self, d_model: int, h: int, num_clusters: int, window_size: int, dropout: float):
        super().__init__()
        # Split heads: half local, half routing
        assert h % 2 == 0, "Heads must be even for split"
        self.h_local = h // 2
        self.h_routing = h // 2
        self.d_model = d_model
        # We reuse the same d_model but split internally? 
        # Usually we want separate projections or run separate blocks and potentialy concat
        
        # Easier: Two separate blocks, each handling half d_model
        self.local_block = local_attention_block(d_model // 2, self.h_local, window_size, dropout)
        self.routing_block = routing_attention_block(d_model // 2, self.h_routing, num_clusters, window_size, dropout)
        
        # Output projection for mixing?
        # The blocks already have output projections (Wo).
        # But we need to project back to d_model size.
        # Actually standard MHA: Input d_model -> split heads -> concat -> Wo -> d_model
        # Here: Input d_model -> 
        #       Split d_model/2 -> Local -> d_model/2
        #       Split d_model/2 -> Routing -> d_model/2
        #       Concat -> d_model
        
        self.proj_mix = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        # We assume q, k, v are (N, L, D)
        # Split features
        d_half = self.d_model // 2
        q_local, q_routing = q[:, :, :d_half], q[:, :, d_half:]
        k_local, k_routing = k[:, :, :d_half], k[:, :, d_half:]
        v_local, v_routing = v[:, :, :d_half], v[:, :, d_half:]
        
        out_local = self.local_block(q_local, k_local, v_local, mask)
        out_routing = self.routing_block(q_routing, k_routing, v_routing, mask)
        
        out = torch.cat([out_local, out_routing], dim=-1)
        out = self.proj_mix(out)
        return out


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
        # Using standard sparse/routing attention for cross might be tricky if lengths differ significantly
        # but generally similar.
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
    
    
class routing_transformer(nn.Module):
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
    
    
def build_routing_transformer(src_vocab_size: int, tgt_vocab_size: int,
                              src_seq_len: int, tgt_seq_len: int,
                              d_model: int = 512, N: int = 6,
                              h: int = 8, dropout: float = 0.1, d_ff: int = 2048,
                              num_clusters: int = 16, window_size: int = 32):

    # create embedding layers 
    src_embed = input_embeddings(d_model, src_vocab_size)
    tgt_embed = input_embeddings(d_model, tgt_vocab_size)
    
    # positional encoding layers 
    src_pos = positional_encoding(d_model, src_seq_len, dropout) 
    tgt_pos = positional_encoding(d_model, tgt_seq_len, dropout)
    
    # create encoder blocks 
    encoder_blocks = []
    for _ in range(N):
        # Use Sparse Multihead Attention (Local + Routing)
        encoder_self_attention_block = sparse_multihead_attention_block(
            d_model, h, num_clusters, window_size, dropout
        )
        feed_forward_block_ = feed_forwardblock(d_model, d_ff, dropout)
        temp_encoder_block = encoder_block(
            encoder_self_attention_block, feed_forward_block_, d_model, dropout
        )
        encoder_blocks.append(temp_encoder_block)
        
    # create decoder blocks 
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = sparse_multihead_attention_block(
            d_model, h, num_clusters, window_size, dropout
        )
        # Cross attention might need to be standard or sparse?
        # Typically global attention is used for cross, or just full attention if seq len is manageable.
        # But if we want consistent Routing Transformer everywhere, we can use sparse.
        decoder_cross_attention_block = sparse_multihead_attention_block(
            d_model, h, num_clusters, window_size, dropout
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
    main_transformer = routing_transformer(
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
