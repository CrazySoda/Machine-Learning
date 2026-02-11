"""
BigBird Sparse Attention Mechanism

This module implements BigBird's sparse attention pattern with three components:
1. Local Window Attention - Sliding window for local context
2. Global Attention - Selected tokens attend to all positions
3. Random Attention - Random connections for information diversity

Reference: "Big Bird: Transformers for Longer Sequences" (Zaheer et al., 2020)
"""

import torch
import torch.nn as nn
import math
from gpu_profiler import GPUProfiler

profiler = GPUProfiler()


class bigbird_attentionblock(nn.Module):
    """
    BigBird Sparse Attention with three components:
    1. Local window attention (sliding window)
    2. Global attention (selected tokens attend to all)
    3. Random attention (random token connections)
    """
    def __init__(self, d_model: int, h: int, dropout: float, 
                 window_size: int = 3, num_global_tokens: int = 2, num_random_tokens: int = 3):
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
        
        # BigBird specific parameters
        self.window_size = window_size  # Local attention window size
        self.num_global_tokens = num_global_tokens  # Number of global tokens
        self.num_random_tokens = num_random_tokens  # Number of random connections per token
    
    def create_bigbird_mask(self, seq_len, device):
        """
        Creates BigBird sparse attention mask combining:
        - Local window attention
        - Global attention 
        - Random attention
        """
        # Start with all zeros (no attention)
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        # 1. LOCAL WINDOW ATTENTION: Each token attends to neighbors within window
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 1
        
        # 2. GLOBAL ATTENTION: First num_global_tokens can attend to all and all attend to them
        if self.num_global_tokens > 0:
            # Global tokens attend to everything
            mask[:self.num_global_tokens, :] = 1
            # Everything attends to global tokens
            mask[:, :self.num_global_tokens] = 1
        
        # 3. RANDOM ATTENTION: Add random connections for diversity
        for i in range(seq_len):
            # Skip if this is a global token (already attends to all)
            if i < self.num_global_tokens:
                continue
                
            # Select random positions to attend to
            random_positions = torch.randperm(seq_len)[:self.num_random_tokens]
            mask[i, random_positions] = 1
        
        return mask
    
    @staticmethod
    def bigbird_attention(query, key, value, bigbird_mask, original_mask, dropout: nn.Dropout):
        """
        Compute attention with BigBird sparse pattern
        """
        d_k = query.shape[-1]
        # Standard attention scores
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply BigBird sparse mask (sets non-attended positions to -inf)
        attention_scores = attention_scores.masked_fill(bigbird_mask == 0, -1e9)
        
        # part of masked attention (apply original mask if provided, e.g., padding mask)
        if original_mask is not None:
            attention_scores = attention_scores.masked_fill(original_mask == 0, -1e9)
            
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
        
        # Create BigBird sparse attention mask
        seq_len = query.shape[2]
        bigbird_mask = self.create_bigbird_mask(seq_len, query.device)
        # Expand for batch and heads: (batch, h, seq_len, seq_len)
        bigbird_mask = bigbird_mask.unsqueeze(0).unsqueeze(0).expand(query.shape[0], self.h, -1, -1)
        
        x, self.attention_scores = bigbird_attentionblock.bigbird_attention(
            query, key, value, bigbird_mask, mask, self.dropout
        )
        
        # (batch, h, seq_len, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        x = self.w_o(x)
        
        profiler.end("BigBirdAttention")
        return x