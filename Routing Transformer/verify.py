import torch
import sys
import os

# Add directory to path
sys.path.append("/workspaces/Machine-Learning/Routing Transformer")

try:
    from model import build_routing_transformer
    print("Successfully imported build_routing_transformer")
    
    # Configuration
    src_vocab_size = 100
    tgt_vocab_size = 100
    src_seq_len = 32
    tgt_seq_len = 32
    d_model = 64
    N = 2
    h = 4
    
    model = build_routing_transformer(
        src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len,
        d_model=d_model, N=N, h=h, num_clusters=4, window_size=4
    )
    
    print("Model built successfully")
    
    # Dummy Input
    src = torch.randint(0, src_vocab_size, (2, src_seq_len)) # Batch 2
    tgt = torch.randint(0, tgt_vocab_size, (2, tgt_seq_len))
    src_mask = None
    tgt_mask = None # Usually causal mask for decoder
    
    # Forward Pass
    encoder_output = model.encode(src, src_mask)
    print(f"Encoder Output Shape: {encoder_output.shape}")
    
    output = model.decode(encoder_output, src_mask, tgt, tgt_mask)
    print(f"Decoder Output Shape: {output.shape}")
    
    projected = model.project(output)
    print(f"Projected Shape: {projected.shape}")
    
    print("Forward pass successful")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

