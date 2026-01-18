# Grouped Query Attention (GQA) Report

## Overview
Grouped-Query Attention (GQA) is an interpolation between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). It was proposed to achieve a better trade-off between inference speed (specifically memory bandwidth loading keys/values) and model quality.

- **MHA (Multi-Head Attention)**: $H$ query heads, $H$ key heads, $H$ value heads. High quality, slow inference (large KV cache).
- **MQA (Multi-Query Attention)**: $H$ query heads, $1$ key head, $1$ value head. Lower quality, fast inference.
- **GQA (Grouped-Query Attention)**: $H$ query heads, $G$ groups. Each group has $1$ key head and $1$ value head (total $G$ KV heads).

## Implementation Details
The implementation follows the standard Transformer architecture with a modified Attention Block.

### Key Logic: `GroupedQueryAttentionBlock`
1. **Inputs**: `d_model`, `h` (heads), `num_groups` (G).
2. **Projections**:
   - Query ($Q$): Projected to $(B, L, H, D_k)$.
   - Key ($K$), Value ($V$): Projected to $(B, L, G, D_k)$.
3. **Broadcasting**:
   - To compute attention scores ($QK^T$), we broadcast the $G$ Key/Value heads to match the $H$ Query heads. 
   - Each Key head is repeated $H/G$ times.
4. **Computation**: Standard Scaled Dot-Product Attention is applied after broadcasting.

## Advantages
- **Memory Efficiency**: Reduces the size of the KV cache by a factor of $H/G$ compared to MHA.
- **Performance**: Retains most of the quality of MHA while approaching the speed of MQA.

## Verification
The implementation can be verified by setting:
- `num_groups = h`: Should behave identical to MHA.
- `num_groups = 1`: Should behave identical to MQA.
