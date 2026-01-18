# Routing Transformer Implementation Report

## Overview
This report documents the implementation of the **Routing Transformer**, a sparse attention model designed to efficiently handle long sequences by reducing the quadratic complexity of self-attention to $O(n^{1.5}d)$.

The implementation mimics the structure of your existing `Transformer` directory but introduces the novel **Routing Attention** and **Local Attention** mechanisms described in the paper *"Efficient Content-Based Sparse Attention with Routing Transformers"*.

## Directory Structure
The project is located in `Routing Transformer/` and contains:

- **`model.py`**: The core model definition.
- **`train_test.py`**: A training script adapted for the Routing Transformer.
- **`gpu_profiler.py`**: Utility for profiling (copied from the original project).

## Key Components

### 1. Routing Attention Block (`routing_attention_block`)
This is the heart of the paper's contribution. Instead of attending to all tokens, the model clusters queries and keys using **Spherical k-means**.
- **Clustering**: Queries and Keys are projected to a unit sphere. We use dot-product similarity (via `torch.einsum`) to find the nearest cluster centroid for each token.
- **Routing**: Tokens are only allowed to attend to other tokens within the same cluster. This is implemented via a mask derived from cluster assignments.
- **Centroid Update**: During training, cluster centroids are updated using an **Exponential Moving Average (EMA)** of the vectors assigned to them, ensuring the clusters evolve with the data.

### 2. Local Attention Block (`local_attention_block`)
To capture immediate local context (which global routing might miss), we implemented a sliding window attention.
- **Mechanism**: A token at position $i$ can only attend to tokens in the range $[i - \text{window\_size}, i]$.
- **Implementation**: Uses a causal mask combined with a band mask (lower triangular matrix with a band limit).

### 3. Sparse Multi-Head Attention (`sparse_multihead_attention_block`)
The paper suggests combining both approaches. We split the attention heads into two groups:
- **Local Heads**: $H/2$ heads perform Local Attention.
- **Routing Heads**: $H/2$ heads perform Routing Attention.
The outputs are concatenated and projected back to the model dimension.

## Hyperparameters
The `train_test.py` script includes new configuration options:
- `NUM_CLUSTERS` (Default: 8): The number of clusters ($k$) for routing.
- `WINDOW_SIZE` (Default: 10): The context window size for local attention.

## How to Run
To train the model on the sample dataset (Opus Books en-de):

```bash
cd "/home/fatin-ishrak-arian/Ml Project(Code)/Machine-Learning/"
python "Routing Transformer/train_test.py"
```

## Comparisons to Standard Transformer
- **Complexity**: The standard Transformer computes attention over $N \times N$ pairs. The Routing Transformer computes it over roughly $N \times (N/k)$ pairs (plus local window), significantly reducing memory usage for long sequences.
- **Drop-in Replacement**: The `routing_transformer` class has the same interface (`encode`, `decode`, `project`) as the standard `transformer`, making it easy to swap into existing pipelines.
