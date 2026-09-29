import numpy as np

def multi_head_attention(X, num_heads=4, d_model=64):
    B, N, _ = X.shape
    d_k = d_model // num_heads
    rng = np.random.default_rng(42)
    W_Q, W_K, W_V = [rng.normal(0, 0.1, size=(d_model, d_model)) for _ in range(3)]
    W_O = rng.normal(0, 0.1, size=(d_model, d_model))
    
    # Linear projection & reshape into (B, num_heads, N, d_k)
    Q = (X @ W_Q).reshape(B, N, num_heads, d_k).transpose(0, 2, 1, 3)
    K = (X @ W_K).reshape(B, N, num_heads, d_k).transpose(0, 2, 1, 3)
    V = (X @ W_V).reshape(B, N, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # Parallel dot-product attention per head
    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    
    # Concatenate heads & final linear projection -> (B, N, d_model)
    heads_out = (weights @ V).transpose(0, 2, 1, 3).reshape(B, N, d_model)
    return heads_out @ W_O

X_in = np.random.default_rng(42).normal(size=(2, 8, 64))  # Batch 2, Seq 8, Dim 64
mha_out = multi_head_attention(X_in, num_heads=4, d_model=64)
print("Multi-Head Attention Output Shape:", mha_out.shape)  # (2, 8, 64)
