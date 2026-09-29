import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)  # Softmax
    return weights @ V, weights

# Demonstrate with 4 tokens of dimension 8
seq_len, d_k = 4, 8
Q = K = V = np.random.default_rng(0).standard_normal((seq_len, d_k))
causal_mask = np.tril(np.ones((seq_len, seq_len)))  # Lower-triangular mask
out, attn = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print("Attention Output Shape:", out.shape)
print("Causal Attention Weights Matrix:\n", np.round(attn, 3))
