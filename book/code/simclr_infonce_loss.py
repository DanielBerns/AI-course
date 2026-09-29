import numpy as np

# 1. Define vectorized InfoNCE Loss with Temperature Scaling (SimCLR style)
def infonce_loss(z1, z2, temperature=0.1):
    """
    z1, z2: (N, d) L2-normalized embedding projections for 2 augmented views
    """
    N = len(z1)
    # Concatenate both views -> 2N x d matrix
    z = np.vstack([z1, z2])
    
    # Compute 2N x 2N cosine similarity matrix (since vectors are L2-normalized)
    sim_matrix = (z @ z.T) / temperature
    
    # Mask out self-similarity along main diagonal
    np.fill_diagonal(sim_matrix, -1e9)
    
    # Positive pair target indices: row i pairs with (i + N) % (2N)
    labels = np.arange(2 * N)
    labels[:N] += N
    labels[N:] -= N
    
    # Numerically stable Log-Softmax Cross-Entropy Loss
    max_sim = np.max(sim_matrix, axis=1, keepdims=True)
    exp_sim = np.exp(sim_matrix - max_sim)
    log_prob = sim_matrix - max_sim - np.log(np.sum(exp_sim, axis=1, keepdims=True))
    
    loss = -np.mean(log_prob[np.arange(2 * N), labels])
    return loss

# 2. Simulate 2 augmented views of 4 samples in 8-dimensional projection space
rng = np.random.default_rng(42)
N, d = 4, 8
# Base representations
z_base = rng.normal(size=(N, d))
z_base /= np.linalg.norm(z_base, axis=1, keepdims=True)

# Positive views (small random augmentation perturbation)
z1 = z_base + 0.05 * rng.normal(size=(N, d))
z2 = z_base + 0.05 * rng.normal(size=(N, d))

# L2 normalize projection vectors
z1 /= np.linalg.norm(z1, axis=1, keepdims=True)
z2 /= np.linalg.norm(z2, axis=1, keepdims=True)

loss_aligned = infonce_loss(z1, z2, temperature=0.1)

# Unaligned random representations baseline
z_rand = rng.normal(size=(N, d))
z_rand /= np.linalg.norm(z_rand, axis=1, keepdims=True)
loss_unaligned = infonce_loss(z1, z_rand, temperature=0.1)

print(f"InfoNCE Loss (Aligned Positive Pairs):    {loss_aligned:.4f}")
print(f"InfoNCE Loss (Random Unaligned Pairs):   {loss_unaligned:.4f}")
