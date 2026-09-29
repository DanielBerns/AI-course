import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Generate non-linear 10D synthetic dataset with 2D intrinsic manifold
rng = np.random.default_rng(42)
t = rng.uniform(-np.pi, np.pi, size=600)
# Non-linear 2D manifold embedded in 10D ambient space
X_raw = np.column_stack([np.sin(t), np.cos(t), t**2, np.sin(2*t), np.cos(2*t),
                         t, np.exp(t/5), np.sin(3*t), np.cos(3*t), t**3/10])
y_labels = (np.sin(t) + np.cos(2*t) > 0).astype(int)

# 2. Define 10D -> 4D -> 2D -> 4D -> 10D Deep Autoencoder in NumPy
W_e1 = rng.normal(0, np.sqrt(2/10), size=(10, 4)); b_e1 = np.zeros((1, 4))
W_e2 = rng.normal(0, np.sqrt(2/4), size=(4, 2));   b_e2 = np.zeros((1, 2))  # Bottleneck z in R^2
W_d1 = rng.normal(0, np.sqrt(2/2), size=(2, 4));   b_d1 = np.zeros((1, 4))
W_d2 = rng.normal(0, np.sqrt(2/4), size=(4, 10));  b_d2 = np.zeros((1, 10))

# 3. Train Autoencoder via SGD with MSE Reconstruction Loss
lr = 0.01
for epoch in range(400):
    # Forward Pass
    h1 = np.maximum(0, X_raw @ W_e1 + b_e1)           # Encoder L1 (ReLU)
    z  = h1 @ W_e2 + b_e2                             # Bottleneck representation z in R^2
    h2 = np.maximum(0, z @ W_d1 + b_d1)               # Decoder L1 (ReLU)
    X_hat = h2 @ W_d2 + b_d2                          # Reconstructed output in R^10
    
    # Backward Pass (Chain rule gradients)
    dX_hat = 2 * (X_hat - X_raw) / len(X_raw)
    dW_d2 = h2.T @ dX_hat;         db_d2 = dX_hat.sum(axis=0, keepdims=True)
    dh2   = (dX_hat @ W_d2.T) * (h2 > 0)
    dW_d1 = z.T @ dh2;            db_d1 = dh2.sum(axis=0, keepdims=True)
    dz    = dh2 @ W_d1.T
    dW_e2 = h1.T @ dz;             db_e2 = dz.sum(axis=0, keepdims=True)
    dh1   = (dz @ W_e2.T) * (h1 > 0)
    dW_e1 = X_raw.T @ dh1;        db_e1 = dh1.sum(axis=0, keepdims=True)
    
    # Parameter Updates
    W_e1 -= lr * dW_e1; b_e1 -= lr * db_e1; W_e2 -= lr * dW_e2; b_e2 -= lr * db_e2
    W_d1 -= lr * dW_d1; b_d1 -= lr * db_d1; W_d2 -= lr * dW_d2; b_d2 -= lr * db_d2

# 4. Extract 2D latent representation z and evaluate linear separability
z_latent = np.maximum(0, X_raw[:450] @ W_e1 + b_e1) @ W_e2 + b_e2
z_test   = np.maximum(0, X_raw[450:] @ W_e1 + b_e1) @ W_e2 + b_e2

clf_raw = LogisticRegression().fit(X_raw[:450], y_labels[:450])
clf_z   = LogisticRegression().fit(z_latent, y_labels[:450])

print(f"Epoch 400 Reconstruction MSE: {np.mean((X_hat - X_raw)**2):.4f}")
print(f"Linear Classifier Acc (Raw 10D Features):   {clf_raw.score(X_raw[450:], y_labels[450:]):.3f}")
print(f"Linear Classifier Acc (Learned 2D Latent z): {clf_z.score(z_test, y_labels[450:]):.3f}")
