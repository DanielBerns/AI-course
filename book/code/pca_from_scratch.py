import numpy as np

# -----------------------------------------------------------------------
# Worked example: PCA on a 2-D dataset with a known correlation structure
# -----------------------------------------------------------------------
rng = np.random.default_rng(42)

# Generate 200 points: strong positive correlation between x1 and x2
mean = np.array([2.0, 3.0])
cov  = np.array([[4.0, 3.2],   # Var(x1)=4, Cov(x1,x2)=3.2
                 [3.2, 4.0]])  # Cov(x1,x2)=3.2, Var(x2)=4
X = rng.multivariate_normal(mean, cov, size=200)

# Step 1 - Centre the data (subtract column means)
X_c = X - X.mean(axis=0)
print("Centred mean (should be ~0):", X_c.mean(axis=0).round(6))

# Step 2 - Compute the (unbiased) sample covariance matrix
n, d = X_c.shape                          # 200 samples, 2 features
Sigma = (X_c.T @ X_c) / (n - 1)          # d x d symmetric matrix
print("Sample covariance:\n", Sigma.round(3))

# Step 3 - Eigen-decompose Sigma (eigh returns ascending order)
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
# Sort descending: largest variance first
idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigenvalues:", eigenvalues.round(3))
print("Explained variance ratio:", (eigenvalues / eigenvalues.sum()).round(3))
print("PC1 direction:", eigenvectors[:, 0].round(3))
print("PC2 direction:", eigenvectors[:, 1].round(3))

# Step 4 - Project onto top-k principal components
k = 1                                     # retain 1 component
W = eigenvectors[:, :k]                   # projection matrix (d x k)
X_proj = X_c @ W                          # (n x k) scores
print(f"\nProjected data shape (n x k): {X_proj.shape}")

# Step 5 - Reconstruct and measure compression error

X_recon = X_proj @ W.T                    # back to d dimensions
recon_error = np.mean((X_c - X_recon)**2)
print(f"Mean squared reconstruction error (k={k}): {recon_error:.4f}")
print(f"  (equals discarded eigenvalue: {eigenvalues[k]:.4f})")

# -----------------------------------------------------------------------
# Scree plot values (for manual inspection)
# -----------------------------------------------------------------------
print("\nScree table:")
for i, ev in enumerate(eigenvalues):
    print(f"  PC{i+1}: eigenvalue={ev:.3f},"
          f" var explained={ev/eigenvalues.sum()*100:.1f}%")
