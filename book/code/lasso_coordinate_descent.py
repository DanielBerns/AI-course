import numpy as np

# Synthetic dataset with collinearity & sparse ground truth
rng = np.random.default_rng(42)
m, n = 100, 5
X = rng.normal(size=(m, n))
X[:, 4] = X[:, 0] + X[:, 1] + 1e-4 * rng.normal(size=m) # collinear feature
true_w = np.array([3.0, -2.5, 1.5, 0.0, 0.0])
y = X @ true_w + 0.2 * rng.normal(size=m)

# 1. Numerically stable OLS Normal Equations via QR & SVD
Q, R = np.linalg.qr(X)
beta_qr = np.linalg.solve(R, Q.T @ y)             # QR decomposition solver
beta_svd = np.linalg.lstsq(X, y, rcond=None)[0]   # SVD solver

# 2. Lasso via Coordinate Descent with Soft-Thresholding
def soft_threshold(z, gamma):
    return np.sign(z) * np.maximum(0, np.abs(z) - gamma)

def lasso_cd(X, y, alpha, max_iter=500):
    m, n = X.shape
    w = np.zeros(n)
    norm_x2 = np.sum(X**2, axis=0)
    for _ in range(max_iter):
        for j in range(n):
            r_j = y - (X @ w - X[:, j] * w[j])     # partial residual
            rho_j = X[:, j] @ r_j
            w[j] = soft_threshold(rho_j, m * alpha) / norm_x2[j]
    return w

w_lasso_np = lasso_cd(X, y, alpha=0.1)

print("True weights: ", true_w)
print("OLS QR:       ", beta_qr.round(2))          # unstable on collinear feature
print("NumPy Lasso:  ", w_lasso_np.round(2))       # zeroes noise & collinear weights
