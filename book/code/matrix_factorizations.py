import numpy as np
from scipy.linalg import lu, qr, cholesky

# --- 1. LU Decomposition (A = P L U) ---
A_square = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]], dtype=float)
P, L, U = lu(A_square)
print("1. LU Factorization:")
print("Reconstruction error:", np.max(np.abs(P @ L @ U - A_square)))
assert np.allclose(P @ L @ U, A_square)

# --- 2. QR Decomposition (A = Q R) ---
A_rect = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
Q, R = qr(A_rect, mode='economic')
print("\n2. QR Factorization:")
print("Q shape:", Q.shape, "R shape:", R.shape)
print("Orthogonality Q^T Q:\n", np.round(Q.T @ Q, 6))
print("Reconstruction error:", np.max(np.abs(Q @ R - A_rect)))
assert np.allclose(Q @ R, A_rect)

# --- 3. Cholesky Decomposition (A = L L^T) for Positive Definite Matrix ---
A_spd = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=float)
L_chol = cholesky(A_spd, lower=True)
print("\n3. Cholesky Factorization:")
print("Reconstruction error:", np.max(np.abs(L_chol @ L_chol.T - A_spd)))
assert np.allclose(L_chol @ L_chol.T, A_spd)

# --- 4. Eigendecomposition (A = V Lambda V^T) for Symmetric Matrix ---
eigenvalues, V = np.linalg.eigh(A_spd)
A_eig_rec = V @ np.diag(eigenvalues) @ V.T
print("\n4. Eigendecomposition:")
print("Reconstruction error:", np.max(np.abs(A_eig_rec - A_spd)))
assert np.allclose(A_eig_rec, A_spd)

# --- 5. Singular Value Decomposition (A = U Sigma V^T) ---
U_svd, s, Vt_svd = np.linalg.svd(A_rect, full_matrices=False)
A_svd_rec = U_svd @ np.diag(s) @ Vt_svd
print("\n5. Singular Value Decomposition (SVD):")
print("Singular values:", s)
print("Reconstruction error:", np.max(np.abs(A_svd_rec - A_rect)))
assert np.allclose(A_svd_rec, A_rect)
