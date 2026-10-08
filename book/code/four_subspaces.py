import numpy as np
from scipy.linalg import null_space

# --- Define a rectangular matrix A (3x4) of rank 2 ---
A = np.array([
    [1, 2, 3, 4],
    [2, 4, 7, 10],
    [3, 6, 10, 14]
], dtype=float)

m, n = A.shape
r = np.linalg.matrix_rank(A)
print(f"Matrix A ({m}x{n}), Rank r = {r}")

# 1. Column Space C(A) in R^m and Left Nullspace N(A^T) in R^m
U, s, Vt = np.linalg.svd(A, full_matrices=True)
col_space = U[:, :r]             # Basis for C(A)
left_null_space = U[:, r:]       # Basis for N(A^T)

# 2. Row Space C(A^T) in R^n and Nullspace N(A) in R^n
row_space = Vt[:r, :].T          # Basis for C(A^T)
null_sp = Vt[r:, :].T            # Basis for N(A)

print(f"1. Column space C(A) dim = {col_space.shape[1]}")
print(f"2. Nullspace N(A) dim = {null_sp.shape[1]}")
print(f"3. Row space C(A^T) dim = {row_space.shape[1]}")
print(f"4. Left Nullspace N(A^T) dim = {left_null_space.shape[1]}")

# --- Verify Rank-Nullity Theorem: dim(C(A^T)) + dim(N(A)) = n ---
assert col_space.shape[1] + null_sp.shape[1] == n
print(f"Rank-Nullity Theorem verified: r ({r}) + (n-r) ({n - r}) = n ({n})")

# --- Verify Orthogonality: C(A^T) perp N(A) in R^n ---
ortho_check_1 = np.max(np.abs(row_space.T @ null_sp))
print(f"Orthogonality check C(A^T) perp N(A) max inner product: {ortho_check_1:.2e}")
assert ortho_check_1 < 1e-12

# --- Verify Orthogonality: C(A) perp N(A^T) in R^m ---
ortho_check_2 = np.max(np.abs(col_space.T @ left_null_space))
print(f"Orthogonality check C(A) perp N(A^T) max inner product: {ortho_check_2:.2e}")
assert ortho_check_2 < 1e-12

print("Strang's Four Fundamental Subspaces successfully verified!")
