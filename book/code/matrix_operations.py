import numpy as np

# --- Basic matrix construction and properties ---
A = np.array([[4, 3], [6, 3]], dtype=float)
print("A =\n", A)
print("Rank:", np.linalg.matrix_rank(A))          # 2  (full rank)
print("Det:", np.linalg.det(A))                   # -6
print("Trace:", np.trace(A))                      # 7  (sum of diagonals)

# --- Transpose, inverse, and matrix product ---
B = np.array([[1, 2], [3, 4]], dtype=float)
print("A @ B =\n", A @ B)                         # standard product
print("A.T =\n", A.T)                             # transpose
print("A^{-1} =\n", np.linalg.inv(A))            # inverse

# --- Solving Ax = b ---
b = np.array([10, 12], dtype=float)
x = np.linalg.solve(A, b)
print("Solution x:", x)                           # [1. 2.]

# --- Singular Value Decomposition (SVD) ---
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print("Singular values:", s)                      # [8.60, 0.70]
# Reconstruct: A = U @ diag(s) @ Vt
A_reconstructed = U @ np.diag(s) @ Vt
print("Reconstruction error:", np.max(np.abs(A - A_reconstructed)))
