import numpy as np

# --- Eigen-decomposition of a symmetric matrix ---
A = np.array([[3, 1], [1, 3]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eigh(A)   # eigh: symmetric solver
# eigenvalues are returned in ASCENDING order by eigh
print("Eigenvalues:", eigenvalues)               # [2., 4.]
print("Eigenvectors (columns):\n", eigenvectors) # orthonormal columns

# --- Verify the defining equation A v = lambda v ---
for i, lam in enumerate(eigenvalues):
    v = eigenvectors[:, i]
    lhs = A @ v
    rhs = lam * v
    print(f"lambda={lam:.1f}: A@v={lhs}, lambda*v={rhs}",
          "  OK" if np.allclose(lhs, rhs) else "  MISMATCH")

# --- Spectral decomposition: A = V diag(lambda) V^T ---
V = eigenvectors
A_reconstructed = V @ np.diag(eigenvalues) @ V.T
print("Reconstruction error:", np.max(np.abs(A - A_reconstructed)))

# --- Rayleigh quotient: measures variance along direction v ---
v_test = np.array([1, 0], dtype=float)          # x-axis direction
rayleigh = (v_test @ A @ v_test) / (v_test @ v_test)
print("Rayleigh quotient along x-axis:", rayleigh)  # = 3
# Maximum Rayleigh quotient equals the largest eigenvalue (4)
