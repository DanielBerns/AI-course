import numpy as np

x_raw = np.array([1.5, 2.0, 3.5, 4.0, 5.5])
y = np.array([2.1, 2.9, 6.1, 8.2, 14.5])

# 1. Prepend bias column of 1s -> X in R^(m x 2)
X_linear = np.column_stack([np.ones(len(x_raw)), x_raw])

# 2. Basis expansion: Polynomial features [1, x, x^2, x^3]
X_poly = np.column_stack([x_raw**p for p in range(4)])

# Closed-form OLS via QR decomposition (R beta = Q^T y) & SVD (avoiding inv(X^T X)):
Q, R = np.linalg.qr(X_poly)
beta_poly_qr = np.linalg.solve(R, Q.T @ y)
beta_poly_svd = np.linalg.lstsq(X_poly, y, rcond=None)[0]

print(f"Linear X shape:   {X_linear.shape} | beta: {np.linalg.lstsq(X_linear, y, rcond=None)[0].round(2)}")
print(f"Poly X (QR):      {X_poly.shape} | beta: {beta_poly_qr.round(2)}")
print(f"Poly X (SVD):     {X_poly.shape} | beta: {beta_poly_svd.round(2)}")
