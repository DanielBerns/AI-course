import numpy as np

# Non-linear XOR / concentric circle dataset
rng = np.random.default_rng(42)
X_b = rng.uniform(-2, 2, size=(200, 2))
y_b = ((X_b[:, 0]**2 + X_b[:, 1]**2) < 1.2).astype(int).reshape(-1, 1)

# Initialize weights with He Normalization (scaling by sqrt(2/fan_in))
W1 = rng.normal(0, np.sqrt(2/2), size=(2, 8)); b1 = np.zeros((1, 8))
W2 = rng.normal(0, np.sqrt(2/8), size=(8, 1)); b2 = np.zeros((1, 1))

for epoch in range(500):
    # Forward pass: Input -> Hidden (ReLU) -> Output (Sigmoid)
    z1 = X_b @ W1 + b1; a1 = np.maximum(0, z1)   # ReLU activation
    z2 = a1 @ W2 + b2;  hat_y = 1 / (1 + np.exp(-np.clip(z2, -500, 500)))
    
    # Backward pass (Gradient computation via chain rule)
    dz2 = (hat_y - y_b) / len(X_b)               # dL/dz2 for BCE loss
    dW2 = a1.T @ dz2; db2 = dz2.sum(axis=0, keepdims=True)
    da1 = dz2 @ W2.T;  dz1 = da1 * (z1 > 0)      # ReLU derivative
    dW1 = X_b.T @ dz1; db1 = dz1.sum(axis=0, keepdims=True)
    
    # Gradient descent parameter updates
    W1 -= 0.1 * dW1; b1 -= 0.1 * db1
    W2 -= 0.1 * dW2; b2 -= 0.1 * db2

bce_loss = -np.mean(y_b*np.log(hat_y+1e-7) + (1-y_b)*np.log(1-hat_y+1e-7))
acc = np.mean((hat_y > 0.5) == y_b)
print(f"Epoch 500 | BCE Loss: {bce_loss:.4f} | Accuracy: {acc:.3f}")
