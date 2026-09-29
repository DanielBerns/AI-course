import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def softmax(Z, temp=1.0):
    # Subtract max logit per row for numerical stability
    Z_scaled = Z / temp
    exp_Z = np.exp(Z_scaled - np.max(Z_scaled, axis=-1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)

# 3 samples, 4 classes (logits Z in R^(3 x 4))
logits = np.array([[2.0, 1.0, 0.1, 3.0],
                   [0.5, 4.0, 1.2, 0.0],
                   [-1.0, 0.0, 5.0, 0.2]])

probs = softmax(logits, temp=1.0)
y_true = np.array([3, 1, 2]) # True targets for each sample

# Categorical Cross-Entropy Loss: -log(p_true)
nll = -np.log(probs[np.arange(len(y_true)), y_true])
print("Probabilities:\n", probs.round(3))
print(f"Per-sample loss: {nll.round(3)} | Mean: {nll.mean():.3f}")
