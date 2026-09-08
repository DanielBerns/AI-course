"""
Multi-Layer Perceptron (MLP) Implemented from Scratch in Pure NumPy.

Implements forward and backward propagation using exact matrix chain-rule calculus
and non-linear activation functions (ReLU, Sigmoid).
"""

from typing import List, Tuple
import numpy as np


class MLPFromScratch:
    """2-Layer Multi-Layer Perceptron with explicit Backpropagation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        np.random.seed(seed)
        self.learning_rate = learning_rate

        # He / Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        # Cache for backpropagation
        self.cache: dict = {}

    def _relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def _relu_derivative(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute forward pass and store activations in cache."""
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self._sigmoid(z2)

        self.cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return a2

    def backward(self, y_true: np.ndarray) -> float:
        """Compute loss and backpropagate gradients layer-by-layer."""
        m = y_true.shape[0]
        X = self.cache["X"]
        a1 = self.cache["a1"]
        z1 = self.cache["z1"]
        a2 = self.cache["a2"]

        # Binary Cross-Entropy Loss
        eps = 1e-12
        loss = float(-np.mean(y_true * np.log(a2 + eps) + (1 - y_true) * np.log(1 - a2 + eps)))

        # Gradient of loss w.r.t z2: delta2 = a2 - y
        dz2 = (a2 - y_true) / m
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Gradient of loss w.r.t z1: delta1 = (dz2 @ W2^T) * relu'(z1)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._relu_derivative(z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Gradient Descent parameter updates
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

        return loss

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200) -> List[float]:
        """Train model over specified epochs and return loss history."""
        loss_history = []
        for epoch in range(epochs):
            preds = self.forward(X)
            loss = self.backward(y)
            loss_history.append(loss)
        return loss_history
