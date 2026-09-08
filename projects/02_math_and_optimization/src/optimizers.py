"""
Optimization Algorithms Implemented from Scratch.

Implements Vanilla Gradient Descent, SGD with Momentum, RMSprop, and Adam
in pure NumPy with trajectory tracking for loss landscape analysis.
"""

from typing import Callable, List, Tuple
import numpy as np


class GradientDescent:
    """Standard (Vanilla) Gradient Descent optimizer."""

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return w - self.lr * grad


class SGDMomentum:
    """Stochastic Gradient Descent with Momentum optimizer."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self.v: np.ndarray = np.array([])

    def step(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.v.size == 0:
            self.v = np.zeros_like(w)
        self.v = self.momentum * self.v + self.lr * grad
        return w - self.v


class RMSprop:
    """RMSprop optimizer with exponential moving average of squared gradients."""

    def __init__(self, lr: float = 0.01, beta: float = 0.99, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v: np.ndarray = np.array([])

    def step(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.v.size == 0:
            self.v = np.zeros_like(w)
        self.v = self.beta * self.v + (1 - self.beta) * (grad**2)
        return w - (self.lr / (np.sqrt(self.v) + self.eps)) * grad


class Adam:
    """Adaptive Moment Estimation (Adam) optimizer."""

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: np.ndarray = np.array([])
        self.v: np.ndarray = np.array([])
        self.t: int = 0

    def step(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m.size == 0:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        self.t += 1
        # Update biased 1st and 2nd moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        # Compute bias-corrected estimates
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        return w - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat


def optimize_trajectory(
    objective_func: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    init_w: np.ndarray,
    optimizer: object,
    steps: int = 100,
) -> Tuple[np.ndarray, List[float]]:
    """Run optimization loop and return parameter trajectory and loss history."""
    trajectory = [init_w.copy()]
    losses = [objective_func(init_w)]
    w = init_w.copy()

    for _ in range(steps):
        grad = grad_func(w)
        w = optimizer.step(w, grad)
        trajectory.append(w.copy())
        losses.append(objective_func(w))

    return np.array(trajectory), losses
