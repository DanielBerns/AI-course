"""
Principal Component Analysis (PCA) Implementation from Scratch.

Implements PCA using covariance matrix eigen-decomposition and SVD,
calculating explained variance ratios and low-rank reconstruction error.
"""

from typing import Tuple
import numpy as np


class PCAFromScratch:
    """Principal Component Analysis using NumPy linear algebra primitives."""

    def __init__(self, n_components: int, method: str = "svd") -> None:
        self.n_components = n_components
        self.method = method.lower()
        self.components_: np.ndarray = np.array([])
        self.explained_variance_: np.ndarray = np.array([])
        self.explained_variance_ratio_: np.ndarray = np.array([])
        self.mean_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray) -> "PCAFromScratch":
        """Compute principal components and explained variance from feature matrix X."""
        n_samples, n_features = X.shape

        # 1. Mean-center the dataset
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        if self.method == "eigh":
            # 2a. Compute covariance matrix: C = (X_centered^T @ X_centered) / (n - 1)
            cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort eigenvalues and eigenvectors in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            self.explained_variance_ = eigenvalues[: self.n_components]
            self.components_ = eigenvectors[:, : self.n_components].T

        elif self.method == "svd":
            # 2b. Compute SVD: X_centered = U @ S @ Vt
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            eigenvalues = (S**2) / (n_samples - 1)

            self.explained_variance_ = eigenvalues[: self.n_components]
            self.components_ = Vt[: self.n_components]
        else:
            raise ValueError(f"Unknown method '{self.method}'. Supported methods: 'svd', 'eigh'.")

        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto top principal components."""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit model and return projected low-dimensional data."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """Reconstruct original high-dimensional representation from low-rank projection."""
        return (X_reduced @ self.components_) + self.mean_

    def compute_reconstruction_error(self, X: np.ndarray) -> float:
        """Compute Mean Squared Error (MSE) between original and reconstructed dataset."""
        X_reduced = self.transform(X)
        X_reconstructed = self.inverse_transform(X_reduced)
        return float(np.mean((X - X_reconstructed) ** 2))
