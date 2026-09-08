"""
PyTorch Deep Learning Inference Engine.

Loads restored checkpoint state dicts and executes high-throughput
batch inference for tabular neural network models.
"""

import os
from typing import Dict, List, Union
import numpy as np
import torch
from model import PyTorchMLP


class PyTorchInferenceEngine:
    """Production batch inference service for PyTorch neural network checkpoints."""

    def __init__(self, checkpoint_path: str, input_dim: int, hidden_dims: List[int] = [32, 16]) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}'.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PyTorchMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return sigmoid probability predictions for input matrix X."""
        x_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        logits = self.model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary class predictions given decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
