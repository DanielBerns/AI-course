"""
PyTorch Custom Dataset & DataLoader Module.

Encapsulates tabular data array primitives into torch.utils.data.Dataset
and builds training, validation, and test DataLoaders.
"""

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    """Custom PyTorch Dataset for tabular classification."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        self.y = y_tensor.unsqueeze(1) if y_tensor.ndim == 1 else y_tensor

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Construct PyTorch DataLoader instances for training and validation splits."""
    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
