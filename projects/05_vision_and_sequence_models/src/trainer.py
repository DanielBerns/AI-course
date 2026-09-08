"""
Unified PyTorch Trainer & Evaluation Engine.

Provides reusable epoch training, validation loops, learning rate scheduling,
and metric tracking for Computer Vision and Sequential models.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class DeepModelTrainer:
    """Trainer engine for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: object = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)

    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x_batch)

            if logits.shape[1] == 1:
                loss = self.criterion(logits, y_batch.float().unsqueeze(1))
                preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
            else:
                loss = self.criterion(logits, y_batch.long())
                preds = torch.argmax(logits, dim=1)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(x_batch)
            correct += (preds == y_batch).sum().item()
            total += len(x_batch)

        if self.scheduler:
            self.scheduler.step()

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model performance."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            logits = self.model(x_batch)

            if logits.shape[1] == 1:
                loss = self.criterion(logits, y_batch.float().unsqueeze(1))
                preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
            else:
                loss = self.criterion(logits, y_batch.long())
                preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * len(x_batch)
            correct += (preds == y_batch).sum().item()
            total += len(x_batch)

        return total_loss / total, correct / total
