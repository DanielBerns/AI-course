"""
Resumable Training & Checkpoint Manager Module.

Saves and restores complete PyTorch model, optimizer, scheduler, epoch, and metric history
to guarantee seamless pause and resume capabilities for deep learning pipelines.
"""

import os
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


class CheckpointManager:
    """Manages PyTorch state dict serialization and training restoration."""

    def __init__(self, checkpoint_dir: str, file_prefix: str = "checkpoint") -> None:
        self.checkpoint_dir = checkpoint_dir
        self.file_prefix = file_prefix
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        val_loss: float,
        train_history: Dict[str, Any],
        scheduler: Optional[Any] = None,
        is_best: bool = False,
    ) -> str:
        """Save full training state dictionary to disk."""
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_loss": val_loss,
            "train_history": train_history,
        }

        latest_path = os.path.join(self.checkpoint_dir, f"{self.file_prefix}_latest.pt")
        torch.save(checkpoint_state, latest_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{self.file_prefix}_best.pt")
            torch.save(checkpoint_state, best_path)

        return latest_path

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        load_best: bool = False,
    ) -> Tuple[int, float, Dict[str, Any]]:
        """Restore model, optimizer, scheduler, epoch, and history states from checkpoint."""
        file_name = f"{self.file_prefix}_best.pt" if load_best else f"{self.file_prefix}_latest.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, file_name)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint file found at '{checkpoint_path}'.")

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        train_history = checkpoint["train_history"]

        print(f" -> Successfully restored checkpoint from '{checkpoint_path}'. Resuming at Epoch {start_epoch}.")
        return start_epoch, best_val_loss, train_history

    def exists(self, load_best: bool = False) -> bool:
        """Check if checkpoint file exists on disk."""
        file_name = f"{self.file_prefix}_best.pt" if load_best else f"{self.file_prefix}_latest.pt"
        return os.path.exists(os.path.join(self.checkpoint_dir, file_name))
