"""
Project 4: Deep Learning Foundations & Resumable Training Pipeline.

Trains NumPy MLP from scratch, builds PyTorch MLP architecture, and demonstrates
interrupted training recovery via state-dict checkpoint restoration across phases.
"""

import os
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from mlp_from_scratch import MLPFromScratch
from dataset import get_dataloaders
from model import PyTorchMLP, create_keras_mlp_equivalent
from checkpoint_manager import CheckpointManager


def train_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> Tuple[float, float]:
    """Train PyTorch model for one epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x_batch)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += len(x_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    """Evaluate PyTorch model on validation dataset."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x_batch, y_batch in loader:
        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * len(x_batch)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += len(x_batch)

    return total_loss / total, correct / total


def main() -> None:
    print("=" * 70)
    print("PROJECT 4: DEEP LEARNING FOUNDATIONS & RESUMABLE TRAINING PIPELINE")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. NumPy MLP Backpropagation from Scratch
    print("\n[Step 1] Training Pure NumPy MLP with Manual Backpropagation...")
    cancer = load_breast_cancer()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cancer.data)
    y = cancer.target.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    mlp_scratch = MLPFromScratch(input_dim=X_scaled.shape[1], hidden_dim=16, output_dim=1, learning_rate=0.05)
    losses_scratch = mlp_scratch.fit(X_train, y_train, epochs=150)
    print(f" - NumPy MLP Initial Loss: {losses_scratch[0]:.4f} -> Final Loss: {losses_scratch[-1]:.4f}")

    # 2. PyTorch Modular Dataset & Model Creation
    print("\n[Step 2] Building PyTorch Architecture & Printing Keras Equivalent Syntax...")
    train_loader, val_loader = get_dataloaders(X_train, y_train, X_test, y_test, batch_size=32)

    model = PyTorchMLP(input_dim=X_scaled.shape[1], hidden_dims=[32, 16], output_dim=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    print("\n --- Equivalent Keras Syntax ---")
    print(create_keras_mlp_equivalent(input_dim=X_scaled.shape[1], hidden_dims=[32, 16]))

    # 3. Demonstration of Resumable Training (Phase 1 & Phase 2)
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir, file_prefix="mlp_tabular")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print("\n[Step 3a] Phase 1 Training (Epochs 1 -> 4) before Simulated Interruption...")
    best_val_loss = float("inf")

    for epoch in range(1, 5):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = evaluate_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        is_best = va_loss < best_val_loss
        if is_best:
            best_val_loss = va_loss

        ckpt_mgr.save_checkpoint(model, optimizer, epoch, va_loss, history, scheduler, is_best=is_best)
        print(f" Epoch {epoch:2d}/4 | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | Val Acc: {va_acc:.2%}")

    print("\n !!! SIMULATING TRAINING INTERRUPTION (Process Killed / Paused) !!!")

    # Re-initialize clean model & optimizer to prove state restoration
    print("\n[Step 3b] Phase 2 Resuming Training from Checkpoint (Epochs 5 -> 8)...")
    restored_model = PyTorchMLP(input_dim=X_scaled.shape[1], hidden_dims=[32, 16], output_dim=1)
    restored_optimizer = optim.AdamW(restored_model.parameters(), lr=0.01, weight_decay=1e-4)
    restored_scheduler = optim.lr_scheduler.StepLR(restored_optimizer, step_size=4, gamma=0.5)

    start_epoch, best_val_loss, history = ckpt_mgr.load_checkpoint(
        restored_model, restored_optimizer, restored_scheduler
    )

    for epoch in range(start_epoch, 9):
        tr_loss, tr_acc = train_epoch(restored_model, train_loader, criterion, restored_optimizer)
        va_loss, va_acc = evaluate_epoch(restored_model, val_loader, criterion)
        restored_scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        is_best = va_loss < best_val_loss
        if is_best:
            best_val_loss = va_loss

        ckpt_mgr.save_checkpoint(restored_model, restored_optimizer, epoch, va_loss, history, restored_scheduler, is_best=is_best)
        print(f" Epoch {epoch:2d}/8 | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | Val Acc: {va_acc:.2%}")

    # Plot continuous loss history across interruption
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs_axis = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs_axis, history["train_loss"], "o-", color="#e67e22", label="Train Loss")
    ax.plot(epochs_axis, history["val_loss"], "s-", color="#263c72", label="Val Loss")
    ax.axvline(x=4.5, color="#6c349e", linestyle="--", label="Interruption & Resumption Point")
    ax.set_title("Resumable Training Loss Curve Across Phase Interruption", fontsize=12, fontweight="bold", color="#263c72")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "resumable_training_curve.png"), dpi=300)

    print("\n" + "=" * 70)
    print("PROJECT 4 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
