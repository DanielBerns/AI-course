import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend suitable for server environments
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """
    Plots training and validation loss, accuracy, and learning rate curves.
    
    Args:
        history: Dict containing 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr' lists.
        save_path: Output file path for PNG image.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Apply style BEFORE creating the figure so it takes effect
    if 'seaborn-v0_8-whitegrid' in plt.style.available:
        plt.style.use('seaborn-v0_8-whitegrid')
    else:
        plt.style.use('default')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Loss
    if "train_loss" in history:
        axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#4F46E5", linewidth=2.5, marker="o")
    if "val_loss" in history and len(history["val_loss"]) > 0:
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#EF4444", linewidth=2.5, linestyle="--", marker="s")
    axes[0].set_title("Loss Curves", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Accuracy
    if "train_acc" in history:
        axes[1].plot(epochs, history["train_acc"], label="Train Accuracy", color="#10B981", linewidth=2.5, marker="o")
    if "val_acc" in history and len(history["val_acc"]) > 0:
        axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="#F59E0B", linewidth=2.5, linestyle="--", marker="s")
    axes[1].set_title("Accuracy Curves (%)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Learning Rate
    if "lr" in history:
        lr_epochs = range(1, len(history["lr"]) + 1)
        axes[2].plot(lr_epochs, history["lr"], label="Learning Rate", color="#8B5CF6", linewidth=2)
        axes[2].set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        axes[2].set_xlabel("Batch / Step", fontsize=12)
        axes[2].set_ylabel("Learning Rate", fontsize=12)
        axes[2].set_yscale("log")
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
