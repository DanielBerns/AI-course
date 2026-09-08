"""
Project 5: Computer Vision & Sequential Architectures Training Script.

Trains PyTorch Residual CNN for vision data and LSTM for sequential time-series data,
tracking accuracy curves and saving checkpoint artifacts.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from cnn_model import PyTorchResNetCNN
from lstm_model import SequentialLSTMClassifier
from trainer import DeepModelTrainer


def main() -> None:
    print("=" * 70)
    print("PROJECT 5: COMPUTER VISION & SEQUENTIAL ARCHITECTURES")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" - Training Hardware Device: {device}")

    # 1. Computer Vision: Residual CNN
    print("\n[Step 1] Training Residual CNN Image Classifier...")
    # Synthetic image tensor batch: (num_samples, channels, height, width) = (400, 1, 28, 28)
    np.random.seed(42)
    torch.manual_seed(42)

    X_img = torch.randn(400, 1, 28, 28)
    y_img = torch.randint(0, 10, (400,))

    train_ds_img = TensorDataset(X_img[:320], y_img[:320])
    val_ds_img = TensorDataset(X_img[320:], y_img[320:])
    train_loader_img = DataLoader(train_ds_img, batch_size=32, shuffle=True)
    val_loader_img = DataLoader(val_ds_img, batch_size=32)

    cnn_model = PyTorchResNetCNN(in_channels=1, num_classes=10, base_filters=16)
    cnn_optimizer = optim.AdamW(cnn_model.parameters(), lr=0.003)
    cnn_scheduler = optim.lr_scheduler.CosineAnnealingLR(cnn_optimizer, T_max=5)
    cnn_trainer = DeepModelTrainer(cnn_model, nn.CrossEntropyLoss(), cnn_optimizer, cnn_scheduler, device=device)

    print(" Training ResNet CNN across 5 epochs...")
    cnn_history = {"train_loss": [], "val_acc": []}
    for ep in range(1, 6):
        tr_loss, tr_acc = cnn_trainer.train_epoch(train_loader_img)
        va_loss, va_acc = cnn_trainer.evaluate(val_loader_img)
        cnn_history["train_loss"].append(tr_loss)
        cnn_history["val_acc"].append(va_acc)
        print(f" Epoch {ep}/5 | Train Loss: {tr_loss:.4f} | Val Accuracy: {va_acc:.2%}")

    torch.save(cnn_model.state_dict(), os.path.join(output_dir, "resnet_cnn.pt"))

    # 2. Sequential Modeling: LSTM
    print("\n[Step 2] Training LSTM Sequential Classifier...")
    # Synthetic time-series tensor batch: (num_samples, seq_len, features) = (400, 20, 5)
    X_seq = torch.randn(400, 20, 5)
    y_seq = torch.randint(0, 2, (400,))

    train_loader_seq = DataLoader(TensorDataset(X_seq[:320], y_seq[:320]), batch_size=32, shuffle=True)
    val_loader_seq = DataLoader(TensorDataset(X_seq[320:], y_seq[320:]), batch_size=32)

    lstm_model = SequentialLSTMClassifier(input_dim=5, hidden_dim=32, num_layers=2, num_classes=2)
    lstm_optimizer = optim.AdamW(lstm_model.parameters(), lr=0.005)
    lstm_trainer = DeepModelTrainer(lstm_model, nn.CrossEntropyLoss(), lstm_optimizer, device=device)

    print(" Training Sequential LSTM across 5 epochs...")
    for ep in range(1, 6):
        tr_loss, tr_acc = lstm_trainer.train_epoch(train_loader_seq)
        va_loss, va_acc = lstm_trainer.evaluate(val_loader_seq)
        print(f" Epoch {ep}/5 | Train Loss: {tr_loss:.4f} | Val Accuracy: {va_acc:.2%}")

    torch.save(lstm_model.state_dict(), os.path.join(output_dir, "lstm_classifier.pt"))

    # Plot performance curves
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, 6), cnn_history["train_loss"], "o-", color="#263c72", label="CNN Loss")
    ax.set_title("Computer Vision Residual CNN Loss Curve", fontsize=11, fontweight="bold", color="#263c72")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy Loss")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "cnn_training_curve.png"), dpi=300)

    print("\n" + "=" * 70)
    print("PROJECT 5 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
