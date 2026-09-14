import os
import time
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CyclicLR,
    OneCycleLR,
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader

from ..models.alexnet import ModernAlexNet
from .logger import TrainingLogger
from .diagnostics import DiagnosticMonitor
from ..utils.visualization import plot_training_history


class Trainer:
    """
    Production-grade Training Engine for ModernAlexNet.
    Features:
    - Config-driven setup & directory separation (data, weights, logs, plots)
    - Checkpoint saving & loading (full state resume)
    - Flexible optimizer & LR scheduling (CyclicLR, OneCycleLR, Cosine, etc.)
    - Underfitting & Overfitting diagnostic alerts
    - Dynamic visualization of training curves
    """

    def __init__(self, config: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader, class_names: list):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names

        # Paths configuration
        paths_cfg = config["paths"]
        self.weights_dir = os.path.abspath(os.path.expanduser(paths_cfg.get("weights_dir", "./weights")))
        self.log_dir = os.path.abspath(os.path.expanduser(paths_cfg.get("log_dir", "./logs")))
        self.plots_dir = os.path.abspath(os.path.expanduser(paths_cfg.get("plots_dir", "./plots")))

        for d in [self.weights_dir, self.log_dir, self.plots_dir]:
            os.makedirs(d, exist_ok=True)

        # Device setup
        device_req = config.get("training", {}).get("device", "auto")
        if device_req == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_req)

        # Logger & Diagnostics
        self.logger = TrainingLogger(self.log_dir)
        diag_cfg = config.get("diagnostics", {})
        self.diagnostics = DiagnosticMonitor(
            overfitting_threshold=diag_cfg.get("overfitting_threshold", 0.20),
            underfitting_loss_threshold=diag_cfg.get("underfitting_loss_threshold", 1.5),
        )

        # Model initialization
        model_cfg = config.get("model", {})
        self.model = ModernAlexNet(
            num_classes=len(class_names),
            in_channels=model_cfg.get("in_channels", 3),
            use_batch_norm=model_cfg.get("use_batch_norm", True),
            dropout=model_cfg.get("dropout", 0.5),
        ).to(self.device)

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Epoch & Run configuration
        self.epochs_per_run = config.get("training", {}).get("epochs", 20)
        self.start_epoch = 1
        self.total_epochs = self.epochs_per_run

        # Scheduler
        self.scheduler = self._build_scheduler()

        # State trackers
        self.best_val_acc = 0.0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

        # Check for resume
        resume_path = config.get("training", {}).get("resume_checkpoint")
        if resume_path:
            self.load_checkpoint(resume_path)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_cfg = self.config.get("optimizer", {})
        name = opt_cfg.get("name", "adamw").lower()
        lr = float(opt_cfg.get("lr", 0.001))
        weight_decay = float(opt_cfg.get("weight_decay", 0.0001))

        if name == "adam":
            return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif name == "adamw":
            return AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif name == "sgd":
            momentum = float(opt_cfg.get("momentum", 0.9))
            return SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer name: {name}")

    def _build_scheduler(self):
        sched_cfg = self.config.get("scheduler", {})
        sched_type = sched_cfg.get("type", "onecycle").lower()

        if sched_type == "cyclic":
            base_lr = float(sched_cfg.get("base_lr", 0.0001))
            max_lr = float(sched_cfg.get("max_lr", 0.005))
            step_size_up = int(sched_cfg.get("step_size_up", 2000))
            return CyclicLR(self.optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, cycle_momentum=False)
        elif sched_type == "onecycle":
            max_lr = float(sched_cfg.get("max_lr", 0.005))
            total_steps = self.epochs_per_run * len(self.train_loader)
            return OneCycleLR(self.optimizer, max_lr=max_lr, total_steps=total_steps)
        elif sched_type == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.epochs_per_run)
        elif sched_type == "step":
            step_size = int(sched_cfg.get("step_size", 5))
            gamma = float(sched_cfg.get("gamma", 0.5))
            return StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif sched_type == "plateau":
            patience = int(sched_cfg.get("patience", 3))
            return ReduceLROnPlateau(self.optimizer, mode="max", patience=patience)
        elif sched_type == "none" or not sched_type:
            return None
        else:
            raise ValueError(f"Unsupported scheduler type: {sched_type}")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # Batch level LR scheduler (for CyclicLR or OneCycleLR)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)
            if isinstance(self.scheduler, (CyclicLR, OneCycleLR)):
                self.scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, loader: DataLoader = None) -> Tuple[float, float]:
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        loss = running_loss / total
        acc = 100.0 * correct / total
        return loss, acc

    def save_checkpoint(self, epoch: int, is_best: bool = False, filename: Optional[str] = None):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_acc": self.best_val_acc,
            "history": self.history,
            "class_names": self.class_names,
            "config": self.config,
        }

        latest_path = os.path.join(self.weights_dir, "latest_checkpoint.pt")
        torch.save(state, latest_path)

        if filename:
            path = os.path.join(self.weights_dir, filename)
            torch.save(state, path)

        if is_best:
            best_path = os.path.join(self.weights_dir, "best_model.pt")
            torch.save(state, best_path)
            self.logger.info(f"[+] Saved new best model checkpoint to: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        self.logger.info(f"[*] Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.total_epochs = self.start_epoch + self.epochs_per_run - 1
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.history = checkpoint.get("history", self.history)

        sched_type = self.config.get("scheduler", {}).get("type", "").lower()
        if sched_type in ("onecycle", "cosine"):
            self.scheduler = self._build_scheduler()
            self.logger.info(f"[*] Re-initialized '{sched_type}' scheduler for {self.epochs_per_run} epochs in this run.")
        elif self.scheduler and checkpoint.get("scheduler_state_dict"):
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                self.logger.warning(f"[!] Could not restore scheduler state dict ({e}). Re-building scheduler.")
                self.scheduler = self._build_scheduler()

        self.logger.info(f"[+] Successfully loaded checkpoint. Resuming from epoch {self.start_epoch} up to epoch {self.total_epochs}")

    def train(self):
        self.logger.info(f"[*] Starting AlexNet training pipeline on device: {self.device}")
        self.logger.info(f"[*] Classes ({len(self.class_names)}): {self.class_names}")

        save_every = self.config.get("training", {}).get("save_every", 5)
        plot_path = os.path.join(self.plots_dir, "training_curves.png")

        for epoch in range(self.start_epoch, self.total_epochs + 1):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Epoch level LR scheduler (StepLR, Cosine, Plateau)
            if isinstance(self.scheduler, (StepLR, CosineAnnealingLR)):
                self.scheduler.step()
            elif isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_acc)

            elapsed = time.time() - start_time

            # Log metrics
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            self.logger.info(
                f"Epoch [{epoch:02d}/{self.total_epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}"
            )
            self.logger.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)

            # Diagnostics check
            warnings = self.diagnostics.check_epoch(epoch, train_loss, val_loss, train_acc, val_acc)
            for warn in warnings:
                self.logger.warning(warn)

            # Save best checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)

            # Save periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, is_best=False, filename=f"checkpoint_epoch_{epoch}.pt")

            # Update loss/accuracy plot
            plot_training_history(self.history, plot_path)

        self.logger.info(f"[*] Training complete! Best Validation Accuracy: {self.best_val_acc:.2f}%")
        self.logger.info(f"[*] Final training curves saved to: {plot_path}")
        self.logger.close()
