from typing import Dict, List, Tuple


class DiagnosticMonitor:
    """
    Monitors training statistics to detect underfitting, overfitting, and learning anomalies.
    """

    def __init__(self, overfitting_threshold: float = 0.20, underfitting_loss_threshold: float = 1.5):
        self.overfitting_threshold = overfitting_threshold
        self.underfitting_loss_threshold = underfitting_loss_threshold
        self.val_loss_history: List[float] = []
        self.train_loss_history: List[float] = []

    def check_epoch(self, epoch: int, train_loss: float, val_loss: float, train_acc: float, val_acc: float) -> List[str]:
        """
        Analyzes current epoch metrics and returns diagnostic warnings/notes.
        """
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        warnings = []

        # 1. Overfitting Detection
        loss_diff = val_loss - train_loss
        if loss_diff > self.overfitting_threshold:
            warnings.append(
                f"[OVERFITTING ALERT] Validation loss ({val_loss:.4f}) exceeds training loss ({train_loss:.4f}) "
                f"by {loss_diff:.4f} (Threshold: {self.overfitting_threshold:.2f}). Consider increasing Dropout or Data Augmentation."
            )

        # Check for rising validation loss over 3 consecutive epochs while train loss decreases
        if len(self.val_loss_history) >= 3:
            if (self.val_loss_history[-1] > self.val_loss_history[-2] > self.val_loss_history[-3] and
                self.train_loss_history[-1] < self.train_loss_history[-2]):
                warnings.append(
                    "[OVERFITTING DIVERGENCE] Validation loss has risen for 3 consecutive epochs while training loss decreased."
                )

        # 2. Underfitting Detection
        if epoch >= 5 and train_loss > self.underfitting_loss_threshold:
            warnings.append(
                f"[UNDERFITTING ALERT] High training loss ({train_loss:.4f}) after epoch {epoch}. "
                f"Model capacity may be constrained or learning rate might be too low/high."
            )

        # 3. Generalization Gap
        gen_gap = train_acc - val_acc
        if gen_gap > 15.0:
            warnings.append(
                f"[GENERALIZATION GAP] Training accuracy ({train_acc:.2f}%) is {gen_gap:.2f}% higher than validation accuracy ({val_acc:.2f}%)."
            )

        return warnings
