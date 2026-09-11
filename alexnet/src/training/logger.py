import os
import logging


class TrainingLogger:
    """
    Unified Logger handling console output and file logging.
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # File logger setup
        log_file = os.path.join(log_dir, "training.log")
        self.logger = logging.getLogger("AlexNetLogger")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear previous handlers

        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def log_metrics(self, epoch: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float, lr: float):
        """Logs epoch metrics to file and console."""
        self.info(
            f"Metrics | Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {lr:.6f}"
        )

    def close(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
