"""
LSTM Recurrent Neural Network for Sequential Data.

Implements multi-layer Gated Recurrent Units / LSTMs for time-series and sequential tasks.
"""

from typing import Tuple
import torch
import torch.nn as nn


class SequentialLSTMClassifier(nn.Module):
    """LSTM sequence classifier with dropout regularization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        # Take hidden state of the final time-step
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)
