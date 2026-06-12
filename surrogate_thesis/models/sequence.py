"""Sequence neural models for short-term load forecasting."""

from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """LSTM surrogate that maps a lookback window to the forecast horizon."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        dropout: float,
    ) -> None:
        """Create the recurrent encoder and regression head."""

        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the final recurrent hidden state as the window representation."""

        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])
