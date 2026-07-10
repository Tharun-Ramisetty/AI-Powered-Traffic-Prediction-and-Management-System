"""LSTM + Attention model for traffic count prediction."""

import torch
import torch.nn as nn


class TrafficLSTM(nn.Module):
    """LSTM model with self-attention for traffic vehicle count prediction.

    Architecture:
        Input -> LSTM (2 layers) -> Multi-Head Self-Attention -> FC -> Output

    The attention mechanism allows the model to focus on specific historical
    time steps that are most relevant for prediction (e.g., same hour yesterday
    is more important than 3 hours ago).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        output_size: int = 6,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_size: Number of input features per timestep.
            hidden_size: LSTM hidden state size.
            num_layers: Number of stacked LSTM layers.
            num_heads: Number of attention heads.
            output_size: Prediction horizon (e.g., 6 = predict 6 hours ahead).
            dropout: Dropout rate.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Predictions of shape (batch, output_size).
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Self-attention over LSTM outputs
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual connection + layer norm
        attn_out = self.layer_norm(attn_out + lstm_out)

        # Use the last timestep for prediction
        last_step = attn_out[:, -1, :]  # (batch, hidden_size)

        # Project to prediction horizon
        output = self.fc(last_step)  # (batch, output_size)

        return output
