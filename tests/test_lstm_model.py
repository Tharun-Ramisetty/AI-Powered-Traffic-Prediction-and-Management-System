"""Tests for LSTM prediction model."""

import pytest
import torch

from src.prediction.lstm_model import TrafficLSTM


class TestTrafficLSTM:
    def test_forward_pass_shape(self):
        """Output shape should be (batch, prediction_horizon)."""
        model = TrafficLSTM(
            input_size=14, hidden_size=64,
            num_layers=2, num_heads=4, output_size=6,
        )
        x = torch.randn(8, 24, 14)  # batch=8, seq_len=24, features=14
        out = model(x)
        assert out.shape == (8, 6)

    def test_single_sample(self):
        """Should work with batch size 1."""
        model = TrafficLSTM(input_size=10, hidden_size=32, output_size=6)
        x = torch.randn(1, 24, 10)
        out = model(x)
        assert out.shape == (1, 6)

    def test_different_sequence_lengths(self):
        """Should work with various sequence lengths."""
        model = TrafficLSTM(input_size=10, hidden_size=32, output_size=6)
        for seq_len in [12, 24, 48, 168]:
            x = torch.randn(4, seq_len, 10)
            out = model(x)
            assert out.shape == (4, 6)

    def test_different_prediction_horizons(self):
        """Should work with various output sizes."""
        for horizon in [1, 6, 12, 24]:
            model = TrafficLSTM(input_size=10, hidden_size=32, output_size=horizon)
            x = torch.randn(4, 24, 10)
            out = model(x)
            assert out.shape == (4, horizon)

    def test_gradient_flow(self):
        """Gradients should flow through the model."""
        model = TrafficLSTM(input_size=10, hidden_size=32, output_size=6)
        x = torch.randn(4, 24, 10)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_eval_mode_no_dropout(self):
        """Model should work in eval mode without errors."""
        model = TrafficLSTM(input_size=10, hidden_size=32, output_size=6, dropout=0.5)
        model.eval()
        x = torch.randn(4, 24, 10)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 6)

    def test_parameter_count(self):
        """Model should have a reasonable number of parameters."""
        model = TrafficLSTM(input_size=14, hidden_size=128, num_layers=2, output_size=6)
        params = sum(p.numel() for p in model.parameters())
        assert params > 0
        assert params < 10_000_000  # Should be well under 10M for this architecture
