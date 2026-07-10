"""Data preprocessing for LSTM traffic prediction training."""

from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from config.settings import LSTMConfig


class TrafficDataPreprocessor:
    """Prepares time-series data for LSTM training and inference."""

    def __init__(self, config: LSTMConfig = None):
        if config is None:
            config = LSTMConfig()
        self.config = config
        self.seq_len = config.sequence_length
        self.pred_horizon = config.prediction_horizon
        self.scaler = MinMaxScaler()
        self.feature_columns = None

    def prepare_dataset(
        self,
        counts_csv: str,
        weather_csv: Optional[str] = None,
        val_split: float = 0.2,
    ) -> Tuple[DataLoader, DataLoader, int]:
        """Prepare training and validation DataLoaders from CSV files.

        Args:
            counts_csv: Path to CSV with timestamp index and count columns.
            weather_csv: Optional path to weather data CSV.
            val_split: Fraction of data for validation.

        Returns:
            Tuple of (train_loader, val_loader, num_features).
        """
        df = pd.read_csv(counts_csv, parse_dates=["timestamp"], index_col="timestamp")

        # Core features
        core_features = [c for c in df.columns if c != "timestamp"]
        if "total" not in core_features:
            df["total"] = df[core_features].sum(axis=1)
            core_features.append("total")

        # Add cyclical time features
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        features = core_features + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

        # Merge weather if available
        if self.config.include_weather and weather_csv:
            weather_df = pd.read_csv(
                weather_csv, parse_dates=["timestamp"], index_col="timestamp"
            )
            weather_cols = ["temperature", "humidity", "rain_1h", "visibility", "wind_speed"]
            available = [c for c in weather_cols if c in weather_df.columns]
            df = df.join(weather_df[available], how="left").ffill().bfill()
            features += available

        self.feature_columns = features
        df = df[features].dropna()

        # Scale
        scaled = self.scaler.fit_transform(df.values)

        # Create sliding windows
        X, y = self._create_windows(scaled)

        # Split
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        return train_loader, val_loader, len(features)

    def _create_windows(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for LSTM."""
        X, y = [], []
        for i in range(len(data) - self.seq_len - self.pred_horizon):
            X.append(data[i : i + self.seq_len])
            # Predict the "total" column (index 0 after reorder, but we use
            # whichever column represents "total")
            total_idx = self.feature_columns.index("total") if self.feature_columns else 0
            y.append(
                data[i + self.seq_len : i + self.seq_len + self.pred_horizon, total_idx]
            )
        return np.array(X), np.array(y)

    def prepare_inference_input(
        self, recent_counts: pd.DataFrame, weather_data: Optional[pd.DataFrame] = None
    ) -> torch.Tensor:
        """Prepare input tensor for inference from recent data.

        Args:
            recent_counts: DataFrame with at least `seq_len` rows.
            weather_data: Optional weather DataFrame aligned with counts.

        Returns:
            Tensor of shape (1, seq_len, num_features).
        """
        df = recent_counts.copy()

        # Add time features
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        if weather_data is not None:
            df = df.join(weather_data, how="left").ffill().bfill()

        # Use same columns as training
        if self.feature_columns:
            available = [c for c in self.feature_columns if c in df.columns]
            df = df[available]

        # Scale using fitted scaler
        scaled = self.scaler.transform(df.values[-self.seq_len :])

        return torch.FloatTensor(scaled).unsqueeze(0)

    def inverse_scale_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse-transform predicted values back to original scale.

        Args:
            predictions: Scaled prediction values.

        Returns:
            Predictions in original scale.
        """
        # Create dummy array with same shape as training data
        n_features = len(self.feature_columns) if self.feature_columns else 1
        total_idx = self.feature_columns.index("total") if self.feature_columns else 0

        dummy = np.zeros((len(predictions), n_features))
        dummy[:, total_idx] = predictions
        unscaled = self.scaler.inverse_transform(dummy)
        return unscaled[:, total_idx]
