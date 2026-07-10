"""Traffic count predictor using trained LSTM model."""

from typing import Optional, List

import numpy as np
import pandas as pd
import torch

from config.settings import LSTMConfig
from .lstm_model import TrafficLSTM
from .data_preprocessor import TrafficDataPreprocessor


class TrafficPredictor:
    """Predicts future vehicle counts using a trained LSTM model."""

    def __init__(
        self,
        model_path: str,
        preprocessor: TrafficDataPreprocessor,
        config: LSTMConfig = None,
    ):
        if config is None:
            config = LSTMConfig()
        self.config = config
        self.preprocessor = preprocessor

        # Determine input size from preprocessor
        n_features = (
            len(preprocessor.feature_columns)
            if preprocessor.feature_columns
            else 14  # default estimate
        )

        # Load model
        self.model = TrafficLSTM(
            input_size=n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            output_size=config.prediction_horizon,
            dropout=0.0,  # No dropout at inference
        )

        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_next_hours(
        self,
        recent_counts: pd.DataFrame,
        weather_forecast: Optional[pd.DataFrame] = None,
        hours: int = None,
    ) -> pd.DataFrame:
        """Predict vehicle counts for the next N hours.

        Args:
            recent_counts: DataFrame with at least `sequence_length` rows of
                           historical count data, indexed by timestamp.
            weather_forecast: Optional weather forecast DataFrame.
            hours: Override for prediction horizon.

        Returns:
            DataFrame with columns: timestamp, predicted_total, confidence_low, confidence_high
        """
        if hours is None:
            hours = self.config.prediction_horizon

        # Prepare input
        input_tensor = self.preprocessor.prepare_inference_input(
            recent_counts, weather_forecast
        )

        # Predict
        with torch.no_grad():
            predictions = self.model(input_tensor)

        pred_values = predictions.cpu().numpy().flatten()

        # Inverse scale
        pred_counts = self.preprocessor.inverse_scale_predictions(pred_values[:hours])
        pred_counts = np.maximum(pred_counts, 0).astype(int)

        # Generate timestamps
        last_ts = recent_counts.index[-1]
        future_timestamps = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1),
            periods=hours,
            freq="h",
        )

        # Confidence intervals (simple heuristic: +/- 15% widening over time)
        confidence_pct = np.linspace(0.10, 0.25, hours)
        low = (pred_counts * (1 - confidence_pct)).astype(int)
        high = (pred_counts * (1 + confidence_pct)).astype(int)

        return pd.DataFrame({
            "timestamp": future_timestamps,
            "predicted_total": pred_counts,
            "confidence_low": np.maximum(low, 0),
            "confidence_high": high,
        }).set_index("timestamp")
