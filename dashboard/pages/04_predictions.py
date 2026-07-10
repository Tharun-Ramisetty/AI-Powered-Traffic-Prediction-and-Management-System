"""Predictions page - LSTM-based traffic count forecasting with real model + weather."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Traffic Predictions", layout="wide")
st.title("Traffic Count Prediction")

st.markdown("""
This page shows **LSTM + Attention** model predictions for future vehicle counts.
The model considers historical traffic patterns and weather conditions.
""")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Prediction Settings")
    prediction_hours = st.slider("Prediction Horizon (hours)", 1, 24, 6)
    include_weather = st.checkbox("Include Weather Features", value=True)
    show_confidence = st.checkbox("Show Confidence Intervals", value=True)

st.markdown("---")

# ─── Real Weather Data ──────────────────────────────────────────────────────
weather_data = None
if include_weather:
    try:
        from src.prediction.weather_features import WeatherClient

        weather_client = WeatherClient()
        weather_data = weather_client.get_current_weather()

        st.markdown("### Current Weather Conditions")
        w1, w2, w3, w4, w5 = st.columns(5)
        with w1:
            st.metric("Temperature", f"{weather_data['temperature']:.1f}°C")
        with w2:
            st.metric("Humidity", f"{weather_data['humidity']:.0f}%")
        with w3:
            st.metric("Rain", f"{weather_data['rain_1h']:.1f} mm/h")
        with w4:
            st.metric("Visibility", f"{weather_data['visibility']:.1f} km")
        with w5:
            st.metric("Wind", f"{weather_data['wind_speed']:.1f} km/h")

        if weather_client.is_configured():
            st.success(f"Live weather from OpenWeatherMap: {weather_data.get('weather_desc', 'N/A')}")
        else:
            st.info("Set `OPENWEATHER_API_KEY` in `.env` for live weather. Using default values.")

        st.markdown("---")
    except Exception as e:
        st.warning(f"Weather module error: {e}")

# ─── Prediction Visualization ───────────────────────────────────────────────
st.markdown("### Traffic Count Prediction")

# Get historical data from session (from live detection)
aggregator = st.session_state.get("aggregator")

if aggregator is not None:
    df = aggregator.get_dataframe()
    if not df.empty and "total" in df.columns:
        hist_counts = df["total"].values
        hist_timestamps = df.index

        # Try to load trained LSTM model
        model_path = Path("trained_models/lstm_traffic_predictor.pth")
        if model_path.exists():
            try:
                from src.prediction.predictor import TrafficPredictor
                from src.prediction.data_preprocessor import TrafficDataPreprocessor
                from config.settings import LSTMConfig

                config = LSTMConfig()
                preprocessor = TrafficDataPreprocessor(config)

                # Fit preprocessor on available data
                preprocessor.fit(df)

                predictor = TrafficPredictor(
                    model_path=str(model_path),
                    preprocessor=preprocessor,
                    config=config,
                )

                # Get weather forecast if available
                weather_forecast = None
                if include_weather:
                    try:
                        weather_forecast = weather_client.get_forecast(prediction_hours)
                    except Exception:
                        pass

                predictions = predictor.predict_next_hours(
                    df, weather_forecast=weather_forecast, hours=prediction_hours
                )

                # Plot historical + predicted
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist_timestamps, y=hist_counts,
                    mode="lines+markers", name="Historical",
                    line=dict(color="blue", width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=predictions.index, y=predictions["predicted_total"],
                    mode="lines+markers", name="Predicted",
                    line=dict(color="red", width=2, dash="dash"),
                ))
                if show_confidence:
                    fig.add_trace(go.Scatter(
                        x=list(predictions.index) + list(predictions.index[::-1]),
                        y=list(predictions["confidence_high"]) + list(predictions["confidence_low"][::-1]),
                        fill="toself", fillcolor="rgba(255,0,0,0.1)",
                        line=dict(color="rgba(255,0,0,0)"),
                        name="Confidence Interval",
                    ))
                fig.update_layout(
                    title="Traffic Count: Historical vs LSTM Predicted",
                    xaxis_title="Time", yaxis_title="Vehicle Count",
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show prediction table
                st.markdown("### Predicted Values")
                st.dataframe(predictions, use_container_width=True)

                st.success("Predictions generated using trained LSTM + Attention model.")

            except Exception as e:
                st.warning(f"LSTM model loading error: {e}")
                st.info("Showing historical data only. Train the LSTM model first for predictions.")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist_timestamps, y=hist_counts,
                    mode="lines+markers", name="Historical Counts",
                    line=dict(color="blue", width=2),
                ))
                fig.update_layout(title="Historical Vehicle Counts", height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"LSTM model not found at `{model_path}`. Train the model first.")
            st.markdown("To train: run the training scripts in `training/` directory.")

            # Show historical data
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_timestamps, y=hist_counts,
                mode="lines+markers", name="Historical Counts",
                line=dict(color="blue", width=2),
            ))
            fig.update_layout(title="Historical Vehicle Counts (No Prediction Model)", height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No count data available. Run detection first.")
else:
    st.info("Run detection on the **Live Detection** page first to generate historical data for predictions.")

# ─── Model Info ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Prediction Model Details")
st.markdown("""
| Parameter | Value |
|-----------|-------|
| Architecture | LSTM (2 layers) + Multi-Head Self-Attention |
| Hidden Size | 128 |
| Attention Heads | 4 |
| Input Window | 24 hours |
| Features | Count + Time (cyclical) + Weather |
| Training | Adam optimizer, MSE loss, early stopping |
""")

st.info("""
**Weather-aware prediction adjusts forecasts based on conditions:**
- **Rain**: Reduces predicted traffic by ~15-25%
- **Low visibility/fog**: Reduces traffic, increases congestion risk
- **High wind**: Minor impact on two-wheelers
""")
