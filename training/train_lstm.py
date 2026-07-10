"""Train LSTM model for traffic count prediction."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config.settings import LSTMConfig
from src.prediction.lstm_model import TrafficLSTM
from src.prediction.data_preprocessor import TrafficDataPreprocessor


def train(
    counts_csv: str = "data/datasets/traffic_timeseries/counts_camera_01.csv",
    weather_csv: str = None,
    config: LSTMConfig = None,
):
    """Train the LSTM traffic prediction model.

    Args:
        counts_csv: Path to time-series count data CSV.
        weather_csv: Optional path to weather data CSV.
        config: LSTM configuration.
    """
    if config is None:
        config = LSTMConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Prepare data
    print("Preparing dataset...")
    preprocessor = TrafficDataPreprocessor(config)
    train_loader, val_loader, num_features = preprocessor.prepare_dataset(
        counts_csv, weather_csv
    )
    print(f"Features: {num_features}, Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Initialize model
    model = TrafficLSTM(
        input_size=num_features,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        output_size=config.prediction_horizon,
        dropout=config.dropout,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    patience_limit = 20

    # Training loop
    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{config.epochs} - "
                  f"Train Loss: {train_loss:.6f} - "
                  f"Val Loss: {val_loss:.6f} - "
                  f"LR: {lr:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_path = Path(config.model_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {config.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM traffic predictor")
    parser.add_argument("--counts", default="data/datasets/traffic_timeseries/counts_camera_01.csv")
    parser.add_argument("--weather", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--pred-horizon", type=int, default=6)
    args = parser.parse_args()

    config = LSTMConfig(
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        sequence_length=args.seq_len,
        prediction_horizon=args.pred_horizon,
    )
    train(counts_csv=args.counts, weather_csv=args.weather, config=config)
