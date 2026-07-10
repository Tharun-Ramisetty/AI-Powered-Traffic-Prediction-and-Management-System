"""Weather data client for traffic prediction enhancement."""

import os
from typing import Dict, Optional

import requests
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from config.settings import WeatherConfig
from src.utils.retry import with_retry

load_dotenv()

_TRANSIENT_HTTP = (requests.ConnectionError, requests.Timeout)


class WeatherClient:
    """Fetches weather data from OpenWeatherMap API."""

    BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self, config: WeatherConfig = None):
        if config is None:
            config = WeatherConfig()
        self.config = config
        self.api_key = os.getenv(config.api_key_env, "")
        self.city = config.city

    def is_configured(self) -> bool:
        """Check if the API key is set."""
        return bool(self.api_key)

    def get_current_weather(self) -> Dict:
        """Fetch current weather conditions.

        Returns:
            Dict with weather features.
        """
        if not self.is_configured():
            return self._default_weather()

        try:
            data = self._fetch_current()
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "rain_1h": data.get("rain", {}).get("1h", 0.0),
                "visibility": data.get("visibility", 10000) / 1000,
                "wind_speed": data["wind"]["speed"],
                "weather_main": data["weather"][0]["main"],
                "weather_desc": data["weather"][0]["description"],
            }
        except _TRANSIENT_HTTP as exc:
            logger.warning("Weather API transient failure ({}): {}",
                           type(exc).__name__, exc)
        except requests.HTTPError as exc:
            logger.error("Weather API HTTP error: {}", exc)
        except (KeyError, ValueError) as exc:
            logger.error("Weather API returned unexpected payload: {}", exc)
        return self._default_weather()

    @with_retry(exceptions=_TRANSIENT_HTTP, max_attempts=3, initial_wait=1.0)
    def _fetch_current(self) -> Dict:
        resp = requests.get(
            f"{self.BASE_URL}/weather",
            params={"q": self.city, "appid": self.api_key, "units": "metric"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def get_forecast(self, hours: int = 24) -> pd.DataFrame:
        """Fetch hourly weather forecast.

        Args:
            hours: Number of hours of forecast.

        Returns:
            DataFrame with weather features indexed by timestamp.
        """
        if not self.is_configured():
            return self._default_forecast(hours)

        try:
            data = self._fetch_forecast(hours)
            records = []
            for entry in data.get("list", []):
                records.append({
                    "timestamp": pd.Timestamp(entry["dt"], unit="s"),
                    "temperature": entry["main"]["temp"],
                    "humidity": entry["main"]["humidity"],
                    "rain_1h": entry.get("rain", {}).get("3h", 0.0) / 3,
                    "visibility": entry.get("visibility", 10000) / 1000,
                    "wind_speed": entry["wind"]["speed"],
                })
            df = pd.DataFrame(records).set_index("timestamp")
            return df.resample("h").interpolate()
        except _TRANSIENT_HTTP as exc:
            logger.warning("Weather forecast transient failure ({}): {}",
                           type(exc).__name__, exc)
        except requests.HTTPError as exc:
            logger.error("Weather forecast HTTP error: {}", exc)
        except (KeyError, ValueError) as exc:
            logger.error("Weather forecast returned unexpected payload: {}", exc)
        return self._default_forecast(hours)

    @with_retry(exceptions=_TRANSIENT_HTTP, max_attempts=3, initial_wait=1.0)
    def _fetch_forecast(self, hours: int) -> Dict:
        resp = requests.get(
            f"{self.BASE_URL}/forecast",
            params={
                "q": self.city,
                "appid": self.api_key,
                "units": "metric",
                "cnt": min(hours, 40),
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _default_weather() -> Dict:
        return {
            "temperature": 25.0,
            "humidity": 60.0,
            "rain_1h": 0.0,
            "visibility": 10.0,
            "wind_speed": 5.0,
            "weather_main": "Clear",
            "weather_desc": "clear sky",
        }

    @staticmethod
    def _default_forecast(hours: int) -> pd.DataFrame:
        timestamps = pd.date_range(
            start=pd.Timestamp.now(), periods=hours, freq="h"
        )
        return pd.DataFrame({
            "timestamp": timestamps,
            "temperature": 25.0,
            "humidity": 60.0,
            "rain_1h": 0.0,
            "visibility": 10.0,
            "wind_speed": 5.0,
        }).set_index("timestamp")
