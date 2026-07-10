# AI Traffic Prediction and Management System

Smart-city traffic platform: YOLO-based detection, ByteTrack / DeepSORT
tracking, LSTM + Attention forecasting, ANPR, accident / emergency-vehicle
detection, email / FCM alerts, and route optimization — all surfaced
through an 11-page Streamlit dashboard.

## Quick start

```bash
# 1. Set up Python environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

# 2. Configure secrets (see docs/INSTALL.md for details)
cp .env.example .env
# edit .env and fill in your API keys

# 3. Run the dashboard
streamlit run dashboard/app.py
```

The dashboard defaults to `http://localhost:8501`.

## Features

| Category           | Modules                                                         |
|--------------------|-----------------------------------------------------------------|
| Core detection     | YOLOv8 / v9 / v10, 9 Indian vehicle classes                     |
| Tracking & counting| ByteTrack, DeepSORT, zone / line counters                       |
| Prediction         | LSTM + Attention, weather-aware forecasts (OpenWeatherMap)      |
| Safety             | Accident detection, ANPR + blacklist, emergency-vehicle detect. |
| Alerts             | Email (SMTP), Firebase Cloud Messaging                          |
| Operations         | OpenRouteService route suggestions                              |

## Security

**The dashboard ships with a shared-password gate.** Set
`DASHBOARD_PASSWORD` in your environment (or `.streamlit/secrets.toml`) before
exposing the UI to anyone. Without it, a banner warns that the app is open.

If the committed `.env.example` ever contained a real secret in your fork,
rotate it immediately — tokens pushed to any remote must be treated as
compromised.

See [docs/INSTALL.md](docs/INSTALL.md) for full configuration details and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a system overview.

## Development

```bash
# Run tests
pytest

# Lint
ruff check src tests config
ruff format src tests config

# Pre-commit hooks (optional)
pre-commit install
```

CI runs lint, unit tests on Python 3.10 / 3.11, and a pip-audit security
scan on every push and pull request — see [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Project layout

```
vehical/
├── config/           # Pydantic-style settings + YAML configs
├── dashboard/        # Streamlit app and 11 feature pages
├── data/             # Local datasets, logs, subscriber files (gitignored)
├── docs/             # INSTALL, ARCHITECTURE, CONTRIBUTING
├── models/           # Trained weights (gitignored)
├── scripts/          # Data-prep CLIs
├── src/              # Library code — see src/* per feature
│   ├── alerts/               # AlertManager + email + in-app
│   ├── accident_detection/
│   ├── anpr/
│   ├── counting/
│   ├── detection/
│   ├── emergency_detection/
│   ├── export/
│   ├── firebase_notifications/
│   ├── maps/                 # Route suggestions + folium overlays
│   ├── pipeline/
│   ├── prediction/           # LSTM + weather features
│   ├── tracking/
│   └── utils/                # env validator, auth, retries, logging
├── tests/            # pytest suite
└── training/         # Model training scripts
```

## License

Proprietary — see individual files for attribution.
