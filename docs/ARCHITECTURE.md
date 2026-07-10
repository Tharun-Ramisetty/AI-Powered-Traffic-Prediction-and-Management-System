# Architecture overview

## High-level flow

```
┌────────────┐   frames    ┌──────────────┐   detections   ┌────────────┐
│ Video feed │ ───────────▶│ YOLO detector│ ──────────────▶│ Tracker    │
│ (file/RTSP)│             │ (v8 / v9 /v10)│               │ (ByteTrack)│
└────────────┘             └──────────────┘                └──────┬─────┘
                                                                  │ tracks
                                                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Pipeline                                     │
│   ┌──────────┐  ┌────────┐  ┌──────────┐  ┌────────────┐  ┌──────────┐ │
│   │ Counting │  │ ANPR   │  │ Accident │  │ Emergency  │  │ Density  │ │
│   └──────────┘  └────────┘  └──────────┘  └────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────┐
                         │ AlertManager     │  priority + cooldown
                         └────────┬─────────┘
           ┌───────┬──────────────┐
           ▼       ▼              ▼
      ┌─────────┐┌────────────┐┌──────────┐
      │ Email   ││ Firebase   ││ In-app   │
      │ (SMTP)  ││ FCM        ││ Notifier │
      └─────────┘└────────────┘└──────────┘
```

## Layering

| Layer | Directory | Responsibility |
|---|---|---|
| Entry | `dashboard/`, `scripts/` | User-facing UI + CLIs |
| Pipeline | `src/pipeline/` | Orchestrates frame processing |
| Feature modules | `src/{detection,tracking,counting,prediction,anpr,…}` | Self-contained domain logic |
| Integrations | `src/{alerts,firebase_notifications,maps}` | External services |
| Cross-cutting | `src/utils/` | Env validation, auth, retries, logging, config schema |
| Config | `config/` | Dataclass settings + YAML files |

Each integration module exposes an `is_configured` property and fails soft
when credentials are absent — the pipeline never crashes because email or
Firebase is unreachable.

## Reliability patterns

* **Retries with exponential backoff** around every external API call
  ([`src/utils/retry.py`](../src/utils/retry.py)). Transient network errors
  are retried up to 3 times; auth / 4xx errors are not.
* **Typed exception handling** replaces bare `except Exception: pass`. Every
  channel in `AlertManager._dispatch` isolates failures so one broken
  transport cannot block the others, and failures are logged with context
  via `loguru`.
* **Fail-fast env validation** at startup surfaces missing secrets
  immediately instead of producing confusing runtime errors.
* **Schema-validated YAML** via [`src/utils/config_loader.py`](../src/utils/config_loader.py)
  catches typos in `density_thresholds.yaml`.

## Security

* Streamlit dashboard is behind a shared-password gate
  ([`src/utils/auth.py`](../src/utils/auth.py)). For production, swap this
  for an IdP-backed auth layer (OAuth / SAML / SSO).
* Subscriber lists, FCM tokens, and flagged-plate data live in gitignored
  JSON under `data/` — they contain PII and must not be committed.
* Secrets live only in `.env` (gitignored) or a secret manager.

## Observability

Logging is centralized in [`src/utils/logging_setup.py`](../src/utils/logging_setup.py).
Call `configure_logging()` once at entry to get:

* A coloured console sink
* A rotating file sink under `logs/app.log` (10 MB × 10, compressed)
* Optional JSON-lines sink for log aggregators (`json_sink=True`)

Set `LOG_LEVEL=DEBUG` to trace individual dispatch calls.

## Testing

* Unit tests live under [`tests/`](../tests/). External services (SMTP email,
  Firebase, OpenWeatherMap, ORS) are never hit in unit tests —
  they are mocked with `unittest.mock` or `responses`.
* Integration tests that exercise real endpoints are guarded by the
  `integration` pytest marker and excluded from CI by default.

## Persistence (SQLite)

The `src/db/` package stores audit / history data in a single SQLite file
(default `data/traffic.db`, override with `DATABASE_PATH`). Tables:

| Table              | Purpose                                            |
|--------------------|----------------------------------------------------|
| `alerts`           | Every dispatched alert, with channels and metadata |
| `notifications`    | In-app feed (replaces `outputs/notifications.json`)|
| `anpr_matches`     | Every plate read, flagged or not                   |
| `vehicle_counts`   | Per-camera, per-class time-series counts           |
| `signal_changes`   | Every traffic-signal state change                  |
| `flagged_vehicles` | Persistent ANPR blacklist                          |
| `schema_version`   | Forward-only migration ledger                      |

Wiring is **opt-in** so existing in-memory behaviour is preserved::

    from src.db import get_default_db, AlertRepository, FlaggedVehicleRepository
    db = get_default_db()
    alert_manager.set_alert_repository(AlertRepository(db))
    illegal_checker.attach_db(
        ANPRMatchRepository(db),
        FlaggedVehicleRepository(db),
    )

WAL journal mode and `synchronous=NORMAL` are enabled for concurrent reader
support. For loads beyond a few hundred writes/sec, swap the SQLite backend
for Postgres — the repository layer is the only thing that needs to change.

## Known limitations

* No REST API. Mobile / external clients should consume a forthcoming
  FastAPI wrapper rather than scraping the Streamlit UI.
* The behavioural accident detector relies on stable vehicle tracks; on
  low-resolution dashcam footage where tracks fragment during impact, use
  the motion-based detector mode instead.
