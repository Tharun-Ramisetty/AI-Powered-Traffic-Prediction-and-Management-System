# Contributing

## Local setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pre-commit install
```

## Workflow

1. Branch from `develop`: `git checkout -b feat/short-name`.
2. Write a test first when changing logic in `src/`.
3. Run the checks locally before pushing:
   ```bash
   ruff check src tests config
   ruff format src tests config
   pytest
   ```
4. Open a PR against `develop`. CI must be green before merge.

## Coding conventions

* Python 3.10+, type-annotated public APIs.
* `loguru` for logging — never `print`.
* No bare `except Exception: pass`. Catch specific types; when a broad
  catch is unavoidable (e.g. subscriber callbacks), use `logger.exception`
  to keep the traceback.
* External API calls go through `src/utils/retry.with_retry` with an
  explicit retryable-exception tuple.
* New env variables must be added to [`src/utils/env_validator.py`](../src/utils/env_validator.py)
  *and* [`.env.example`](../.env.example).

## Tests

* Unit tests only in CI — mock Twilio, Firebase, weather, ORS.
* Mark slow or network-hitting tests with `@pytest.mark.integration` or
  `@pytest.mark.slow` so they are skipped by default.
* Target ≥ 70% coverage for new `src/utils` and `src/alerts` code.

## Security

* Never commit real API keys, even temporarily.
* Rotate anything that leaks into Git history immediately.
* `pre-commit` runs `detect-private-key` and `check-added-large-files` to
  catch accidents early.
