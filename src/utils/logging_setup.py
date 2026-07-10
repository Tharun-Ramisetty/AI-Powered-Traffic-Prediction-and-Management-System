"""Structured logging setup (loguru-based).

Call :func:`configure_logging` once at application entry (Streamlit app, CLI,
worker). Subsequent ``from loguru import logger`` calls across the codebase
will automatically use the configured sinks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

_CONFIGURED = False


def configure_logging(
    level: Optional[str] = None,
    log_dir: Optional[Path] = None,
    json_sink: bool = False,
) -> None:
    """Configure application-wide logging.

    Args:
        level: Log level (DEBUG/INFO/WARNING/ERROR). Defaults to the
            ``LOG_LEVEL`` env var, then ``INFO``.
        log_dir: Directory for rotating file logs. Defaults to ``logs/``
            under the project root (created on demand).
        json_sink: When True, also write structured JSON lines to
            ``logs/app.jsonl`` for ingestion by log aggregators.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    effective_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logger.remove()

    logger.add(
        sys.stderr,
        level=effective_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}:{function}:{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    if log_dir is None:
        log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "app.log",
        level=effective_level,
        rotation="10 MB",
        retention=10,
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    if json_sink:
        logger.add(
            log_dir / "app.jsonl",
            level=effective_level,
            rotation="10 MB",
            retention=10,
            serialize=True,
            enqueue=True,
        )

    _CONFIGURED = True
    logger.debug("Logging configured at level {} (log_dir={})",
                 effective_level, log_dir)
