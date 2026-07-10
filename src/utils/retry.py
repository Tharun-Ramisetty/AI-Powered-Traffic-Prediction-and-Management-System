"""Retry helpers for external API calls.

Prefers ``tenacity`` when installed (production). Falls back to a minimal
exponential-backoff decorator so tests and constrained environments still
work without the extra dependency.
"""

from __future__ import annotations

import functools
import random
import time
from typing import Any, Callable, Tuple, Type

from loguru import logger

try:
    from tenacity import (
        retry as _tenacity_retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log,
    )
    import logging
    _HAS_TENACITY = True
except ImportError:  # pragma: no cover - exercised in minimal envs
    _HAS_TENACITY = False


def with_retry(
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
):
    """Retry ``func`` with exponential backoff on the given exception types.

    Transient network failures (timeouts, 5xx responses, DNS hiccups) usually
    resolve on a second try. Permanent failures (auth errors, 4xx) are not
    retried because callers pass only the exception types that are worth
    retrying.
    """
    if _HAS_TENACITY:
        return _tenacity_retry(
            retry=retry_if_exception_type(exceptions),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=initial_wait, max=max_wait),
            before_sleep=before_sleep_log(
                logging.getLogger(__name__), logging.WARNING
            ),
            reraise=True,
        )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            wait = initial_wait
            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt >= max_attempts:
                        raise
                    sleep_for = min(wait, max_wait) * (0.5 + random.random())
                    logger.warning(
                        "{} failed (attempt {}/{}): {}. Retrying in {:.1f}s",
                        func.__name__, attempt, max_attempts, exc, sleep_for,
                    )
                    time.sleep(sleep_for)
                    wait *= 2

        return wrapper

    return decorator
