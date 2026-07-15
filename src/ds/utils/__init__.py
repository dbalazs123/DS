"""Small cross-cutting helpers used across the toolkit."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from ds.logging import get_logger


@contextmanager
def timer(label: str = "block") -> Iterator[None]:
    """Context manager that logs the wall-clock time of the wrapped block.

    Args:
        label: A human-readable name for the timed section.

    Yields:
        Control to the wrapped block.
    """
    logger = get_logger(__name__)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s took %.3fs", label, elapsed)


__all__ = ["timer"]
