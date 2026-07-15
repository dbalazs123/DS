"""Consistent logging setup for library and notebook use alike."""

from __future__ import annotations

import logging

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str | int = "INFO") -> None:
    """Configure root logging once, idempotently.

    Safe to call from notebooks or scripts repeatedly; only the first call
    installs handlers.

    Args:
        level: Logging level name or numeric value.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(level=level, format=_FORMAT)
    _CONFIGURED = True


def get_logger(name: str, level: str | int | None = None) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name, conventionally ``__name__`` of the caller.
        level: Optional level override for this logger.

    Returns:
        A ready-to-use :class:`logging.Logger`.
    """
    configure_logging()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
