"""Tests for cross-cutting modules: config, logging, reproducibility."""

from __future__ import annotations

import logging

import numpy as np

import ds
from ds.config import Settings, get_settings
from ds.reproducibility import seed_everything


def test_version_exposed() -> None:
    assert ds.__version__ == "0.1.0"


def test_settings_defaults_and_paths() -> None:
    settings = Settings()
    assert settings.random_seed == 42
    assert settings.raw_dir == settings.data_dir / "raw"
    assert settings.processed_dir == settings.data_dir / "processed"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_seed_everything_is_deterministic() -> None:
    seed_everything(123)
    first = np.random.rand(3)
    seed_everything(123)
    second = np.random.rand(3)
    assert np.allclose(first, second)


def test_get_logger_returns_logger() -> None:
    logger = ds.get_logger("ds.test", level=logging.DEBUG)
    assert logger.level == logging.DEBUG
