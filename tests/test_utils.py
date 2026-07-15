"""Tests for the utils subpackage."""

from __future__ import annotations

import logging

import pytest

from ds.utils import timer


def test_timer_logs_elapsed(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="ds.utils"), timer("unit-test-block"):
        total = sum(range(1000))
    assert total == 499500
    assert any("unit-test-block took" in record.message for record in caplog.records)
