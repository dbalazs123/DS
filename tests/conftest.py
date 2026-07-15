"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small, mixed-type frame used across tests."""
    return pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-06", "2024-01-07"],
            "Total Sales ($)": [100.0, 200.0, 300.0],
            "constant": [1, 1, 1],
            "category": ["a", "b", "a"],
        }
    )
