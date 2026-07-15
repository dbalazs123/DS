"""Data-quality and schema checks.

Lightweight, dependency-free guards for the assumptions a pipeline makes about
its inputs. For richer, declarative schemas reach for ``pandera`` (a core
dependency) directly; these helpers cover the common inline cases.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


class DataValidationError(ValueError):
    """Raised when a DataFrame violates an expected constraint."""


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Assert that ``df`` contains every column in ``columns``.

    Args:
        df: The DataFrame to check.
        columns: Column names that must be present.

    Returns:
        The same DataFrame, to support fluent chaining.

    Raises:
        DataValidationError: If any required column is missing.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    return df


def assert_no_nulls(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Assert that the given columns (or the whole frame) contain no nulls.

    Args:
        df: The DataFrame to check.
        columns: Columns to check; ``None`` checks all columns.

    Returns:
        The same DataFrame, to support fluent chaining.

    Raises:
        DataValidationError: If nulls are found.
    """
    subset = df if columns is None else df[list(columns)]
    null_counts = subset.isna().sum()
    offenders = null_counts[null_counts > 0]
    if not offenders.empty:
        raise DataValidationError(f"Null values found in: {offenders.to_dict()}")
    return df


__all__ = ["DataValidationError", "assert_no_nulls", "require_columns"]
