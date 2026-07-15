"""Feature engineering across tabular, time-series and text data."""

from __future__ import annotations

import pandas as pd


def add_datetime_features(df: pd.DataFrame, column: str, *, drop: bool = False) -> pd.DataFrame:
    """Expand a datetime column into calendar features.

    Adds ``<column>_year``, ``_month``, ``_day``, ``_dayofweek`` and
    ``_is_weekend`` columns — the workhorse features for most time-series and
    tabular models with a temporal component.

    Args:
        df: The source DataFrame.
        column: Name of a datetime (or datetime-parseable) column.
        drop: If ``True``, drop the original column from the result.

    Returns:
        A new DataFrame with the added feature columns.

    Raises:
        KeyError: If ``column`` is not present.
    """
    if column not in df.columns:
        raise KeyError(column)
    out = df.copy()
    ts = pd.to_datetime(out[column])
    out[f"{column}_year"] = ts.dt.year
    out[f"{column}_month"] = ts.dt.month
    out[f"{column}_day"] = ts.dt.day
    out[f"{column}_dayofweek"] = ts.dt.dayofweek
    out[f"{column}_is_weekend"] = ts.dt.dayofweek.isin((5, 6))
    if drop:
        out = out.drop(columns=[column])
    return out


__all__ = ["add_datetime_features"]
