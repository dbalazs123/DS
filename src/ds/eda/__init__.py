"""Exploratory data analysis helpers."""

from __future__ import annotations

import pandas as pd


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column summary useful for a first look at a dataset.

    The result has one row per column with its dtype, non-null count, null
    fraction, number of unique values and (for numeric columns) the mean.

    Args:
        df: The DataFrame to profile.

    Returns:
        A summary DataFrame indexed by the original column names.
    """
    n = len(df)
    records = []
    for col in df.columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        records.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "null_frac": float(series.isna().mean()) if n else 0.0,
                "n_unique": int(series.nunique(dropna=True)),
                "mean": float(series.mean()) if is_numeric and n else None,
            }
        )
    return pd.DataFrame.from_records(records).set_index("column")


__all__ = ["summarize"]
