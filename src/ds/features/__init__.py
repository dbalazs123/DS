"""Feature engineering across tabular, time-series and text data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import pandas as pd

ScaleMethod = Literal["standard", "minmax"]
BinMethod = Literal["width", "quantile"]


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


def _categorical_columns(df: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    """Resolve which columns to encode, validating any explicit names."""
    if columns is None:
        return list(df.select_dtypes(include=["object", "category"]).columns)
    resolved = list(columns)
    missing = [col for col in resolved if col not in df.columns]
    if missing:
        raise KeyError(missing)
    return resolved


def one_hot_encode(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    drop_first: bool = False,
    dummy_na: bool = False,
) -> pd.DataFrame:
    """One-hot encode categorical columns into indicator columns.

    A typed wrapper over :func:`pandas.get_dummies` that leaves numeric columns
    untouched. New columns are named ``<column>_<value>``.

    Args:
        df: The source DataFrame.
        columns: Columns to encode; ``None`` encodes every ``object``/``category``
            column.
        drop_first: Drop the first level of each column to avoid collinearity.
        dummy_na: Add an indicator column for missing values.

    Returns:
        A new DataFrame with the selected columns replaced by indicators.

    Raises:
        KeyError: If a name in ``columns`` is not a column of ``df``.
    """
    resolved = _categorical_columns(df, columns)
    return pd.get_dummies(df, columns=resolved, drop_first=drop_first, dummy_na=dummy_na)


def ordinal_encode(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    categories: Mapping[str, Sequence[object]] | None = None,
) -> pd.DataFrame:
    """Encode categorical columns as integer codes.

    Each value maps to its position in an ordered category list — either the one
    supplied in ``categories`` or, by default, the sorted unique values. Missing
    or unseen values encode as ``-1``, matching :class:`pandas.Categorical`.

    Args:
        df: The source DataFrame.
        columns: Columns to encode; ``None`` encodes every ``object``/``category``
            column.
        categories: Optional per-column ordering of the categories, giving the
            codes a meaningful rank (e.g. ``{"size": ["S", "M", "L"]}``).

    Returns:
        A new DataFrame with the selected columns replaced by integer codes.

    Raises:
        KeyError: If a name in ``columns`` is not a column of ``df``.
    """
    resolved = _categorical_columns(df, columns)
    lookup = categories or {}
    out = df.copy()
    for col in resolved:
        order = list(lookup[col]) if col in lookup else sorted(df[col].dropna().unique())
        codes = pd.Categorical(df[col], categories=order, ordered=True).codes
        out[col] = pd.Series(codes, index=df.index, dtype="int64")
    return out


def _numeric_columns(df: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    """Resolve the numeric columns to scale, validating any explicit names."""
    if columns is None:
        return list(df.select_dtypes("number").columns)
    resolved = list(columns)
    missing = [col for col in resolved if col not in df.columns]
    if missing:
        raise KeyError(missing)
    non_numeric = [col for col in resolved if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"non-numeric columns cannot be scaled: {non_numeric}")
    return resolved


def scale_features(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    method: ScaleMethod = "standard",
) -> pd.DataFrame:
    """Scale numeric columns to a common range.

    ``"standard"`` centres each column and divides by its (sample) standard
    deviation; ``"minmax"`` rescales it to ``[0, 1]``. A constant column maps to
    all zeros rather than producing ``inf``/``NaN``.

    Args:
        df: The source DataFrame.
        columns: Numeric columns to scale; ``None`` uses every numeric column.
        method: ``"standard"`` (z-score) or ``"minmax"``.

    Returns:
        A new DataFrame with the selected columns scaled.

    Raises:
        KeyError: If a name in ``columns`` is not a column of ``df``.
        ValueError: If a named column is not numeric.
    """
    resolved = _numeric_columns(df, columns)
    out = df.copy()
    for col in resolved:
        series = out[col].astype(float)
        if method == "standard":
            spread = float(series.std())
            centre = float(series.mean())
            out[col] = (series - centre) / spread if spread else 0.0
        else:
            low = float(series.min())
            span = float(series.max()) - low
            out[col] = (series - low) / span if span else 0.0
    return out


def bin_column(
    df: pd.DataFrame,
    column: str,
    *,
    bins: int | Sequence[float],
    method: BinMethod = "width",
    labels: Sequence[str] | None = None,
    drop: bool = False,
) -> pd.DataFrame:
    """Discretize a numeric column into a categorical ``<column>_bin`` column.

    ``"width"`` cuts the value range into equal-width intervals
    (:func:`pandas.cut`); ``"quantile"`` cuts into equal-frequency intervals
    (:func:`pandas.qcut`), dropping duplicate edges when the data is degenerate.

    Args:
        df: The source DataFrame.
        column: Name of the numeric column to bin.
        bins: Number of bins, or explicit edges for ``"width"`` /
            quantile fractions for ``"quantile"``.
        method: ``"width"`` (equal-width) or ``"quantile"`` (equal-frequency).
        labels: Optional labels for the resulting bins.
        drop: If ``True``, drop the original column from the result.

    Returns:
        A new DataFrame with the added ``<column>_bin`` column.

    Raises:
        KeyError: If ``column`` is not present.
    """
    if column not in df.columns:
        raise KeyError(column)
    out = df.copy()
    bin_labels: Sequence[str] | None = list(labels) if labels is not None else None
    if method == "width":
        out[f"{column}_bin"] = pd.cut(out[column], bins=bins, labels=bin_labels)
    else:
        out[f"{column}_bin"] = pd.qcut(out[column], q=bins, labels=bin_labels, duplicates="drop")
    if drop:
        out = out.drop(columns=[column])
    return out


__all__ = [
    "add_datetime_features",
    "bin_column",
    "one_hot_encode",
    "ordinal_encode",
    "scale_features",
]
