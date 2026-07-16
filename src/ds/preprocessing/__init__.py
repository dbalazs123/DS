"""Cleaning and reshaping utilities."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

_NON_SNAKE = re.compile(r"[^0-9a-zA-Z]+")

OutlierMethod = Literal["iqr", "zscore"]
ImputeStrategy = Literal["mean", "median", "most_frequent", "constant"]

# Sensible per-method spread multipliers when the caller does not pick one:
# 1.5 IQRs is Tukey's classic fence, 3 standard deviations the usual z-score cut.
_DEFAULT_FACTOR: dict[OutlierMethod, float] = {"iqr": 1.5, "zscore": 3.0}


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``snake_case`` column names.

    Spaces and punctuation collapse to single underscores and everything is
    lower-cased, so ``"Total Sales ($)"`` becomes ``"total_sales"``.

    Args:
        df: The DataFrame whose columns to rename.

    Returns:
        A new DataFrame with cleaned column names.
    """
    renamed = {col: _NON_SNAKE.sub("_", str(col)).strip("_").lower() for col in df.columns}
    return df.rename(columns=renamed)


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that hold a single unique value (ignoring nulls).

    Constant columns carry no signal for modeling and only add noise.

    Args:
        df: The DataFrame to prune.

    Returns:
        A new DataFrame without constant columns.
    """
    keep = [col for col in df.columns if df[col].nunique(dropna=True) > 1]
    return df[keep].copy()


def drop_duplicate_rows(
    df: pd.DataFrame,
    subset: Sequence[str] | None = None,
    *,
    keep: Literal["first", "last", False] = "first",
) -> pd.DataFrame:
    """Drop duplicate rows, optionally comparing only a subset of columns.

    A thin, typed wrapper over :meth:`pandas.DataFrame.drop_duplicates` that
    always returns a fresh frame so the original is never mutated.

    Args:
        df: The DataFrame to de-duplicate.
        subset: Columns to consider when identifying duplicates; ``None`` uses
            every column.
        keep: Which duplicate to retain — ``"first"``, ``"last"`` or ``False``
            to drop every row that has a duplicate.

    Returns:
        A new DataFrame without the duplicate rows.

    Raises:
        KeyError: If any name in ``subset`` is not a column of ``df``.
    """
    columns = list(subset) if subset is not None else None
    if columns is not None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise KeyError(missing)
    return df.drop_duplicates(subset=columns, keep=keep).copy()


def coerce_dtypes(
    df: pd.DataFrame,
    dtypes: Mapping[str, str | type],
    *,
    errors: Literal["raise", "ignore"] = "raise",
) -> pd.DataFrame:
    """Cast the given columns to the requested dtypes.

    Loaders often hand back everything as ``object``; this pins each column to
    the dtype the rest of the pipeline expects. With ``errors="raise"`` a value
    that will not convert stops the pipeline; with ``errors="ignore"`` the
    offending column is left untouched.

    Args:
        df: The source DataFrame.
        dtypes: Mapping of column name to a dtype (e.g. ``"int64"``,
            ``"category"``, ``float``).
        errors: ``"raise"`` to propagate conversion errors, ``"ignore"`` to
            leave a column unchanged when it cannot be cast.

    Returns:
        A new DataFrame with the requested columns re-typed.

    Raises:
        KeyError: If a key in ``dtypes`` is not a column of ``df``.
    """
    missing = [col for col in dtypes if col not in df.columns]
    if missing:
        raise KeyError(missing)
    return df.astype(dict(dtypes), errors=errors)


def _numeric_columns(df: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    """Resolve the numeric columns to operate on, validating explicit names."""
    if columns is None:
        return list(df.select_dtypes("number").columns)
    resolved = list(columns)
    missing = [col for col in resolved if col not in df.columns]
    if missing:
        raise KeyError(missing)
    non_numeric = [col for col in resolved if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"non-numeric columns cannot be treated as outliers: {non_numeric}")
    return resolved


def _outlier_bounds(series: pd.Series, method: OutlierMethod, factor: float) -> tuple[float, float]:
    """Return the ``(lower, upper)`` acceptance bounds for a numeric series."""
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return -np.inf, np.inf
    if method == "iqr":
        q1 = float(np.nanpercentile(values, 25))
        q3 = float(np.nanpercentile(values, 75))
        spread = q3 - q1
        return q1 - factor * spread, q3 + factor * spread
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    return mean - factor * std, mean + factor * std


def flag_outliers(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    method: OutlierMethod = "iqr",
    factor: float | None = None,
) -> pd.DataFrame:
    """Flag out-of-range values in each numeric column.

    Bounds come from either Tukey's IQR fence (``"iqr"``) or a z-score cut
    (``"zscore"``). Missing values are never flagged, so the result composes
    with :func:`impute_missing`.

    Args:
        df: The DataFrame to inspect.
        columns: Numeric columns to check; ``None`` uses every numeric column.
        method: ``"iqr"`` (bounds at ``factor`` IQRs beyond the quartiles) or
            ``"zscore"`` (``factor`` standard deviations from the mean).
        factor: Spread multiplier; defaults to ``1.5`` for ``"iqr"`` and ``3.0``
            for ``"zscore"``.

    Returns:
        A boolean DataFrame aligned to ``df``'s index with one column per
        checked column, ``True`` where the value is an outlier.

    Raises:
        KeyError: If a name in ``columns`` is not a column of ``df``.
        ValueError: If a named column is not numeric.
    """
    resolved = _numeric_columns(df, columns)
    spread = _DEFAULT_FACTOR[method] if factor is None else factor
    flags = pd.DataFrame(False, index=df.index, columns=resolved)
    for col in resolved:
        lower, upper = _outlier_bounds(df[col], method, spread)
        below_or_above = (df[col] < lower) | (df[col] > upper)
        flags[col] = below_or_above.fillna(False)
    return flags


def clip_outliers(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    method: OutlierMethod = "iqr",
    factor: float | None = None,
) -> pd.DataFrame:
    """Winsorize numeric columns to their outlier bounds.

    Values outside the bounds used by :func:`flag_outliers` are pulled in to the
    nearest bound rather than dropped, keeping the row count intact. Missing
    values pass through unchanged.

    Args:
        df: The source DataFrame.
        columns: Numeric columns to clip; ``None`` uses every numeric column.
        method: ``"iqr"`` or ``"zscore"`` — see :func:`flag_outliers`.
        factor: Spread multiplier; defaults to ``1.5`` for ``"iqr"`` and ``3.0``
            for ``"zscore"``.

    Returns:
        A new DataFrame with the selected columns winsorized.

    Raises:
        KeyError: If a name in ``columns`` is not a column of ``df``.
        ValueError: If a named column is not numeric.
    """
    resolved = _numeric_columns(df, columns)
    spread = _DEFAULT_FACTOR[method] if factor is None else factor
    out = df.copy()
    for col in resolved:
        lower, upper = _outlier_bounds(df[col], method, spread)
        out[col] = out[col].clip(lower=lower, upper=upper)
    return out


def impute_missing(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    strategy: ImputeStrategy = "mean",
    fill_value: object = None,
) -> pd.DataFrame:
    """Fill missing values column by column.

    Each column is filled independently from its own non-null values, so a
    single call can clean a whole frame. The ``"mean"`` and ``"median"``
    strategies apply only to numeric columns; ``"most_frequent"`` and
    ``"constant"`` work on any dtype.

    Args:
        df: The source DataFrame.
        columns: Columns to impute; ``None`` selects every numeric column for
            ``"mean"``/``"median"`` and every column otherwise.
        strategy: ``"mean"``, ``"median"``, ``"most_frequent"`` or
            ``"constant"``.
        fill_value: The value used by the ``"constant"`` strategy (also the
            fallback when an all-null column has no mode).

    Returns:
        A new DataFrame with missing values filled.

    Raises:
        KeyError: If a name in ``columns`` is not a column of ``df``.
        ValueError: If ``"mean"``/``"median"`` is used on a non-numeric column.
    """
    numeric_only = strategy in ("mean", "median")
    if columns is None:
        resolved = list(df.select_dtypes("number").columns if numeric_only else df.columns)
    else:
        resolved = list(columns)
        missing = [col for col in resolved if col not in df.columns]
        if missing:
            raise KeyError(missing)
        if numeric_only:
            bad = [col for col in resolved if not pd.api.types.is_numeric_dtype(df[col])]
            if bad:
                raise ValueError(f"{strategy} imputation needs numeric columns: {bad}")

    out = df.copy()
    for col in resolved:
        series = out[col]
        if strategy == "mean":
            fill: Any = series.mean()
        elif strategy == "median":
            fill = series.median()
        elif strategy == "most_frequent":
            modes = series.mode(dropna=True)
            fill = modes.iloc[0] if not modes.empty else fill_value
        else:  # constant
            fill = fill_value
        out[col] = series.fillna(fill)
    return out


__all__ = [
    "clip_outliers",
    "coerce_dtypes",
    "drop_constant_columns",
    "drop_duplicate_rows",
    "flag_outliers",
    "impute_missing",
    "standardize_column_names",
]
