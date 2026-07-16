"""Cleaning and reshaping utilities."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

_NON_SNAKE = re.compile(r"[^0-9a-zA-Z]+")

OutlierMethod = Literal["iqr", "zscore"]
ImputeStrategy = Literal["mean", "median", "most_frequent", "constant"]

# Sensible per-method spread multipliers when the caller does not pick one:
# 1.5 IQRs is Tukey's classic fence, 3 standard deviations the usual z-score cut.
_DEFAULT_FACTOR: dict[OutlierMethod, float] = {"iqr": 1.5, "zscore": 3.0}


@dataclass(frozen=True)
class OutlierBounds:
    """Per-column ``(lower, upper)`` acceptance bounds learned by :func:`fit_outlier_bounds`.

    Attributes:
        bounds: Mapping of column name to its ``(lower, upper)`` bounds.
        method: The method the bounds were fitted with (``"iqr"`` or ``"zscore"``).
        factor: The spread multiplier the bounds were fitted with.
    """

    bounds: Mapping[str, tuple[float, float]]
    method: OutlierMethod
    factor: float


@dataclass(frozen=True)
class ImputeValues:
    """Per-column fill values learned by :func:`fit_impute_values`.

    Attributes:
        fill_values: Mapping of column name to the value used to fill its gaps.
        strategy: The strategy the values were fitted with.
    """

    fill_values: Mapping[str, Any]
    strategy: ImputeStrategy


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


def fit_outlier_bounds(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    method: OutlierMethod = "iqr",
    factor: float | None = None,
) -> OutlierBounds:
    """Learn per-column outlier bounds without applying them.

    Bounds come from either Tukey's IQR fence (``"iqr"``) or a z-score cut
    (``"zscore"``). Fit them on the training split, then reuse them on test
    data or new rows via :func:`apply_flag_outliers` / :func:`apply_clip_outliers`
    so held-out data never influences the bounds.

    Args:
        df: The DataFrame to learn bounds from (typically the training split).
        columns: Numeric columns to fit; ``None`` uses every numeric column.
        method: ``"iqr"`` (bounds at ``factor`` IQRs beyond the quartiles) or
            ``"zscore"`` (``factor`` standard deviations from the mean).
        factor: Spread multiplier; defaults to ``1.5`` for ``"iqr"`` and ``3.0``
            for ``"zscore"``.

    Returns:
        The learned :class:`OutlierBounds`.

    Raises:
        KeyError: If a name in ``columns`` is not a column of ``df``.
        ValueError: If a named column is not numeric.
    """
    resolved = _numeric_columns(df, columns)
    spread = _DEFAULT_FACTOR[method] if factor is None else factor
    bounds = {col: _outlier_bounds(df[col], method, spread) for col in resolved}
    return OutlierBounds(bounds=bounds, method=method, factor=spread)


def apply_flag_outliers(df: pd.DataFrame, bounds: OutlierBounds) -> pd.DataFrame:
    """Flag values outside previously learned bounds.

    The split-safe counterpart of :func:`flag_outliers`: the bounds come from
    :func:`fit_outlier_bounds` rather than from ``df`` itself. Missing values
    are never flagged, so the result composes with :func:`apply_impute_missing`.

    Args:
        df: The DataFrame to inspect.
        bounds: Bounds learned by :func:`fit_outlier_bounds`.

    Returns:
        A boolean DataFrame aligned to ``df``'s index with one column per
        fitted column, ``True`` where the value is an outlier.

    Raises:
        KeyError: If a fitted column is not a column of ``df``.
        ValueError: If a fitted column is not numeric in ``df``.
    """
    resolved = _numeric_columns(df, list(bounds.bounds))
    flags = pd.DataFrame(False, index=df.index, columns=resolved)
    for col in resolved:
        lower, upper = bounds.bounds[col]
        below_or_above = (df[col] < lower) | (df[col] > upper)
        flags[col] = below_or_above.fillna(False)
    return flags


def apply_clip_outliers(df: pd.DataFrame, bounds: OutlierBounds) -> pd.DataFrame:
    """Winsorize columns to previously learned bounds.

    The split-safe counterpart of :func:`clip_outliers`: values outside the
    bounds learned by :func:`fit_outlier_bounds` are pulled in to the nearest
    bound, keeping the row count intact. Missing values pass through unchanged.

    Args:
        df: The DataFrame to clip.
        bounds: Bounds learned by :func:`fit_outlier_bounds`.

    Returns:
        A new DataFrame with the fitted columns winsorized.

    Raises:
        KeyError: If a fitted column is not a column of ``df``.
        ValueError: If a fitted column is not numeric in ``df``.
    """
    resolved = _numeric_columns(df, list(bounds.bounds))
    out = df.copy()
    for col in resolved:
        lower, upper = bounds.bounds[col]
        out[col] = out[col].clip(lower=lower, upper=upper)
    return out


def flag_outliers(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    method: OutlierMethod = "iqr",
    factor: float | None = None,
) -> pd.DataFrame:
    """Flag out-of-range values in each numeric column.

    Convenience wrapper that fits bounds on ``df`` and flags the same frame —
    fine for exploration or pre-split cleaning. For a train/test workflow use
    :func:`fit_outlier_bounds` + :func:`apply_flag_outliers` so the bounds come
    from the training split only. Missing values are never flagged, so the
    result composes with :func:`impute_missing`.

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
    return apply_flag_outliers(df, fit_outlier_bounds(df, columns, method=method, factor=factor))


def clip_outliers(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    method: OutlierMethod = "iqr",
    factor: float | None = None,
) -> pd.DataFrame:
    """Winsorize numeric columns to their outlier bounds.

    Convenience wrapper that fits bounds on ``df`` and clips the same frame —
    fine for exploration or pre-split cleaning. For a train/test workflow use
    :func:`fit_outlier_bounds` + :func:`apply_clip_outliers` so the bounds come
    from the training split only. Values outside the bounds are pulled in to
    the nearest bound rather than dropped, keeping the row count intact.
    Missing values pass through unchanged.

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
    return apply_clip_outliers(df, fit_outlier_bounds(df, columns, method=method, factor=factor))


def fit_impute_values(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    strategy: ImputeStrategy = "mean",
    fill_value: object = None,
) -> ImputeValues:
    """Learn per-column fill values without applying them.

    Each column's fill value comes from its own non-null values. Fit on the
    training split, then reuse the values on test data or new rows via
    :func:`apply_impute_missing` so held-out data never influences the fills.
    The ``"mean"`` and ``"median"`` strategies apply only to numeric columns;
    ``"most_frequent"`` and ``"constant"`` work on any dtype.

    Args:
        df: The DataFrame to learn fill values from (typically the training
            split).
        columns: Columns to fit; ``None`` selects every numeric column for
            ``"mean"``/``"median"`` and every column otherwise.
        strategy: ``"mean"``, ``"median"``, ``"most_frequent"`` or
            ``"constant"``.
        fill_value: The value used by the ``"constant"`` strategy (also the
            fallback when an all-null column has no mode).

    Returns:
        The learned :class:`ImputeValues`.

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

    fill_values: dict[str, Any] = {}
    for col in resolved:
        series = df[col]
        if strategy == "mean":
            fill: Any = series.mean()
        elif strategy == "median":
            fill = series.median()
        elif strategy == "most_frequent":
            modes = series.mode(dropna=True)
            fill = modes.iloc[0] if not modes.empty else fill_value
        else:  # constant
            fill = fill_value
        fill_values[col] = fill
    return ImputeValues(fill_values=fill_values, strategy=strategy)


def apply_impute_missing(df: pd.DataFrame, values: ImputeValues) -> pd.DataFrame:
    """Fill missing values with previously learned fill values.

    The split-safe counterpart of :func:`impute_missing`: the fills come from
    :func:`fit_impute_values` rather than from ``df`` itself, so test data and
    new rows are filled with training-split statistics.

    Args:
        df: The DataFrame to fill.
        values: Fill values learned by :func:`fit_impute_values`.

    Returns:
        A new DataFrame with missing values filled.

    Raises:
        KeyError: If a fitted column is not a column of ``df``.
    """
    missing = [col for col in values.fill_values if col not in df.columns]
    if missing:
        raise KeyError(missing)
    out = df.copy()
    for col, fill in values.fill_values.items():
        out[col] = out[col].fillna(fill)
    return out


def impute_missing(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    strategy: ImputeStrategy = "mean",
    fill_value: object = None,
) -> pd.DataFrame:
    """Fill missing values column by column.

    Convenience wrapper that fits fill values on ``df`` and fills the same
    frame — fine for exploration or pre-split cleaning. For a train/test
    workflow use :func:`fit_impute_values` + :func:`apply_impute_missing` so
    the fills come from the training split only. The ``"mean"`` and
    ``"median"`` strategies apply only to numeric columns; ``"most_frequent"``
    and ``"constant"`` work on any dtype.

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
    return apply_impute_missing(
        df, fit_impute_values(df, columns, strategy=strategy, fill_value=fill_value)
    )


__all__ = [
    "ImputeValues",
    "OutlierBounds",
    "apply_clip_outliers",
    "apply_flag_outliers",
    "apply_impute_missing",
    "clip_outliers",
    "coerce_dtypes",
    "drop_constant_columns",
    "drop_duplicate_rows",
    "fit_impute_values",
    "fit_outlier_bounds",
    "flag_outliers",
    "impute_missing",
    "standardize_column_names",
]
