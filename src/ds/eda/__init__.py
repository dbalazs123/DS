"""Exploratory data analysis helpers."""

from __future__ import annotations

import itertools
from typing import Literal

import pandas as pd

CorrMethod = Literal["pearson", "kendall", "spearman"]


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


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report the columns that contain missing values, worst first.

    Unlike :func:`summarize`, which lists every column, this narrows to just the
    columns with gaps — the ones worth deciding what to do about.

    Args:
        df: The DataFrame to inspect.

    Returns:
        A DataFrame indexed by column name with ``n_missing`` and
        ``frac_missing`` columns, sorted by ``frac_missing`` descending. Empty
        when nothing is missing.
    """
    n = len(df)
    counts = df.isna().sum().astype(int)
    report = pd.DataFrame(
        {
            "n_missing": counts,
            "frac_missing": (counts / n) if n else 0.0,
        }
    )
    report.index.name = "column"
    report = report[report["n_missing"] > 0]
    return report.sort_values("frac_missing", ascending=False)


def top_correlations(
    df: pd.DataFrame, *, n: int = 10, method: CorrMethod = "pearson"
) -> pd.DataFrame:
    """Return the most strongly correlated pairs of numeric columns.

    Handy for spotting redundant features or potential target leakage. Each
    unordered pair appears once and pairs are ranked by absolute correlation, so
    strong negative relationships surface alongside strong positive ones.

    Args:
        df: The DataFrame to analyze; non-numeric columns are ignored.
        n: Maximum number of pairs to return.
        method: Correlation method passed to :meth:`pandas.DataFrame.corr`
            (``"pearson"``, ``"spearman"`` or ``"kendall"``).

    Returns:
        A DataFrame with ``feature_a``, ``feature_b`` and ``correlation``
        columns. Empty (but with those columns) when fewer than two numeric
        columns are present.
    """
    columns = list(df.select_dtypes("number").columns)
    if len(columns) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "correlation"])

    values = df[columns].corr(method=method).to_numpy()
    records = [
        {
            "feature_a": columns[i],
            "feature_b": columns[j],
            "correlation": float(values[i, j]),
        }
        for i, j in itertools.combinations(range(len(columns)), 2)
    ]
    result = pd.DataFrame.from_records(records)
    order = result["correlation"].abs().sort_values(ascending=False).index
    return result.loc[order].head(n).reset_index(drop=True)


def target_rate_by_category(
    df: pd.DataFrame, column: str, target: str, *, min_count: int = 1
) -> pd.DataFrame:
    """Summarize how a categorical column moves a numeric target, level by level.

    The categorical counterpart to :func:`top_correlations`, which sees only
    numeric columns: for each level of ``column`` this reports how many rows
    carry it, what fraction of the data that is, and the mean of ``target``
    within the level — the "target rate" (a 0/1 target's positive rate, or the
    group mean for any numeric target). Levels are sorted by ``target_rate``
    descending, so the most- and least-associated categories sit at the ends,
    and the overall target mean rides along as a constant ``baseline`` column to
    compare each level against. Missing values in ``column`` are reported as
    their own level (``NaN``) rather than dropped, since a category being absent
    can itself move the target.

    This is a *descriptive* summary, not a fitted feature. Averaging the target
    over all rows and feeding the result back as a model input is textbook
    target leakage; when a rate is meant to inform a modeling decision, compute
    it on the training split only, like every other fitted statistic in the
    toolkit.

    Args:
        df: The DataFrame to summarize.
        column: The categorical column to group by; its distinct values (plus
            ``NaN`` when present) become the rows.
        target: A numeric column whose per-level mean is the target rate.
        min_count: Drop levels seen fewer than this many times, so a sparse tail
            of near-empty categories does not clutter the ranking. Defaults to
            ``1`` (keep every observed level).

    Returns:
        A DataFrame indexed by category level with ``count``, ``frac``,
        ``target_rate`` and ``baseline`` columns, sorted by ``target_rate``
        descending.

    Raises:
        KeyError: If ``column`` or ``target`` is not a column of ``df``.
        ValueError: If ``target`` is not numeric.
    """
    for name in (column, target):
        if name not in df.columns:
            raise KeyError(name)
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError(f"target column {target!r} must be numeric")

    n = len(df)
    grouped = df.groupby(column, dropna=False, observed=True)[target]
    report = pd.DataFrame(
        {
            "count": grouped.size().astype(int),
            "target_rate": grouped.mean().astype(float),
        }
    )
    report["frac"] = (report["count"] / n) if n else 0.0
    report["baseline"] = float(df[target].mean()) if n else 0.0
    report = report[report["count"] >= min_count]
    report = report[["count", "frac", "target_rate", "baseline"]]
    report.index.name = column
    return report.sort_values("target_rate", ascending=False)


__all__ = [
    "missing_value_report",
    "summarize",
    "target_rate_by_category",
    "top_correlations",
]
