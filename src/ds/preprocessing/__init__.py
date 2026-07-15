"""Cleaning and reshaping utilities."""

from __future__ import annotations

import re

import pandas as pd

_NON_SNAKE = re.compile(r"[^0-9a-zA-Z]+")


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


__all__ = ["drop_constant_columns", "standardize_column_names"]
