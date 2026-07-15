"""Classical tabular modeling helpers."""

from __future__ import annotations

import pandas as pd


def split_features_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into a feature matrix ``X`` and target vector ``y``.

    Args:
        df: The full modeling frame.
        target: Name of the target column.

    Returns:
        A ``(X, y)`` tuple where ``X`` excludes the target column.

    Raises:
        KeyError: If ``target`` is not a column of ``df``.
    """
    if target not in df.columns:
        raise KeyError(target)
    x = df.drop(columns=[target])
    y = df[target]
    return x, y


__all__ = ["split_features_target"]
