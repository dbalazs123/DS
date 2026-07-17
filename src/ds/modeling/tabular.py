"""Classical tabular modeling helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


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


def train_test_split_random(
    df: pd.DataFrame, *, test_size: float = 0.2, stratify: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly split a DataFrame into train and test sets.

    The order-free counterpart to
    :func:`ds.modeling.timeseries.train_test_split_by_time`: rows are
    shuffled before splitting, so use it only when they carry no time axis
    (a random split on temporal data trains on the future). Shuffling draws
    from numpy's global generator, so :func:`ds.seed_everything` makes the
    split reproducible.

    Args:
        df: The source DataFrame.
        test_size: Fraction of rows to hold out (0 < x < 1).
        stratify: Optional column whose class proportions both halves must
            preserve — pass the target on classification data so an
            imbalanced class stays equally represented. The column itself
            stays in both frames.

    Returns:
        A ``(train, test)`` tuple.

    Raises:
        KeyError: If ``stratify`` names a missing column.
        ValueError: If ``test_size`` is not in the open interval (0, 1), or
            (from scikit-learn) if a ``stratify`` class has fewer than two
            members.
    """
    if stratify is not None and stratify not in df.columns:
        raise KeyError(stratify)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[stratify] if stratify is not None else None
    )
    return train, test


__all__ = ["split_features_target", "train_test_split_random"]
