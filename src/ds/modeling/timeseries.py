"""Time-series modeling helpers."""

from __future__ import annotations

import pandas as pd


def train_test_split_by_time(
    df: pd.DataFrame, time_column: str, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split a DataFrame into train and test sets.

    Unlike a random split, this preserves temporal order so the test set is
    strictly in the future relative to the train set — the only valid protocol
    for forecasting evaluation.

    Args:
        df: The source DataFrame.
        time_column: Column to sort by before splitting.
        test_size: Fraction of the most recent rows to hold out (0 < x < 1).

    Returns:
        A ``(train, test)`` tuple.

    Raises:
        KeyError: If ``time_column`` is missing.
        ValueError: If ``test_size`` is not in the open interval (0, 1).
    """
    if time_column not in df.columns:
        raise KeyError(time_column)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")
    ordered = df.sort_values(time_column)
    split_at = int(len(ordered) * (1.0 - test_size))
    train = ordered.iloc[:split_at].copy()
    test = ordered.iloc[split_at:].copy()
    return train, test


__all__ = ["train_test_split_by_time"]
