"""Time-series modeling helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
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


def forecast_recursive(
    model: Any,
    history: Sequence[float],
    *,
    lags: Sequence[int],
    steps: int,
) -> list[float]:
    """Forecast ``steps`` values ahead from a lag-feature model, recursively.

    A model trained on autoregressive features (see
    :func:`ds.features.add_lagged_features`) predicts one step from a row of the
    series' own recent values. To forecast *further* than one step it has to
    consume its own predictions as the lags of later steps — there are no
    observed values past the end of ``history`` — so this feeds each prediction
    back into the window and slides forward, the multi-step forecast a single
    ``model.predict`` cannot produce.

    At every step the lag vector handed to the model is ``[buffer[-k] for k in
    lags]``, where ``buffer`` is ``history`` extended by the predictions made so
    far. ``lags`` must therefore list the same offsets, in the same order, the
    model was trained on, and ``model`` must predict from exactly those features:
    a pure autoregression, since any exogenous feature would need a value for the
    future rows this cannot supply. (This is one-step recursion compounded, so
    error accumulates with the horizon — long-range forecasts of a noisy series
    decay toward its mean, which is honest, not a bug.)

    Args:
        model: A fitted estimator with a scikit-learn ``predict`` interface,
            trained on lag features in the order ``lags`` lists. If it exposes
            ``feature_names_in_`` (a model fitted on a named DataFrame), each
            step is passed a matching one-row frame so no feature-name warning
            is raised.
        history: The observed series in time order; its tail seeds the recursion
            and must be at least ``max(lags)`` values long.
        lags: The positive lag offsets the model expects, in feature order.
        steps: How many future values to forecast (``> 0``).

    Returns:
        A list of ``steps`` forecast values, in time order.

    Raises:
        ValueError: If ``steps`` is not positive, ``lags`` is empty or names a
            non-positive lag, or ``history`` is shorter than the largest lag.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    lag_list = list(lags)
    if not lag_list:
        raise ValueError("lags must name at least one lag")
    if min(lag_list) < 1:
        raise ValueError("lags must be positive (a lag of k looks k steps back)")
    if len(history) < max(lag_list):
        raise ValueError(
            f"history has {len(history)} values but the largest lag is {max(lag_list)}"
        )

    names = getattr(model, "feature_names_in_", None)
    buffer = [float(value) for value in history]
    predictions: list[float] = []
    for _ in range(steps):
        row = np.array([[buffer[-k] for k in lag_list]], dtype=float)
        features: Any = pd.DataFrame(row, columns=names) if names is not None else row
        yhat = float(np.asarray(model.predict(features))[0])
        predictions.append(yhat)
        buffer.append(yhat)
    return predictions


__all__ = ["forecast_recursive", "train_test_split_by_time"]
