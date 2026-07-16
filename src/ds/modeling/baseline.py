"""Naive baseline models every real model must beat.

A metric means nothing in isolation: an r² of 0.7 is only good if predicting
the mean scores worse. These baselines give every project's first metric a
reference point without hand-rolling one (a friction item from the real-data
taxi-fare project, where the train-mean baseline was built inline).

The baselines are deliberately tiny frozen objects, not scikit-learn
estimators: they need no feature matrix — only the training target — so
``fit_baseline(y_train)`` then ``.predict(len(y_test))`` is the whole
protocol, and the result feeds straight into :mod:`ds.evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

BaselineStrategy = Literal["mean", "naive_last", "seasonal_naive"]


@dataclass(frozen=True)
class Baseline:
    """Fitted naive-baseline state learned by :func:`fit_baseline`.

    Attributes:
        strategy: The strategy the baseline was fitted with.
        values: The values predictions cycle through — a single value for
            ``"mean"``/``"naive_last"``, the last observed season (in
            chronological order) for ``"seasonal_naive"``.
    """

    strategy: BaselineStrategy
    values: tuple[float, ...]

    def predict(self, n: int) -> list[float]:
        """Predict the next ``n`` values.

        ``"mean"`` and ``"naive_last"`` repeat their single fitted value;
        ``"seasonal_naive"`` repeats the last fitted season cyclically
        (prediction ``i`` is the value one season before it).

        Args:
            n: Number of predictions to produce (the length of the frame or
                window being scored).

        Returns:
            A list of ``n`` predicted values.

        Raises:
            ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        return [self.values[i % len(self.values)] for i in range(n)]


def fit_baseline(
    y: pd.Series,
    *,
    strategy: BaselineStrategy = "mean",
    season_length: int | None = None,
) -> Baseline:
    """Fit a naive baseline on a training target.

    Strategies:
        - ``"mean"`` — predict the training mean (the floor for any
          regression metric).
        - ``"naive_last"`` — predict the last training value; the standard
          random-walk baseline for time series (fit on a chronologically
          ordered target, e.g. after :func:`ds.modeling.timeseries.
          train_test_split_by_time`).
        - ``"seasonal_naive"`` — predict the value one season ago, cycling
          the last ``season_length`` training values; the baseline to beat
          for seasonal data (e.g. ``season_length=7`` for daily data with a
          weekly cycle).

    Args:
        y: The training target, in chronological order for the naive
            strategies. Missing values are ignored for ``"mean"`` and
            rejected in the fitted window of the naive strategies.
        strategy: One of the strategies above.
        season_length: Required for ``"seasonal_naive"``: the cycle length,
            at least 1 and at most ``len(y)``. Must be omitted otherwise.

    Returns:
        The fitted :class:`Baseline`.

    Raises:
        ValueError: If ``y`` is empty (or all-null for ``"mean"``), if
            ``season_length`` is missing/invalid for ``"seasonal_naive"`` or
            supplied for another strategy, or if the values the naive
            strategies would repeat contain nulls.
    """
    if len(y) == 0:
        raise ValueError("cannot fit a baseline on an empty series")
    if strategy != "seasonal_naive" and season_length is not None:
        raise ValueError(f"season_length only applies to 'seasonal_naive', not {strategy!r}")

    if strategy == "mean":
        mean = y.mean()
        if pd.isna(mean):
            raise ValueError("cannot fit a 'mean' baseline on an all-null series")
        return Baseline(strategy=strategy, values=(float(mean),))

    if strategy == "naive_last":
        tail = y.iloc[-1:]
    else:
        if season_length is None:
            raise ValueError("'seasonal_naive' requires season_length")
        if not 1 <= season_length <= len(y):
            raise ValueError(
                f"season_length must be between 1 and len(y)={len(y)}, got {season_length}"
            )
        tail = y.iloc[-season_length:]
    if tail.isna().any():
        raise ValueError(f"the last values a {strategy!r} baseline repeats contain nulls")
    return Baseline(strategy=strategy, values=tuple(float(value) for value in tail))


__all__ = ["Baseline", "BaselineStrategy", "fit_baseline"]
