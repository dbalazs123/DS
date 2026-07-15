"""Model evaluation: metrics and reporting."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn import metrics


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Compute common regression metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        A dict with ``mae``, ``rmse`` and ``r2``.
    """
    mse = float(metrics.mean_squared_error(y_true, y_pred))
    return {
        "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
        "rmse": mse**0.5,
        "r2": float(metrics.r2_score(y_true, y_pred)),
    }


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], *, average: str = "binary"
) -> dict[str, float]:
    """Compute common classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        average: Averaging strategy for multiclass precision/recall/F1.

    Returns:
        A dict with ``accuracy``, ``precision``, ``recall`` and ``f1``.
    """
    return {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "precision": float(
            metrics.precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(metrics.recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(metrics.f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


__all__ = ["classification_metrics", "regression_metrics"]
