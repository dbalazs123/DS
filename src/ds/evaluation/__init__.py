"""Model evaluation: metrics and reporting."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
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


def confusion_frame(y_true: Sequence[int], y_pred: Sequence[int]) -> pd.DataFrame:
    """Confusion matrix as a labeled DataFrame.

    Rows are the true labels, columns the predicted ones, so ``frame.loc[t, p]``
    is the number of samples of true class ``t`` predicted as ``p``. Labels are
    the sorted union of those seen in ``y_true`` and ``y_pred``.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        A square DataFrame indexed by true label (index name ``"true"``) with
        predicted labels as columns (column name ``"predicted"``).
    """
    labels = sorted(set(y_true) | set(y_pred))
    matrix = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=pd.Index(labels, name="true"),
        columns=pd.Index(labels, name="predicted"),
    )


def per_class_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> pd.DataFrame:
    """Per-class precision, recall, F1 and support.

    Where :func:`classification_metrics` gives one averaged number per metric,
    this breaks the same metrics out for every class — the view you need to see
    which class a model is failing on.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        A DataFrame indexed by class label (index name ``"label"``) with
        ``precision``, ``recall``, ``f1`` and ``support`` columns.
    """
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support},
        index=pd.Index(labels, name="label"),
    )


__all__ = [
    "classification_metrics",
    "confusion_frame",
    "per_class_metrics",
    "regression_metrics",
]
