"""Model evaluation: metrics, cross-validation and model comparison."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold

from ds.pipeline import Pipeline


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
        average: Averaging strategy for multiclass precision/recall/F1. The
            ``"binary"`` default raises beyond two classes — pass ``"macro"``
            (or ``"weighted"``) there, and for the two-argument ``metrics_fn``
            hooks (:func:`cross_validate_kfold`, :func:`compare_models`) bind
            it with ``functools.partial(classification_metrics, average="macro")``.

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


def confusion_frame(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Confusion matrix as a labeled DataFrame.

    Rows are the true labels, columns the predicted ones, so ``frame.loc[t, p]``
    is the number of samples of true class ``t`` predicted as ``p``. Labels are
    the sorted union of those seen in ``y_true`` and ``y_pred``.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Optional display names per integer code, applied to both axes
            after the matrix is computed — the metric math stays on the int
            codes. Codes absent from the mapping keep their integer form.

    Returns:
        A square DataFrame indexed by true label (index name ``"true"``) with
        predicted labels as columns (column name ``"predicted"``).
    """
    codes = sorted(set(y_true) | set(y_pred))
    matrix = metrics.confusion_matrix(y_true, y_pred, labels=codes)
    frame = pd.DataFrame(
        matrix,
        index=pd.Index(codes, name="true"),
        columns=pd.Index(codes, name="predicted"),
    )
    if labels is not None:
        frame = frame.rename(index=dict(labels), columns=dict(labels))
    return frame


def per_class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Per-class precision, recall, F1 and support.

    Where :func:`classification_metrics` gives one averaged number per metric,
    this breaks the same metrics out for every class — the view you need to see
    which class a model is failing on.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Optional display names per integer code, applied to the index
            after the metrics are computed — the metric math stays on the int
            codes. Codes absent from the mapping keep their integer form.

    Returns:
        A DataFrame indexed by class label (index name ``"label"``) with
        ``precision``, ``recall``, ``f1`` and ``support`` columns.
    """
    codes = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        y_true, y_pred, labels=codes, zero_division=0
    )
    frame = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support},
        index=pd.Index(codes, name="label"),
    )
    if labels is not None:
        frame = frame.rename(index=dict(labels))
    return frame


# Any fold-scoring function with the shape of `regression_metrics` /
# `classification_metrics`: two aligned sequences in, named scores out.
MetricsFunction = Callable[[Sequence[Any], Sequence[Any]], Mapping[str, float]]


def _fold_boundaries(n_rows: int, n_blocks: int) -> list[int]:
    """Split ``range(n_rows)`` into ``n_blocks`` contiguous, non-empty blocks.

    Returns the ``n_blocks + 1`` boundary offsets (first ``0``, last
    ``n_rows``), sized as evenly as integer division allows.
    """
    base, remainder = divmod(n_rows, n_blocks)
    boundaries = [0]
    for block in range(n_blocks):
        boundaries.append(boundaries[-1] + base + (1 if block < remainder else 0))
    return boundaries


def cross_validate_by_time(
    df: pd.DataFrame,
    *,
    time_column: str,
    target: str,
    make_model: Callable[[], Any],
    make_pipeline: Callable[[pd.DataFrame], Pipeline] | None = None,
    n_splits: int = 5,
    metrics_fn: MetricsFunction = regression_metrics,
) -> pd.DataFrame:
    """Rolling-origin cross-validation for time-ordered data.

    The frame is sorted by ``time_column`` and cut into ``n_splits + 1``
    contiguous blocks; fold ``i`` trains on the first ``i`` blocks and tests
    on block ``i + 1``, so every test window is strictly in its training
    data's future — the repeated-fold counterpart to
    :func:`ds.modeling.timeseries.train_test_split_by_time`, and the only
    valid protocol for forecasting evaluation (a shuffled k-fold would train
    on the future). ``time_column`` and ``target`` are excluded from the
    feature matrix.

    Args:
        df: The modeling frame (features + target + time column). With
            ``make_pipeline`` set, pass the frame *before* the fit-based
            transforms — the point is to re-fit them inside each fold.
        time_column: Column to order by.
        target: Name of the target column.
        make_model: Zero-argument factory returning a **fresh** unfitted
            model with scikit-learn's ``fit``/``predict`` protocol (e.g.
            ``lambda: LinearRegression()``); a new instance is built per fold
            so no state leaks between folds.
        make_pipeline: Factory fitting a **fresh** transform
            :class:`~ds.pipeline.Pipeline` on a training frame — typically
            ``lambda frame: fit_pipeline(frame, plan)`` with the same plan the
            training run uses. Called once per fold with that fold's expanding
            training window (time and target columns included); the fitted
            pipeline is applied to both fold halves before the model sees
            them, so every fold's statistics (imputation fills, scale
            parameters, learned category vocabularies, …) are learned from its
            own past only — the rolling-origin twin of
            :func:`cross_validate_kfold`'s parameter. Without it, handing this
            function an already-transformed frame leaks: each fold's future
            test block has influenced the statistics its training rows were
            transformed with.
        n_splits: Number of folds (each fold adds one more block of history).
        metrics_fn: Fold-scoring function; defaults to
            :func:`regression_metrics` (use :func:`classification_metrics`
            for classifiers).

    Returns:
        A DataFrame with one row per fold (index name ``"fold"``, starting at
        1) carrying ``train_size``, ``test_size`` and one column per metric.

    Raises:
        KeyError: If ``time_column`` or ``target`` is missing.
        ValueError: If ``n_splits < 1`` or ``df`` has fewer than
            ``n_splits + 1`` rows.
    """
    missing = [col for col in (time_column, target) if col not in df.columns]
    if missing:
        raise KeyError(missing)
    if n_splits < 1:
        raise ValueError(f"n_splits must be at least 1, got {n_splits}")
    if len(df) < n_splits + 1:
        raise ValueError(
            f"need at least n_splits + 1 = {n_splits + 1} rows for non-empty folds, got {len(df)}"
        )

    ordered = df.sort_values(time_column)
    boundaries = _fold_boundaries(len(ordered), n_splits + 1)
    rows: list[dict[str, float]] = []
    for fold in range(1, n_splits + 1):
        train = ordered.iloc[: boundaries[fold]]
        test = ordered.iloc[boundaries[fold] : boundaries[fold + 1]]
        if make_pipeline is not None:
            pipeline = make_pipeline(train)
            train = pipeline.apply(train)
            test = pipeline.apply(test)
        model = make_model()
        model.fit(train.drop(columns=[time_column, target]), train[target])
        predictions = model.predict(test.drop(columns=[time_column, target]))
        scores = metrics_fn(test[target].tolist(), list(predictions))
        rows.append({"train_size": float(len(train)), "test_size": float(len(test)), **scores})
    return pd.DataFrame(rows, index=pd.RangeIndex(1, n_splits + 1, name="fold"))


def cross_validate_kfold(
    df: pd.DataFrame,
    *,
    target: str,
    make_model: Callable[[], Any],
    make_pipeline: Callable[[pd.DataFrame], Pipeline] | None = None,
    n_splits: int = 5,
    shuffle: bool = True,
    stratify: bool = False,
    metrics_fn: MetricsFunction = regression_metrics,
) -> pd.DataFrame:
    """Plain k-fold cross-validation for order-free tabular data.

    Wraps :class:`sklearn.model_selection.KFold` (or
    :class:`~sklearn.model_selection.StratifiedKFold` with ``stratify=True``)
    and scores every fold with the stage's own metric helpers, returning the
    same per-fold frame as :func:`cross_validate_by_time`. Use that function
    instead whenever the rows are time-ordered — a shuffled k-fold on
    temporal data trains on the future. Shuffling draws from numpy's global
    generator, so :func:`ds.seed_everything` makes the folds reproducible.

    Args:
        df: The modeling frame (features + target). With ``make_pipeline``
            set, pass the frame *before* the fit-based transforms — the
            point is to re-fit them inside each fold.
        target: Name of the target column.
        make_model: Zero-argument factory returning a **fresh** unfitted
            model with scikit-learn's ``fit``/``predict`` protocol; a new
            instance is built per fold.
        make_pipeline: Factory fitting a **fresh** transform
            :class:`~ds.pipeline.Pipeline` on a training frame — typically
            ``lambda frame: fit_pipeline(frame, plan)`` with the same plan
            the training run uses. Called once per fold with that fold's
            training rows (target column included); the fitted pipeline is
            applied to both fold halves before the model sees them. Without
            it, handing this function an already-transformed frame leaks:
            every fold's test rows have influenced the statistics (imputation
            fills, scale parameters, …) its training rows were transformed
            with.
        n_splits: Number of folds.
        shuffle: Whether to shuffle rows before splitting.
        stratify: Keep every fold's target class balance at the frame's —
            the option for classification targets (composes with
            ``metrics_fn=classification_metrics``), where plain ``KFold``
            lets fold class proportions drift. scikit-learn warns (via
            ``UserWarning``) when a class has fewer than ``n_splits``
            members, leaving some folds without that class.
        metrics_fn: Fold-scoring function; defaults to
            :func:`regression_metrics` (use :func:`classification_metrics`
            for classifiers).

    Returns:
        A DataFrame with one row per fold (index name ``"fold"``, starting at
        1) carrying ``train_size``, ``test_size`` and one column per metric.

    Raises:
        KeyError: If ``target`` is missing.
        ValueError: If ``n_splits`` is not between 2 and ``len(df)``.
    """
    if target not in df.columns:
        raise KeyError(target)
    if not 2 <= n_splits <= len(df):
        raise ValueError(f"n_splits must be between 2 and len(df)={len(df)}, got {n_splits}")

    rows: list[dict[str, float]] = []
    folds = (
        StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
        if stratify
        else KFold(n_splits=n_splits, shuffle=shuffle)
    )
    for train_idx, test_idx in folds.split(df.drop(columns=[target]), df[target]):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        if make_pipeline is not None:
            pipeline = make_pipeline(train)
            train = pipeline.apply(train)
            test = pipeline.apply(test)
        model = make_model()
        model.fit(train.drop(columns=[target]), train[target])
        predictions = model.predict(test.drop(columns=[target]))
        scores = metrics_fn(test[target].tolist(), list(predictions))
        rows.append(
            {"train_size": float(len(train_idx)), "test_size": float(len(test_idx)), **scores}
        )
    return pd.DataFrame(rows, index=pd.RangeIndex(1, n_splits + 1, name="fold"))


def compare_models(
    y_true: Sequence[Any],
    predictions: Mapping[str, Sequence[Any]],
    *,
    metrics_fn: MetricsFunction = regression_metrics,
) -> pd.DataFrame:
    """Score several models' predictions on one target, side by side.

    The named-row companion to the single-model metric helpers: score each
    candidate (including a :func:`ds.modeling.baseline.fit_baseline`
    reference — a model only means something relative to the naive floor)
    and get one frame to read or hand to
    :func:`ds.viz.plot_model_comparison`.

    Args:
        y_true: Ground-truth target values.
        predictions: Mapping of model name to that model's predictions,
            each aligned with ``y_true``. Row order follows the mapping.
        metrics_fn: Scoring function; defaults to :func:`regression_metrics`.

    Returns:
        A DataFrame with one row per model (index name ``"model"``) and one
        column per metric.

    Raises:
        ValueError: If ``predictions`` is empty or a prediction sequence's
            length differs from ``y_true``'s.
    """
    if not predictions:
        raise ValueError("predictions must name at least one model")
    mismatched = {
        name: len(y_pred) for name, y_pred in predictions.items() if len(y_pred) != len(y_true)
    }
    if mismatched:
        raise ValueError(
            f"predictions must align with y_true (length {len(y_true)}); mismatched: {mismatched}"
        )
    rows = {name: dict(metrics_fn(y_true, y_pred)) for name, y_pred in predictions.items()}
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "model"
    return frame


__all__ = [
    "MetricsFunction",
    "classification_metrics",
    "compare_models",
    "confusion_frame",
    "cross_validate_by_time",
    "cross_validate_kfold",
    "per_class_metrics",
    "regression_metrics",
]
