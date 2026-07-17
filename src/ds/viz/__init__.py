"""Plotting helpers with a consistent, readable default theme."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes

from ds.eda import missing_value_report
from ds.evaluation import confusion_frame
from ds.preprocessing import OutlierMethod, flag_outliers

# A restrained, colour-blind-friendly categorical palette (Okabe-Ito).
PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]


def set_theme(context: str = "notebook") -> None:
    """Apply the toolkit's default matplotlib theme.

    Sets a clean, legible style and the shared colour cycle so every chart in
    the project looks like it belongs to the same report.

    Args:
        context: One of ``"notebook"``, ``"talk"`` or ``"paper"``; controls the
            base font size.
    """
    font_sizes = {"paper": 10.0, "notebook": 12.0, "talk": 15.0}
    mpl.rcParams.update(
        {
            "figure.figsize": (8.0, 5.0),
            "figure.dpi": 100,
            "font.size": font_sizes.get(context, 12.0),
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
            "axes.prop_cycle": cycler(color=PALETTE),
        }
    )


def _resolve_ax(ax: Axes | None) -> Axes:
    """Return ``ax`` if given, otherwise a fresh Axes on a new figure."""
    if ax is not None:
        return ax
    _, created = plt.subplots()
    return created


def plot_missingness(df: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    """Plot the fraction of missing values per column, worst first.

    A horizontal bar chart of :func:`ds.eda.missing_value_report`; columns with
    no missing values are omitted.

    Args:
        df: The DataFrame to inspect.
        ax: Existing Axes to draw on; a new figure is created when omitted.

    Returns:
        The Axes the chart was drawn on.
    """
    report = missing_value_report(df)
    ax = _resolve_ax(ax)
    ax.barh(list(report.index), report["frac_missing"].to_numpy(), color=PALETTE[0])
    ax.set_xlabel("fraction missing")
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()  # keep the worst column at the top
    ax.set_title("Missing values by column")
    return ax


def plot_outliers(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    method: OutlierMethod = "iqr",
    factor: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot the count of outliers per numeric column, worst first.

    A horizontal bar chart of :func:`ds.preprocessing.flag_outliers`; columns
    with no flagged values are omitted, mirroring how :func:`plot_missingness`
    visualizes the missing-value report.

    Args:
        df: The DataFrame to inspect.
        columns: Numeric columns to check; ``None`` uses every numeric column.
        method: ``"iqr"`` or ``"zscore"`` — see
            :func:`ds.preprocessing.flag_outliers`.
        factor: Spread multiplier passed through to ``flag_outliers``.
        ax: Existing Axes to draw on; a new figure is created when omitted.

    Returns:
        The Axes the chart was drawn on.
    """
    counts = flag_outliers(df, columns, method=method, factor=factor).sum()
    counts = counts[counts > 0].sort_values(ascending=False)
    ax = _resolve_ax(ax)
    ax.barh(list(counts.index), counts.to_numpy(), color=PALETTE[3])
    ax.set_xlabel("outlier count")
    ax.invert_yaxis()  # keep the worst column at the top
    ax.set_title(f"Outliers by column ({method})")
    return ax


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    ax: Axes | None = None,
    normalize: bool = False,
) -> Axes:
    """Plot a confusion matrix as an annotated heatmap.

    Wraps :func:`ds.evaluation.confusion_frame`; rows are true labels and
    columns predicted labels.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        ax: Existing Axes to draw on; a new figure is created when omitted.
        normalize: If ``True``, show each cell as a fraction of its true-class
            row rather than a raw count.

    Returns:
        The Axes the heatmap was drawn on.
    """
    frame = confusion_frame(y_true, y_pred)
    matrix = frame.to_numpy(dtype=float)
    if normalize:
        totals = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals != 0)

    ax = _resolve_ax(ax)
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0)
    labels = list(frame.index)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")

    fmt = ".2f" if normalize else ".0f"
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], fmt),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )
    ax.figure.colorbar(image, ax=ax)
    return ax


def plot_residuals(
    y_true: Sequence[float], y_pred: Sequence[float], *, ax: Axes | None = None
) -> Axes:
    """Plot regression residuals against predicted values.

    A residual-vs-predicted scatter with a reference line at zero — the standard
    diagnostic for spotting bias or heteroscedasticity in a regression model.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.
        ax: Existing Axes to draw on; a new figure is created when omitted.

    Returns:
        The Axes the scatter was drawn on.
    """
    predicted = np.asarray(y_pred, dtype=float)
    residuals = np.asarray(y_true, dtype=float) - predicted
    ax = _resolve_ax(ax)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.scatter(predicted, residuals, alpha=0.7, color=PALETTE[0])
    ax.set_xlabel("predicted")
    ax.set_ylabel("residual (actual − predicted)")
    ax.set_title("Residuals vs predicted")
    return ax


def plot_model_comparison(
    comparison: pd.DataFrame, *, metric: str | None = None, ax: Axes | None = None
) -> Axes:
    """Plot one metric from a model-comparison frame, best-labeled first.

    A horizontal bar chart of a :func:`ds.evaluation.compare_models` frame
    (one row per model), mirroring how :func:`plot_missingness` visualizes
    the missing-value report. Models are shown in the frame's row order, so
    sort the frame first if a ranking is wanted.

    Args:
        comparison: Frame with models as the index and metrics as columns.
        metric: Column to plot; defaults to the frame's first column.
        ax: Existing Axes to draw on; a new figure is created when omitted.

    Returns:
        The Axes the chart was drawn on.

    Raises:
        KeyError: If ``metric`` is not a column of ``comparison``.
        ValueError: If ``comparison`` has no columns to plot.
    """
    if comparison.columns.empty:
        raise ValueError("comparison frame has no metric columns to plot")
    if metric is None:
        metric = str(comparison.columns[0])
    elif metric not in comparison.columns:
        raise KeyError(metric)
    ax = _resolve_ax(ax)
    names = [str(name) for name in comparison.index]
    ax.barh(names, comparison[metric].to_numpy(), color=PALETTE[0])
    ax.set_xlabel(metric)
    ax.invert_yaxis()  # keep the frame's first model at the top
    ax.set_title(f"Model comparison — {metric}")
    return ax


def plot_series(
    time: Sequence[object] | pd.Series,
    values: Sequence[float] | pd.Series,
    *,
    predictions: Mapping[str, Sequence[float] | pd.Series] | None = None,
    label: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot one series over time, optionally overlaid with prediction series.

    The time-series workhorse: a solid line for the observed values and, when
    ``predictions`` is given, one dashed line per named prediction series over
    the same time axis — the standard forecast-vs-actual visual. Lines take
    their colours from the Axes' colour cycle (the shared palette after
    :func:`set_theme`), so repeated calls on the same ``ax`` compose without
    colliding — e.g. draw the training tail first, then the held-out window
    with its forecasts:

    .. code-block:: python

        ax = plot_series(history["date"], history["y"], label="history")
        plot_series(
            test["date"],
            y_test,
            predictions={"model": preds, "seasonal naive": naive_preds},
            label="actual",
            ax=ax,
        )

    Args:
        time: Time-axis values (datetimes, or anything matplotlib can order).
        values: Observed values, aligned with ``time``.
        predictions: Optional mapping of series name to predicted values, each
            aligned with ``time``; drawn dashed, named in the legend.
        label: Legend label for the observed series; it stays out of the
            legend when omitted.
        ax: Existing Axes to draw on; a new figure is created when omitted.

    Returns:
        The Axes the series was drawn on.

    Raises:
        ValueError: If ``values`` (or any prediction series) differs in length
            from ``time``.
    """
    named = dict(predictions or {})
    if len(values) != len(time):
        raise ValueError(f"values has {len(values)} points but time has {len(time)}")
    for name, preds in named.items():
        if len(preds) != len(time):
            raise ValueError(
                f"predictions[{name!r}] has {len(preds)} points but time has {len(time)}"
            )
    # A Series keeps datetimes as datetime64, which matplotlib's date
    # converter understands regardless of how `time` was passed in.
    axis = pd.Series(time)
    ax = _resolve_ax(ax)
    ax.plot(axis, values, label=label)
    for name, preds in named.items():
        ax.plot(axis, preds, linestyle="--", label=name)
    if label is not None or named:
        ax.legend()
    return ax


__all__ = [
    "PALETTE",
    "plot_confusion_matrix",
    "plot_missingness",
    "plot_model_comparison",
    "plot_outliers",
    "plot_residuals",
    "plot_series",
    "set_theme",
]
