"""Tests for the plotting helpers.

The backend is switched to the headless ``Agg`` at runtime so these run without
a display and never open a window.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ds.viz import (
    plot_confusion_matrix,
    plot_missingness,
    plot_outliers,
    plot_pr_curve,
    plot_residuals,
    plot_roc_curve,
    plot_series,
    plot_target_rate,
)


@pytest.fixture(autouse=True)
def _headless() -> Iterator[None]:
    plt.switch_backend("Agg")
    yield
    plt.close("all")


def test_plot_missingness_bars_only_missing_columns() -> None:
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    ax = plot_missingness(df)
    # One bar, for column "a" only.
    assert len(ax.patches) == 1
    assert [label.get_text() for label in ax.get_yticklabels()] == ["a"]


def test_plot_outliers_bars_only_flagged_columns() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100], "clean": [1, 2, 3, 4, 5, 6]})
    ax = plot_outliers(df)
    # One bar, for column "x" only (the 100 is an outlier).
    assert len(ax.patches) == 1
    assert [label.get_text() for label in ax.get_yticklabels()] == ["x"]


def test_plot_confusion_matrix_annotates_every_cell() -> None:
    ax = plot_confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0])
    # 2 classes -> 2x2 = 4 annotated cells.
    assert len(ax.texts) == 4
    assert ax.get_xlabel() == "predicted"
    assert ax.get_ylabel() == "true"


def test_plot_confusion_matrix_normalizes_rows() -> None:
    ax = plot_confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], normalize=True)
    texts = {t.get_text() for t in ax.texts}
    # True-class 1 was always predicted 1 -> a normalized 1.00 cell exists.
    assert "1.00" in texts


def test_plot_confusion_matrix_display_labels_name_the_ticks() -> None:
    ax = plot_confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0], labels={0: "no", 1: "yes"})
    assert [label.get_text() for label in ax.get_xticklabels()] == ["no", "yes"]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["no", "yes"]
    assert len(ax.texts) == 4  # counts still annotated per cell


def test_plot_pr_curve_draws_curve_and_prevalence_floor() -> None:
    ax = plot_pr_curve([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], label="model")
    assert ax.get_xlabel() == "recall"
    assert ax.get_ylabel() == "precision"
    # A dashed no-skill line at the positive rate (2/4 = 0.5).
    baseline = next(line for line in ax.get_lines() if line.get_linestyle() == "--")
    assert baseline.get_ydata()[0] == pytest.approx(0.5)


def test_plot_roc_curve_draws_curve_and_chance_diagonal() -> None:
    ax = plot_roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert ax.get_xlabel() == "false positive rate"
    # The chance diagonal runs corner to corner.
    diagonal = next(line for line in ax.get_lines() if line.get_linestyle() == "--")
    assert list(diagonal.get_xdata()) == [0.0, 1.0]
    assert list(diagonal.get_ydata()) == [0.0, 1.0]


def test_plot_residuals_draws_points_and_zero_line() -> None:
    ax = plot_residuals([1.0, 2.0, 3.0], [1.5, 1.5, 3.5])
    assert ax.collections  # the scatter
    assert ax.get_ylabel().startswith("residual")


def test_plot_series_overlays_dashed_predictions_with_a_legend() -> None:
    time = pd.date_range("2024-01-01", periods=4, freq="MS")
    ax = plot_series(
        time,
        [1.0, 2.0, 3.0, 4.0],
        predictions={"model": [1.1, 2.1, 2.9, 4.2]},
        label="actual",
    )
    observed, predicted = ax.lines
    assert observed.get_linestyle() == "-"
    assert predicted.get_linestyle() == "--"
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["actual", "model"]


def test_plot_series_composes_history_and_forecast_window() -> None:
    history = pd.date_range("2024-01-01", periods=3, freq="MS")
    future = pd.date_range("2024-04-01", periods=2, freq="MS")
    ax = plot_series(history, [1.0, 2.0, 3.0], label="history")
    returned = plot_series(future, [4.0, 5.0], predictions={"model": [3.9, 5.2]}, ax=ax)
    assert returned is ax
    assert len(ax.lines) == 3
    # The colour cycle advances across calls, so composed lines stay distinct.
    assert len({line.get_color() for line in ax.lines}) == 3


def test_plot_series_stays_out_of_the_legend_when_unlabeled() -> None:
    ax = plot_series([1, 2, 3], [1.0, 2.0, 3.0])
    assert ax.get_legend() is None


def test_plot_series_rejects_misaligned_lengths() -> None:
    with pytest.raises(ValueError, match="values has 2"):
        plot_series([1, 2, 3], [1.0, 2.0])
    with pytest.raises(ValueError, match="predictions\\['model'\\]"):
        plot_series([1, 2], [1.0, 2.0], predictions={"model": [1.0]})


def test_plot_target_rate_bars_each_level_with_a_baseline_line() -> None:
    df = pd.DataFrame(
        {
            "grp": ["a", "a", "b", "b"],
            "y": [1, 1, 0, 1],  # a: 1.0, b: 0.5, baseline 0.75
        }
    )
    ax = plot_target_rate(df, "grp", "y")
    assert len(ax.patches) == 2  # one bar per level
    assert [label.get_text() for label in ax.get_yticklabels()] == ["a", "b"]
    # The dashed baseline reference line is drawn and legended.
    legend = ax.get_legend()
    assert legend is not None
    assert "baseline" in {text.get_text() for text in legend.get_texts()}


def test_plots_accept_an_existing_axes() -> None:
    _, ax = plt.subplots()
    returned = plot_residuals([1.0, 2.0], [1.0, 1.0], ax=ax)
    assert returned is ax
