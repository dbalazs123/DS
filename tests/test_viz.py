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
    plot_residuals,
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


def test_plot_residuals_draws_points_and_zero_line() -> None:
    ax = plot_residuals([1.0, 2.0, 3.0], [1.5, 1.5, 3.5])
    assert ax.collections  # the scatter
    assert ax.get_ylabel().startswith("residual")


def test_plots_accept_an_existing_axes() -> None:
    _, ax = plt.subplots()
    returned = plot_residuals([1.0, 2.0], [1.0, 1.0], ax=ax)
    assert returned is ax
