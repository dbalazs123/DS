"""Tests for the flights pipeline.

Run from the repo root with::

    uv run pytest projects/flights --no-cov

The end-to-end test downloads the real dataset once into a temporary data
directory; it skips (rather than fails) when the network is unavailable.
"""

from __future__ import annotations

import importlib.util
import urllib.error
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

PIPELINE_PATH = Path(__file__).resolve().parent.parent / "pipeline.py"


def _load_pipeline() -> ModuleType:
    """Load the sibling ``pipeline.py`` by path (no package import needed)."""
    spec = importlib.util.spec_from_file_location("flights_pipeline", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_time_axis_parses_and_sorts() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame(
        {
            "year": [1950, 1949, 1949],
            "month": ["January", "February", "January"],
            "passengers": [115, 118, 112],
        }
    )
    out = pipeline.build_time_axis(df)
    assert out["date"].tolist() == [
        pd.Timestamp("1949-01-01"),
        pd.Timestamp("1949-02-01"),
        pd.Timestamp("1950-01-01"),
    ]
    # Sorting reorders the rows; the target must travel with its month.
    assert out["passengers"].tolist() == [112, 118, 115]


def test_build_time_axis_rejects_duplicated_months() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame(
        {
            "year": [1949, 1949],
            "month": ["January", "January"],
            "passengers": [112, 113],
        }
    )
    with pytest.raises(ValueError, match="1949-01"):
        pipeline.build_time_axis(df)


def test_engineer_trend_and_calendar_keeps_signal_drops_noise() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame(
        {
            "year": [1949, 1949, 1950],
            "month": ["January", "February", "January"],
            "passengers": [112, 118, 115],
            "date": pd.to_datetime(["1949-01-01", "1949-02-01", "1950-01-01"]),
        }
    )
    out = pipeline.engineer_trend_and_calendar(df)
    # month_index is the monotone months-elapsed counter: +1 per month,
    # +12 per year.
    assert (out["month_index"] - out["month_index"].iloc[0]).tolist() == [0, 1, 12]
    # The weekday-of-the-1st artifacts and the columns consumed into
    # month_index are gone; the one-hot source and the time axis stay.
    for dropped in ("year", "date_year", "date_month", "date_dayofweek", "date_is_weekend"):
        assert dropped not in out.columns
    assert {"date", "month", "passengers", "month_index"} <= set(out.columns)


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    from ds import Settings
    from ds.io import load_params
    from ds.pipeline import Pipeline

    pipeline = _load_pipeline()
    settings = Settings(data_dir=tmp_path / "data")
    try:
        pipeline.fetch_raw(settings)
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"dataset download unavailable: {exc}")

    out = tmp_path / "out"
    metrics = pipeline.run(out, settings=settings)

    # The model must add information over both naive references on the
    # strictly future held-out window — the point of a forecasting protocol.
    assert metrics["mae"] < metrics["seasonal_naive_mae"]
    assert metrics["mae"] < metrics["naive_last_mae"]
    assert metrics["r2"] > 0.5

    # The comparison frame carries all three contenders, scored with
    # regression metrics.
    comparison = pd.read_csv(out / "model_comparison.csv", index_col=0)
    assert {"linear_regression", "seasonal_naive", "naive_last"} <= set(comparison.index)
    assert {"mae", "rmse", "r2"} <= set(comparison.columns)

    # Five rolling-origin folds with growing training windows — every fold
    # trains only on its test block's past (cross_validate_by_time's first
    # real consumer).
    cv_scores = pd.read_csv(out / "cv_folds.csv", index_col=0)
    assert len(cv_scores) == 5
    assert {"mae", "rmse", "r2", "train_size", "test_size"} <= set(cv_scores.columns)
    assert cv_scores["train_size"].is_monotonic_increasing
    assert cv_scores["mae"].mean() < metrics["seasonal_naive_mae"]

    # The seasonal profile covers all twelve calendar months and shows the
    # summer peak the month one-hots exist to capture.
    seasonality = pd.read_csv(out / "seasonality.csv", index_col=0)
    assert len(seasonality) == 12
    assert seasonality["passengers"].idxmax() in ("July", "August")

    # Artifacts: EDA reports, figures, processed data, persisted pipeline
    # and model.
    for name in (
        "summary.csv",
        "seasonality.csv",
        "series.png",
        "cv_folds.csv",
        "model_comparison.csv",
        "model_comparison.png",
        "residuals.png",
        "forecast.png",
    ):
        assert (out / name).exists()
    assert (settings.processed_dir / "flights_features.parquet").exists()
    assert (settings.processed_dir / "params" / "flights_model.joblib").exists()

    # The persisted scoring pipeline reloads with the single fitted step —
    # the month vocabulary, fixed at the twelve calendar months.
    scoring = load_params(settings.processed_dir / "params" / "flights_scoring.json", Pipeline)
    assert [step.kind for step in scoring.steps] == ["one_hot_encode"]
    (categories,) = [step.params.categories["month"] for step in scoring.steps]
    assert len(categories) == 12
