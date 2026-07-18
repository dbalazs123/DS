"""Tests for the sunspots pipeline.

Run from the repo root with::

    uv run pytest projects/sunspots --no-cov

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
    spec = importlib.util.spec_from_file_location("sunspots_pipeline", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_time_axis_parses_and_sorts() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame({"month": ["1749-03", "1749-01", "1749-02"], "sunspots": [70.0, 58.0, 62.6]})
    out = pipeline.build_time_axis(df)
    assert out["date"].tolist() == [
        pd.Timestamp("1749-01-01"),
        pd.Timestamp("1749-02-01"),
        pd.Timestamp("1749-03-01"),
    ]
    # Sorting reorders rows; the target must travel with its month.
    assert out["sunspots"].tolist() == [58.0, 62.6, 70.0]


def test_build_time_axis_rejects_duplicated_months() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame({"month": ["1749-01", "1749-01"], "sunspots": [58.0, 59.0]})
    with pytest.raises(ValueError, match="1749-01"):
        pipeline.build_time_axis(df)


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    from ds import Settings

    pipeline = _load_pipeline()
    settings = Settings(data_dir=tmp_path / "data")
    try:
        pipeline.fetch_raw(settings)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        pytest.skip(f"dataset download unavailable: {exc}")

    out = tmp_path / "out"
    metrics = pipeline.run(out, settings=settings)

    # The one-step-ahead AR model is strong and beats every reference — the
    # point of choosing autoregressive features over the calendar approach that
    # is useless on a non-calendar cycle.
    assert metrics["r2"] > 0.8
    assert metrics["mae"] < metrics["seasonal_naive_mae"]
    assert metrics["mae"] < metrics["naive_last_mae"]
    assert metrics["mae"] < metrics["recursive_mae"]

    # The recursive multi-step forecast decays over the decade horizon (error
    # compounds), so it is honestly weaker than the one-step forecast — yet even
    # decayed it still beats the two calendar-naive references.
    assert metrics["recursive_r2"] < metrics["r2"]
    assert metrics["recursive_mae"] < metrics["seasonal_naive_mae"]
    assert metrics["recursive_mae"] < metrics["naive_last_mae"]

    # The comparison frame carries all four contenders, scored with regression
    # metrics.
    comparison = pd.read_csv(out / "model_comparison.csv", index_col=0)
    assert {"ar_one_step", "ar_recursive", "seasonal_naive", "naive_last"} <= set(comparison.index)
    assert {"mae", "rmse", "r2"} <= set(comparison.columns)

    # Five rolling-origin folds with growing training windows.
    cv_scores = pd.read_csv(out / "cv_folds.csv", index_col=0)
    assert len(cv_scores) == 5
    assert cv_scores["train_size"].is_monotonic_increasing

    # The by-calendar-month means are near-flat — the evidence that month
    # carries no signal, so the calendar features are dropped for lags.
    seasonality = pd.read_csv(out / "seasonality.csv", index_col=0)
    assert len(seasonality) == 12
    spread = seasonality["sunspots"].max() - seasonality["sunspots"].min()
    assert spread < 0.2 * seasonality["sunspots"].mean()

    # Artifacts: EDA reports, figures, processed data and the persisted model.
    # There is no scoring Pipeline JSON here — a pure-AR forecaster has no
    # fit-based frame transform to persist (a deliberate scope finding).
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
    assert (settings.processed_dir / "sunspots_features.parquet").exists()
    assert (settings.processed_dir / "params" / "sunspots_model.joblib").exists()
    assert not (settings.processed_dir / "params" / "sunspots_scoring.json").exists()
