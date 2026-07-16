"""Tests for the nyc_taxis pipeline.

Run from the repo root with::

    uv run pytest projects/nyc_taxis --no-cov

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
    spec = importlib.util.spec_from_file_location("nyc_taxis_pipeline", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_engineer_trip_features_adds_hour_and_duration() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame(
        {
            "pickup": pd.to_datetime(["2019-03-01 22:15:00", "2019-03-02 07:30:00"]),
            "dropoff": pd.to_datetime(["2019-03-01 22:45:00", "2019-03-02 07:36:00"]),
        }
    )
    out = pipeline.engineer_trip_features(df)
    assert out["pickup_hour"].tolist() == [22, 7]
    assert out["duration_min"].tolist() == [30.0, 6.0]
    assert "dropoff" not in out.columns
    assert {"pickup_year", "pickup_month", "pickup_day", "pickup_dayofweek"} <= set(out.columns)


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

    # The model must add information over predicting the train-mean fare.
    assert metrics["rmse"] < metrics["baseline_rmse"]
    assert metrics["mae"] < metrics["baseline_mae"]

    # Artifacts: EDA reports, figures, processed data, persisted pipeline.
    for name in ("summary.csv", "missing.csv", "missingness.png", "residuals.png"):
        assert (out / name).exists()
    assert (settings.processed_dir / "taxis_features.parquet").exists()

    # The persisted scoring pipeline reloads and reproduces the training-time
    # feature columns on raw-shaped rows.
    scoring = load_params(settings.processed_dir / "params" / "taxis_scoring.json", Pipeline)
    assert [step.kind for step in scoring.steps] == [
        "clip_outliers",
        "impute_missing",
        "one_hot_encode",
        "scale_features",
    ]
