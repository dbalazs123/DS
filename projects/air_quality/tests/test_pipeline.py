"""Tests for the air_quality pipeline.

Run from the repo root with::

    uv run pytest projects/air_quality --no-cov

The end-to-end test downloads the real dataset once into a temporary data
directory; it skips (rather than fails) when the network is unavailable. A
mirror that serves the wrong bytes is *not* a skip — the pinned checksum
failing on a reachable network is exactly the failure the fetch exists to
make loud.
"""

from __future__ import annotations

import importlib.util
import urllib.error
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

PIPELINE_PATH = Path(__file__).resolve().parent.parent / "pipeline.py"


def _load_pipeline() -> ModuleType:
    """Load the sibling ``pipeline.py`` by path (no package import needed)."""
    spec = importlib.util.spec_from_file_location("air_quality_pipeline", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measurement_frame(pipeline: ModuleType, n_rows: int) -> pd.DataFrame:
    """A minimal valid frame with every measurement column filled."""
    return pd.DataFrame({column: [1.0] * n_rows for column in pipeline.MEASUREMENT_COLUMNS})


def test_trim_raw_drops_junk_columns_and_empty_rows() -> None:
    pipeline = _load_pipeline()
    df = _measurement_frame(pipeline, 3)
    df["Unnamed: 15"] = [np.nan] * 3
    df.iloc[2] = np.nan  # an all-empty trailing row, like the file's last 114
    out = pipeline.trim_raw(df, expected_rows=2)
    assert len(out) == 2
    assert not any(column.startswith("Unnamed") for column in out.columns)
    # The input frame is never mutated.
    assert len(df) == 3


def test_trim_raw_rejects_unexpected_row_count() -> None:
    pipeline = _load_pipeline()
    df = _measurement_frame(pipeline, 5)
    with pytest.raises(ValueError, match="expected 2 data rows"):
        pipeline.trim_raw(df, expected_rows=2)


def test_mask_sentinels_hits_measurement_columns_only() -> None:
    pipeline = _load_pipeline()
    df = _measurement_frame(pipeline, 2)
    df.loc[0, "co_gt"] = -200.0
    df["date"] = ["10/03/2004", "10/03/2004"]
    out = pipeline.mask_sentinels(df)
    assert np.isnan(out.loc[0, "co_gt"])
    assert out.loc[1, "co_gt"] == 1.0
    assert out["date"].notna().all()
    assert df.loc[0, "co_gt"] == -200.0  # input frame untouched


def test_build_time_axis_parses_the_dotted_format_and_sorts() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame(
        {
            "date": ["10/03/2004", "10/03/2004"],
            "time": ["19.00.00", "18.00.00"],
            "co_gt": [2.0, 2.6],
        }
    )
    out = pipeline.build_time_axis(df)
    assert list(out["timestamp"]) == [
        pd.Timestamp("2004-03-10 18:00:00"),
        pd.Timestamp("2004-03-10 19:00:00"),
    ]
    # Sorted chronologically, day-first (the 10th of March, not October 3rd).
    assert out["co_gt"].tolist() == [2.6, 2.0]
    assert "date" not in out.columns and "time" not in out.columns


def test_build_time_axis_rejects_duplicate_hours() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame(
        {
            "date": ["10/03/2004", "10/03/2004"],
            "time": ["18.00.00", "18.00.00"],
        }
    )
    with pytest.raises(ValueError, match="duplicated hours"):
        pipeline.build_time_axis(df)


def test_drop_offline_rows_requires_every_channel_missing() -> None:
    pipeline = _load_pipeline()
    df = _measurement_frame(pipeline, 3)
    df.loc[0, pipeline.DEVICE_COLUMNS] = np.nan  # fully offline: dropped
    df.loc[1, "pt08_s1_co"] = np.nan  # partial gap: kept for the impute step
    out = pipeline.drop_offline_rows(df)
    assert len(out) == 2
    assert np.isnan(out.loc[0, "pt08_s1_co"])


def test_same_hour_yesterday_reference_aligns_by_timestamp() -> None:
    pipeline = _load_pipeline()
    hours = pd.date_range("2004-03-10", periods=30, freq="h")
    frame = pd.DataFrame({"timestamp": hours, "co_gt": [float(i) for i in range(30)]})
    # Ask for hours 24..26 (yesterday exists) and one hour with no lag row.
    wanted = pd.Series([hours[24], hours[25], hours[26], pd.Timestamp("2004-03-20 00:00:00")])
    out = pipeline.same_hour_yesterday_reference(frame, wanted, fallback=-1.0)
    assert out == [0.0, 1.0, 2.0, -1.0]


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

    # The model must add information over both references: the training mean
    # (the no-information floor) and the station's own reading 24 hours
    # earlier (the cheap strong reference).
    assert metrics["mae"] < metrics["yesterday_mae"]
    assert metrics["mae"] < metrics["train_mean_mae"]
    assert metrics["r2"] > metrics["yesterday_r2"]
    assert metrics["r2"] > 0.8

    # The comparison frame carries all three contenders.
    comparison = pd.read_csv(out / "model_comparison.csv", index_col=0)
    assert {"ridge", "same_hour_yesterday", "train_mean"} <= set(comparison.index)
    assert {"mae", "rmse", "r2"} <= set(comparison.columns)

    # Five rolling-origin folds, plus the would-be per-fold fitted state the
    # single up-front transform cannot re-fit (the item recorded in
    # ROADMAP.md): the impute medians and scale centre genuinely vary.
    cv_scores = pd.read_csv(out / "cv_folds.csv", index_col=0)
    assert len(cv_scores) == 5
    assert {"mae", "rmse", "r2", "train_size", "test_size"} <= set(cv_scores.columns)
    fit_state = pd.read_csv(out / "cv_fold_fit_state.csv", index_col=0)
    assert len(fit_state) == 5
    assert fit_state["nox_gt_median"].nunique() > 1
    assert fit_state["pt08_s1_co_center"].nunique() > 1

    # The missing-value triage evidence: the report ranks the 90%-missing
    # column first, and the correlation table carries the near-identity that
    # justified excluding c6h6_gt.
    missing = pd.read_csv(out / "missing_report.csv", index_col=0)
    assert missing.index[0] == "nmhc_gt"
    assert missing.loc["nmhc_gt", "frac_missing"] > 0.9
    correlations = pd.read_csv(out / "top_correlations.csv")
    top_pair = set(correlations.iloc[0][["feature_a", "feature_b"]])
    assert top_pair == {"c6h6_gt", "pt08_s2_nmhc"}

    # Artifacts: EDA reports, figures, processed data, persisted pipeline and
    # model.
    for name in (
        "missing_report.csv",
        "missingness.png",
        "summary.csv",
        "top_correlations.csv",
        "cv_folds.csv",
        "cv_fold_fit_state.csv",
        "model_comparison.csv",
        "model_comparison.png",
        "residuals.png",
        "reconstruction_week.png",
    ):
        assert (out / name).exists()
    assert (settings.processed_dir / "air_quality_features.parquet").exists()
    assert (settings.processed_dir / "params" / "air_quality_model.joblib").exists()

    # The persisted scoring pipeline reloads from strict JSON and holds the
    # three fitted steps in plan order.
    scoring = load_params(settings.processed_dir / "params" / "air_quality_scoring.json", Pipeline)
    assert [step.kind for step in scoring.steps] == [
        "impute_missing",
        "one_hot_encode",
        "scale_features",
    ]
