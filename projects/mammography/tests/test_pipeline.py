"""Tests for the mammography pipeline.

Run from the repo root with::

    uv run pytest projects/mammography --no-cov

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

from ds.validation import DataValidationError

PIPELINE_PATH = Path(__file__).resolve().parent.parent / "pipeline.py"


def _load_pipeline() -> ModuleType:
    """Load the sibling ``pipeline.py`` by path (no package import needed)."""
    spec = importlib.util.spec_from_file_location("mammography_pipeline", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_encode_target_strips_quotes_and_maps_calcification_to_one() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame({"severity": ["'-1'", "'1'", "'-1'"]})
    out = pipeline.encode_target(df)
    assert out["severity"].tolist() == [0, 1, 0]
    assert out["severity"].dtype == "int64"


def test_encode_target_rejects_unexpected_label() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame({"severity": ["'-1'", "'2'", "'1'"]})
    with pytest.raises(DataValidationError):
        pipeline.encode_target(df)


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

    # The honest headline is *probabilistic*: at 2.3% prevalence accuracy is a
    # trap (the majority floor scores ~0.98 while finding no calcifications), so
    # accuracy is deliberately NOT the bar. The model must rank calcifications
    # far above chance — ROC-AUC well over 0.5 and average precision many times
    # the prevalence floor.
    assert metrics["roc_auc"] > 0.9
    assert metrics["average_precision"] > 0.3
    assert metrics["majority_recall"] == 0.0

    # The operating-point story: tuning to the recall budget (the returned
    # top-level metrics) must catch far more calcifications than the default 0.5
    # cut, which is the whole reason the threshold is tuned rather than trusted.
    assert metrics["recall"] >= pipeline._RECALL_BUDGET
    assert metrics["recall"] > metrics["default_recall"]
    # And the recall budget spends precision to buy that recall (the screening
    # trade), so it sits below the conservative default's precision.
    assert metrics["precision"] < metrics["default_precision"]
    # The tuned thresholds are strictly below the naive 0.5 that misses positives.
    assert metrics["threshold_recall_budget"] < metrics["threshold_f1_optimal"] < 0.5

    # The probabilistic comparison frame carries the model and the prevalence
    # floor, scored with the ranking metrics; the floor ranks at exactly chance.
    probabilistic = pd.read_csv(out / "probabilistic_comparison.csv", index_col=0)
    assert {"logistic_regression", "prevalence_rate"} == set(probabilistic.index)
    assert {"roc_auc", "average_precision", "brier"} <= set(probabilistic.columns)
    assert probabilistic.loc["prevalence_rate", "roc_auc"] == 0.5

    # The operating-points frame carries the three tuned points plus the floor.
    operating = pd.read_csv(out / "operating_points.csv", index_col=0)
    assert {"default_0.5", "f1_optimal", "recall_budget", "majority_class"} == set(operating.index)
    assert {"accuracy", "precision", "recall", "f1"} <= set(operating.columns)

    # Five stratified k-fold rows, scored with classification metrics, with the
    # scale plan re-fitted inside each fold via make_pipeline.
    cv_scores = pd.read_csv(out / "cv_folds.csv", index_col=0)
    assert len(cv_scores) == 5
    assert {"accuracy", "recall", "train_size", "test_size"} <= set(cv_scores.columns)

    # The confusion matrix and per-class frame carry the benign/calcification
    # *names* on the axes while the metric math stayed on codes.
    confusion = pd.read_csv(out / "confusion_matrix.csv", index_col=0)
    assert list(confusion.index) == ["benign", "calcification"]
    per_class = pd.read_csv(out / "per_class_metrics.csv", index_col=0)
    assert list(per_class.index) == ["benign", "calcification"]
    assert {"precision", "recall", "f1", "support"} <= set(per_class.columns)
    assert confusion.to_numpy().sum() == per_class["support"].sum()

    # Artifacts: EDA reports, the two operating-point curves, the confusion
    # figure, processed data and the persisted pipeline and model.
    for name in (
        "summary.csv",
        "missing.csv",
        "top_correlations.csv",
        "cv_folds.csv",
        "operating_points.csv",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "per_class_metrics.csv",
        "probabilistic_comparison.csv",
        "probabilistic_comparison.png",
        "pr_curve.png",
        "roc_curve.png",
    ):
        assert (out / name).exists()
    assert (settings.processed_dir / "mammography_features.parquet").exists()
    assert (settings.processed_dir / "params" / "mammography_model.joblib").exists()

    # The persisted scoring pipeline reloads with its single training-time step.
    scoring = load_params(settings.processed_dir / "params" / "mammography_scoring.json", Pipeline)
    assert [step.kind for step in scoring.steps] == ["scale_features"]
