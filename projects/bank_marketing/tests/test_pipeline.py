"""Tests for the bank_marketing pipeline.

Run from the repo root with::

    uv run pytest projects/bank_marketing --no-cov

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
    spec = importlib.util.spec_from_file_location("bank_marketing_pipeline", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_encode_target_maps_yes_to_one() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame({"y": ["no", "yes", "no"]})
    out = pipeline.encode_target(df)
    assert out["y"].tolist() == [0, 1, 0]
    assert out["y"].dtype == "int64"


def test_encode_target_rejects_unexpected_label() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame({"y": ["no", "maybe", "yes"]})
    with pytest.raises(DataValidationError):
        pipeline.encode_target(df)


def test_encode_pdays_sentinel_becomes_flag_and_drops_column() -> None:
    pipeline = _load_pipeline()
    # 999 is "never previously contacted" -> flag 0; any real gap -> flag 1.
    df = pd.DataFrame({"pdays": [999, 3, 999, 0], "other": [1, 2, 3, 4]})
    out = pipeline.encode_pdays_sentinel(df)
    assert "pdays" not in out.columns
    assert out["was_previously_contacted"].tolist() == [0, 1, 0, 1]
    assert out["was_previously_contacted"].dtype == "int64"
    assert out["other"].tolist() == [1, 2, 3, 4]  # untouched columns survive


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

    # The honest headline is *probabilistic*: at 11% prevalence the model's
    # accuracy is actually below the majority floor (predicting "no" for
    # everyone scores its prevalence), so accuracy is deliberately NOT the bar.
    # The model must rank subscribers far above chance — ROC-AUC well over the
    # 0.5 floor and average precision well over the prevalence floor — and it
    # must recover real recall the majority floor has none of.
    assert metrics["roc_auc"] > 0.75
    assert metrics["roc_auc"] > metrics["prevalence_roc_auc"]
    assert metrics["average_precision"] > 2 * metrics["prevalence_average_precision"]
    assert metrics["recall"] > 0.5
    assert metrics["majority_recall"] == 0.0

    # On hard labels the model must beat both the majority floor (F1 zero, since
    # it finds no positives) and the interpretable prior-success rule on F1.
    assert metrics["f1"] > metrics["majority_f1"]
    assert metrics["f1"] > metrics["prior_success_f1"]

    # The probabilistic comparison frame carries the model and the prevalence
    # floor, scored with the ranking metrics; the floor ranks at exactly chance.
    probabilistic = pd.read_csv(out / "probabilistic_comparison.csv", index_col=0)
    assert {"logistic_regression", "prevalence_rate"} == set(probabilistic.index)
    assert {"roc_auc", "average_precision", "brier"} <= set(probabilistic.columns)
    assert probabilistic.loc["prevalence_rate", "roc_auc"] == 0.5

    # The hard-label comparison frame carries all three contenders.
    comparison = pd.read_csv(out / "model_comparison.csv", index_col=0)
    assert {"logistic_regression", "prior_success_rule", "majority_class"} <= set(comparison.index)
    assert {"accuracy", "precision", "recall", "f1"} <= set(comparison.columns)

    # Five stratified k-fold rows, scored with classification metrics, with the
    # transform plan re-fitted inside each fold via make_pipeline.
    cv_scores = pd.read_csv(out / "cv_folds.csv", index_col=0)
    assert len(cv_scores) == 5
    assert {"accuracy", "f1", "train_size", "test_size"} <= set(cv_scores.columns)
    assert cv_scores["f1"].between(0.0, 1.0).all()

    # The confusion matrix and per-class frame carry the yes/no *names* on the
    # axes (the labels= display mapping) while the metric math stayed on codes.
    confusion = pd.read_csv(out / "confusion_matrix.csv", index_col=0)
    assert list(confusion.index) == ["no", "yes"]
    per_class = pd.read_csv(out / "per_class_metrics.csv", index_col=0)
    assert list(per_class.index) == ["no", "yes"]
    assert {"precision", "recall", "f1", "support"} <= set(per_class.columns)
    assert confusion.to_numpy().sum() == per_class["support"].sum()

    # Artifacts: EDA reports, figures, processed data, persisted pipeline and
    # model.
    for name in (
        "summary.csv",
        "missing.csv",
        "top_correlations.csv",
        "missingness.png",
        "target_rate_poutcome.csv",
        "target_rate_poutcome.png",
        "cv_folds.csv",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "per_class_metrics.csv",
        "model_comparison.csv",
        "model_comparison.png",
        "probabilistic_comparison.csv",
        "probabilistic_comparison.png",
    ):
        assert (out / name).exists()
    assert (settings.processed_dir / "bank_marketing_features.parquet").exists()
    assert (settings.processed_dir / "params" / "bank_marketing_model.joblib").exists()

    # The persisted scoring pipeline reloads with the training-time steps in
    # order: wide one-hot, then scale.
    scoring = load_params(
        settings.processed_dir / "params" / "bank_marketing_scoring.json", Pipeline
    )
    assert [step.kind for step in scoring.steps] == ["one_hot_encode", "scale_features"]
