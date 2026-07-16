"""Tests for the titanic pipeline.

Run from the repo root with::

    uv run pytest projects/titanic --no-cov

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
    spec = importlib.util.spec_from_file_location("titanic_pipeline", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_drop_leaky_and_derived_verifies_then_drops() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame(
        {
            "survived": [0, 1, 0],
            "alive": ["no", "yes", "no"],
            "pclass": [3, 1, 2],
            "class": ["Third", "First", "Second"],
            "who": ["man", "woman", "man"],
            "adult_male": [True, False, True],
            "embark_town": ["Southampton", "Cherbourg", "Southampton"],
            "alone": [True, False, True],
        }
    )
    out = pipeline.drop_leaky_and_derived(df)
    assert set(out.columns) == {"survived", "pclass"}
    assert len(out) == 3


def test_drop_leaky_and_derived_rejects_non_redundant_columns() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame(
        {
            "survived": [0, 1, 0],
            # "no" maps to both target values -> not the target respelled.
            "alive": ["no", "yes", "yes"],
            "pclass": [3, 1, 2],
            "class": ["Third", "First", "Second"],
            "who": ["man", "woman", "man"],
            "adult_male": [True, False, True],
            "embark_town": ["Southampton", "Cherbourg", "Southampton"],
            "alone": [True, False, True],
        }
    )
    with pytest.raises(ValueError, match="alive"):
        pipeline.drop_leaky_and_derived(df)


def test_engineer_passenger_features_turns_deck_into_indicator() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame({"deck": ["C", None, "B", None]})
    out = pipeline.engineer_passenger_features(df)
    assert out["deck_known"].tolist() == [1, 0, 1, 0]
    assert "deck" not in out.columns


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

    # The model must add information over predicting the majority class
    # (which never predicts a survivor, so its F1 for the positive class
    # is zero) and over the classic sex-only rule.
    assert metrics["accuracy"] > metrics["majority_accuracy"]
    assert metrics["f1"] > metrics["majority_f1"]
    assert metrics["f1"] > 0.6

    # The comparison frame carries all three contenders, scored with
    # classification metrics.
    comparison = pd.read_csv(out / "model_comparison.csv", index_col=0)
    assert {"logistic_regression", "sex_only_rule", "majority_class"} <= set(comparison.index)
    assert {"accuracy", "precision", "recall", "f1"} <= set(comparison.columns)

    # Five k-fold rows, scored with classification metrics — the first real
    # composition of cross_validate_kfold with metrics_fn=classification_metrics.
    cv_scores = pd.read_csv(out / "cv_folds.csv", index_col=0)
    assert len(cv_scores) == 5
    assert {"accuracy", "f1", "train_size", "test_size"} <= set(cv_scores.columns)
    assert cv_scores["accuracy"].between(0.0, 1.0).all()
    assert cv_scores["accuracy"].mean() > 0.65

    # The confusion matrix is the square 2x2 frame of the binary target, and
    # per-class metrics break out both classes with matching support.
    confusion = pd.read_csv(out / "confusion_matrix.csv", index_col=0)
    assert list(confusion.index) == [0, 1]
    per_class = pd.read_csv(out / "per_class_metrics.csv", index_col=0)
    assert list(per_class.index) == [0, 1]
    assert {"precision", "recall", "f1", "support"} <= set(per_class.columns)
    assert confusion.to_numpy().sum() == per_class["support"].sum()

    # Artifacts: EDA reports, figures, processed data, persisted pipeline
    # and model.
    for name in (
        "summary.csv",
        "missing.csv",
        "missingness.png",
        "cv_folds.csv",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "per_class_metrics.csv",
        "model_comparison.csv",
        "model_comparison.png",
    ):
        assert (out / name).exists()
    assert (settings.processed_dir / "titanic_features.parquet").exists()
    assert (settings.processed_dir / "params" / "titanic_model.joblib").exists()

    # The persisted scoring pipeline reloads with the training-time steps in
    # order — including the two impute steps with different strategies.
    scoring = load_params(settings.processed_dir / "params" / "titanic_scoring.json", Pipeline)
    assert [step.kind for step in scoring.steps] == [
        "clip_outliers",
        "impute_missing",
        "impute_missing",
        "one_hot_encode",
        "scale_features",
    ]
