"""Tests for the adult_income pipeline.

Run from the repo root with::

    uv run pytest projects/adult_income --no-cov

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
    spec = importlib.util.spec_from_file_location("adult_income_pipeline", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_sentinels_replaces_question_marks_with_nan() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame(
        {
            "workclass": ["Private", "?", "State-gov"],
            "occupation": ["?", "Sales", "?"],
            "native_country": ["United-States", "Mexico", "?"],
            # A non-sentinel column that legitimately contains a "?" stays put.
            "note": ["?", "ok", "ok"],
        }
    )
    out = pipeline.decode_sentinels(df)
    assert out["workclass"].isna().tolist() == [False, True, False]
    assert out["occupation"].isna().tolist() == [True, False, True]
    assert out["native_country"].isna().tolist() == [False, False, True]
    assert out["note"].tolist() == ["?", "ok", "ok"]


def test_encode_target_maps_high_income_to_one() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame({"income": ["<=50K", ">50K", "<=50K"]})
    out = pipeline.encode_target(df)
    assert out["income"].tolist() == [0, 1, 0]
    assert out["income"].dtype == "int64"


def test_encode_target_rejects_unexpected_label() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame({"income": ["<=50K", ">50K.", ">50K"]})
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

    # The model must add information over the majority-class floor (all-<=50K:
    # high accuracy on this imbalance, zero recall so F1 is zero) and over the
    # interpretable marital-status rule, on both accuracy and F1.
    assert metrics["accuracy"] > metrics["majority_accuracy"]
    assert metrics["accuracy"] > metrics["married_accuracy"]
    assert metrics["f1"] > metrics["majority_f1"]
    assert metrics["f1"] > metrics["married_f1"]
    assert metrics["f1"] > 0.6

    # The comparison frame carries all three contenders, scored with
    # classification metrics.
    comparison = pd.read_csv(out / "model_comparison.csv", index_col=0)
    assert {"logistic_regression", "married_rule", "majority_class"} <= set(comparison.index)
    assert {"accuracy", "precision", "recall", "f1"} <= set(comparison.columns)

    # Five stratified k-fold rows, scored with classification metrics, with the
    # transform plan re-fitted inside each fold via make_pipeline (the fitted
    # state — occupation modes, kept top-k countries, scale centres — genuinely
    # varies fold to fold).
    cv_scores = pd.read_csv(out / "cv_folds.csv", index_col=0)
    assert len(cv_scores) == 5
    assert {"accuracy", "f1", "train_size", "test_size"} <= set(cv_scores.columns)
    assert cv_scores["f1"].between(0.0, 1.0).all()
    assert cv_scores["f1"].mean() > 0.55

    # The confusion matrix and per-class frame carry the income *names* on the
    # axes (the labels= display mapping) while the metric math stayed on codes.
    confusion = pd.read_csv(out / "confusion_matrix.csv", index_col=0)
    assert list(confusion.index) == ["<=50K", ">50K"]
    per_class = pd.read_csv(out / "per_class_metrics.csv", index_col=0)
    assert list(per_class.index) == ["<=50K", ">50K"]
    assert {"precision", "recall", "f1", "support"} <= set(per_class.columns)
    assert confusion.to_numpy().sum() == per_class["support"].sum()

    # The capital-column outlier report exists and flags real values without
    # removing them (the flag-not-clip path): both columns carry outliers and
    # the processed frame keeps the full row count.
    capital = pd.read_csv(out / "capital_outliers.csv", index_col=0)
    assert {"capital_gain", "capital_loss"} == set(capital.index)
    assert (capital["outlier_count"] > 0).all()

    # Artifacts: EDA reports, figures, processed data, persisted pipeline and
    # model.
    for name in (
        "summary.csv",
        "missing.csv",
        "top_correlations.csv",
        "missingness.png",
        "capital_outliers.csv",
        "capital_outliers.png",
        "cv_folds.csv",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "per_class_metrics.csv",
        "model_comparison.csv",
        "model_comparison.png",
    ):
        assert (out / name).exists()
    assert (settings.processed_dir / "adult_income_features.parquet").exists()
    assert (settings.processed_dir / "params" / "adult_income_model.joblib").exists()

    # The persisted scoring pipeline reloads with the training-time steps in
    # order: collapse the tails, mode-impute the sentinels, wide one-hot, scale.
    scoring = load_params(settings.processed_dir / "params" / "adult_income_scoring.json", Pipeline)
    assert [step.kind for step in scoring.steps] == [
        "collapse_categories",
        "impute_missing",
        "one_hot_encode",
        "scale_features",
    ]
