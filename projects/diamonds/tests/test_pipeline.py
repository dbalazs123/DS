"""Tests for the diamonds pipeline.

Run from the repo root with::

    uv run pytest projects/diamonds --no-cov

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

CUT_NAMES = ["Fair", "Good", "Very Good", "Premium", "Ideal"]


def _load_pipeline() -> ModuleType:
    """Load the sibling ``pipeline.py`` by path (no package import needed)."""
    spec = importlib.util.spec_from_file_location("diamonds_pipeline", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_drop_impossible_dimensions_removes_only_bad_rows() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame(
        {
            "x": [3.95, 0.0, 4.05, 5.0],
            "y": [3.98, 4.0, 0.0, 5.1],
            "z": [2.43, 2.5, 2.6, 3.1],
            "price": [326, 327, 328, 329],
        }
    )
    out = pipeline.drop_impossible_dimensions(df)
    assert out["price"].tolist() == [326, 329]
    assert list(out.index) == [0, 1]  # reindexed from zero


def test_encode_cut_maps_grades_to_ordered_codes() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame({"cut": ["Ideal", "Fair", "Very Good", "Premium", "Good"]})
    out = pipeline.encode_cut(df)
    assert out["cut"].tolist() == [4, 0, 2, 3, 1]
    # Round-trip: the code indexes back into the grade order.
    assert [pipeline.CUT_ORDER[code] for code in out["cut"]] == [
        "Ideal",
        "Fair",
        "Very Good",
        "Premium",
        "Good",
    ]


def test_proportions_rule_grades_by_band() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame(
        {
            # ideal band; premium band; very-good band; neither.
            "depth": [61.5, 59.0, 63.8, 70.0],
            "table": [56.0, 58.5, 61.0, 65.0],
        }
    )
    assert pipeline.proportions_rule(df) == [
        pipeline.CUT_ORDER.index("Ideal"),
        pipeline.CUT_ORDER.index("Premium"),
        pipeline.CUT_ORDER.index("Very Good"),
        pipeline.CUT_ORDER.index("Good"),
    ]


def test_macro_metrics_cover_five_classes() -> None:
    pipeline = _load_pipeline()
    scores = pipeline.macro_classification_metrics([0, 1, 2, 3, 4], [0, 1, 2, 3, 3])
    assert scores["accuracy"] == 0.8
    # Macro average counts the missed class 4 at zero, not weighted away.
    assert scores["recall"] == 0.8


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    from ds import Settings
    from ds.features import OrdinalCategories
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

    # The model must add information over both references: the majority
    # class (predict Ideal for everything) and the depth/table proportions
    # rule that reads the two columns cut grade is defined by.
    assert metrics["accuracy"] > metrics["proportions_accuracy"]
    assert metrics["accuracy"] > metrics["majority_accuracy"]
    assert metrics["f1"] > metrics["proportions_f1"]
    assert metrics["f1"] > metrics["majority_f1"]
    assert metrics["f1"] > 0.4

    # The comparison frame carries all three contenders, scored with the
    # macro-averaged classification metrics.
    comparison = pd.read_csv(out / "model_comparison.csv", index_col=0)
    assert {"logistic_regression", "proportions_rule", "majority_class"} <= set(comparison.index)
    assert {"accuracy", "precision", "recall", "f1"} <= set(comparison.columns)

    # Five stratified k-fold rows, the transform plan re-fitted inside each
    # fold via make_pipeline, scored with macro metrics.
    cv_scores = pd.read_csv(out / "cv_folds.csv", index_col=0)
    assert len(cv_scores) == 5
    assert {"accuracy", "f1", "train_size", "test_size"} <= set(cv_scores.columns)
    assert cv_scores["f1"].between(0.0, 1.0).all()
    assert cv_scores["f1"].mean() > 0.4

    # The confusion matrix is the square 5x5 frame of the multiclass target,
    # persisted with grade names rather than integer codes, and per-class
    # metrics break out all five classes with matching support.
    confusion = pd.read_csv(out / "confusion_matrix.csv", index_col=0)
    assert list(confusion.index) == CUT_NAMES
    assert list(confusion.columns) == CUT_NAMES
    per_class = pd.read_csv(out / "per_class_metrics.csv", index_col=0)
    assert list(per_class.index) == CUT_NAMES
    assert {"precision", "recall", "f1", "support"} <= set(per_class.columns)
    assert confusion.to_numpy().sum() == per_class["support"].sum()

    # Artifacts: EDA reports, figures, processed data, persisted pipeline
    # and model.
    for name in (
        "summary.csv",
        "correlations.csv",
        "carat_band_cut_mix.csv",
        "outliers.png",
        "cv_folds.csv",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "per_class_metrics.csv",
        "model_comparison.csv",
        "model_comparison.png",
    ):
        assert (out / name).exists()
    assert (settings.processed_dir / "diamonds_features.parquet").exists()
    assert (settings.processed_dir / "params" / "diamonds_model.joblib").exists()

    # The persisted scoring pipeline reloads with the training-time steps in
    # order — clip the dimensions, ordinal-encode the graded scales, scale —
    # and the reloaded ordinal step still carries the explicit worst-to-best
    # color order it was fitted with.
    scoring = load_params(settings.processed_dir / "params" / "diamonds_scoring.json", Pipeline)
    assert [step.kind for step in scoring.steps] == [
        "clip_outliers",
        "ordinal_encode",
        "scale_features",
    ]
    ordinal = scoring.steps[1].params
    assert isinstance(ordinal, OrdinalCategories)
    assert ordinal.categories["color"] == ("J", "I", "H", "G", "F", "E", "D")
