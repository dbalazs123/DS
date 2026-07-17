"""Tests for the sms_spam pipeline.

Run from the repo root with::

    uv run pytest projects/sms_spam --no-cov

The end-to-end test downloads the real dataset once into a temporary data
directory; it skips (rather than fails) when the network is unavailable.
Nothing here asserts an exact ``token_count`` value: that column degrades to
a whitespace count without tiktoken (or its vocabulary), so the assertions
stay environment-independent — structure and baselines-beaten, per the
pipeline's determinism note.
"""

from __future__ import annotations

import importlib.util
import urllib.error
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

PIPELINE_PATH = Path(__file__).resolve().parent.parent / "pipeline.py"

LABEL_NAMES = ["ham", "spam"]


def _load_pipeline() -> ModuleType:
    """Load the sibling ``pipeline.py`` by path (no package import needed)."""
    spec = importlib.util.spec_from_file_location("sms_spam_pipeline", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_encode_label_maps_ham_to_zero_spam_to_one() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame({"label": ["spam", "ham", "ham", "spam"]})
    out = pipeline.encode_label(df)
    assert out["label"].tolist() == [1, 0, 0, 1]
    # Round-trip: the code indexes back into the label order.
    assert [pipeline.LABEL_ORDER[code] for code in out["label"]] == ["spam", "ham", "ham", "spam"]


def test_keyword_rule_flags_spam_vocabulary_only() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame(
        {
            "message": [
                "URGENT! You have won a FREE prize, txt CLAIM to 81010 now",
                "Ok lar... Joking wif u oni...",
                "Sorry, I'll call later",
                "Congratulations - claim your cash award today",
            ]
        }
    )
    assert pipeline.keyword_rule(df) == [1, 0, 0, 1]


def test_add_text_features_counts_are_positive_ints() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame({"message": ["Ok lar", "Free entry in 2 a wkly comp to win"]})
    out = pipeline.add_text_features(df)
    # char_count is deterministic everywhere; token_count depends on whether
    # tiktoken (and its vocabulary) is available, so assert bounds, not values.
    assert out["char_count"].tolist() == [6, 34]
    assert (out["token_count"] >= 1).all()
    assert out["token_count"].dtype.kind == "i"
    # The input frame is never mutated.
    assert "char_count" not in df.columns


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

    # The model must add information over both references: the majority class
    # (predict ham for everything — 0 on every spam-positive metric) and the
    # keyword rule reading the classic spam vocabulary.
    assert metrics["accuracy"] > metrics["keyword_accuracy"]
    assert metrics["accuracy"] > metrics["majority_accuracy"]
    assert metrics["f1"] > metrics["keyword_f1"]
    assert metrics["f1"] > metrics["majority_f1"]
    assert metrics["majority_f1"] == 0.0
    assert metrics["f1"] > 0.8

    # The comparison frame carries all three contenders, scored with the
    # binary (spam-positive) classification metrics.
    comparison = pd.read_csv(out / "model_comparison.csv", index_col=0)
    assert {"tfidf_logistic", "keyword_rule", "majority_class"} <= set(comparison.index)
    assert {"accuracy", "precision", "recall", "f1"} <= set(comparison.columns)

    # Five stratified k-fold rows, the scale step re-fitted inside each fold
    # via make_pipeline (and the vectorizer per fold via make_model).
    cv_scores = pd.read_csv(out / "cv_folds.csv", index_col=0)
    assert len(cv_scores) == 5
    assert {"accuracy", "f1", "train_size", "test_size"} <= set(cv_scores.columns)
    assert cv_scores["f1"].between(0.0, 1.0).all()
    assert cv_scores["f1"].mean() > 0.8

    # The confusion matrix is the 2x2 frame of the binary target, persisted
    # with label names rather than integer codes — the labels= mapping's
    # second consumer — and per-class metrics break out both classes with
    # matching support.
    confusion = pd.read_csv(out / "confusion_matrix.csv", index_col=0)
    assert list(confusion.index) == LABEL_NAMES
    assert list(confusion.columns) == LABEL_NAMES
    per_class = pd.read_csv(out / "per_class_metrics.csv", index_col=0)
    assert list(per_class.index) == LABEL_NAMES
    assert {"precision", "recall", "f1", "support"} <= set(per_class.columns)
    assert confusion.to_numpy().sum() == per_class["support"].sum()

    # Artifacts: EDA reports, figures, processed data, persisted pipeline and
    # model.
    for name in (
        "summary.csv",
        "length_by_label.csv",
        "length_band_label_mix.csv",
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
    assert (settings.processed_dir / "sms_spam_features.parquet").exists()
    assert (settings.processed_dir / "params" / "sms_spam_model.joblib").exists()

    # The persisted ds scoring pipeline reloads — and holds only the
    # char_count scaler. The TF-IDF vocabulary, the fitted heart of a text
    # pipeline, has no step kind and lives in the model joblib instead: the
    # split this project exists to record.
    scoring = load_params(settings.processed_dir / "params" / "sms_spam_scoring.json", Pipeline)
    assert [step.kind for step in scoring.steps] == ["scale_features"]
