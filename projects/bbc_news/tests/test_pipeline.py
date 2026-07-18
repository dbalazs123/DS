"""Tests for the bbc_news pipeline.

Run from the repo root with::

    uv run pytest projects/bbc_news --no-cov

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
    spec = importlib.util.spec_from_file_location("bbc_pipeline", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_add_text_features_emits_the_four_length_columns() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame({"text": ["one two three", "solo"]})
    out = pipeline.add_text_features(df)
    for column in pipeline._NUMERIC_FEATURES:
        assert column in out.columns
    assert list(out["text_word_count"]) == [3, 1]
    assert list(out["text_char_count"]) == [13, 4]


def test_encode_target_maps_topics_to_stable_codes() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame({"category": ["tech", "business", "sport"]})
    out = pipeline.encode_target(df)
    # business=0, entertainment=1, politics=2, sport=3, tech=4 (CLASS_ORDER).
    assert list(out["category"]) == [4, 0, 3]


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    from ds import Settings
    from ds.io import load_params
    from ds.pipeline import Pipeline

    pipeline = _load_pipeline()
    settings = Settings(data_dir=tmp_path / "data")
    try:
        pipeline.fetch_raw(settings)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        pytest.skip(f"dataset download unavailable: {exc}")

    out = tmp_path / "out"
    metrics = pipeline.run(out, settings=settings)

    # The TF-IDF model is strong and beats both references by a wide margin —
    # the point that the learned vocabulary, not document length, carries the
    # topic signal. The macro-F1 bound is loose enough to hold whether
    # count_tokens ran BPE or its whitespace fallback.
    assert metrics["f1"] > 0.9
    assert metrics["accuracy"] > 0.9
    assert metrics["f1"] > metrics["length_only_f1"] + 0.3
    assert metrics["f1"] > metrics["majority_f1"]

    # Five stratified folds, macro-scored.
    cv_scores = pd.read_csv(out / "cv_folds.csv", index_col=0)
    assert len(cv_scores) == 5
    assert cv_scores["f1"].mean() > 0.9

    # The comparison frame carries all three contenders, macro-scored.
    comparison = pd.read_csv(out / "model_comparison.csv", index_col=0)
    assert {"tfidf_logreg", "length_only", "majority_class"} <= set(comparison.index)

    # The confusion frame is 5x5 with the topic names on the axes.
    confusion = pd.read_csv(out / "confusion_matrix.csv", index_col=0)
    assert confusion.shape == (5, 5)
    assert set(pipeline.CLASS_ORDER) <= set(confusion.index.astype(str))

    # Artifacts: EDA reports, figures, processed data, persisted pipeline + model.
    for name in (
        "summary.csv",
        "length_by_topic.csv",
        "cv_folds.csv",
        "model_comparison.csv",
        "model_comparison.png",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "per_class_metrics.csv",
    ):
        assert (out / name).exists()
    assert (settings.processed_dir / "bbc_news_features.parquet").exists()
    assert (settings.processed_dir / "params" / "bbc_news_model.joblib").exists()

    # The persisted scoring pipeline reloads with the single fitted step — the
    # numeric-length scaler; the TF-IDF vectorizer lives in the model joblib
    # instead (the model-side-transform convention).
    scoring = load_params(settings.processed_dir / "params" / "bbc_news_scoring.json", Pipeline)
    assert [step.kind for step in scoring.steps] == ["scale_features"]
