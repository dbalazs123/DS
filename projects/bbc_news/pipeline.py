"""BBC News — classify a news article into one of five topics from its text.

The ninth project on *real* data, and the **second text** one: the 2,225 BBC
News articles of the classic topic-classification set, each labelled with one of
five sections — business, entertainment, politics, sport, tech. It was picked,
after ``sms_spam``, to stress the text surface a *second* time and decide the
triggers that first project parked:

- **Text feature helpers (backlog item 21).** ``sms_spam`` hand-rolled its
  length features because the features stage had none; a second text project
  hand-rolling the same family is exactly the trigger that item recorded. So the
  friction is served here and the project consumes it: ``ds.features.text_features``
  emits the character/word/word-length columns that vary sharply by topic (tech
  and politics run long, sport short).
- **``count_tokens`` gets a real *modeling* consumer.** ``sms_spam`` kept its
  ``token_count`` descriptive-only, wary of the extras-dependent value. Here it
  is one coarse length feature among a TF-IDF heart, and the classifier is
  robust to which counting path runs (BPE vs whitespace), so it feeds the model —
  the accurate-count path's first modeling consumer, with the tests asserting
  loose, path-independent bounds.
- **The vectorization-step question (backlog item 18), re-checked.** The TF-IDF
  vectorizer is again the fitted heart, and again it lives *inside* the
  scikit-learn estimator (a ``ColumnTransformer``) while the ``ds`` scoring
  ``Pipeline`` carries the frame-shaped scale step — the model-side-transform
  convention P11 settled. This second consumer confirms it suffices; no
  first-class vectorize step is built (that would smuggle a pickle into the
  strict-JSON ``save_params`` story).

The pipeline runs the full lifecycle on ``ds`` + scikit-learn alone: fetch →
validate the boundary → drop verbatim-duplicate articles → text features →
explore (length by topic) → encode the target → stratified split → fit the
one-step scale plan (``fit_pipeline``) → stratified 5-fold CV with the plan
re-fitted per fold → persist the scoring pipeline and the fitted model → score
the held-out split from the reloaded model → evaluate (macro-averaged) against a
length-only model and the majority class → visualize. Friction it surfaced and
served is recorded in ``ROADMAP.md``.

**Determinism across CI jobs:** ``count_tokens`` returns BPE counts only when
tiktoken and its vocabulary are available, else a whitespace count. CI runs the
suite with and without extras; the model is robust to the difference (it is one
feature beside thousands of TF-IDF terms), so the tests assert macro-F1 bounds
and baselines-beaten, not token-dependent values.

Run it with::

    uv run ds run bbc_news

The raw CSV is downloaded once into ``<data_dir>/raw/`` (git-ignored) and reused
on later runs. The mirror is a live third-party GitHub repo, so the fetch pins
its sha256 and verifies the download before trusting it.
"""

from __future__ import annotations

import functools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline

from ds import Settings, get_logger, get_settings, seed_everything
from ds.eda import summarize
from ds.evaluation import (
    classification_metrics,
    compare_models,
    confusion_frame,
    cross_validate_kfold,
    per_class_metrics,
)
from ds.features import fit_scale_params, ordinal_encode, text_features
from ds.io import fetch_dataset, load_raw, save_params, save_processed
from ds.modeling.baseline import fit_baseline
from ds.modeling.nlp import count_tokens
from ds.modeling.persistence import load_model, save_model
from ds.modeling.tabular import split_features_target, train_test_split_random
from ds.pipeline import FitStep, fit_pipeline
from ds.preprocessing import drop_duplicate_rows, standardize_column_names
from ds.validation import (
    assert_in_set,
    assert_no_nulls,
    assert_row_count,
    require_columns,
)
from ds.viz import plot_confusion_matrix, plot_model_comparison, set_theme

logger = get_logger(__name__)

# A live third-party mirror of the BBC News topic set (columns "category" and
# "text"). Because it is a live upstream repo, ds.io.fetch_dataset pins the
# sha256 below and verifies the download. (A second byte-identical mirror would
# slot straight into this tuple.)
DATA_URLS = (
    "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv",
)
RAW_NAME = "bbc-text.csv"
RAW_SHA256 = "fdaee0f7451cd8db2709d00e992886fe1c387ee332b8c7b7c1554ed3d3e3382e"

EXPECTED_ROWS = 2225  # before de-duplication

_TARGET = "category"
_TEXT = "text"
# The five sections, sorted so the ordinal codes are stable and reproducible.
CLASS_ORDER = ("business", "entertainment", "politics", "sport", "tech")
# Code -> class name, for the labels= display mapping on the metric frames and
# the confusion plot (the metric math stays on the int codes).
CLASS_NAMES = dict(enumerate(CLASS_ORDER))

# The numeric features that ride alongside the TF-IDF vectors: three
# encoding-independent length features from ds.features.text_features, plus
# token_count from count_tokens (extras-dependent value, robust as a coarse
# length signal). All are scaled by the ds pipeline and passed through the
# ColumnTransformer into the classifier.
_TEXT_LENGTH_FEATURES = [f"{_TEXT}_char_count", f"{_TEXT}_word_count", f"{_TEXT}_avg_word_length"]
_NUMERIC_FEATURES = [*_TEXT_LENGTH_FEATURES, "token_count"]

# Macro-averaged metrics for the five balanced-ish classes (item 17's idiom:
# bind the average for the two-argument metrics_fn hooks).
_macro_metrics = functools.partial(classification_metrics, average="macro")


def fetch_raw(settings: Settings) -> Path:
    """Download the BBC News CSV into ``settings.raw_dir`` and verify its checksum.

    A thin binding of this project's dataset (mirror, filename, pinned digest)
    to :func:`ds.io.fetch_dataset`.

    Args:
        settings: Resolves the raw-data directory.

    Returns:
        Path to the verified local copy of the dataset.

    Raises:
        ValueError: If the mirror does not serve a file matching the checksum.
        urllib.error.URLError: If the mirror is unreachable.
    """
    return fetch_dataset(RAW_NAME, DATA_URLS, sha256=RAW_SHA256, settings=settings)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the length features the model reads alongside the TF-IDF vectors.

    The three encoding-independent features (``char_count``, ``word_count``,
    ``avg_word_length``) come from :func:`ds.features.text_features` — the helper
    this project's friction promoted, replacing the hand-rolled columns
    ``sms_spam`` carried. ``token_count`` comes from
    :func:`ds.modeling.nlp.count_tokens`; its BPE/whitespace value depends on the
    installed extras, but as one coarse length signal among thousands of TF-IDF
    terms the classifier is robust to it, so — unlike ``sms_spam`` — it feeds the
    model here (count_tokens' first modeling consumer).

    Args:
        df: Frame with a string ``text`` column.

    Returns:
        A new frame with the four numeric length columns appended.
    """
    out = text_features(df, _TEXT)
    out["token_count"] = [count_tokens(document) for document in out[_TEXT]]
    return out


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the ``category`` label as integer codes in :data:`CLASS_ORDER`.

    Uses the stage's ordinal encoder with the explicit order, so the coding is
    supplied rather than learned and the call is safe before the split. Int codes
    are what the evaluation stack and ``fit_baseline("majority")`` accept.

    Args:
        df: Frame whose ``category`` values are all in :data:`CLASS_ORDER`
            (guaranteed by the vocabulary validation upstream).

    Returns:
        A new frame with ``category`` replaced by its integer code.
    """
    return ordinal_encode(df, [_TARGET], categories={_TARGET: list(CLASS_ORDER)})


def _make_model() -> SklearnPipeline:
    """Build the fresh, unfitted topic classifier.

    A scikit-learn pipeline: TF-IDF over the raw ``text`` plus the (already
    ds-scaled) numeric length features, into a multinomial logistic regression.
    The vectorizer lives *inside* the model object — the model-side-transform
    convention (backlog item 18): ``ds.pipeline``'s step vocabulary has no
    vectorization step, so the fitted TF-IDF vocabulary persists in the model
    joblib, outside the strict-JSON ``ds`` scoring pipeline.
    """
    features = ColumnTransformer(
        [
            (
                "tfidf",
                TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=5000),
                _TEXT,
            ),
            ("length", "passthrough", _NUMERIC_FEATURES),
        ]
    )
    return SklearnPipeline(
        [("features", features), ("classifier", LogisticRegression(max_iter=1000))]
    )


def _length_only_model() -> SklearnPipeline:
    """A reference classifier that reads only the length features, no text.

    The point of comparison that shows the TF-IDF text is the heart of the model:
    document length alone separates the topics only weakly, so beating this by a
    wide margin is what justifies learning a vocabulary.
    """
    features = ColumnTransformer([("length", "passthrough", _NUMERIC_FEATURES)])
    return SklearnPipeline(
        [("features", features), ("classifier", LogisticRegression(max_iter=1000))]
    )


def run(output_dir: Path, settings: Settings | None = None) -> dict[str, float]:
    """Run the topic-classification pipeline end to end.

    Args:
        output_dir: Directory to write artifacts (figures, reports) into.
        settings: Data-directory configuration; defaults to the shared one.

    Returns:
        Macro-averaged classification metrics on the stratified held-out split,
        plus ``length_only_``-prefixed counterparts from the text-free reference
        and ``majority_``-prefixed ones from the majority-class baseline.
    """
    seed_everything()
    settings = settings or get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire — one idempotent, checksum-verified download (fetch_dataset's
    # fourth consumer).
    fetch_raw(settings)
    df = load_raw(RAW_NAME, settings=settings)

    # 2. Validate the boundary: the published row count (before de-duplication),
    # both columns, no nulls, and the exact five-topic vocabulary.
    df = standardize_column_names(df)
    require_columns(df, [_TARGET, _TEXT])
    assert_row_count(df, EXPECTED_ROWS)
    assert_no_nulls(df)
    assert_in_set(df, _TARGET, list(CLASS_ORDER))

    # 3. Clean — drop exact duplicate rows (99 articles appear verbatim more than
    # once). An identical article straddling the split would hand the vectorizer
    # the test answer word for word — the diamonds/sms_spam leak; with no
    # per-article identifier a verbatim repeat is indistinguishable from a
    # re-entry, so it goes.
    before = len(df)
    df = drop_duplicate_rows(df)
    logger.info("Dropped %d exact duplicate articles", before - len(df))

    # 4. Feature — the length features (text_features + count_tokens); see
    # add_text_features for the determinism note on token_count.
    df = add_text_features(df)

    # 5. Explore — the profile the modeling choices rest on: the summary and the
    # length story by topic (tech/politics run long, sport short — real signal
    # the length features carry beside the TF-IDF vectors).
    summarize(df).to_csv(output_dir / "summary.csv")
    df.groupby(_TARGET)[_NUMERIC_FEATURES].mean().to_csv(output_dir / "length_by_topic.csv")

    # 6. Encode the target — stateless (coding supplied, not learned), safe
    # before the split.
    df = encode_target(df)

    # 7. Split before anything fit-based; stratified so all five topics keep
    # their share in both halves. Keep only the columns the model reads.
    model_frame = df[[_TEXT, *_NUMERIC_FEATURES, _TARGET]]
    train, test = train_test_split_random(model_frame, test_size=0.2, stratify=_TARGET)

    # 8. Fit on train, apply to both. The ds plan holds the one frame-shaped
    # fitted step — standardizing the numeric length features; the TF-IDF
    # vectorizer, the fitted heart, lives in the sklearn model instead
    # (see _make_model).
    plan = [
        FitStep("scale_features", lambda frame: fit_scale_params(frame, columns=_NUMERIC_FEATURES)),
    ]
    scoring = fit_pipeline(train, plan)

    # 9. Cross-validate on the raw training split — the scale step re-fitted per
    # fold via make_pipeline, the vectorizer re-fitted per fold as make_model
    # builds a fresh sklearn pipeline; stratified across the five topics; scored
    # macro-averaged.
    cv_scores = cross_validate_kfold(
        train,
        target=_TARGET,
        make_model=_make_model,
        make_pipeline=lambda frame: fit_pipeline(frame, plan),
        stratify=True,
        metrics_fn=_macro_metrics,
    )
    cv_scores.to_csv(output_dir / "cv_folds.csv")
    logger.info(
        "5-fold macro CV f1: %.3f (+/- %.3f)",
        float(cv_scores["f1"].mean()),
        float(cv_scores["f1"].std()),
    )

    # 10. Apply the fitted plan to both windows, persist the processed frame, the
    # scoring pipeline and the model; then score the held-out split from the
    # reloaded model.
    train = scoring.apply(train)
    test = scoring.apply(test)
    assert_no_nulls(train)
    assert_no_nulls(test)
    save_processed(pd.concat([train, test]), "bbc_news_features.parquet", settings=settings)
    params_dir = settings.processed_dir / "params"
    save_params(scoring, params_dir / "bbc_news_scoring.json")

    x_train, y_train = split_features_target(train, _TARGET)
    x_test, y_test = split_features_target(test, _TARGET)
    save_model(_make_model().fit(x_train, y_train), params_dir / "bbc_news_model.joblib")
    model = load_model(params_dir / "bbc_news_model.joblib")
    preds = [int(value) for value in model.predict(x_test)]

    # 11. Evaluate (macro-averaged) against two references: a length-only model
    # (no TF-IDF — the text is the heart, so this should lose by a wide margin)
    # and the majority class (fit_baseline).
    length_only = _length_only_model().fit(x_train, y_train)
    length_only_preds = [int(value) for value in length_only.predict(x_test)]
    majority = fit_baseline(y_train, strategy="majority")
    majority_preds = [int(value) for value in majority.predict(len(y_test))]
    comparison = compare_models(
        y_test.tolist(),
        {
            "tfidf_logreg": preds,
            "length_only": length_only_preds,
            "majority_class": majority_preds,
        },
        metrics_fn=_macro_metrics,
    )
    comparison.to_csv(output_dir / "model_comparison.csv")
    confusion_frame(y_test.tolist(), preds, labels=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    per_class_metrics(y_test.tolist(), preds, labels=CLASS_NAMES).to_csv(
        output_dir / "per_class_metrics.csv"
    )

    metrics = _macro_metrics(y_test.tolist(), preds)
    length_scores = _macro_metrics(y_test.tolist(), length_only_preds)
    metrics.update({f"length_only_{name}": value for name, value in length_scores.items()})
    majority_scores = _macro_metrics(y_test.tolist(), majority_preds)
    metrics.update({f"majority_{name}": value for name, value in majority_scores.items()})
    logger.info("Held-out macro metrics vs references: %s", metrics)

    # 12. Visualize — the confusion matrix (with topic names on the axes) and the
    # model comparison bars.
    set_theme("notebook")
    fig, ax = plt.subplots()
    plot_confusion_matrix(y_test.tolist(), preds, labels=CLASS_NAMES, ax=ax)
    ax.set_title("BBC topic confusion matrix - held-out split")
    fig.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots()
    plot_model_comparison(comparison, metric="f1", ax=ax2)
    fig2.savefig(output_dir / "model_comparison.png", bbox_inches="tight")
    plt.close(fig2)

    return metrics


def main() -> None:
    settings = get_settings()
    metrics = run(settings.processed_dir / "bbc_news", settings=settings)
    print("Pipeline finished. Held-out macro metrics (vs length-only and majority class):")
    for name, value in metrics.items():
        print(f"  {name:>24}: {value:,.3f}")


if __name__ == "__main__":
    main()
