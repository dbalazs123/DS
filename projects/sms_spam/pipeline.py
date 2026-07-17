"""SMS Spam — flag a text message as spam from its content.

The fifth project on *real* data, and the first **text** one: the 5,574
labelled messages of the SMS Spam Collection (ham/spam, ~13% spam). It is the
first data to stress the surfaces no real project has consumed: the
``nlp``/tiktoken extra (``ds.modeling.nlp.count_tokens`` gets its first real
consumer, and its graceful-degradation contract gets a real test), the
``labels=`` display mapping's second consumer after diamonds (here
``{0: "ham", 1: "spam"}``), and — by *absence* — the text gaps: the features
stage has no text helpers, and ``ds.pipeline``'s closed step vocabulary has no
vectorization step, so the TF-IDF vectorizer's fitted state must live in the
scikit-learn model object outside the persisted ``ds`` ``Pipeline``. That
friction is recorded in ``ROADMAP.md`` — the regenerated backlog is as much
the deliverable as the classifier.

The pipeline runs the full lifecycle on ``ds`` + scikit-learn alone: fetch →
validate at the boundary (the raw file is a headerless TSV whose quotes are
literal text — ``quoting=csv.QUOTE_NONE`` or two rows silently vanish) → drop
exact duplicate rows → explore (length profiles by label, length-band label
mix via ``bin_column``) → engineer text features → encode the target →
stratified split → fit the one-step transform plan (``fit_pipeline``) →
stratified 5-fold cross-validation with the plan re-fitted per fold → persist
the scoring pipeline and the fitted model → score the held-out split from the
reloaded model → evaluate against the majority class and a keyword rule →
visualize.

**Determinism across CI jobs:** ``count_tokens`` returns real BPE token counts
only when tiktoken is importable *and* its vocabulary is loadable; otherwise
it degrades to a whitespace count. CI runs the suite both without extras and
with ``--extra all``, so the ``token_count`` column is **descriptive only** —
it feeds the exploration artifacts, never the model. The model reads the
TF-IDF text features plus ``char_count`` (deterministic everywhere), so
fitted models and metrics are identical in both jobs, and the tests assert
structure and baselines-beaten rather than token-dependent values.

Run it with::

    uv run ds run sms_spam

The raw TSV is downloaded once into ``<data_dir>/raw/`` (git-ignored) and
reused on later runs.
"""

from __future__ import annotations

import csv
import urllib.request
from collections.abc import Callable
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
from ds.features import bin_column, fit_scale_params, ordinal_encode
from ds.io import load_raw, save_params, save_processed
from ds.modeling.baseline import fit_baseline
from ds.modeling.nlp import count_tokens
from ds.modeling.persistence import load_model, save_model
from ds.modeling.tabular import split_features_target, train_test_split_random
from ds.pipeline import FitStep, fit_pipeline
from ds.preprocessing import drop_duplicate_rows
from ds.validation import (
    assert_in_range,
    assert_in_set,
    assert_no_nulls,
    require_columns,
)
from ds.viz import plot_confusion_matrix, plot_model_comparison, plot_outliers, set_theme

logger = get_logger(__name__)

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
RAW_NAME = "sms.tsv"

_TARGET = "label"
# ham first so the spam-detection task gets the conventional coding: spam = 1
# is the positive class the binary metrics score.
LABEL_ORDER = ("ham", "spam")
# Code -> label name, for the labels= display mapping on the metric frames and
# the confusion plot (the metric math stays on the int codes).
LABEL_NAMES = dict(enumerate(LABEL_ORDER))

# Classic SMS-spam markers (prize-claim and premium-SMS vocabulary), written
# down from domain knowledge rather than tuned on this data — the honest
# reference the model must beat to justify learning a vocabulary.
_SPAM_KEYWORDS = ("free", "win", "won", "prize", "claim", "urgent", "txt", "cash", "award")


def fetch_raw(settings: Settings) -> Path:
    """Download the SMS Spam Collection TSV into ``settings.raw_dir`` if absent.

    Args:
        settings: Resolves the raw-data directory.

    Returns:
        Path to the local copy of the dataset.
    """
    destination = settings.raw_dir / RAW_NAME
    if destination.exists():
        return destination
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", DATA_URL, destination)
    with urllib.request.urlopen(DATA_URL) as response:
        destination.write_bytes(response.read())
    return destination


def load_messages(settings: Settings) -> pd.DataFrame:
    """Load the raw TSV into a two-column ``label``/``message`` frame.

    The file is headerless and its double quotes are message text, not CSV
    quoting — several messages open with ``"`` and never close it. Under
    pandas' default quoting two physical rows are silently swallowed into a
    neighbour (5,572 rows instead of 5,574, one with an embedded newline), so
    the read must disable quote handling. Everything here rides on
    ``load_raw`` forwarding pandas keywords: the ``ds.io`` readers have no
    first-class notion of a headerless file.

    Args:
        settings: Resolves the raw-data directory.

    Returns:
        The messages frame with ``label`` and ``message`` string columns.
    """
    return load_raw(
        RAW_NAME,
        settings=settings,
        header=None,
        names=["label", "message"],
        quoting=csv.QUOTE_NONE,
    )


def _resolve_token_counter() -> Callable[[str], int]:
    """Probe once whether ``count_tokens``' accurate path is live.

    ``count_tokens`` degrades gracefully — but *per call*: with tiktoken
    installed and its vocabulary endpoint unreachable, every call re-attempts
    the download before falling back (~0.4 s observed), which would turn the
    5,000-message feature map below into a half-hour stall. The caller cannot
    tell from a returned count which path produced it, so the guard has to
    reproduce the library's own probe: try the encoding once, and on any
    failure use the same documented whitespace fallback wholesale. Recorded
    as backlog friction in ``ROADMAP.md``.

    Returns:
        ``count_tokens`` when the accurate path works, else the whitespace
        fallback.
    """
    try:
        import tiktoken

        tiktoken.get_encoding("cl100k_base")  # count_tokens' default encoding
    except Exception:
        return lambda text: len(text.split())
    return count_tokens


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-message length features: ``char_count`` and ``token_count``.

    Both are hand-rolled here because the features stage has no text helpers.
    ``char_count`` (message length in characters) is deterministic everywhere
    and may feed the model. ``token_count`` comes from
    :func:`ds.modeling.nlp.count_tokens` (probed once — see
    :func:`_resolve_token_counter`), whose value depends on whether tiktoken
    and its vocabulary are available — it is kept descriptive-only (see the
    module docstring) so the model never sees it.

    Args:
        df: Frame with a string ``message`` column.

    Returns:
        A new frame with the two integer count columns appended.
    """
    out = df.copy()
    out["char_count"] = out["message"].str.len().astype(int)
    counter = _resolve_token_counter()
    out["token_count"] = [counter(message) for message in out["message"]]
    return out


def encode_label(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the target as integer codes: ham -> 0, spam -> 1.

    Uses the stage's ordinal encoder with the explicit order, so the coding is
    supplied rather than learned and the call is safe before the train/test
    split. Int codes are what the evaluation stage and
    ``fit_baseline("majority")`` accept, and spam = 1 makes spam the positive
    class of the binary metrics.

    Args:
        df: Frame whose ``label`` values are all in :data:`LABEL_ORDER`
            (guaranteed by the vocabulary validation upstream).

    Returns:
        A new frame with ``label`` replaced by its integer code.
    """
    return ordinal_encode(df, [_TARGET], categories={_TARGET: list(LABEL_ORDER)})


def keyword_rule(df: pd.DataFrame) -> list[int]:
    """Flag a message as spam if it contains any classic spam keyword.

    The domain heuristic the classifier must beat — the SMS analogue of the
    diamonds proportions rule. Case-insensitive substring containment against
    :data:`_SPAM_KEYWORDS`; scores ~0.66 F1 on the full collection, well above
    the majority baseline's 0 but far from what a learned vocabulary reaches.

    Args:
        df: Frame with a raw string ``message`` column.

    Returns:
        One predicted label code (0 ham / 1 spam) per row.
    """
    lowered = df["message"].str.lower()
    return [int(any(keyword in message for keyword in _SPAM_KEYWORDS)) for message in lowered]


def _make_model() -> SklearnPipeline:
    """Build the fresh, unfitted spam classifier.

    A scikit-learn pipeline: TF-IDF over the raw ``message`` text plus the
    (already ds-scaled) ``char_count`` column, into a logistic regression.
    The vectorizer lives *inside* the model object because ``ds.pipeline``'s
    step vocabulary has no vectorization step — its fitted vocabulary
    persists in the model joblib, outside the ``ds`` scoring pipeline (the
    friction this project exists to record). Anything else in the frame
    (``token_count``) is dropped by the column selection.
    """
    features = ColumnTransformer(
        [
            ("tfidf", TfidfVectorizer(), "message"),
            ("length", "passthrough", ["char_count"]),
        ]
    )
    return SklearnPipeline(
        [("features", features), ("classifier", LogisticRegression(max_iter=1000))]
    )


def run(output_dir: Path, settings: Settings | None = None) -> dict[str, float]:
    """Run the spam-detection pipeline end to end.

    Args:
        output_dir: Directory to write artifacts (figures, reports) into.
        settings: Data-directory configuration; defaults to the shared one.

    Returns:
        Binary classification metrics (spam = positive class) on the
        stratified held-out split, plus ``keyword_``-prefixed counterparts
        from the keyword rule and ``majority_``-prefixed ones from the
        predict-ham majority reference.
    """
    seed_everything()
    settings = settings or get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire — one idempotent download into the git-ignored data tree,
    # then the quote-disabled headerless read (see load_messages).
    fetch_raw(settings)
    df = load_messages(settings)

    # 2. Validate at the boundary: both columns present, no nulls, the label
    # vocabulary exact. This dataset has zero missing values — imputation
    # stays with the projects whose data demands it. The engineered
    # char_count is range-checked below, which doubles as a no-empty-messages
    # assertion.
    require_columns(df, [_TARGET, "message"])
    assert_no_nulls(df)
    assert_in_set(df, _TARGET, list(LABEL_ORDER))

    # 3. Clean — exact duplicate rows are dropped. The collection has ~400
    # (chain-letter spam sent verbatim to many recipients, stock phrases like
    # "Sorry, I'll call later"), and with no sender or timestamp column a
    # repeat is indistinguishable from a re-entry. More to the point, an
    # identical message straddling the split would hand the vectorizer the
    # test answer verbatim — the text analogue of the diamonds leak.
    # (Contrast titanic, which keeps its duplicates: those rows carry
    # identifying columns and are demonstrably distinct people.)
    before = len(df)
    df = drop_duplicate_rows(df)
    logger.info("Dropped %d exact duplicate messages", before - len(df))

    # 4. Feature — the two hand-rolled length features (no text helpers in
    # the features stage). The range check on char_count asserts every
    # surviving message is non-empty.
    df = add_text_features(df)
    assert_in_range(df, "char_count", min_value=1)

    # 5. Explore — persist the profile the modeling choices rest on: the
    # summary, the length story by label (spam is long: ~139 chars vs ~72 —
    # padded toward the 160-char SMS limit), the label mix across
    # char_count quantile bands (bin_column's second consumer, again as an
    # exploration device rather than a model feature), and the length
    # outlier profile (the extreme lengths are real multi-part messages, not
    # errors). token_count appears here and only here — descriptive use, per
    # the module docstring.
    summarize(df).to_csv(output_dir / "summary.csv")
    df.groupby(_TARGET)[["char_count", "token_count"]].agg(["mean", "median", "max"]).to_csv(
        output_dir / "length_by_label.csv"
    )
    banded = bin_column(df, "char_count", bins=5, method="quantile")
    label_mix = pd.crosstab(banded["char_count_bin"], banded[_TARGET], normalize="index")
    label_mix.to_csv(output_dir / "length_band_label_mix.csv")
    set_theme("notebook")
    fig, ax = plt.subplots()
    plot_outliers(df, columns=["char_count", "token_count"], ax=ax)
    ax.set_title("Length outliers — long tails are real multi-part messages")
    fig.savefig(output_dir / "outliers.png", bbox_inches="tight")
    plt.close(fig)

    # 6. Encode the target — stateless (the coding is supplied, not learned),
    # so safe before the split.
    df = encode_label(df)

    # 7. Split before anything fit-based; stratified so both halves keep the
    # imbalanced ham/spam mix (~13% spam).
    train, test = train_test_split_random(df, test_size=0.2, stratify=_TARGET)

    # 8. Fit on train, apply to both. The plan holds the *one* step the
    # closed vocabulary can express for this data — standardizing char_count
    # for the linear model. The TF-IDF vectorizer, the actual fitted heart of
    # a text pipeline, cannot be a step: it lives inside the sklearn model
    # instead (see _make_model), splitting the fitted state across two
    # artifacts.
    plan = [
        FitStep("scale_features", lambda df: fit_scale_params(df, columns=["char_count"])),
    ]
    scoring = fit_pipeline(train, plan)

    # 9. Cross-validate on the raw training split — the scale step re-fitted
    # per fold via make_pipeline, the vectorizer re-fitted per fold as a side
    # effect of make_model building a fresh sklearn pipeline; stratified so
    # every fold keeps the 13% spam share; scored with the binary metrics
    # (spam = positive class).
    cv_scores = cross_validate_kfold(
        train,
        target=_TARGET,
        make_model=_make_model,
        make_pipeline=lambda frame: fit_pipeline(frame, plan),
        stratify=True,
        metrics_fn=classification_metrics,
    )
    cv_scores.to_csv(output_dir / "cv_folds.csv")
    logger.info("5-fold CV spam F1: %.3f (+/- %.3f)", *_mean_std(cv_scores["f1"]))

    # The keyword rule reads raw message text, so its held-out predictions
    # come from the untransformed test split.
    rule_preds = keyword_rule(test)

    train = scoring.apply(train)
    test = scoring.apply(test)
    assert_no_nulls(train)
    assert_no_nulls(test)

    # 10. Persist the processed data and the scoring pipeline. Note what the
    # pipeline holds: the char_count scaler and nothing else — reloading it
    # alone cannot score a message; the joblib below carries the vocabulary.
    save_processed(pd.concat([train, test]), "sms_spam_features.parquet", settings=settings)
    params_dir = settings.processed_dir / "params"
    save_params(scoring, params_dir / "sms_spam_scoring.json")

    # 11. Model — TF-IDF + logistic regression, fitted once, persisted next
    # to the scoring pipeline, and the held-out split scored from the
    # *reloaded* copy, proving a later run needs only the files on disk.
    x_train, y_train = split_features_target(train, _TARGET)
    x_test, y_test = split_features_target(test, _TARGET)
    save_model(_make_model().fit(x_train, y_train), params_dir / "sms_spam_model.joblib")
    model = load_model(params_dir / "sms_spam_model.joblib")
    preds = [int(value) for value in model.predict(x_test)]

    # 12. Evaluate — the binary metric surface with the display mapping. The
    # majority reference (predict ham for everything) scores 0 on every
    # spam-positive metric except accuracy — exactly why accuracy alone is
    # the wrong lens at 13% positives. The keyword rule is the domain
    # heuristic the model must beat to justify learning a vocabulary.
    majority = fit_baseline(y_train, strategy="majority")
    majority_preds = [int(value) for value in majority.predict(len(y_test))]
    comparison = compare_models(
        y_test.tolist(),
        {
            "tfidf_logistic": preds,
            "keyword_rule": rule_preds,
            "majority_class": majority_preds,
        },
        metrics_fn=classification_metrics,
    )
    comparison.to_csv(output_dir / "model_comparison.csv")
    confusion_frame(y_test.tolist(), preds, labels=LABEL_NAMES).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    per_class_metrics(y_test.tolist(), preds, labels=LABEL_NAMES).to_csv(
        output_dir / "per_class_metrics.csv"
    )
    metrics = classification_metrics(y_test.tolist(), preds)
    rule_scores = classification_metrics(y_test.tolist(), rule_preds)
    metrics.update({f"keyword_{name}": value for name, value in rule_scores.items()})
    majority_scores = classification_metrics(y_test.tolist(), majority_preds)
    metrics.update({f"majority_{name}": value for name, value in majority_scores.items()})
    logger.info("Held-out metrics vs baselines: %s", metrics)

    # 13. Visualize.
    fig2, ax2 = plt.subplots()
    plot_confusion_matrix(y_test.tolist(), preds, ax=ax2, labels=LABEL_NAMES)
    ax2.set_title("Spam confusion matrix — held-out split")
    fig2.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    plot_confusion_matrix(y_test.tolist(), preds, ax=ax3, normalize=True, labels=LABEL_NAMES)
    ax3.set_title("Spam confusion matrix — row-normalized")
    fig3.savefig(output_dir / "confusion_matrix_normalized.png", bbox_inches="tight")
    plt.close(fig3)

    fig4, ax4 = plt.subplots()
    plot_model_comparison(comparison, metric="f1", ax=ax4)
    fig4.savefig(output_dir / "model_comparison.png", bbox_inches="tight")
    plt.close(fig4)

    return metrics


def _mean_std(values: pd.Series) -> tuple[float, float]:
    """Mean and standard deviation of a numeric series, for log lines."""
    return float(values.mean()), float(values.std())


def main() -> None:
    settings = get_settings()
    metrics = run(settings.processed_dir / "sms_spam", settings=settings)
    print("Pipeline finished. Held-out binary metrics (vs keyword rule and majority class):")
    for name, value in metrics.items():
        print(f"  {name:>20}: {value:,.3f}")


if __name__ == "__main__":
    main()
