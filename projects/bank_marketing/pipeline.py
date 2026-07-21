"""Bank Marketing — predict who subscribes to a term deposit (a rare "yes").

The eleventh project on *real* data, and the first deliberately **imbalanced /
rare-event** one: the 41,188-row UCI Bank Marketing dataset (the
``bank-additional-full`` variant), one row per phone contact in a Portuguese
bank's 2008–2010 term-deposit campaign. Each row carries the client's
demographics (age, job, marital status, education), their existing products
(``default`` / ``housing`` / ``loan``), how this campaign reached them
(``contact``, ``month``, ``day_of_week``, ``campaign`` count), their prior
campaign history (``previous``, ``poutcome``, and a ``pdays`` "days since last
contact" whose 999 means *never*), and five macro-economic indicators
(``emp_var_rate`` … ``nr_employed``). The task is the classic one — did the
client subscribe? — but only **11.3%** did, and that skew is the whole point.

The dataset was chosen by the established rule — grep which library surfaces
still have no real consumer — and it stresses a *shape* every prior
classification project (titanic, adult_income, diamonds, sms_spam, bbc_news)
lacked: a rare positive class where **accuracy is a trap**. A model that
predicts "no" for everyone scores 0.887 accuracy while finding *not one*
subscriber, so hard-label metrics can't tell an honest story here. That gap is
exactly what this project surfaced in the library — there was no way to score a
classifier's *probabilities* (its ranking of who is likeliest to say yes,
independent of any threshold) — and it pulled the new
:func:`ds.evaluation.probability_metrics` (ROC-AUC / average precision / Brier)
into the toolkit. The imbalance itself is handled idiomatically with
``class_weight="balanced"`` on the estimator, so its 0.5 threshold still yields
a meaningful confusion matrix. Friction it surfaced is recorded in
``ROADMAP_ARCHIVE.md`` — that regenerated backlog is as much the deliverable as
the model.

Two boundary decisions are load-bearing and specific to this dataset:

- ``duration`` (the call length in seconds) is **dropped as leakage**: it is
  unknown until *after* the call that determines the outcome, and a call of
  zero seconds is a "no" by construction, so any model that keeps it scores
  spectacularly and predicts nothing usable. The UCI documentation is explicit
  that it must be excluded for a realistic model; this is the first project to
  drop a feature purely for leakage (the fnlwgt / education drops in
  adult_income were redundancy, not leakage).
- ``pdays == 999`` is an in-band "never previously contacted" sentinel covering
  ~96% of rows; left numeric it would drag the scale. It becomes a clean binary
  ``was_previously_contacted`` feature and the raw column is dropped (the same
  decode-a-sentinel pattern as air_quality's −200 and adult_income's "?", third
  consumer).

The pipeline runs the full lifecycle on ``ds`` + scikit-learn alone: fetch →
validate the boundary and pin the numeric dtypes → encode the target and the
``pdays`` sentinel → explore (with the per-level subscribe rate that makes
``poutcome`` and ``contact`` legible) → stratified split → a two-step transform
plan (wide one-hot / scale) fitted on train and re-fitted inside every CV fold
→ persist the scoring pipeline and the class-weighted model → score from the
reloaded model → evaluate **probabilistically** (ROC-AUC / PR-AUC vs a
prevalence floor) *and* on hard labels (vs a majority-class floor and an
interpretable prior-success rule) → visualize.

Run it with::

    uv run ds run bank_marketing

The raw CSV is downloaded once into ``<data_dir>/raw/`` (git-ignored) and
reused on later runs. The UCI archive is not reachable from every network, so
the fetch pins a GitHub mirror and verifies its sha256 before trusting it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ds import Settings, get_logger, get_settings, seed_everything
from ds.eda import (
    missing_value_report,
    summarize,
    target_rate_by_category,
    top_correlations,
)
from ds.evaluation import (
    classification_metrics,
    compare_models,
    confusion_frame,
    cross_validate_kfold,
    per_class_metrics,
    probability_metrics,
)
from ds.features import fit_one_hot_categories, fit_scale_params
from ds.io import fetch_dataset, load_raw, save_params, save_processed
from ds.modeling.baseline import fit_baseline
from ds.modeling.persistence import load_model, save_model
from ds.modeling.tabular import split_features_target, train_test_split_random
from ds.pipeline import FitStep, fit_pipeline
from ds.preprocessing import standardize_column_names
from ds.validation import (
    assert_in_set,
    assert_no_nulls,
    assert_row_count,
    check_schema,
    require_columns,
)
from ds.viz import (
    plot_confusion_matrix,
    plot_missingness,
    plot_model_comparison,
    plot_target_rate,
    set_theme,
)

logger = get_logger(__name__)

# The UCI archive is not reachable from every network, so the fetch uses a
# GitHub mirror of the bank-additional-full variant — 20 features + the y
# target, ";"-separated — and ds.io.fetch_dataset verifies the checksum below
# before trusting the download. (A second byte-identical mirror would slot
# straight into this tuple.)
DATA_URLS = ("https://raw.githubusercontent.com/selva86/datasets/master/bank-full.csv",)
RAW_NAME = "bank_marketing.csv"
RAW_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"

# The published size of the bank-additional-full dataset: 41,188 records.
EXPECTED_ROWS = 41188

_TARGET = "y"

# duration is the call length in seconds — a leakage feature: it is only known
# after the call whose outcome we predict (a 0-second call is a "no" by
# construction), and the UCI docs are explicit it must be excluded for a
# realistic model. Dropped before anything else touches the frame.
_LEAKAGE = "duration"

# pdays == 999 is an in-band "never previously contacted" sentinel (~96% of
# rows); it becomes a binary was_previously_contacted feature and the raw
# numeric column is dropped so its 999s don't drag the scale.
_PDAYS = "pdays"
_PDAYS_SENTINEL = 999
_CONTACTED_FLAG = "was_previously_contacted"

# Every categorical feature that becomes indicator columns under one-hot.
_CATEGORICAL = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

# The numeric features the scaler standardizes. duration and pdays are handled
# above (dropped / turned into the flag); the flag itself is already 0/1 and
# left off the scaler.
_NUMERIC_FEATURES = [
    "age",
    "campaign",
    "previous",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
]

# The label names for the consumer-facing confusion/per-class artifacts; the
# metric math stays on the 0/1 codes.
_LABELS = {0: "no", 1: "yes"}


def fetch_raw(settings: Settings) -> Path:
    """Download the bank-marketing CSV into ``settings.raw_dir`` and verify it.

    A thin binding of this project's dataset (mirrors, filename, pinned digest)
    to :func:`ds.io.fetch_dataset`, which does the multi-mirror download,
    checksum verification and cache re-verify (ROADMAP_ARCHIVE.md item 27's shared dance).

    Args:
        settings: Resolves the raw-data directory.

    Returns:
        Path to the verified local copy of the dataset.

    Raises:
        ValueError: If no mirror serves a file matching the pinned checksum.
        urllib.error.URLError: If every mirror is unreachable.
    """
    return fetch_dataset(RAW_NAME, DATA_URLS, sha256=RAW_SHA256, settings=settings)


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the ``y`` label as 1 for ``"yes"`` (the rare positive), else 0.

    The frozen ``Baseline`` and the classification metrics work on integer
    codes, and ``"yes"`` — the ~11% minority the campaign exists to find — is
    the natural positive class, so it maps to 1. The string values are
    validated first: an unexpected label would otherwise be silently coded 0.

    Args:
        df: Frame with a string ``y`` column of ``"yes"`` / ``"no"``.

    Returns:
        A new frame with ``y`` as an ``int64`` 0/1 column.

    Raises:
        DataValidationError: If ``y`` carries a value outside the two expected
            labels.
    """
    assert_in_set(df, _TARGET, {"yes", "no"})
    out = df.copy()
    out[_TARGET] = (out[_TARGET] == "yes").astype("int64")
    return out


def encode_pdays_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the ``pdays == 999`` "never contacted" sentinel into a binary flag.

    ``pdays`` is days since the client was last contacted in a *prior*
    campaign, but 999 (about 96% of rows) is an in-band sentinel meaning
    "never previously contacted", not a real 999-day gap. Left numeric it is a
    huge false magnitude the scaler would spread the genuine 0–27 range against.
    The signal that actually matters — *was* there a prior contact at all — is
    kept as a clean 0/1 ``was_previously_contacted`` column, and the raw
    ``pdays`` is dropped.

    Args:
        df: Frame with a numeric ``pdays`` column.

    Returns:
        A new frame with ``was_previously_contacted`` added and ``pdays``
        removed.
    """
    out = df.copy()
    out[_CONTACTED_FLAG] = (out[_PDAYS] != _PDAYS_SENTINEL).astype("int64")
    return out.drop(columns=[_PDAYS])


def run(output_dir: Path, settings: Settings | None = None) -> dict[str, float]:
    """Run the term-deposit classification pipeline end to end.

    Args:
        output_dir: Directory to write artifacts (figures, reports) into.
        settings: Paths to read/write data under; resolved from the
            environment when omitted. Tests pass a temporary one so a run
            never touches the shared data tree.

    Returns:
        Hard-label classification metrics on the stratified held-out split,
        plus ``roc_auc`` / ``average_precision`` / ``brier`` from the
        probabilistic evaluation, ``prior_success_``-prefixed counterparts
        from the interpretable rule and ``majority_``/``prevalence_``-prefixed
        ones from the naive floors.
    """
    seed_everything()
    settings = settings or get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire — one idempotent, checksum-verified download into the
    # git-ignored data tree. The mirror is ";"-separated, so load_raw forwards
    # the separator to the reader; no project-local loader is needed.
    fetch_raw(settings)
    df = load_raw(RAW_NAME, settings=settings, sep=";")

    # 2. Validate the boundary: the published row count (a silently-wrong parse
    # would land elsewhere), the columns we depend on, and — the parse pin — the
    # numeric columns actually numeric. check_schema coerces them so a stray
    # stringified column fails here, loudly, rather than surfacing as a mangled
    # scale far downstream. (pdays is validated before it becomes the flag.)
    df = standardize_column_names(df)
    assert_row_count(df, EXPECTED_ROWS)
    require_columns(df, [_TARGET, _LEAKAGE, _PDAYS, *_CATEGORICAL, *_NUMERIC_FEATURES])
    df = check_schema(
        df, {**dict.fromkeys(_NUMERIC_FEATURES, "float64"), _PDAYS: "int64"}, coerce=True
    )

    # 3. Clean — drop the duration leakage feature, encode the target to 0/1 and
    # fold the pdays 999 sentinel into a binary contacted flag. No
    # drop_duplicate_rows: with only these coarse attributes and no client
    # identifier, identical vectors are expected distinct contacts, so
    # deduplicating would delete real records (the titanic/adult_income
    # precedent).
    df = df.drop(columns=[_LEAKAGE])
    df = encode_target(df)
    df = encode_pdays_sentinel(df)

    # 4. Explore — persist the profile the modeling rests on: the summary, the
    # missing-value report (this dataset has no NaN — the "unknown" category is
    # a legal level, not a gap), the numeric correlations with the target (the
    # macro indicators nr_employed / euribor3m lead), and the categorical read.
    set_theme("notebook")
    summarize(df).to_csv(output_dir / "summary.csv")
    missing_value_report(df).to_csv(output_dir / "missing.csv")
    top_correlations(df, n=15).to_csv(output_dir / "top_correlations.csv", index=False)

    fig_missing, ax_missing = plt.subplots()
    plot_missingness(df, ax=ax_missing)
    ax_missing.set_title("Missing values — none (the 'unknown' level is a category)")
    fig_missing.savefig(output_dir / "missingness.png", bbox_inches="tight")
    plt.close(fig_missing)

    # top_correlations is numeric-only, so it cannot rank the categorical
    # predictors — which on this dataset carry much of the signal.
    # target_rate_by_category is the categorical read: the subscribe rate per
    # level of poutcome (a prior "success" subscribes ~65% of the time — this is
    # what makes the prior_success_rule below a real classifier) and per contact
    # method (cellular beats telephone). Descriptive only: it never becomes a
    # model feature, so computing it over the whole frame is a profile, not
    # leakage.
    for feature in ("poutcome", "contact"):
        target_rate_by_category(df, feature, _TARGET).to_csv(
            output_dir / f"target_rate_{feature}.csv"
        )
    fig_rate, ax_rate = plt.subplots()
    plot_target_rate(df, "poutcome", _TARGET, ax=ax_rate)
    ax_rate.set_title("Subscribe rate by prior-campaign outcome")
    fig_rate.savefig(output_dir / "target_rate_poutcome.png", bbox_inches="tight")
    plt.close(fig_rate)

    # 5. Split before anything fit-based. No usable time axis (the year is not
    # in the file), so a shuffled split (seeded by seed_everything), stratified
    # on the target to keep the ~11/89 class balance in both halves. The
    # interpretable reference is read off the *raw* test split now, before the
    # transforms dissolve poutcome into indicator columns.
    train, test = train_test_split_random(df, test_size=0.2, stratify=_TARGET)
    prior_success_preds = [int(value) for value in test["poutcome"] == "success"]
    y_test_raw = test[_TARGET].tolist()

    # 6. Fit the two-step transform plan on the training split only, re-fit
    # inside every CV fold via the same object: one-hot every categorical with
    # drop_first (a linear model needs the reference level dropped to stay
    # full-rank), then scale the numerics for the penalized logistic regression.
    plan = [
        FitStep(
            "one_hot_encode",
            lambda frame: fit_one_hot_categories(frame, columns=_CATEGORICAL, drop_first=True),
        ),
        FitStep("scale_features", lambda frame: fit_scale_params(frame, columns=_NUMERIC_FEATURES)),
    ]
    scoring = fit_pipeline(train, plan)

    # class_weight="balanced" is how the imbalance is handled: it up-weights the
    # rare "yes" in the loss so the fitted 0.5 threshold still yields a
    # meaningful confusion matrix (an unweighted fit would predict "no" almost
    # always). This is a scikit-learn estimator argument, not a library gap.
    def make_model() -> LogisticRegression:
        return LogisticRegression(max_iter=1000, class_weight="balanced")

    # 7. Cross-validate on the *raw* training split, the transform chain re-fit
    # inside each fold and stratified to hold the class balance. The fitted
    # state varies fold to fold (the learned one-hot vocabulary, the scale
    # centres), so re-fitting per fold keeps each fold's test rows out of its
    # own transform statistics.
    cv_scores = cross_validate_kfold(
        train,
        target=_TARGET,
        make_model=make_model,
        make_pipeline=lambda frame: fit_pipeline(frame, plan),
        stratify=True,
        metrics_fn=classification_metrics,
    )
    cv_scores.to_csv(output_dir / "cv_folds.csv")
    logger.info(
        "5-fold CV f1: %.3f (+/- %.3f)",
        float(cv_scores["f1"].mean()),
        float(cv_scores["f1"].std()),
    )

    # 8. Apply the fitted plan to both windows and persist everything a later
    # scoring run needs: the processed frame, the strict-JSON scoring pipeline,
    # and the model.
    train = scoring.apply(train)
    test = scoring.apply(test)
    assert_no_nulls(train)
    assert_no_nulls(test)
    save_processed(pd.concat([train, test]), "bank_marketing_features.parquet", settings=settings)
    params_dir = settings.processed_dir / "params"
    save_params(scoring, params_dir / "bank_marketing_scoring.json")

    x_train, y_train = split_features_target(train, _TARGET)
    x_test, y_test = split_features_target(test, _TARGET)
    save_model(make_model().fit(x_train, y_train), params_dir / "bank_marketing_model.joblib")
    model = load_model(params_dir / "bank_marketing_model.joblib")
    preds = [int(value) for value in model.predict(x_test)]
    # The positive-class probability — the input the probabilistic metrics score.
    scores = [float(p) for p in model.predict_proba(x_test)[:, 1]]

    # 9. Evaluate. Two views, because at 11% prevalence one is a trap:
    #
    #   (a) Probabilistic (threshold-free) — the honest headline. roc_auc /
    #       average_precision score how well the model *ranks* subscribers,
    #       against a prevalence floor (a constant = the training positive rate,
    #       which by construction ranks at chance: ROC-AUC 0.5, AP = prevalence).
    #       This is the new ds.evaluation.probability_metrics, the surface this
    #       project pulled.
    #
    #   (b) Hard-label — the operating-point view. The class-weighted model at
    #       its 0.5 threshold, against the majority-class floor (all-"no": 0.887
    #       accuracy, zero recall — the trap made concrete) and the
    #       interpretable prior-success rule (predict "yes" iff a prior campaign
    #       succeeded).
    prevalence = fit_baseline(y_train, strategy="mean")
    prevalence_scores = [float(v) for v in prevalence.predict(len(y_test))]
    probabilistic = compare_models(
        y_test_raw,
        {"logistic_regression": scores, "prevalence_rate": prevalence_scores},
        metrics_fn=probability_metrics,
    )
    probabilistic.to_csv(output_dir / "probabilistic_comparison.csv")

    majority = fit_baseline(y_train, strategy="majority")
    majority_preds = [int(value) for value in majority.predict(len(y_test))]
    comparison = compare_models(
        y_test_raw,
        {
            "logistic_regression": preds,
            "prior_success_rule": prior_success_preds,
            "majority_class": majority_preds,
        },
        metrics_fn=classification_metrics,
    )
    comparison.to_csv(output_dir / "model_comparison.csv")
    confusion_frame(y_test_raw, preds, labels=_LABELS).to_csv(output_dir / "confusion_matrix.csv")
    per_class_metrics(y_test_raw, preds, labels=_LABELS).to_csv(
        output_dir / "per_class_metrics.csv"
    )

    metrics = classification_metrics(y_test_raw, preds)
    metrics.update(probability_metrics(y_test_raw, scores))
    prior_scores = classification_metrics(y_test_raw, prior_success_preds)
    metrics.update({f"prior_success_{name}": value for name, value in prior_scores.items()})
    majority_scores = classification_metrics(y_test_raw, majority_preds)
    metrics.update({f"majority_{name}": value for name, value in majority_scores.items()})
    metrics.update(
        {
            f"prevalence_{name}": value
            for name, value in probability_metrics(y_test_raw, prevalence_scores).items()
        }
    )
    logger.info("Held-out metrics vs references: %s", metrics)

    # 10. Visualize — the confusion matrix (raw and row-normalized, with the
    # yes/no labels on the axes), the hard-label model comparison bars, and the
    # probabilistic ranking bars (average precision, the imbalance-robust one).
    fig_cm, ax_cm = plt.subplots()
    plot_confusion_matrix(y_test_raw, preds, labels=_LABELS, ax=ax_cm)
    ax_cm.set_title("Term-deposit confusion matrix — held-out split")
    fig_cm.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig_cm)

    fig_cmn, ax_cmn = plt.subplots()
    plot_confusion_matrix(y_test_raw, preds, labels=_LABELS, ax=ax_cmn, normalize=True)
    ax_cmn.set_title("Term-deposit confusion matrix — row-normalized")
    fig_cmn.savefig(output_dir / "confusion_matrix_normalized.png", bbox_inches="tight")
    plt.close(fig_cmn)

    fig_cmp, ax_cmp = plt.subplots()
    plot_model_comparison(comparison, metric="f1", ax=ax_cmp)
    fig_cmp.savefig(output_dir / "model_comparison.png", bbox_inches="tight")
    plt.close(fig_cmp)

    fig_ap, ax_ap = plt.subplots()
    plot_model_comparison(probabilistic, metric="average_precision", ax=ax_ap)
    ax_ap.set_title("Ranking quality (average precision) vs prevalence floor")
    fig_ap.savefig(output_dir / "probabilistic_comparison.png", bbox_inches="tight")
    plt.close(fig_ap)

    return metrics


def main() -> None:
    settings = get_settings()
    metrics = run(settings.processed_dir / "bank_marketing", settings=settings)
    print("Pipeline finished. Held-out metrics (vs prior-success rule and naive floors):")
    for name, value in metrics.items():
        print(f"  {name:>28}: {value:,.3f}")


if __name__ == "__main__":
    main()
