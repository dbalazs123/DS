"""Adult Income — predict whether a 1994 census respondent earns over $50K.

The seventh project on *real* data, and the first **heavily-categorical** one:
the 32,560-row training split of the UCI Adult / Census Income dataset, drawn
from the 1994 US Current Population Survey. Each row is one respondent — age,
work class, education, marital status, occupation, race, sex, weekly hours, a
capital-gain/-loss pair and country of origin — and the task is the classic
binary one: does this person's income exceed $50K a year (about 24% do)?

The dataset was chosen by the established rule — grep which library surfaces
still have no real consumer — and it earns its keep on the whole *categorical*
cluster that was thin or unused: high-cardinality ``native_country`` (41
countries, 90% United-States) and ``occupation`` (14 trades with a sparse
tail) are the second real consumer of ``fit_topk_categories`` /
``collapse_categories``; the wide indicator matrix they and five more
categoricals expand into is the first real consumer of one-hot
``drop_first=True`` (the dummy-trap guard a linear model needs); the
zero-inflated ``capital_gain`` / ``capital_loss`` are the first real consumer
of ``flag_outliers`` on its *flag*-not-clip path (the extremes are signal, not
error). Friction it surfaced in the library is recorded in ``ROADMAP.md`` —
that regenerated backlog is as much the deliverable as the model.

The pipeline runs the full lifecycle on ``ds`` + scikit-learn alone: fetch →
validate the boundary and pin the numeric dtypes → decode the ``?`` sentinels
and encode the target → explore → stratified split → a four-step transform
plan (collapse tails / mode-impute the sentinels / wide one-hot / scale) fitted
on the training split and re-fitted inside every CV fold → persist the scoring
pipeline and the fitted model → score from the reloaded model → evaluate with
the classification stack against a majority-class floor and an interpretable
marital-status rule → visualize.

Run it with::

    uv run ds run adult_income

The raw CSV is downloaded once into ``<data_dir>/raw/`` (git-ignored) and
reused on later runs. The UCI archive is not reachable from every network, so
the fetch pins a GitHub mirror of the training split and verifies its sha256
before trusting the download.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ds import Settings, get_logger, get_settings, seed_everything
from ds.eda import missing_value_report, summarize, top_correlations
from ds.evaluation import (
    classification_metrics,
    compare_models,
    confusion_frame,
    cross_validate_kfold,
    per_class_metrics,
)
from ds.features import fit_one_hot_categories, fit_scale_params, fit_topk_categories
from ds.io import load_raw, save_params, save_processed
from ds.modeling.baseline import fit_baseline
from ds.modeling.persistence import load_model, save_model
from ds.modeling.tabular import split_features_target, train_test_split_random
from ds.pipeline import FitStep, fit_pipeline
from ds.preprocessing import fit_impute_values, flag_outliers, standardize_column_names
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
    plot_outliers,
    set_theme,
)

logger = get_logger(__name__)

# The UCI archive (dataset 2) is not reachable from every network, so the fetch
# uses a GitHub mirror of the original adult.data training split — same 15
# columns, whitespace-padded values, "?" sentinels and CRLF line endings — and
# verifies the checksum below before trusting the download. (A second
# byte-identical mirror would slot straight into this tuple; one plus the pin
# matches the seaborn-mirror projects' fetch, the checksum matches air_quality.)
DATA_URLS = ("https://raw.githubusercontent.com/dsrscientist/dataset1/master/census_income.csv",)
RAW_NAME = "census_income.csv"
RAW_SHA256 = "833cc71e1409363daa6a882d48d582c2e043da10e712e8fa8e7b50833cd5c15f"

# The published size of the training split: 32,561 rows in the raw file, one of
# which is a blank separator line the CSV read drops, leaving 32,560 records.
EXPECTED_ROWS = 32560

_TARGET = "income"

# fnlwgt is the census sampling weight (how many people the row represents) —
# an artifact of the survey design, not an attribute of the person, so it is no
# predictor. education duplicates education_num, which is its ordinal code
# (Preschool=1 … Doctorate=16); the numeric form is kept and the 16-way string
# dropped rather than one-hot-expanded into a redundant block.
_DROPPED = ["fnlwgt", "education"]

# The three columns whose missing values arrive as a literal "?" string.
_SENTINEL_COLUMNS = ["workclass", "occupation", "native_country"]

# The two genuinely high-cardinality categoricals: native_country is 41
# countries dominated by United-States with a long <100-row tail, occupation 14
# trades with a sparse tail (Armed-Forces has nine rows). Both get their rare
# levels collapsed to "other" before one-hot so the linear model spends
# coefficients on levels with support (friction item 4's second consumer).
_HIGH_CARDINALITY = ["native_country", "occupation"]
_TOP_K = 10

# Every categorical feature that becomes indicator columns. workclass /
# occupation / native_country still carry the sentinel gaps the impute step
# fills; all seven expand under one-hot.
_CATEGORICAL = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
_NUMERIC_FEATURES = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

# The label names for the consumer-facing confusion/per-class artifacts; the
# metric math stays on the 0/1 codes.
_LABELS = {0: "<=50K", 1: ">50K"}


def fetch_raw(settings: Settings) -> Path:
    """Download the census CSV into ``settings.raw_dir`` and verify it.

    Tries each mirror in :data:`DATA_URLS` in order and checks the download
    against :data:`RAW_SHA256` — the mirror is a personal repository, so a
    silently drifted copy must fail loudly, not parse strangely. A cached copy
    is re-verified rather than trusted (a partial earlier download would
    otherwise poison every later run).

    Args:
        settings: Resolves the raw-data directory.

    Returns:
        Path to the verified local copy of the dataset.

    Raises:
        ValueError: If no mirror serves a file matching the pinned checksum.
        urllib.error.URLError: If every mirror is unreachable.
    """
    destination = settings.raw_dir / RAW_NAME
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == RAW_SHA256:
            return destination
        logger.warning("Cached %s fails its checksum; re-downloading", destination)
        destination.unlink()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in DATA_URLS:
        logger.info("Downloading %s -> %s", url, destination)
        try:
            with urllib.request.urlopen(url) as response:
                payload = response.read()
        except OSError as exc:  # URLError subclasses OSError
            last_error = exc
            continue
        if hashlib.sha256(payload).hexdigest() == RAW_SHA256:
            destination.write_bytes(payload)
            return destination
        logger.warning("Mirror %s served a file that fails the pinned checksum", url)
        last_error = ValueError(f"checksum mismatch from {url}")
    if isinstance(last_error, ValueError):
        raise last_error
    raise urllib.error.URLError(f"no mirror reachable for {RAW_NAME}") from last_error


def decode_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the file's ``"?"`` missing-value strings into real NaN.

    Three categorical columns tag an unknown value with a literal ``"?"``. Left
    in place it is a legal-looking category that one-hot would expand into its
    own indicator and impute would never touch; as NaN the gaps become visible
    to ``missing_value_report`` and fillable by the mode-impute step. (The
    whitespace that padded every value is already gone — ``skipinitialspace``
    stripped it at read time — so the sentinel is the bare ``"?"``.)

    This is the second project in the workspace to decode an in-band missing
    sentinel (air_quality's was a numeric −200); the string flavor here is the
    same one-line ``replace`` on a different spelling, recorded against
    ROADMAP item 26.

    Args:
        df: Frame with the sentinel columns present.

    Returns:
        A new frame with ``"?"`` replaced by NaN in those columns.
    """
    out = df.copy()
    out[_SENTINEL_COLUMNS] = out[_SENTINEL_COLUMNS].replace("?", np.nan)
    return out


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the ``income`` label as 1 for ``>50K`` (the positive class), else 0.

    The frozen ``Baseline`` and the classification metrics work on integer
    codes (friction item 6's numeric-label contract), and ``>50K`` — the ~24%
    minority the model exists to find — is the natural positive class, so it
    maps to 1. The string values are validated first: an unexpected label would
    otherwise be silently coded 0.

    Args:
        df: Frame with a string ``income`` column of ``<=50K`` / ``>50K``.

    Returns:
        A new frame with ``income`` as an ``int64`` 0/1 column.

    Raises:
        DataValidationError: If ``income`` carries a value outside the two
            expected labels.
    """
    assert_in_set(df, _TARGET, {"<=50K", ">50K"})
    out = df.copy()
    out[_TARGET] = (out[_TARGET] == ">50K").astype("int64")
    return out


def run(output_dir: Path, settings: Settings | None = None) -> dict[str, float]:
    """Run the income-classification pipeline end to end.

    Args:
        output_dir: Directory to write artifacts (figures, reports) into.
        settings: Paths to read/write data under; resolved from the
            environment when omitted. Tests pass a temporary one so a run
            never touches the shared data tree.

    Returns:
        Classification metrics on the stratified held-out split, plus
        ``married_``-prefixed counterparts from the marital-status rule and
        ``majority_``-prefixed ones from the majority-class reference.
    """
    seed_everything()
    settings = settings or get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire — one idempotent, checksum-verified download into the
    # git-ignored data tree. skipinitialspace strips the leading blank that pads
    # every categorical value (" Private" -> "Private"); load_raw forwards it to
    # the reader, so no project-local loader is needed.
    fetch_raw(settings)
    df = load_raw(RAW_NAME, settings=settings, skipinitialspace=True)

    # 2. Validate the boundary: the published row count (a silently-wrong parse
    # would land at a different number), the columns we depend on, and — the
    # parse pin — the six numeric columns actually numeric. check_schema coerces
    # them so a stray stringified column fails here, loudly, rather than
    # surfacing as a mangled scale or one-hot block far downstream.
    df = standardize_column_names(df)
    assert_row_count(df, EXPECTED_ROWS)
    require_columns(df, [_TARGET, *_DROPPED, *_CATEGORICAL, *_NUMERIC_FEATURES])
    df = check_schema(df, dict.fromkeys(_NUMERIC_FEATURES, "int64"), coerce=True)

    # 3. Clean — decode the "?" sentinels to NaN (so the gaps are visible and
    # imputable), encode the target to 0/1, and drop the two non-features. No
    # drop_duplicate_rows: 24 rows are exact duplicates, but with only these
    # coarse attributes and no respondent identifier, identical vectors are
    # expected distinct people, so deduplicating would delete real records
    # (the titanic precedent, the deliberate opposite of the diamonds drop).
    df = decode_sentinels(df)
    df = encode_target(df)
    df = df.drop(columns=_DROPPED)

    # 4. Explore — persist the profile the modeling choices rest on: the
    # summary, the missing-value report and plot (the three sentinel columns,
    # ~1.8k/1.8k/0.6k gaps), the numeric correlations with the target (capital
    # gain and education_num lead), and the outlier view that decides the
    # capital columns are flagged, not clipped.
    set_theme("notebook")
    summarize(df).to_csv(output_dir / "summary.csv")
    missing_value_report(df).to_csv(output_dir / "missing.csv")
    top_correlations(df, n=15).to_csv(output_dir / "top_correlations.csv", index=False)

    fig, ax = plt.subplots()
    plot_missingness(df, ax=ax)
    ax.set_title("Missing values — the '?' sentinel columns")
    fig.savefig(output_dir / "missingness.png", bbox_inches="tight")
    plt.close(fig)

    # capital_gain/capital_loss are ~92% zero with a long right tail (gains run
    # to 99,999). flag_outliers *reports* how many values sit beyond the IQR
    # fence without touching them — clipping would erase exactly the large-gain
    # signal that most cleanly separates the classes, so the plan below keeps
    # them intact. This is the flag path's first real consumer (the clip path
    # serves the earlier projects).
    outlier_flags = flag_outliers(df, columns=["capital_gain", "capital_loss"])
    outlier_flags.sum().to_frame("outlier_count").to_csv(output_dir / "capital_outliers.csv")
    fig2, ax2 = plt.subplots()
    plot_outliers(df, columns=["capital_gain", "capital_loss"], ax=ax2)
    ax2.set_title("Capital gain/loss outliers — flagged, not clipped")
    fig2.savefig(output_dir / "capital_outliers.png", bbox_inches="tight")
    plt.close(fig2)

    # 5. Split before anything fit-based. No time axis, so a shuffled split
    # (friction item 7's helper, seeded by seed_everything), stratified on the
    # target to keep the ~24/76 class balance in both halves. The interpretable
    # reference is read off the *raw* test split now, before the transforms
    # dissolve marital_status into indicator columns.
    train, test = train_test_split_random(df, test_size=0.2, stratify=_TARGET)
    married_preds = [int(value) for value in test["marital_status"] == "Married-civ-spouse"]
    y_test_raw = test[_TARGET].tolist()

    # 6. Fit the four-step transform plan on the training split only, and re-fit
    # it inside every CV fold via the same object. Order is load-bearing:
    # collapse the high-cardinality tails first (so impute/one-hot see the
    # reduced vocabulary), mode-impute the sentinel gaps, one-hot every
    # categorical with drop_first (a linear model needs the reference level
    # dropped to stay full-rank — the first drop_first consumer), then scale the
    # numerics for the penalized logistic regression.
    plan = [
        FitStep(
            "collapse_categories",
            lambda frame: fit_topk_categories(frame, columns=_HIGH_CARDINALITY, k=_TOP_K),
        ),
        FitStep(
            "impute_missing",
            lambda frame: fit_impute_values(
                frame, columns=_SENTINEL_COLUMNS, strategy="most_frequent"
            ),
        ),
        FitStep(
            "one_hot_encode",
            lambda frame: fit_one_hot_categories(frame, columns=_CATEGORICAL, drop_first=True),
        ),
        FitStep("scale_features", lambda frame: fit_scale_params(frame, columns=_NUMERIC_FEATURES)),
    ]
    scoring = fit_pipeline(train, plan)

    # 7. Cross-validate on the *raw* training split, the transform chain re-fit
    # inside each fold (friction item 9) and stratified to hold the class
    # balance (friction item 8). The fitted state genuinely varies fold to fold
    # here — the mode of an occupation, the kept top-k countries, the scale
    # centres — so re-fitting per fold keeps each fold's test rows out of its
    # own transform statistics.
    cv_scores = cross_validate_kfold(
        train,
        target=_TARGET,
        make_model=lambda: LogisticRegression(max_iter=1000),
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
    save_processed(pd.concat([train, test]), "adult_income_features.parquet", settings=settings)
    params_dir = settings.processed_dir / "params"
    save_params(scoring, params_dir / "adult_income_scoring.json")

    x_train, y_train = split_features_target(train, _TARGET)
    x_test, y_test = split_features_target(test, _TARGET)
    save_model(
        LogisticRegression(max_iter=1000).fit(x_train, y_train),
        params_dir / "adult_income_model.joblib",
    )
    model = load_model(params_dir / "adult_income_model.joblib")
    preds = [int(value) for value in model.predict(x_test)]

    # 9. Evaluate — the classification stack against two references the model
    # must beat: the majority-class floor (fit_baseline, all-<=50K: high
    # accuracy on this imbalance, zero recall) and the interpretable
    # marital-status rule (predict >50K iff married-civ-spouse — the single
    # strongest demographic split, recovering real recall at the cost of
    # accuracy). The model beats both on accuracy and F1.
    majority = fit_baseline(y_train, strategy="majority")
    majority_preds = [int(value) for value in majority.predict(len(y_test))]
    comparison = compare_models(
        y_test_raw,
        {
            "logistic_regression": preds,
            "married_rule": married_preds,
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
    married_scores = classification_metrics(y_test_raw, married_preds)
    metrics.update({f"married_{name}": value for name, value in married_scores.items()})
    majority_scores = classification_metrics(y_test_raw, majority_preds)
    metrics.update({f"majority_{name}": value for name, value in majority_scores.items()})
    logger.info("Held-out metrics vs references: %s", metrics)

    # 10. Visualize — the confusion matrix (raw and row-normalized, with the
    # income labels on the axes) and the model comparison bars.
    fig3, ax3 = plt.subplots()
    plot_confusion_matrix(y_test_raw, preds, labels=_LABELS, ax=ax3)
    ax3.set_title("Income confusion matrix — held-out split")
    fig3.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig3)

    fig4, ax4 = plt.subplots()
    plot_confusion_matrix(y_test_raw, preds, labels=_LABELS, ax=ax4, normalize=True)
    ax4.set_title("Income confusion matrix — row-normalized")
    fig4.savefig(output_dir / "confusion_matrix_normalized.png", bbox_inches="tight")
    plt.close(fig4)

    fig5, ax5 = plt.subplots()
    plot_model_comparison(comparison, metric="f1", ax=ax5)
    fig5.savefig(output_dir / "model_comparison.png", bbox_inches="tight")
    plt.close(fig5)

    return metrics


def main() -> None:
    settings = get_settings()
    metrics = run(settings.processed_dir / "adult_income", settings=settings)
    print("Pipeline finished. Held-out metrics (vs married-civ-spouse rule and majority class):")
    for name, value in metrics.items():
        print(f"  {name:>20}: {value:,.3f}")


if __name__ == "__main__":
    main()
