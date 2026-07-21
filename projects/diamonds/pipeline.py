"""Diamonds — grade the cut of a diamond from its measurements.

The fourth project on *real* data, and the first **multiclass** one: the
53,940 diamonds of the classic ggplot2 ``diamonds`` dataset (seaborn-data
mirror), graded into five ordered cut classes (Fair < Good < Very Good <
Premium < Ideal). It is the first data to stress three surfaces the library
carries but no real project has consumed: the ordinal-encoding pair with an
explicit domain ordering (``color`` and ``clarity`` are genuinely ranked
scales), the outlier pair and ``plot_outliers`` on data with genuine
measurement errors (physically impossible zero dimensions, a 58.9 mm width on
a 2-carat stone), and the classification metric/plot surface beyond two
classes (``per_class_metrics``, ``confusion_frame``,
``plot_confusion_matrix``, ``classification_metrics(average="macro")``).

The pipeline runs the full lifecycle on ``ds`` + scikit-learn alone: fetch →
validate at the boundary → drop the physically impossible rows and exact
duplicates → explore (correlations, carat-band cut shares via ``bin_column``,
the outlier profile) → encode the ordered target → stratified split → fit the
three-step transform plan on the training split (``ds.pipeline.fit_pipeline``)
→ stratified 5-fold cross-validation with the plan re-fitted per fold →
persist the scoring pipeline and the fitted model → score the held-out split
from the reloaded model → evaluate against the majority class and a
proportions-only rule → visualize. Friction it surfaced in the library is
recorded in ``ROADMAP_ARCHIVE.md`` — that regenerated backlog is as much the
deliverable as the classifier.

Run it with::

    uv run ds run diamonds

The raw CSV is downloaded once into ``<data_dir>/raw/`` (git-ignored) and
reused on later runs.
"""

from __future__ import annotations

import urllib.request
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ds import Settings, get_logger, get_settings, seed_everything
from ds.eda import summarize, top_correlations
from ds.evaluation import (
    classification_metrics,
    compare_models,
    confusion_frame,
    cross_validate_kfold,
    per_class_metrics,
)
from ds.features import (
    bin_column,
    fit_ordinal_categories,
    fit_scale_params,
    ordinal_encode,
)
from ds.io import load_raw, save_params, save_processed
from ds.modeling.baseline import fit_baseline
from ds.modeling.persistence import load_model, save_model
from ds.modeling.tabular import split_features_target, train_test_split_random
from ds.pipeline import FitStep, fit_pipeline
from ds.preprocessing import drop_duplicate_rows, fit_outlier_bounds, standardize_column_names
from ds.validation import (
    assert_in_range,
    assert_in_set,
    assert_no_nulls,
    check_schema,
    require_columns,
)
from ds.viz import plot_confusion_matrix, plot_model_comparison, plot_outliers, set_theme

logger = get_logger(__name__)

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
RAW_NAME = "diamonds.csv"

_TARGET = "cut"
# The three graded scales, worst to best, as the GIA defines them. Supplying
# the order explicitly is the point: the default (sorted unique values) would
# rank color D..J backwards and interleave the clarity grades alphabetically.
CUT_ORDER = ("Fair", "Good", "Very Good", "Premium", "Ideal")
COLOR_ORDER = ("J", "I", "H", "G", "F", "E", "D")
CLARITY_ORDER = ("I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF")
# Code -> grade name, for the labels= display mapping on the metric frames
# and the confusion plot (the metric math stays on the int codes).
CUT_LABELS = dict(enumerate(CUT_ORDER))

_DIMENSIONS = ("x", "y", "z")  # length / width / depth, mm
_NUMERIC_RAW = ("carat", "depth", "table", "price", *_DIMENSIONS)
# Every model input, post-encoding; the scale step must name them explicitly
# or it would standardize the int-coded target too.
_FEATURES = ("carat", "color", "clarity", "depth", "table", "price", *_DIMENSIONS)


def fetch_raw(settings: Settings) -> Path:
    """Download the diamonds CSV into ``settings.raw_dir`` if not already there.

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


def drop_impossible_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any physical dimension is zero or negative.

    A diamond with a zero length, width or depth cannot exist — the dataset's
    20 such rows are recording errors, not extreme stones, so they are removed
    before the range validation downstream (which then asserts strict
    positivity for every surviving row). Hand-rolled mask: the validation
    stage can only assert, and no cleaning helper filters rows by a range.

    Args:
        df: Frame with the ``x``/``y``/``z`` dimension columns.

    Returns:
        A new frame with only the rows where all three dimensions are
        strictly positive, reindexed from zero.
    """
    valid = (df[list(_DIMENSIONS)] > 0).all(axis=1)
    dropped = int((~valid).sum())
    if dropped:
        logger.info("Dropping %d rows with a zero/negative physical dimension", dropped)
    return df.loc[valid].reset_index(drop=True)


def encode_cut(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the target's five cut grades as ordered integer codes 0-4.

    Uses the stage's ordinal encoder with the explicit quality order, so
    ``Fair`` maps to 0 and ``Ideal`` to 4. Because the order is supplied
    rather than learned from the frame, nothing here is fitted state and the
    call is safe before the train/test split. Int codes are also what the
    evaluation stage and ``fit_baseline("majority")`` accept — the metric
    surface is typed for integer labels.

    Args:
        df: Frame whose ``cut`` values are all in :data:`CUT_ORDER`
            (guaranteed by the vocabulary validation upstream).

    Returns:
        A new frame with ``cut`` replaced by its integer code.
    """
    return ordinal_encode(df, [_TARGET], categories={_TARGET: list(CUT_ORDER)})


def proportions_rule(df: pd.DataFrame) -> list[int]:
    """Grade cut from ``depth`` and ``table`` alone — the honest reference.

    Cut grade is *defined* by a stone's proportions, so the classifier must
    beat a rule that reads only the two proportion columns: near-ideal depth
    and table → ``Ideal``, progressively wider bands → ``Premium`` /
    ``Very Good``, everything else → ``Good``. (``Fair`` is never predicted —
    at 3% support the rule gains nothing from a fourth band, which keeps it
    honest rather than tuned.) Bands follow the standard round-brilliant
    grading charts, not a fit to this data, and read raw units — the rule
    must see the frame *before* scaling.

    Args:
        df: Frame with raw ``depth`` and ``table`` columns.

    Returns:
        One predicted cut code (0-4) per row.
    """
    ideal = (df["depth"].between(60.5, 62.5)) & (df["table"] <= 57)
    premium = (df["depth"] <= 63.5) & (df["table"] <= 59)
    very_good = df["depth"].between(61.0, 64.0)
    codes = pd.Series(CUT_ORDER.index("Good"), index=df.index)
    codes[very_good] = CUT_ORDER.index("Very Good")
    codes[premium] = CUT_ORDER.index("Premium")
    codes[ideal] = CUT_ORDER.index("Ideal")
    return [int(code) for code in codes]


# The idiom for the two-argument metrics_fn hooks beyond two classes:
# classification_metrics defaults to average="binary", which raises at five.
# Macro (unweighted per-class mean) is the right average here — with 40% of
# stones graded Ideal and 3% Fair, a weighted average would hide the minority
# classes this project exists to look at.
macro_classification_metrics = partial(classification_metrics, average="macro")


def run(output_dir: Path, settings: Settings | None = None) -> dict[str, float]:
    """Run the cut-grading pipeline end to end.

    Args:
        output_dir: Directory to write artifacts (figures, reports) into.
        settings: Data-directory configuration; defaults to the shared one.

    Returns:
        Macro-averaged classification metrics on the stratified held-out
        split, plus ``proportions_``-prefixed counterparts from the
        depth/table-only rule and ``majority_``-prefixed ones from the
        predict-Ideal majority reference.
    """
    seed_everything()
    settings = settings or get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire — one idempotent download into the git-ignored data tree.
    fetch_raw(settings)
    df = load_raw(RAW_NAME, settings=settings)

    # 2. Validate at the boundary: column presence, parseable dtypes, no
    # nulls (this dataset has none — imputation stays with the projects whose
    # data demands it), the three known grade vocabularies, and plausibility
    # ranges. The zero-dimension rows must be dropped *before* the dimension
    # range checks can assert strict positivity — validation only flags, so a
    # hand-rolled mask filters.
    df = standardize_column_names(df)
    require_columns(df, [_TARGET, "color", "clarity", *_NUMERIC_RAW])
    df = check_schema(df, {"price": "int64", "carat": "float64"}, coerce=True)
    assert_no_nulls(df)
    assert_in_set(df, _TARGET, list(CUT_ORDER))
    assert_in_set(df, "color", list(COLOR_ORDER))
    assert_in_set(df, "clarity", list(CLARITY_ORDER))
    df = drop_impossible_dimensions(df)
    assert_in_range(df, "carat", min_value=0.0, inclusive="right")
    assert_in_range(df, "price", min_value=1)
    assert_in_range(df, "depth", min_value=40, max_value=100)
    assert_in_range(df, "table", min_value=40, max_value=100)
    for dimension in _DIMENSIONS:
        assert_in_range(df, dimension, min_value=0.0, max_value=60.0, inclusive="right")

    # 3. Clean — exact duplicates are re-entries here, not distinct stones:
    # with no identifier column, ties across all ten measured columns (price
    # to the dollar, dimensions to 0.01 mm) don't plausibly describe two
    # diamonds, and a duplicate pair straddling the split would leak test
    # rows into training. (Contrast titanic, which deliberately *keeps* its
    # duplicates — those are real people.)
    before = len(df)
    df = drop_duplicate_rows(df)
    logger.info("Dropped %d exact duplicate rows", before - len(df))

    # 4. Explore — persist the profile the modeling choices below rest on:
    # the summary, the correlation pairs (carat and the three dimensions are
    # one near-collinear size block), the cut-grade mix per carat band
    # (bin_column's first consumer — quantile bins as an exploration device
    # on the full frame, not a model feature), and the outlier profile
    # (plot_outliers' first real data: price/carat right skew is genuine
    # signal, while the extreme y/z values are measurement errors).
    summarize(df).to_csv(output_dir / "summary.csv")
    top_correlations(df).to_csv(output_dir / "correlations.csv", index=False)
    banded = bin_column(df, "carat", bins=5, method="quantile")
    cut_mix = pd.crosstab(banded["carat_bin"], banded[_TARGET], normalize="index")
    cut_mix[list(CUT_ORDER)].to_csv(output_dir / "carat_band_cut_mix.csv")
    set_theme("notebook")
    fig, ax = plt.subplots()
    plot_outliers(df, columns=list(_NUMERIC_RAW), ax=ax)
    ax.set_title("IQR outliers per column — skew is real, y/z extremes are errors")
    fig.savefig(output_dir / "outliers.png", bbox_inches="tight")
    plt.close(fig)

    # 5. Encode the ordered target — stateless (the order is supplied, not
    # learned), so safe before the split.
    df = encode_cut(df)

    # 6. Split before anything fit-based; stratified so both halves keep the
    # imbalanced five-class mix (40% Ideal .. 3% Fair).
    train, test = train_test_split_random(df, test_size=0.2, stratify=_TARGET)

    # 7. Fit on train, apply to both. Three steps: clip the physical
    # dimensions (where extremes are measurement errors — deliberately *not*
    # depth/table, whose extremes are exactly the poor proportions that make
    # a cut Fair, nor price/carat, whose skew is real), ordinal-encode the
    # two graded scales with their explicit worst-to-best orders (the pair's
    # first real consumer), and standardize every feature for the
    # multinomial model.
    plan = [
        FitStep(
            "clip_outliers",
            lambda df: fit_outlier_bounds(df, columns=list(_DIMENSIONS)),
        ),
        FitStep(
            "ordinal_encode",
            lambda df: fit_ordinal_categories(
                df,
                ["color", "clarity"],
                categories={"color": COLOR_ORDER, "clarity": CLARITY_ORDER},
            ),
        ),
        FitStep("scale_features", lambda df: fit_scale_params(df, columns=list(_FEATURES))),
    ]
    scoring = fit_pipeline(train, plan)

    # 8. Cross-validate on the *raw* training split — the plan is re-fitted
    # inside each fold via make_pipeline, stratified so every fold keeps all
    # five classes at the frame's balance, scored with the macro metrics the
    # imbalance demands.
    cv_scores = cross_validate_kfold(
        train,
        target=_TARGET,
        make_model=lambda: LogisticRegression(max_iter=2000),
        make_pipeline=lambda frame: fit_pipeline(frame, plan),
        stratify=True,
        metrics_fn=macro_classification_metrics,
    )
    cv_scores.to_csv(output_dir / "cv_folds.csv")
    logger.info("5-fold CV macro F1: %.3f (+/- %.3f)", *_mean_std(cv_scores["f1"]))

    # The proportions rule reads raw depth/table bands, so its held-out
    # predictions are computed from the untransformed test split, before the
    # scoring pipeline scales those columns.
    rule_preds = proportions_rule(test)

    train = scoring.apply(train)
    test = scoring.apply(test)
    assert_no_nulls(train)
    assert_no_nulls(test)

    # 9. Persist the processed data and the whole scoring pipeline.
    save_processed(pd.concat([train, test]), "diamonds_features.parquet", settings=settings)
    params_dir = settings.processed_dir / "params"
    save_params(scoring, params_dir / "diamonds_scoring.json")

    # 10. Model — multinomial logistic regression on the scaled features,
    # fitted once, persisted next to the scoring pipeline, and the held-out
    # split scored from the *reloaded* copy, proving a later run needs only
    # the files on disk.
    x_train, y_train = split_features_target(train, _TARGET)
    x_test, y_test = split_features_target(test, _TARGET)
    save_model(
        LogisticRegression(max_iter=2000).fit(x_train, y_train),
        params_dir / "diamonds_model.joblib",
    )
    model = load_model(params_dir / "diamonds_model.joblib")
    preds = [int(value) for value in model.predict(x_test)]

    # 11. Evaluate — the multiclass stack this project exists to exercise.
    # The majority reference (predict Ideal for everything) comes from the
    # library's baseline — the int-coded target is exactly the numeric-label
    # scope it was built for. The proportions rule is the domain heuristic
    # the model must beat to justify reading the other seven columns.
    majority = fit_baseline(y_train, strategy="majority")
    majority_preds = [int(value) for value in majority.predict(len(y_test))]
    comparison = compare_models(
        y_test.tolist(),
        {
            "logistic_regression": preds,
            "proportions_rule": rule_preds,
            "majority_class": majority_preds,
        },
        metrics_fn=macro_classification_metrics,
    )
    comparison.to_csv(output_dir / "model_comparison.csv")
    confusion_frame(y_test.tolist(), preds, labels=CUT_LABELS).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    per_class_metrics(y_test.tolist(), preds, labels=CUT_LABELS).to_csv(
        output_dir / "per_class_metrics.csv"
    )
    metrics = macro_classification_metrics(y_test.tolist(), preds)
    rule_scores = macro_classification_metrics(y_test.tolist(), rule_preds)
    metrics.update({f"proportions_{name}": value for name, value in rule_scores.items()})
    majority_scores = macro_classification_metrics(y_test.tolist(), majority_preds)
    metrics.update({f"majority_{name}": value for name, value in majority_scores.items()})
    logger.info("Held-out metrics vs baselines: %s", metrics)

    # 12. Visualize.
    fig2, ax2 = plt.subplots()
    plot_confusion_matrix(y_test.tolist(), preds, ax=ax2, labels=CUT_LABELS)
    ax2.set_title("Cut confusion matrix — held-out split")
    fig2.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    plot_confusion_matrix(y_test.tolist(), preds, ax=ax3, normalize=True, labels=CUT_LABELS)
    ax3.set_title("Cut confusion matrix — row-normalized")
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
    metrics = run(settings.processed_dir / "diamonds", settings=settings)
    print("Pipeline finished. Held-out macro metrics (vs proportions rule and majority class):")
    for name, value in metrics.items():
        print(f"  {name:>22}: {value:,.3f}")


if __name__ == "__main__":
    main()
