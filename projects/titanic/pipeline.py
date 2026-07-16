"""Titanic — predict a passenger's survival from real manifest records.

The second project on *real* data, and the first **classification** one: the
891 passengers of the seaborn ``titanic`` dataset (the classic Kaggle/RMS
Titanic manifest). Like ``projects/nyc_taxis``, nothing here was generated to
fit the toolkit — the data brings its own quirks: missing values at three very
different severities (``age`` ~20%, ``deck`` ~77%, ``embarked`` two rows), a
target respelled as a feature (``alive``), derived duplicates (``class``,
``who``, ``adult_male``, ``embark_town``, ``alone``), and a heavily skewed
``fare``.

The pipeline runs the full lifecycle on ``ds`` + scikit-learn alone: fetch →
validate → explore → clean → stratified split → fit-on-train/apply-to-both →
persist the scoring pipeline and the fitted model → score from the reloaded
model → evaluate with the classification stack (k-fold cross-validation,
confusion matrix, per-class metrics, model comparison) against a
majority-class baseline and the classic sex-only rule → visualize. Friction it
surfaced in the library is recorded in ``ROADMAP.md`` — that regenerated
backlog is as much the deliverable as the model.

Run it with::

    uv run ds run titanic

The raw CSV is downloaded once into ``<data_dir>/raw/`` (git-ignored) and
reused on later runs.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ds import Settings, get_logger, get_settings, seed_everything
from ds.eda import missing_value_report, summarize
from ds.evaluation import (
    classification_metrics,
    compare_models,
    confusion_frame,
    cross_validate_kfold,
    per_class_metrics,
)
from ds.features import fit_one_hot_categories, fit_scale_params
from ds.io import load_raw, save_params, save_processed
from ds.modeling.persistence import load_model, save_model
from ds.modeling.tabular import split_features_target
from ds.pipeline import Pipeline, PipelineStep
from ds.preprocessing import (
    apply_clip_outliers,
    apply_impute_missing,
    fit_impute_values,
    fit_outlier_bounds,
    standardize_column_names,
)
from ds.validation import assert_no_nulls, check_schema, require_columns
from ds.viz import (
    plot_confusion_matrix,
    plot_missingness,
    plot_model_comparison,
    set_theme,
)

logger = get_logger(__name__)

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
RAW_NAME = "titanic.csv"

# `alive` is the target respelled and `class` is `pclass` respelled — kept
# long enough to *verify* the redundancy, then dropped (see
# drop_leaky_and_derived). The rest are deterministic functions of retained
# columns: who/adult_male of sex+age, embark_town of embarked, alone of
# sibsp+parch.
_LEAKY_OR_DERIVED = ["alive", "class", "who", "adult_male", "embark_town", "alone"]
_CATEGORICAL = ["sex", "embarked"]
_NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
_TARGET = "survived"


def fetch_raw(settings: Settings) -> Path:
    """Download the titanic CSV into ``settings.raw_dir`` if not already there.

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


def drop_leaky_and_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the target respellings and derived duplicates, verifying first.

    ``alive`` must correspond one-to-one with the target ``survived`` and
    ``class`` one-to-one with ``pclass`` — if either assumption breaks, the
    columns are *not* the redundant respellings this project takes them for,
    and silently dropping them would discard information.

    Args:
        df: Frame carrying all raw manifest columns.

    Returns:
        A new frame without the leaky/derived columns.

    Raises:
        ValueError: If ``alive`` does not map one-to-one onto ``survived`` or
            ``class`` does not map one-to-one onto ``pclass``.
    """
    for respelled, of in (("alive", _TARGET), ("class", "pclass")):
        pairs = df[[respelled, of]].drop_duplicates()
        if len(pairs) != df[of].nunique(dropna=False):
            raise ValueError(
                f"expected {respelled!r} to be {of!r} respelled, but the columns"
                f" do not map one-to-one ({len(pairs)} distinct pairs)"
            )
    return df.drop(columns=_LEAKY_OR_DERIVED)


def engineer_passenger_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the project-specific features the generic helpers don't cover.

    ``deck`` is ~77% missing — far too sparse to impute a level into — but
    *whether* a cabin deck was recorded is itself informative (deck survives
    in the manifest mostly for first-class passengers), so it becomes a
    ``deck_known`` 0/1 indicator and the raw column is dropped.

    Args:
        df: Frame with the ``deck`` column.

    Returns:
        A new frame with ``deck_known`` added and ``deck`` dropped.
    """
    out = df.copy()
    out["deck_known"] = out["deck"].notna().astype(int)
    return out.drop(columns=["deck"])


def run(output_dir: Path, settings: Settings | None = None) -> dict[str, float]:
    """Run the survival-classification pipeline end to end.

    Args:
        output_dir: Directory to write artifacts (figures, reports) into.
        settings: Data-directory configuration; defaults to the shared one.

    Returns:
        Classification metrics on the stratified held-out split, plus
        ``sex_only_``-prefixed counterparts from the classic
        women-survive rule and ``majority_``-prefixed ones from a
        predict-the-majority-class reference.
    """
    seed_everything()
    settings = settings or get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire — one idempotent download into the git-ignored data tree.
    fetch_raw(settings)
    df = load_raw(RAW_NAME, settings=settings)

    # 2. Validate at the boundary. Nulls are allowed here — they are real and
    # get imputed split-safely below; what must hold is column presence and
    # parseable dtypes.
    df = standardize_column_names(df)
    require_columns(df, [_TARGET, *_LEAKY_OR_DERIVED, *_CATEGORICAL, *_NUMERIC_FEATURES, "deck"])
    df = check_schema(
        df,
        {
            _TARGET: "int64",
            "pclass": "int64",
            "age": "float64",
            "fare": "float64",
        },
        coerce=True,
    )

    # 3. Explore — persist the profile the modeling choices below rest on.
    summarize(df).to_csv(output_dir / "summary.csv")
    missing_value_report(df).to_csv(output_dir / "missing.csv")
    set_theme("notebook")
    fig, ax = plt.subplots()
    plot_missingness(df, ax=ax)
    ax.set_title("Missing values — age, deck, embarked")
    fig.savefig(output_dir / "missingness.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Structural clean and stateless features — nothing here learns
    # statistics, so it is safe before the split. Deliberately *no*
    # drop_duplicate_rows: 107 rows are exact duplicates yet are distinct
    # passengers (the manifest has no identifier column), so deduplicating
    # would silently delete real people from the sample.
    df = drop_leaky_and_derived(df)
    df = engineer_passenger_features(df)

    # 5. Split before anything fit-based. The manifest has no time axis, so
    # this is a shuffled split, stratified on the target to keep the 62/38
    # class balance in both halves (scikit-learn's, seeded by
    # seed_everything; ds.modeling only ships the chronological splitter).
    train, test = train_test_split(df, test_size=0.2, stratify=df[_TARGET])

    # 6. Fit on train, apply to both. Each parameter set is fitted on the
    # train frame as transformed by the steps before it, then the whole
    # ordered chain is applied through one Pipeline. Two impute steps because
    # age (median) and embarked (most frequent) need different strategies.
    bounds = fit_outlier_bounds(train, columns=["fare"])
    clipped = apply_clip_outliers(train, bounds)
    age_fill = fit_impute_values(clipped, columns=["age"], strategy="median")
    imputed = apply_impute_missing(clipped, age_fill)
    embarked_fill = fit_impute_values(imputed, columns=["embarked"], strategy="most_frequent")
    imputed = apply_impute_missing(imputed, embarked_fill)
    vocabulary = fit_one_hot_categories(imputed, columns=_CATEGORICAL)
    scaling = fit_scale_params(imputed, columns=_NUMERIC_FEATURES)

    scoring = Pipeline(
        steps=(
            PipelineStep("clip_outliers", bounds),
            PipelineStep("impute_missing", age_fill),
            PipelineStep("impute_missing", embarked_fill),
            PipelineStep("one_hot_encode", vocabulary),
            PipelineStep("scale_features", scaling),
        )
    )
    train = scoring.apply(train)
    test = scoring.apply(test)
    assert_no_nulls(train)
    assert_no_nulls(test)

    # 7. Persist the processed data and the whole scoring pipeline.
    save_processed(pd.concat([train, test]), "titanic_features.parquet", settings=settings)
    params_dir = settings.processed_dir / "params"
    save_params(scoring, params_dir / "titanic_scoring.json")

    # 8. Cross-validate on the training split before committing to a model —
    # the first real composition of cross_validate_kfold with
    # classification_metrics. Two caveats, both recorded as friction in
    # ROADMAP.md: the folds reuse transforms fitted on the *whole* training
    # frame (the ds Pipeline cannot re-fit per fold), and KFold cannot
    # stratify, so fold class balance drifts on a 62/38 target.
    cv_scores = cross_validate_kfold(
        train,
        target=_TARGET,
        make_model=lambda: LogisticRegression(max_iter=1000),
        metrics_fn=classification_metrics,
    )
    cv_scores.to_csv(output_dir / "cv_folds.csv")
    logger.info("5-fold CV accuracy: %.3f (+/- %.3f)", *_mean_std(cv_scores["accuracy"]))

    # 9. Model — fitted once, persisted next to the scoring pipeline, and the
    # held-out split scored from the *reloaded* copy, proving a later run
    # needs only the files on disk.
    x_train, y_train = split_features_target(train, _TARGET)
    x_test, y_test = split_features_target(test, _TARGET)
    save_model(
        LogisticRegression(max_iter=1000).fit(x_train, y_train),
        params_dir / "titanic_model.joblib",
    )
    model = load_model(params_dir / "titanic_model.joblib")
    preds = model.predict(x_test)

    # 10. Evaluate — the classification stack this project exists to exercise.
    # The majority-class reference is hand-rolled: ds.modeling.fit_baseline is
    # regression-shaped (its "mean" strategy would predict 0.38, not a class
    # label) — recorded as friction in ROADMAP.md rather than built here. The
    # sex-only rule (predict survival iff female) is the classic strong
    # heuristic every Titanic model must beat to justify its features.
    majority = int(y_train.mode().iloc[0])
    majority_preds = [majority] * len(y_test)
    sex_only_preds = [int(value) for value in x_test["sex_female"] > 0]
    comparison = compare_models(
        y_test.tolist(),
        {
            "logistic_regression": [int(value) for value in preds],
            "sex_only_rule": sex_only_preds,
            "majority_class": majority_preds,
        },
        metrics_fn=classification_metrics,
    )
    comparison.to_csv(output_dir / "model_comparison.csv")
    confusion_frame(y_test.tolist(), preds.tolist()).to_csv(output_dir / "confusion_matrix.csv")
    per_class_metrics(y_test.tolist(), preds.tolist()).to_csv(output_dir / "per_class_metrics.csv")
    metrics = classification_metrics(y_test.tolist(), preds.tolist())
    sex_only_scores = classification_metrics(y_test.tolist(), sex_only_preds)
    metrics.update({f"sex_only_{name}": value for name, value in sex_only_scores.items()})
    majority_scores = classification_metrics(y_test.tolist(), majority_preds)
    metrics.update({f"majority_{name}": value for name, value in majority_scores.items()})
    logger.info("Held-out metrics vs baselines: %s", metrics)

    # 11. Visualize.
    fig2, ax2 = plt.subplots()
    plot_confusion_matrix(y_test.tolist(), preds.tolist(), ax=ax2)
    ax2.set_title("Survival confusion matrix — held-out split")
    fig2.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    plot_confusion_matrix(y_test.tolist(), preds.tolist(), ax=ax3, normalize=True)
    ax3.set_title("Survival confusion matrix — row-normalized")
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
    metrics = run(settings.processed_dir / "titanic", settings=settings)
    print("Pipeline finished. Held-out metrics (vs sex-only rule and majority class):")
    for name, value in metrics.items():
        print(f"  {name:>18}: {value:,.3f}")


if __name__ == "__main__":
    main()
