"""Flights — forecast monthly airline passengers from the classic series.

The third project on *real* data, and the first **forecasting** one: the
144 monthly totals of international airline passengers, 1949–1960 (the
Box & Jenkins "AirPassengers" series, mirrored as seaborn's ``flights``
dataset). It is the canonical seasonal series — a strong upward trend with a
pronounced yearly cycle (summer peaks) whose amplitude grows with the level —
and it is the first data to stress the toolkit's time-series surface:
``train_test_split_by_time`` gains its second consumer, and
``cross_validate_by_time`` and ``fit_baseline``'s ``"naive_last"`` /
``"seasonal_naive"`` strategies their first.

The pipeline runs the full lifecycle on ``ds`` + scikit-learn alone: fetch →
validate → explore → build the time axis and select the datetime features
that apply (month effects + the library's elapsed-months trend) →
chronological split → fit the one-step transform plan on the training window
(``ds.pipeline.fit_pipeline``) → cross-validate with rolling-origin folds →
persist the scoring pipeline and the fitted model → score the held-out window
from the reloaded model → evaluate against the naive-last and seasonal-naive
baselines → visualize with ``ds.viz.plot_series``. Friction it surfaced in
the library was recorded in ``ROADMAP_ARCHIVE.md`` — that regenerated backlog was as
much the deliverable as the forecast, and this pipeline now consumes the
helpers it demanded (items 10–12).

Run it with::

    uv run ds run flights

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
from sklearn.linear_model import LinearRegression

from ds import Settings, get_logger, get_settings, seed_everything
from ds.eda import summarize
from ds.evaluation import compare_models, cross_validate_by_time, regression_metrics
from ds.features import add_datetime_features, fit_one_hot_categories
from ds.io import load_raw, save_params, save_processed
from ds.modeling.baseline import fit_baseline
from ds.modeling.persistence import load_model, save_model
from ds.modeling.tabular import split_features_target
from ds.modeling.timeseries import train_test_split_by_time
from ds.pipeline import FitStep, fit_pipeline
from ds.preprocessing import standardize_column_names
from ds.validation import (
    assert_in_range,
    assert_in_set,
    assert_no_nulls,
    assert_unique,
    check_schema,
    require_columns,
)
from ds.viz import plot_model_comparison, plot_residuals, plot_series, set_theme

logger = get_logger(__name__)

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv"
RAW_NAME = "flights.csv"

_MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_TARGET = "passengers"
SEASON_LENGTH = 12  # monthly data with a yearly cycle


def fetch_raw(settings: Settings) -> Path:
    """Download the flights CSV into ``settings.raw_dir`` if not already there.

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


def build_time_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble a ``date`` column from the ``year`` + month-name columns.

    The raw file carries the time axis in two pieces (an integer ``year`` and
    a spelled-out ``month``); everything temporal downstream — the
    chronological split, the rolling-origin folds, the calendar features —
    needs one sortable datetime column. Each month is stamped to its first
    day. The assembled axis must be unique: a duplicated (year, month) pair
    would mean corrupted input, and sorting would silently interleave the
    duplicates — the ``assert_unique`` guard (ROADMAP_ARCHIVE.md item 24; this and
    air_quality are its two consumers) that raw ``to_datetime`` does not do.

    Args:
        df: Frame with ``year`` (int) and ``month`` (full English month name)
            columns.

    Returns:
        A new frame with a ``date`` datetime column, sorted chronologically.

    Raises:
        DataValidationError: If any (year, month) pair occurs more than once.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(
        out["year"].astype(str) + "-" + out["month"].astype(str), format="%Y-%B"
    )
    assert_unique(out, "date")
    return out.sort_values("date").reset_index(drop=True)


def engineer_trend_and_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add the trend term and keep only the datetime signal that applies here.

    On a monthly series most of the full calendar set is constant or noise
    (``date_dayofweek``/``date_is_weekend`` would carry the weekday the 1st
    of each month lands on), so the expansion is scoped to exactly what the
    model uses instead of hand-pruned afterwards: the trend enters as the
    library's ``date_elapsed_months`` — the monotone, stateless
    months-since-a-fixed-epoch counter that replaced this project's
    hand-rolled ``month_index``. The seasonal shape rides on the raw
    ``month`` name column into the one-hot step (a *numeric* month would
    wrongly order December next to nothing), and the raw ``year`` column is
    dropped as collinear with the trend.

    Args:
        df: Frame with ``date``, ``year`` and ``month`` columns.

    Returns:
        A new frame with ``date_elapsed_months`` added and the raw ``year``
        removed.
    """
    out = add_datetime_features(df, "date", features=["elapsed_months"])
    return out.drop(columns=["year"])


def run(output_dir: Path, settings: Settings | None = None) -> dict[str, float]:
    """Run the passenger-forecasting pipeline end to end.

    Args:
        output_dir: Directory to write artifacts (figures, reports) into.
        settings: Data-directory configuration; defaults to the shared one.

    Returns:
        Regression metrics on the chronologically held-out window, plus
        ``seasonal_naive_``- and ``naive_last_``-prefixed counterparts from
        the two naive references the model must beat.
    """
    seed_everything()
    settings = settings or get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire — one idempotent download into the git-ignored data tree.
    fetch_raw(settings)
    df = load_raw(RAW_NAME, settings=settings)

    # 2. Validate at the boundary: column presence, parseable dtypes, no
    # gaps (this series is complete), a known month vocabulary, and strictly
    # positive counts.
    df = standardize_column_names(df)
    require_columns(df, ["year", "month", _TARGET])
    df = check_schema(df, {"year": "int64", _TARGET: "int64"}, coerce=True)
    assert_no_nulls(df)
    assert_in_set(df, "month", _MONTH_NAMES)
    assert_in_range(df, _TARGET, min_value=1)

    # 3. Time axis — the two-piece year + month-name axis becomes one sorted
    # datetime column (hand-rolled; nothing in the library assembles a
    # datetime from parts).
    df = build_time_axis(df)

    # 4. Explore — persist the profile the modeling choices below rest on:
    # the summary, the seasonal shape (mean passengers per calendar month),
    # and the series itself (ds.viz.plot_series — the plot this project's
    # friction item 10 demanded).
    summarize(df).to_csv(output_dir / "summary.csv")
    seasonal_profile = df.groupby("month")[_TARGET].mean().reindex(_MONTH_NAMES)
    seasonal_profile.to_csv(output_dir / "seasonality.csv")
    set_theme("notebook")
    fig, ax = plt.subplots()
    plot_series(df["date"], df[_TARGET], ax=ax)
    ax.set_xlabel("month")
    ax.set_ylabel("passengers (thousands)")
    ax.set_title("International airline passengers, 1949-1960")
    fig.savefig(output_dir / "series.png", bbox_inches="tight")
    plt.close(fig)

    # 5. Stateless features — nothing here learns statistics, so it is safe
    # before the split. The scoped selection emits only what the model uses,
    # so there is nothing left for drop_constant_columns to catch.
    df = engineer_trend_and_calendar(df)

    # 6. Chronological split — the held-out window (the last ~29 months,
    # 1958-08 through 1960-12) is strictly in the training data's future,
    # the only valid protocol for forecasting evaluation.
    train, test = train_test_split_by_time(df, "date", test_size=0.2)

    # 7. Fit on train, apply to both. The plan has exactly one fit-based
    # step: the month one-hot vocabulary. The series needs no imputation
    # (complete), no outlier clipping (the "outliers" are the seasonal peaks
    # the model must learn) and no scaling (plain OLS is scale-free) — a
    # scope finding about fit_pipeline recorded in ROADMAP_ARCHIVE.md, not a gap.
    plan = [
        FitStep("one_hot_encode", lambda df: fit_one_hot_categories(df, columns=["month"])),
    ]
    scoring = fit_pipeline(train, plan)

    # 8. Cross-validate on the training window with rolling-origin folds —
    # cross_validate_by_time's first real consumer; every fold's test block
    # is strictly in its training blocks' future. The frame is transformed
    # once up front: cross_validate_by_time has no make_pipeline, but here
    # per-fold re-fitting would be a no-op anyway — the only fitted state is
    # the month vocabulary, and every fold's training window spans more than
    # a year, so each would re-learn the identical 12 calendar months
    # (measured; see ROADMAP_ARCHIVE.md).
    cv_scores = cross_validate_by_time(
        scoring.apply(train),
        time_column="date",
        target=_TARGET,
        make_model=lambda: LinearRegression(),
    )
    cv_scores.to_csv(output_dir / "cv_folds.csv")
    logger.info("Rolling-origin CV mae: %.1f (+/- %.1f)", *_mean_std(cv_scores["mae"]))

    train = scoring.apply(train)
    test = scoring.apply(test)
    assert_no_nulls(train)
    assert_no_nulls(test)

    # 9. Persist the processed data and the whole scoring pipeline.
    save_processed(pd.concat([train, test]), "flights_features.parquet", settings=settings)
    params_dir = settings.processed_dir / "params"
    save_params(scoring, params_dir / "flights_scoring.json")

    # 10. Model — a linear trend + month effects, fitted once, persisted next
    # to the scoring pipeline, and the held-out window scored from the
    # *reloaded* copy, proving a later run needs only the files on disk.
    # ``date`` orders the rows but is not a feature.
    x_train, y_train = split_features_target(train.drop(columns=["date"]), _TARGET)
    x_test, y_test = split_features_target(test.drop(columns=["date"]), _TARGET)
    save_model(LinearRegression().fit(x_train, y_train), params_dir / "flights_model.joblib")
    model = load_model(params_dir / "flights_model.joblib")
    preds = model.predict(x_test)

    # 11. Evaluate against the two naive references (fit_baseline's first
    # real consumers). Both are fitted on the chronologically ordered
    # training target; because the held-out window starts right after the
    # training window ends, seasonal_naive's cycle stays aligned with the
    # test months (prediction i repeats the value 12 months before it).
    naive_last = fit_baseline(y_train, strategy="naive_last")
    seasonal = fit_baseline(y_train, strategy="seasonal_naive", season_length=SEASON_LENGTH)
    seasonal_preds = seasonal.predict(len(y_test))
    naive_last_preds = naive_last.predict(len(y_test))
    linear_preds = [float(value) for value in preds]
    comparison = compare_models(
        y_test.tolist(),
        {
            "linear_regression": linear_preds,
            "seasonal_naive": seasonal_preds,
            "naive_last": naive_last_preds,
        },
    )
    comparison.to_csv(output_dir / "model_comparison.csv")
    metrics = regression_metrics(y_test.tolist(), linear_preds)
    seasonal_scores = regression_metrics(y_test.tolist(), seasonal_preds)
    metrics.update({f"seasonal_naive_{name}": value for name, value in seasonal_scores.items()})
    naive_scores = regression_metrics(y_test.tolist(), naive_last_preds)
    metrics.update({f"naive_last_{name}": value for name, value in naive_scores.items()})
    logger.info("Held-out metrics vs baselines: %s", metrics)

    # 12. Visualize. The forecast-vs-actual plot composes two plot_series
    # calls on one Axes: the training tail first, then the held-out window
    # with the model and seasonal-naive forecasts overlaid.
    fig2, ax2 = plt.subplots()
    plot_residuals(y_test.tolist(), linear_preds, ax=ax2)
    ax2.set_title("Residuals vs predicted - held-out window")
    fig2.savefig(output_dir / "residuals.png", bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    plot_model_comparison(comparison, metric="mae", ax=ax3)
    fig3.savefig(output_dir / "model_comparison.png", bbox_inches="tight")
    plt.close(fig3)

    fig4, ax4 = plt.subplots()
    history = train.iloc[-2 * SEASON_LENGTH :]
    plot_series(history["date"], history[_TARGET], label="history", ax=ax4)
    plot_series(
        test["date"],
        y_test,
        predictions={"linear regression": linear_preds, "seasonal naive": seasonal_preds},
        label="actual",
        ax=ax4,
    )
    ax4.set_xlabel("month")
    ax4.set_ylabel("passengers (thousands)")
    ax4.set_title("Held-out window - forecasts vs actual")
    fig4.savefig(output_dir / "forecast.png", bbox_inches="tight")
    plt.close(fig4)

    return metrics


def _mean_std(values: pd.Series) -> tuple[float, float]:
    """Mean and standard deviation of a numeric series, for log lines."""
    return float(values.mean()), float(values.std())


def main() -> None:
    settings = get_settings()
    metrics = run(settings.processed_dir / "flights", settings=settings)
    print("Pipeline finished. Held-out metrics (vs seasonal-naive and naive-last):")
    for name, value in metrics.items():
        print(f"  {name:>20}: {value:,.3f}")


if __name__ == "__main__":
    main()
