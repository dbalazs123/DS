"""NYC Taxis — predict a ride's fare from real trip records.

The first project on *real* data: 6,433 NYC yellow- and green-cab rides from
March 2019 (the seaborn ``taxis`` dataset, sampled from the NYC TLC trip
records). Unlike ``projects/_example``, nothing here was generated to fit the
toolkit — the data brings its own quirks: genuinely missing payment types and
boroughs, ~200-level pickup/dropoff zone columns, and fares whose strongest
temporal signal is the hour of day.

The pipeline runs the full lifecycle on ``ds`` + scikit-learn alone: fetch →
validate → explore → clean → chronological split → fit-on-train/apply-to-both →
persist the scoring pipeline and the fitted model → score from the reloaded
model → evaluate against a naive baseline → visualize. Friction it surfaced in
the library is recorded in ``ROADMAP.md``.

Run it with::

    uv run ds run nyc_taxis

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
from ds.eda import missing_value_report, summarize
from ds.evaluation import regression_metrics
from ds.features import add_datetime_features, fit_one_hot_categories, fit_scale_params
from ds.io import load_raw, save_params, save_processed
from ds.modeling.persistence import load_model, save_model
from ds.modeling.tabular import split_features_target
from ds.modeling.timeseries import train_test_split_by_time
from ds.pipeline import Pipeline, PipelineStep
from ds.preprocessing import (
    apply_clip_outliers,
    drop_duplicate_rows,
    fit_impute_values,
    fit_outlier_bounds,
    standardize_column_names,
)
from ds.validation import assert_no_nulls, check_schema, require_columns
from ds.viz import plot_missingness, plot_residuals, set_theme

logger = get_logger(__name__)

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"
RAW_NAME = "taxis.csv"

# The zone columns carry ~200 distinct values each — too many levels to one-hot
# and with no meaningful order to ordinal-encode — so the model falls back to
# the borough columns. tip/tolls/total are post-ride outcomes (total *contains*
# the fare), so using them to predict the fare would be leakage.
_DROPPED = ["pickup_zone", "dropoff_zone", "tip", "tolls", "total"]
_CATEGORICAL = ["color", "payment", "pickup_borough", "dropoff_borough"]
_NUMERIC_FEATURES = ["distance", "duration_min", "passengers", "pickup_hour"]
_TARGET = "fare"


def fetch_raw(settings: Settings) -> Path:
    """Download the taxis CSV into ``settings.raw_dir`` if not already there.

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


def engineer_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the project-specific features the generic helpers don't cover.

    ``ds.features.add_datetime_features`` expands year/month/day/dayofweek but
    not hour — and hour of day is the strongest temporal signal in taxi fares
    (night surcharges, rush hours) — so it is derived here. Trip duration in
    minutes is likewise a domain feature built from the two timestamps.

    Args:
        df: Frame with ``pickup``/``dropoff`` datetime columns.

    Returns:
        A new frame with the calendar features, ``pickup_hour`` and
        ``duration_min`` added and the raw ``dropoff`` timestamp dropped
        (it is fully determined by pickup + duration).
    """
    out = add_datetime_features(df, "pickup", drop=False)
    out["pickup_hour"] = out["pickup"].dt.hour
    out["duration_min"] = (out["dropoff"] - out["pickup"]).dt.total_seconds() / 60.0
    return out.drop(columns=["dropoff"])


def run(output_dir: Path, settings: Settings | None = None) -> dict[str, float]:
    """Run the fare-prediction pipeline end to end.

    Args:
        output_dir: Directory to write artifacts (figures, reports) into.
        settings: Data-directory configuration; defaults to the shared one.

    Returns:
        Regression metrics on the held-out (chronologically last) window,
        plus ``baseline_``-prefixed counterparts from a predict-the-train-mean
        reference model.
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
    require_columns(df, ["pickup", "dropoff", _TARGET, *_CATEGORICAL])
    df = check_schema(
        df,
        {
            "pickup": "datetime64[ns]",
            "dropoff": "datetime64[ns]",
            "passengers": "int64",
            "distance": "float64",
            _TARGET: "float64",
        },
        coerce=True,
    )

    # 3. Explore — persist the profile the modeling choices below rest on.
    summarize(df).to_csv(output_dir / "summary.csv")
    missing_value_report(df).to_csv(output_dir / "missing.csv")
    set_theme("notebook")
    fig, ax = plt.subplots()
    plot_missingness(df, ax=ax)
    ax.set_title("Missing values — real, not synthetic")
    fig.savefig(output_dir / "missingness.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Structural clean and stateless features — nothing here learns
    # statistics, so it is safe before the split.
    df = drop_duplicate_rows(df)
    df = df.drop(columns=_DROPPED)
    df = engineer_trip_features(df)

    # 5. Chronological split before anything fit-based: the held-out window is
    # strictly later in March than every training ride.
    train, test = train_test_split_by_time(df, "pickup")
    train = train.drop(columns=["pickup"])
    test = test.drop(columns=["pickup"])

    # 6. Fit on train, apply to both. Each parameter set is fitted on the
    # train frame as transformed by the steps before it, then the whole
    # ordered chain is applied through one Pipeline.
    bounds = fit_outlier_bounds(train, columns=["distance", "duration_min"])
    clipped = apply_clip_outliers(train, bounds)
    fills = fit_impute_values(clipped, columns=_CATEGORICAL, strategy="most_frequent")
    vocabulary = fit_one_hot_categories(clipped, columns=_CATEGORICAL)
    scaling = fit_scale_params(clipped, columns=_NUMERIC_FEATURES)

    scoring = Pipeline(
        steps=(
            PipelineStep("clip_outliers", bounds),
            PipelineStep("impute_missing", fills),
            PipelineStep("one_hot_encode", vocabulary),
            PipelineStep("scale_features", scaling),
        )
    )
    train = scoring.apply(train)
    test = scoring.apply(test)
    assert_no_nulls(train)
    assert_no_nulls(test)

    # 7. Persist the processed data and the whole scoring pipeline.
    save_processed(pd.concat([train, test]), "taxis_features.parquet", settings=settings)
    params_dir = settings.processed_dir / "params"
    save_params(scoring, params_dir / "taxis_scoring.json")

    # 8. Model — fitted once, persisted next to the scoring pipeline, and the
    # held-out window scored from the *reloaded* copy, proving a later run
    # needs only the files on disk (closing this project's friction item 1).
    x_train, y_train = split_features_target(train, _TARGET)
    x_test, y_test = split_features_target(test, _TARGET)
    save_model(LinearRegression().fit(x_train, y_train), params_dir / "taxis_model.joblib")
    model = load_model(params_dir / "taxis_model.joblib")
    preds = model.predict(x_test)

    # 9. Evaluate — against a naive predict-the-train-mean baseline, built by
    # hand because ds.modeling has no baseline estimators yet (ROADMAP.md).
    metrics = regression_metrics(y_test.tolist(), preds.tolist())
    baseline_preds = [float(y_train.mean())] * len(y_test)
    baseline = regression_metrics(y_test.tolist(), baseline_preds)
    metrics.update({f"baseline_{name}": value for name, value in baseline.items()})
    logger.info("Held-out metrics vs baseline: %s", metrics)

    # 10. Visualize.
    fig2, ax2 = plt.subplots()
    plot_residuals(y_test.tolist(), preds.tolist(), ax=ax2)
    ax2.set_title("Fare model residuals — held-out window")
    fig2.savefig(output_dir / "residuals.png", bbox_inches="tight")
    plt.close(fig2)

    return metrics


def main() -> None:
    settings = get_settings()
    metrics = run(settings.processed_dir / "nyc_taxis", settings=settings)
    print("Pipeline finished. Held-out metrics (vs naive train-mean baseline):")
    for name, value in metrics.items():
        print(f"  {name:>14}: {value:,.3f}")


if __name__ == "__main__":
    main()
