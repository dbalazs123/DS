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
the library is recorded in ``ROADMAP.md``; the zone columns, originally dropped
for their cardinality, are consumed via ``fit_topk_categories`` (friction
item 4, since promoted into the library) and evaluated against a boroughs-only
variant to check they earn their place.

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
from ds.evaluation import compare_models, regression_metrics
from ds.features import (
    add_datetime_features,
    fit_one_hot_categories,
    fit_scale_params,
    fit_topk_categories,
)
from ds.io import load_raw, save_params, save_processed
from ds.modeling.baseline import fit_baseline
from ds.modeling.persistence import load_model, save_model
from ds.modeling.tabular import split_features_target
from ds.modeling.timeseries import train_test_split_by_time
from ds.pipeline import FitStep, fit_pipeline
from ds.preprocessing import (
    drop_duplicate_rows,
    fit_impute_values,
    fit_outlier_bounds,
    standardize_column_names,
)
from ds.validation import assert_no_nulls, check_schema, require_columns
from ds.viz import plot_missingness, plot_model_comparison, plot_residuals, set_theme

logger = get_logger(__name__)

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"
RAW_NAME = "taxis.csv"

# tip/tolls/total are post-ride outcomes (total *contains* the fare), so using
# them to predict the fare would be leakage.
_DROPPED = ["tip", "tolls", "total"]
_CATEGORICAL = ["color", "payment", "pickup_borough", "dropoff_borough"]
# The zone columns carry ~200 distinct values each — too many levels to one-hot
# directly and with no meaningful order to ordinal-encode (friction item 4).
# fit_topk_categories collapses each to its most frequent levels + "other",
# after which the ordinary one-hot machinery handles them.
_ZONES = ["pickup_zone", "dropoff_zone"]
_ZONE_TOP_K = 15
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

    ``ds.features.add_datetime_features`` covers the calendar features —
    including ``pickup_hour``, originally hand-rolled here and since promoted
    into the library (friction item 2). Trip duration in minutes remains a
    domain feature built from the two timestamps.

    Args:
        df: Frame with ``pickup``/``dropoff`` datetime columns.

    Returns:
        A new frame with the calendar features and ``duration_min`` added and
        the raw ``dropoff`` timestamp dropped (it is fully determined by
        pickup + duration).
    """
    out = add_datetime_features(df, "pickup", drop=False)
    out["duration_min"] = (out["dropoff"] - out["pickup"]).dt.total_seconds() / 60.0
    return out.drop(columns=["dropoff"])


def run(output_dir: Path, settings: Settings | None = None) -> dict[str, float]:
    """Run the fare-prediction pipeline end to end.

    Args:
        output_dir: Directory to write artifacts (figures, reports) into.
        settings: Data-directory configuration; defaults to the shared one.

    Returns:
        Regression metrics on the held-out (chronologically last) window,
        plus ``boroughs_only_``-prefixed counterparts from a variant without
        the top-k zone indicators and ``baseline_``-prefixed ones from a
        predict-the-train-mean reference model.
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
    require_columns(df, ["pickup", "dropoff", _TARGET, *_CATEGORICAL, *_ZONES])
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

    # 6. Fit on train, apply to both. fit_pipeline runs the ordered
    # fit → apply → fit chain (each parameter set fitted on the train frame
    # as transformed by the steps before it) that this project used to
    # hand-string — friction item 5.
    plan = [
        FitStep(
            "clip_outliers",
            lambda df: fit_outlier_bounds(df, columns=["distance", "duration_min"]),
        ),
        FitStep(
            "collapse_categories",
            lambda df: fit_topk_categories(df, columns=_ZONES, k=_ZONE_TOP_K),
        ),
        FitStep(
            "impute_missing",
            lambda df: fit_impute_values(
                df, columns=_CATEGORICAL + _ZONES, strategy="most_frequent"
            ),
        ),
        FitStep(
            "one_hot_encode",
            lambda df: fit_one_hot_categories(df, columns=_CATEGORICAL + _ZONES),
        ),
        FitStep("scale_features", lambda df: fit_scale_params(df, columns=_NUMERIC_FEATURES)),
    ]
    scoring = fit_pipeline(train, plan)
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

    # 9. Evaluate — side by side with the naive train-mean baseline
    # (ds.modeling.baseline replaced the hand-rolled version this project
    # originally needed — friction item 3, since promoted into the library)
    # and with a boroughs-only variant, to check whether the top-k zone
    # indicators actually earn their place over the coarser borough columns.
    zone_columns = [
        col for col in x_train.columns if col.startswith(("pickup_zone_", "dropoff_zone_"))
    ]
    borough_model = LinearRegression().fit(x_train.drop(columns=zone_columns), y_train)
    borough_preds = borough_model.predict(x_test.drop(columns=zone_columns))
    baseline = fit_baseline(y_train, strategy="mean")
    comparison = compare_models(
        y_test.tolist(),
        {
            "linear_regression": [float(value) for value in preds],
            "boroughs_only": [float(value) for value in borough_preds],
            "train_mean_baseline": baseline.predict(len(y_test)),
        },
    )
    comparison.to_csv(output_dir / "model_comparison.csv")
    metrics = regression_metrics(y_test.tolist(), preds.tolist())
    borough_scores = regression_metrics(y_test.tolist(), borough_preds.tolist())
    metrics.update({f"boroughs_only_{name}": value for name, value in borough_scores.items()})
    baseline_scores = regression_metrics(y_test.tolist(), baseline.predict(len(y_test)))
    metrics.update({f"baseline_{name}": value for name, value in baseline_scores.items()})
    logger.info("Held-out metrics vs baseline: %s", metrics)

    # 10. Visualize.
    fig2, ax2 = plt.subplots()
    plot_residuals(y_test.tolist(), preds.tolist(), ax=ax2)
    ax2.set_title("Fare model residuals — held-out window")
    fig2.savefig(output_dir / "residuals.png", bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    plot_model_comparison(comparison, metric="mae", ax=ax3)
    fig3.savefig(output_dir / "model_comparison.png", bbox_inches="tight")
    plt.close(fig3)

    return metrics


def main() -> None:
    settings = get_settings()
    metrics = run(settings.processed_dir / "nyc_taxis", settings=settings)
    print("Pipeline finished. Held-out metrics (vs naive train-mean baseline):")
    for name, value in metrics.items():
        print(f"  {name:>14}: {value:,.3f}")


if __name__ == "__main__":
    main()
