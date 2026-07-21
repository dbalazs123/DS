"""Air Quality — reconstruct the reference CO measurement from the rest of the station.

The sixth project on *real* data: the UCI Air Quality dataset — 9,357 hourly
rows from a multi-sensor device co-located with certified reference analyzers
at road level in an Italian city, March 2004 to April 2005. The reference CO
analyzer was down for 18% of those hours, and that gap is the task: predict
its reading (``co_gt``, mg/m³) from what the rest of the station saw that
hour — the five metal-oxide sensor channels, temperature/humidity, and the
NOx/NO2 reference channels — so the held-out evaluation mimics back-filling
an instrument outage from its neighbours.

The dataset was chosen by the established rule — grep which library surfaces
still have no real consumer — and it earns its keep on exactly those:
missingness with real teeth at last (−200 sentinels: one column 90% missing,
the target 18%, the feature channels partially and *independently* gapped),
first real consumers for ``assert_dtypes`` (the raw file is
semicolon-separated with decimal commas — a half-right read silently parses
every measurement as strings), and a rolling-origin cross-validation whose
per-fold fitted state genuinely varies (the parked item-9 trigger). Friction
it surfaced in the library is recorded in ``ROADMAP_ARCHIVE.md`` — the regenerated
backlog is as much the deliverable as the model.

The pipeline runs the full lifecycle on ``ds`` + scikit-learn alone: fetch →
trim the raw file's junk and pin the parse → sentinels to NaN → the
missing-value triage (drop a column, drop offline rows, drop unlabeled rows,
impute the rest) → hand-assembled hourly time axis → calendar features →
chronological split → three-step fit plan (impute / hour one-hot / scale)
fitted on the training window → rolling-origin CV → pipeline + model
persisted and the held-out window scored from reloaded state → evaluated
against the train-mean and same-hour-yesterday references → visualized.

Run it with::

    uv run ds run air_quality

The raw CSV is downloaded once into ``<data_dir>/raw/`` (git-ignored) and
reused on later runs. The UCI archive itself is not reachable from every
network, so the fetch pins two byte-identical GitHub mirrors of the original
file and verifies its sha256 before trusting either.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ds import Settings, get_logger, get_settings, seed_everything
from ds.eda import missing_value_report, summarize, top_correlations
from ds.evaluation import compare_models, cross_validate_by_time, regression_metrics
from ds.features import add_datetime_features, fit_one_hot_categories, fit_scale_params
from ds.io import fetch_dataset, load_raw, save_params, save_processed
from ds.modeling.baseline import fit_baseline
from ds.modeling.persistence import load_model, save_model
from ds.modeling.tabular import split_features_target
from ds.modeling.timeseries import train_test_split_by_time
from ds.pipeline import FitStep, fit_pipeline
from ds.preprocessing import fit_impute_values, standardize_column_names
from ds.validation import (
    assert_dtypes,
    assert_in_range,
    assert_no_nulls,
    assert_row_count,
    assert_unique,
    require_columns,
)
from ds.viz import (
    plot_missingness,
    plot_model_comparison,
    plot_residuals,
    plot_series,
    set_theme,
)

logger = get_logger(__name__)

# The UCI archive (dataset 360) is not reachable from every network, so the
# fetch uses byte-identical GitHub mirrors of the original AirQualityUCI.csv
# — same semicolons, decimal commas, CRLF line endings and trailing junk —
# and ds.io.fetch_dataset verifies the checksum below before trusting either.
DATA_URLS = (
    "https://raw.githubusercontent.com/shrikumarp/airquality/master/AirQualityUCI.csv",
    "https://raw.githubusercontent.com/asharvi1/UCI-Air-Quality-Data/master/AirQualityUCI.csv",
)
RAW_NAME = "AirQualityUCI.csv"
RAW_SHA256 = "13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996"

# The published size of the dataset: 9,357 hourly rows. The raw file carries
# 114 additional all-empty trailing rows (plus two empty trailing columns)
# that a half-right read silently keeps.
EXPECTED_ROWS = 9357

_TARGET = "co_gt"

# Every measurement column, post-standardize_column_names, in file order.
# −200 is the file's missing-value sentinel in all of them.
MEASUREMENT_COLUMNS = [
    "co_gt",
    "pt08_s1_co",
    "nmhc_gt",
    "c6h6_gt",
    "pt08_s2_nmhc",
    "nox_gt",
    "pt08_s3_nox",
    "no2_gt",
    "pt08_s4_no2",
    "pt08_s5_o3",
    "t",
    "rh",
    "ah",
]

# What the multi-sensor device itself records each hour. These are missing
# all-or-nothing: an offline hour blanks every one of them at once.
DEVICE_COLUMNS = [
    "pt08_s1_co",
    "pt08_s2_nmhc",
    "pt08_s3_nox",
    "pt08_s4_no2",
    "pt08_s5_o3",
    "t",
    "rh",
    "ah",
]

# The model's measurement features: the device channels plus the NOx/NO2
# reference analyzers, whose outages are independent of both the device's and
# the CO analyzer's — the genuinely cell-level gaps the impute step serves.
# Excluded: nmhc_gt (90% missing — dropped whole) and c6h6_gt (published as a
# transform of pt08_s2_nmhc; the correlation report shows the near-identity).
FEATURE_COLUMNS = DEVICE_COLUMNS + ["nox_gt", "no2_gt"]

HOURS_PER_DAY = 24


def fetch_raw(settings: Settings) -> Path:
    """Download the air-quality CSV into ``settings.raw_dir`` and verify its checksum.

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


def trim_raw(df: pd.DataFrame, expected_rows: int = EXPECTED_ROWS) -> pd.DataFrame:
    """Cut the raw file's structural junk and check what remains is the dataset.

    The original file ends every line with two empty fields (read as
    ``Unnamed:`` columns) and carries 114 all-empty trailing rows. Both are
    dropped, and the surviving row count is checked against the published size
    with ``assert_row_count`` — the second time in this workspace a boundary
    file has needed an expected-shape check to make a silently-wrong parse
    loud (the trap here: a plausible-looking frame that is 114 rows of NaN too
    long), which earned that guard its place (ROADMAP_ARCHIVE.md item 25).

    Args:
        df: The frame as read from the raw file with the correct separator.
        expected_rows: Rows the trimmed frame must have.

    Returns:
        A new frame without the junk columns and empty rows.

    Raises:
        DataValidationError: If the trimmed frame does not have
            ``expected_rows`` rows.
    """
    out = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    out = out.dropna(how="all").reset_index(drop=True)
    return assert_row_count(out, expected_rows)


def mask_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the file's −200 missing-value sentinels into real NaN.

    The dataset documents −200 as its missing tag in every measurement
    column. Left in place it is a legal-looking float that would silently
    poison every statistic downstream; as NaN the gaps become visible to
    ``missing_value_report`` and the impute step. (Hand-rolled: the library
    has no sentinel-to-NaN helper — recorded in the friction backlog.)

    Args:
        df: Frame with the measurement columns present.

    Returns:
        A new frame with sentinels replaced by NaN.
    """
    out = df.copy()
    out[MEASUREMENT_COLUMNS] = out[MEASUREMENT_COLUMNS].replace(-200.0, np.nan)
    return out


def build_time_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble one ``timestamp`` column from the ``date`` + ``time`` strings.

    The raw file carries the time axis in two pieces — ``10/03/2004`` (day
    first) and ``18.00.00`` (dot-separated) — and everything temporal
    downstream needs one sortable datetime column. The assembled axis must be
    unique: a duplicated hour would mean corrupted input, and sorting would
    silently interleave the duplicates — the guard raw ``to_datetime`` does
    not do, now ``assert_unique`` (ROADMAP_ARCHIVE.md item 24, the second project to
    hand-assemble a time axis; the concatenation and format stay project-local).

    Args:
        df: Frame with ``date`` and ``time`` string columns.

    Returns:
        A new frame with a ``timestamp`` column replacing ``date``/``time``,
        sorted chronologically.

    Raises:
        DataValidationError: If any timestamp occurs more than once.
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["date"] + " " + out["time"], format="%d/%m/%Y %H.%M.%S")
    assert_unique(out, "timestamp")
    out = out.drop(columns=["date", "time"])
    return out.sort_values("timestamp").reset_index(drop=True)


def drop_offline_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the hours where the multi-sensor device recorded nothing at all.

    The device's gaps are all-or-nothing: an offline hour blanks every one of
    its eight channels at once, so there is nothing left to predict *from* —
    imputing an entire feature row out of training medians would manufacture
    data. Rows where at least one device channel reports stay, and the impute
    step fills only those genuine partial gaps. (Hand-rolled row mask;
    recorded against ROADMAP_ARCHIVE.md item 15's trigger.)

    Args:
        df: Frame with the device columns present.

    Returns:
        A new frame without the all-channels-missing rows.
    """
    offline = df[DEVICE_COLUMNS].isna().all(axis=1)
    if offline.any():
        logger.info("Dropping %d device-offline hours (all channels missing)", offline.sum())
    return df.loc[~offline].reset_index(drop=True)


def same_hour_yesterday_reference(
    frame: pd.DataFrame, timestamps: pd.Series, fallback: float
) -> list[float]:
    """The station's strongest cheap reference: yesterday's reading, same hour.

    For each requested timestamp, look up the actual ``co_gt`` measured 24
    hours earlier anywhere in the labeled data; where the analyzer was also
    down then, fall back to ``fallback`` (the training mean). This is the
    honest stand-in for ``fit_baseline``'s ``"seasonal_naive"``: that
    strategy aligns *positionally*, and on this axis — hourly with the
    unlabeled rows removed — position ``i - 24`` is usually not the same hour
    yesterday, so the positional baseline would be scored on a
    misalignment. (Recorded in the friction backlog.)

    Args:
        frame: Labeled rows with ``timestamp`` and ``co_gt`` columns.
        timestamps: The timestamps to produce reference predictions for.
        fallback: Value used where no reading exists 24 hours earlier.

    Returns:
        One reference prediction per requested timestamp.
    """
    by_time = frame.set_index("timestamp")[_TARGET]
    lagged = timestamps - pd.Timedelta(hours=HOURS_PER_DAY)
    return [float(by_time.get(when, fallback)) for when in lagged]


def run(output_dir: Path, settings: Settings | None = None) -> dict[str, float]:
    """Run the CO-reconstruction pipeline end to end.

    Args:
        output_dir: Directory to write artifacts (figures, reports) into.
        settings: Paths to read/write data under; resolved from the
            environment when omitted. Tests pass a temporary one so a run
            never touches the shared data tree.

    Returns:
        Regression metrics on the chronologically held-out window, plus
        ``yesterday_``- and ``train_mean_``-prefixed counterparts from the
        two references the model must beat.
    """
    seed_everything()
    settings = settings or get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire — one idempotent, checksum-verified download into the
    # git-ignored data tree, then the one correct read: semicolon-separated
    # with decimal commas. (sep=";" alone parses every decimal-comma column
    # as strings; the dtype pin below is what makes that mistake loud.)
    fetch_raw(settings)
    df = load_raw(RAW_NAME, settings=settings, sep=";", decimal=",")

    # 2. Trim the file's structural junk and validate the boundary: the
    # published row count, the column set, and — the parse pin — every
    # measurement column already numeric.
    df = trim_raw(df)
    df = standardize_column_names(df)
    require_columns(df, ["date", "time", *MEASUREMENT_COLUMNS])
    assert_dtypes(df, dict.fromkeys(MEASUREMENT_COLUMNS, "float64"))

    # 3. Sentinels to NaN, then range-check the physical claims that survive:
    # sensor resistances are positive readings, relative humidity is a
    # percentage, absolute humidity is positive. (With the sentinels still in
    # place every one of these checks would be unsatisfiable.)
    df = mask_sentinels(df)
    for column in ("pt08_s1_co", "pt08_s2_nmhc", "pt08_s3_nox", "pt08_s4_no2", "pt08_s5_o3"):
        assert_in_range(df, column, min_value=0, inclusive="neither")
    assert_in_range(df, "rh", min_value=0, max_value=100)
    assert_in_range(df, "ah", min_value=0, inclusive="neither")
    assert_in_range(df, _TARGET, min_value=0)

    # 4. Time axis — the two-piece date + dotted-time axis becomes one sorted
    # datetime column (hand-rolled; nothing in the library assembles a
    # datetime from parts — the second project to do this dance).
    df = build_time_axis(df)

    # 5. Explore — persist the evidence the triage below rests on: the
    # missing-value report (one column 90% gone, the target 18%, the feature
    # channels ~4–6%), its plot, the summary, and the correlation pairs
    # (which show c6h6_gt as pt08_s2_nmhc's near-identity, hence excluded).
    set_theme("notebook")
    missing_value_report(df).to_csv(output_dir / "missing_report.csv")
    fig, ax = plt.subplots()
    plot_missingness(df, ax=ax)
    fig.savefig(output_dir / "missingness.png", bbox_inches="tight")
    plt.close(fig)
    summarize(df).to_csv(output_dir / "summary.csv")
    top_correlations(df, n=15).to_csv(output_dir / "top_correlations.csv", index=False)

    # 6. The missingness triage, three different tools for three structures:
    # nmhc_gt is 90% missing (drop the column — nothing to learn from a 10%
    # remnant of the first weeks); c6h6_gt is published as a transform of
    # pt08_s2_nmhc (drop as redundant); device-offline hours have no
    # features at all (drop the rows); unlabeled hours have no target (drop
    # the rows — they are the deployment condition, not training data). The
    # genuine partial gaps that remain are the impute step's job.
    df = df.drop(columns=["nmhc_gt", "c6h6_gt"])
    df = drop_offline_rows(df)
    unlabeled = df[_TARGET].isna()
    logger.info("Dropping %d unlabeled hours (reference CO analyzer down)", unlabeled.sum())
    df = df.loc[~unlabeled].reset_index(drop=True)

    # 7. Stateless calendar features — hour of day carries the traffic cycle
    # (one-hot below: CO's daily shape is bimodal, not linear in the hour),
    # is_weekend the lighter weekend traffic, and elapsed_months the
    # months-long sensor drift this dataset is known for, as a linear trend.
    df = add_datetime_features(df, "timestamp", features=["hour", "is_weekend", "elapsed_months"])

    # 8. Chronological split — the held-out window (the last ~20% of labeled
    # hours) is strictly in the training data's future, matching how a
    # deployed reconstruction would meet its data.
    train, test = train_test_split_by_time(df, "timestamp", test_size=0.2)

    # 9. Fit the three-step plan on the training window only: median fills
    # for the measurement channels (the reference channels carry the real
    # gaps), the 24-level hour vocabulary, and standardization (the channels
    # sit on wildly different scales — sensor resistances in the thousands,
    # absolute humidity around one — and Ridge's penalty is scale-sensitive).
    plan = [
        FitStep(
            "impute_missing",
            lambda frame: fit_impute_values(frame, FEATURE_COLUMNS, strategy="median"),
        ),
        FitStep(
            "one_hot_encode",
            lambda frame: fit_one_hot_categories(frame, columns=["timestamp_hour"]),
        ),
        FitStep(
            "scale_features",
            lambda frame: fit_scale_params(frame, [*FEATURE_COLUMNS, "timestamp_elapsed_months"]),
        ),
    ]
    scoring = fit_pipeline(train, plan)

    # 10. Cross-validate on the training window with rolling-origin folds,
    # re-fitting the transform plan inside each fold. Unlike the earlier
    # projects' single-vocabulary plans, this pipeline's fitted state
    # genuinely varies fold to fold — the impute medians and scale centres are
    # learned from seasons of differing missingness and level (nox_gt's median
    # swings ~28% across the folds) — so make_pipeline (ROADMAP_ARCHIVE.md item 22) is
    # what keeps each fold's statistics on its own past. The raw training
    # frame goes in; each fold fits its own pipeline from the same plan the
    # scoring run uses. This dissolved the hand-rolled fold-boundary
    # reproduction the earlier version needed to measure that drift out-of-band.
    cv_scores = cross_validate_by_time(
        train,
        time_column="timestamp",
        target=_TARGET,
        make_model=lambda: Ridge(alpha=1.0),
        make_pipeline=lambda frame: fit_pipeline(frame, plan),
    )
    cv_scores.to_csv(output_dir / "cv_folds.csv")
    logger.info(
        "Rolling-origin CV mae: %.3f (+/- %.3f)",
        float(cv_scores["mae"].mean()),
        float(cv_scores["mae"].std()),
    )

    # 11. Apply the fitted plan to both windows and persist everything a
    # later scoring run needs: the processed frame, the strict-JSON scoring
    # pipeline, and the model.
    train = scoring.apply(train)
    test = scoring.apply(test)
    assert_no_nulls(train)
    assert_no_nulls(test)
    save_processed(pd.concat([train, test]), "air_quality_features.parquet", settings=settings)
    params_dir = settings.processed_dir / "params"
    save_params(scoring, params_dir / "air_quality_scoring.json")

    x_train, y_train = split_features_target(train.drop(columns=["timestamp"]), _TARGET)
    x_test, y_test = split_features_target(test.drop(columns=["timestamp"]), _TARGET)
    save_model(Ridge(alpha=1.0).fit(x_train, y_train), params_dir / "air_quality_model.joblib")
    model = load_model(params_dir / "air_quality_model.joblib")
    predictions = [float(value) for value in model.predict(x_test)]

    # 12. Evaluate against two references the model must beat: the training
    # mean (fit_baseline — the no-information floor) and the station's own
    # reading 24 hours earlier (hand-rolled time-indexed lookup; the
    # positional seasonal_naive cannot align on this gapped axis).
    train_mean = fit_baseline(y_train, strategy="mean")
    mean_predictions = train_mean.predict(len(y_test))
    yesterday_predictions = same_hour_yesterday_reference(
        pd.concat([train, test])[["timestamp", _TARGET]],
        test["timestamp"],
        fallback=mean_predictions[0],
    )
    comparison = compare_models(
        y_test.tolist(),
        {
            "ridge": predictions,
            "same_hour_yesterday": yesterday_predictions,
            "train_mean": mean_predictions,
        },
    )
    comparison.to_csv(output_dir / "model_comparison.csv")
    metrics = regression_metrics(y_test.tolist(), predictions)
    yesterday_scores = regression_metrics(y_test.tolist(), yesterday_predictions)
    metrics.update({f"yesterday_{name}": value for name, value in yesterday_scores.items()})
    mean_scores = regression_metrics(y_test.tolist(), list(mean_predictions))
    metrics.update({f"train_mean_{name}": value for name, value in mean_scores.items()})
    logger.info("Held-out metrics vs references: %s", metrics)

    # 13. Visualize: the comparison bars, the residual diagnostic, and one
    # readable week of the held-out window — reconstruction vs what the
    # analyzer actually measured.
    fig2, ax2 = plt.subplots()
    plot_model_comparison(comparison, metric="mae", ax=ax2)
    fig2.savefig(output_dir / "model_comparison.png", bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    plot_residuals(y_test.tolist(), predictions, ax=ax3)
    ax3.set_title("Residuals vs predicted - held-out window")
    fig3.savefig(output_dir / "residuals.png", bbox_inches="tight")
    plt.close(fig3)

    week = 7 * HOURS_PER_DAY
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    plot_series(
        test["timestamp"].iloc[:week],
        y_test.iloc[:week],
        predictions={"ridge reconstruction": predictions[:week]},
        label="reference analyzer",
        ax=ax4,
    )
    ax4.set_xlabel("hour")
    ax4.set_ylabel("CO (mg/m^3)")
    ax4.set_title("First held-out week - reconstruction vs reference")
    fig4.savefig(output_dir / "reconstruction_week.png", bbox_inches="tight")
    plt.close(fig4)

    return metrics


def main() -> None:
    settings = get_settings()
    metrics = run(settings.processed_dir / "air_quality", settings=settings)
    print("Pipeline finished. Held-out metrics (vs same-hour-yesterday and train-mean):")
    for name, value in metrics.items():
        print(f"  {name:>20}: {value:,.3f}")


if __name__ == "__main__":
    main()
