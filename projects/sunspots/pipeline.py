"""Sunspots — forecast the monthly sunspot number with an autoregressive model.

The eighth project on *real* data, and the **second forecasting** one — chosen,
after ``flights``, precisely for the kind of series a calendar-feature + naive
approach handles *badly*. The monthly mean sunspot number, 1749–1983 (2,820
months of the Zurich/SILSO record), rises and falls on the ~11-year solar cycle,
whose length wanders between roughly 9 and 14 years and is aligned to *nothing*
on the calendar. So the toolkit's existing time-series surface — one-hot
calendar months, an elapsed-months trend, a ``seasonal_naive`` of period 12 — is
near-useless here: month has no effect (the Explore step shows the by-month
means are flat), and last-year's value is a poor guide to this one. What *does*
predict the series is its own recent history, so this is the first project to
need **autoregressive** features and a **recursive multi-step forecast** — the
two library gaps it surfaced and now consumes (``ds.features.add_lagged_features``
and ``ds.modeling.timeseries.forecast_recursive``).

The pipeline runs the full lifecycle on ``ds`` + scikit-learn alone: fetch →
validate → build the time axis → explore (show month carries no signal) → lag
features → chronological split → cross-validate the one-step AR model with
rolling-origin folds → persist the model → from the reloaded model, both a
one-step-ahead forecast (each step reads the *true* recent values) and a
recursive multi-step forecast (each step feeds the model its own predictions)
→ evaluate against the naive references → visualize. Friction it surfaced is
recorded in ``ROADMAP_ARCHIVE.md`` — that regenerated backlog is as much the deliverable
as the forecast.

Run it with::

    uv run ds run sunspots

The raw CSV is downloaded once into ``<data_dir>/raw/`` (git-ignored) and reused
on later runs. The single mirror is a live third-party GitHub repo, so the fetch
pins its sha256 and verifies the download before trusting it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

from ds import Settings, get_logger, get_settings, seed_everything
from ds.eda import summarize
from ds.evaluation import compare_models, cross_validate_by_time, regression_metrics
from ds.features import add_lagged_features
from ds.io import fetch_dataset, load_raw, save_processed
from ds.modeling.baseline import fit_baseline
from ds.modeling.persistence import load_model, save_model
from ds.modeling.tabular import split_features_target
from ds.modeling.timeseries import forecast_recursive, train_test_split_by_time
from ds.preprocessing import standardize_column_names
from ds.validation import (
    assert_in_range,
    assert_no_nulls,
    assert_row_count,
    assert_unique,
    check_schema,
    require_columns,
)
from ds.viz import plot_model_comparison, plot_residuals, plot_series, set_theme

logger = get_logger(__name__)

# A live third-party mirror of the monthly Zurich sunspot record (columns
# "Month" = YYYY-MM and "Sunspots" = monthly mean number). Because it is a live
# upstream repo, ds.io.fetch_dataset pins the sha256 below and verifies the
# download before trusting it. (A second byte-identical mirror would slot
# straight into this tuple.)
DATA_URLS = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv",)
RAW_NAME = "monthly-sunspots.csv"
RAW_SHA256 = "c4ec8cc57d9f6fb6ecdb3e1f37f25b6f4badd5124c55931052b0cd2fc3bc71f3"

EXPECTED_ROWS = 2820  # 1749-01 through 1983-12, monthly, no gaps
_TARGET = "sunspots"

# The autoregressive features: the value 1, 2, 3, 6 and 12 months back. Short
# lags carry the month-to-month momentum; the 12-month lag lets the linear model
# lean on the (weak, non-calendar) year-over-year persistence. The cycle itself
# is far longer than any fixed lag, which is exactly why a recursive forecast
# decays — see the evaluation notes.
_LAGS = [1, 2, 3, 6, 12]

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

# The forecast horizon for the held-out window: 120 months (a decade, a little
# under one mean solar cycle). A forecasting evaluation holds out a *future*
# window, and a decade is a meaningful ask; over it the recursive forecast has
# room to track the cycle's current phase before its error compounds.
HORIZON = 120


def fetch_raw(settings: Settings) -> Path:
    """Download the sunspots CSV into ``settings.raw_dir`` and verify its checksum.

    A thin binding of this project's dataset (mirror, filename, pinned digest)
    to :func:`ds.io.fetch_dataset`, which does the download, checksum
    verification and cache re-verify.

    Args:
        settings: Resolves the raw-data directory.

    Returns:
        Path to the verified local copy of the dataset.

    Raises:
        ValueError: If the mirror does not serve a file matching the checksum.
        urllib.error.URLError: If the mirror is unreachable.
    """
    return fetch_dataset(RAW_NAME, DATA_URLS, sha256=RAW_SHA256, settings=settings)


def build_time_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the ``month`` strings into a sorted, unique ``date`` column.

    The raw file carries the time axis as ``YYYY-MM`` strings; everything
    downstream (the chronological split, the rolling-origin folds, the lag
    features taken by row position) needs one sortable datetime column, stamped
    to the first of each month. The axis must be unique — a duplicated month
    would corrupt the by-position lags — which the ``assert_unique`` guard checks
    and raw ``to_datetime`` does not.

    Args:
        df: Frame with a ``month`` column of ``YYYY-MM`` strings.

    Returns:
        A new frame with a ``date`` datetime column, sorted chronologically.

    Raises:
        DataValidationError: If any month occurs more than once.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["month"], format="%Y-%m")
    assert_unique(out, "date")
    return out.sort_values("date").reset_index(drop=True)


def run(output_dir: Path, settings: Settings | None = None) -> dict[str, float]:
    """Run the sunspot-forecasting pipeline end to end.

    Args:
        output_dir: Directory to write artifacts (figures, reports) into.
        settings: Data-directory configuration; defaults to the shared one.

    Returns:
        Regression metrics for the one-step-ahead forecast on the held-out
        window (``mae``/``rmse``/``r2``), plus ``recursive_``-, ``seasonal_naive_``-
        and ``naive_last_``-prefixed counterparts from the recursive multi-step
        forecast and the two naive references.
    """
    seed_everything()
    settings = settings or get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire — one idempotent, checksum-verified download (fetch_dataset's
    # third consumer, after air_quality and adult_income).
    fetch_raw(settings)
    df = load_raw(RAW_NAME, settings=settings)

    # 2. Validate at the boundary: the published row count, the two columns we
    # depend on, a numeric target, no gaps (the record is complete), and a
    # non-negative range (a sunspot number cannot be below zero).
    df = standardize_column_names(df)
    require_columns(df, ["month", _TARGET])
    assert_row_count(df, EXPECTED_ROWS)
    df = check_schema(df, {_TARGET: "float64"}, coerce=True)
    assert_no_nulls(df)
    assert_in_range(df, _TARGET, min_value=0.0)

    # 3. Time axis — the YYYY-MM strings become one sorted, unique datetime
    # column (hand-rolled parse; nothing in the library assembles it).
    df = build_time_axis(df)

    # 4. Explore — persist the profile the modeling choices rest on, and the
    # evidence for the model *shape*: the summary, the by-calendar-month mean
    # (near-flat: month carries essentially no signal, which is why the calendar
    # features flights leans on are dropped here), and the series itself (the
    # ~11-year cycle the AR model must ride).
    set_theme("notebook")
    summarize(df).to_csv(output_dir / "summary.csv")
    by_month = df.assign(month_name=df["date"].dt.month_name())
    seasonal_profile = by_month.groupby("month_name")[_TARGET].mean().reindex(_MONTH_NAMES)
    seasonal_profile.to_csv(output_dir / "seasonality.csv")
    logger.info(
        "By-month range %.1f-%.1f vs overall mean %.1f (month carries little signal)",
        float(seasonal_profile.min()),
        float(seasonal_profile.max()),
        float(df[_TARGET].mean()),
    )
    fig, ax = plt.subplots()
    plot_series(df["date"], df[_TARGET], ax=ax)
    ax.set_xlabel("year")
    ax.set_ylabel("monthly sunspot number")
    ax.set_title("Monthly mean sunspot number, 1749-1983")
    fig.savefig(output_dir / "series.png", bbox_inches="tight")
    plt.close(fig)

    # 5. Autoregressive features — the value 1/2/3/6/12 months back. This is a
    # stateless transform (a row's lags are the rows already beside it), so it is
    # safe before the split; the warm-up rows with no complete history are
    # dropped. This is add_lagged_features' first consumer — the friction this
    # project surfaced.
    lagged = add_lagged_features(df, _TARGET, _LAGS)
    lag_columns = [f"{_TARGET}_lag_{k}" for k in _LAGS]

    # 6. Chronological split — hold out the last HORIZON months as a strictly
    # future window (the only valid forecasting protocol). test_size is the
    # horizon as a fraction of the lagged frame.
    test_size = HORIZON / len(lagged)
    train, test = train_test_split_by_time(lagged, "date", test_size=test_size)

    # 7. Model — a pure autoregression: LinearRegression on the lag features
    # alone. No fit-based frame transform is needed or honest here — the lags are
    # stateless, the series is complete (no impute), its swings are the signal
    # (no clip), and OLS is scale-free (no scale). So, unlike the other projects,
    # there is no ds.pipeline scoring Pipeline to persist (a scope finding, not a
    # gap — flights already had a one-step plan; this AR model has none); only
    # the model is persisted. The model is fitted, saved, and reloaded, so both
    # forecasts below score from the on-disk copy.
    x_train, y_train = split_features_target(train[[*lag_columns, _TARGET]], _TARGET)
    x_test, y_test = split_features_target(test[[*lag_columns, _TARGET]], _TARGET)
    save_processed(pd.concat([train, test]), "sunspots_features.parquet", settings=settings)
    params_dir = settings.processed_dir / "params"
    save_model(LinearRegression().fit(x_train, y_train), params_dir / "sunspots_model.joblib")
    model = load_model(params_dir / "sunspots_model.joblib")

    # 8. Cross-validate the one-step AR model with rolling-origin folds on the
    # training window — cross_validate_by_time again (its date column and target
    # excluded from the features). Each fold's test block is one-step-ahead: it
    # reads the true recent values, the realistic "predict next month" task.
    cv_scores = cross_validate_by_time(
        train[["date", *lag_columns, _TARGET]],
        time_column="date",
        target=_TARGET,
        make_model=lambda: LinearRegression(),
    )
    cv_scores.to_csv(output_dir / "cv_folds.csv")
    logger.info("Rolling-origin one-step CV mae: %.2f (+/- %.2f)", *_mean_std(cv_scores["mae"]))

    # 9. Forecast the held-out window two ways from the *reloaded* model:
    #   - one-step-ahead: each prediction reads the true lag values (what you
    #     have when forecasting next month), a plain model.predict.
    #   - recursive multi-step: forecast the whole HORIZON from the end of
    #     training, feeding each prediction back as later steps' lags
    #     (forecast_recursive) — the realistic "forecast the next decade now"
    #     task, and this project's second surfaced helper.
    one_step_preds = [float(value) for value in model.predict(x_test)]
    recursive_preds = forecast_recursive(
        model, train[_TARGET].tolist(), lags=_LAGS, steps=len(y_test)
    )

    # 10. Evaluate against the two naive references the calendar approach would
    # reach for — both weak here: naive_last repeats one arbitrary phase of the
    # cycle, seasonal_naive repeats the value 12 months back (the cycle is far
    # longer than a year). One-step AR should beat both comfortably; the
    # recursive forecast is honest about compounding error over a decade.
    naive_last = fit_baseline(y_train, strategy="naive_last")
    seasonal = fit_baseline(y_train, strategy="seasonal_naive", season_length=12)
    naive_last_preds = naive_last.predict(len(y_test))
    seasonal_preds = seasonal.predict(len(y_test))
    comparison = compare_models(
        y_test.tolist(),
        {
            "ar_one_step": one_step_preds,
            "ar_recursive": recursive_preds,
            "seasonal_naive": seasonal_preds,
            "naive_last": naive_last_preds,
        },
    )
    comparison.to_csv(output_dir / "model_comparison.csv")

    metrics = regression_metrics(y_test.tolist(), one_step_preds)
    for name, preds in (
        ("recursive", recursive_preds),
        ("seasonal_naive", seasonal_preds),
        ("naive_last", naive_last_preds),
    ):
        scores = regression_metrics(y_test.tolist(), preds)
        metrics.update({f"{name}_{key}": value for key, value in scores.items()})
    logger.info("Held-out metrics (one-step vs recursive vs naive): %s", metrics)

    # 11. Visualize — residuals of the one-step forecast, the model comparison
    # bars, and the forecast-vs-actual plot: the training tail, then the held-out
    # window with both the one-step and recursive forecasts overlaid.
    fig2, ax2 = plt.subplots()
    plot_residuals(y_test.tolist(), one_step_preds, ax=ax2)
    ax2.set_title("One-step residuals vs predicted - held-out window")
    fig2.savefig(output_dir / "residuals.png", bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    plot_model_comparison(comparison, metric="mae", ax=ax3)
    fig3.savefig(output_dir / "model_comparison.png", bbox_inches="tight")
    plt.close(fig3)

    fig4, ax4 = plt.subplots()
    history = train.iloc[-2 * HORIZON :]
    plot_series(history["date"], history[_TARGET], label="history", ax=ax4)
    plot_series(
        test["date"],
        y_test,
        predictions={"one-step": one_step_preds, "recursive": recursive_preds},
        label="actual",
        ax=ax4,
    )
    ax4.set_xlabel("year")
    ax4.set_ylabel("monthly sunspot number")
    ax4.set_title("Held-out decade - forecasts vs actual")
    fig4.savefig(output_dir / "forecast.png", bbox_inches="tight")
    plt.close(fig4)

    return metrics


def _mean_std(values: pd.Series) -> tuple[float, float]:
    """Mean and standard deviation of a numeric series, for log lines."""
    return float(values.mean()), float(values.std())


def main() -> None:
    settings = get_settings()
    metrics = run(settings.processed_dir / "sunspots", settings=settings)
    print("Pipeline finished. Held-out metrics (one-step vs recursive vs naive references):")
    for name, value in metrics.items():
        print(f"  {name:>24}: {value:,.3f}")


if __name__ == "__main__":
    main()
