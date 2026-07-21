"""Store sales — forecast daily units sold across a store x item **panel**.

The tenth project on *real* data, and the first on a **panel**: not one flat
table and not one univariate series, but many series stacked in one frame — the
daily unit sales of a subset of the classic "Store Item Demand" dataset (10
stores, 50 items, 2013-2017). Every project before this was single-entity, so
the toolkit's time-series surface had only ever seen one series at a time. A
panel breaks that assumption at exactly one place, and finding it was the point:

    ``ds.features.add_lagged_features`` lagged **by row position over the whole
    frame**, so stacking store 2 beneath store 1 made store 2's first rows read
    *store 1's tail* as their history — history bled across every entity edge.

The fix this project pulled is the helper's ``group=`` parameter: lags taken
independently *within* each (store, item), so no value ever crosses an entity
boundary. That is the one library change of this loop; the rest of the panel's
friction (a chronological split, naive references and a rolling backtest that are
all still single-series) is recorded in ``ROADMAP_ARCHIVE.md`` and handled inline
here, because on a *shared-calendar* panel each has a clean workaround: a global
date cutoff splits every entity at once, and each entity's naive forecast is
simply its own lag column.

The pipeline runs the full lifecycle on ``ds`` + scikit-learn alone: fetch →
validate → select the panel and order it within each entity → explore (the
weekly and yearly shape every series shares) → calendar + **grouped** lag
features → split on a date cutoff (train 2013-2016, forecast 2017) → fit the
one-hot plan on the training window (``ds.pipeline.fit_pipeline``) → a pooled
linear model with entity + calendar effects and autoregressive lags → evaluate
one-step-ahead against the naive-last and weekly-seasonal-naive references (the
``sales_lag_1`` and ``sales_lag_7`` columns themselves) → visualize one entity's
forecast.

Run it with::

    uv run ds run store_sales

The raw CSV is downloaded once into ``<data_dir>/raw/`` (git-ignored) and reused
on later runs. The single mirror is a live third-party GitHub repo, so the fetch
pins its sha256 and verifies the download before trusting it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

from ds import Settings, get_logger, get_settings, seed_everything
from ds.eda import summarize
from ds.evaluation import compare_models, regression_metrics
from ds.features import add_datetime_features, add_lagged_features, fit_one_hot_categories
from ds.io import fetch_dataset, load_raw, save_params, save_processed
from ds.modeling.persistence import load_model, save_model
from ds.modeling.tabular import split_features_target
from ds.pipeline import FitStep, fit_pipeline
from ds.preprocessing import standardize_column_names
from ds.validation import (
    DataValidationError,
    assert_in_range,
    assert_no_nulls,
    assert_row_count,
    check_schema,
    require_columns,
)
from ds.viz import plot_model_comparison, plot_residuals, plot_series, set_theme

logger = get_logger(__name__)

# A live third-party mirror of the Kaggle "Store Item Demand Forecasting
# Challenge" training file (columns date, store, item, sales — 500 daily series,
# 10 stores x 50 items, 2013-01-01 through 2017-12-31). Because it is a live
# upstream repo, ds.io.fetch_dataset pins the sha256 below and verifies the
# download before trusting it. (A second byte-identical mirror would slot
# straight into this tuple.)
DATA_URLS = (
    "https://raw.githubusercontent.com/DharitShah13/"
    "Kaggle-Store-Item-Demand-Forecasting-Challenge/master/train.csv",
)
RAW_NAME = "store_item_demand.csv"
RAW_SHA256 = "038f25690a65149c94f86ddd3deceda20c037a5cfd754cafdfc539a72992f2ed"

EXPECTED_ROWS = 913_000  # 500 series x 1,826 days, no gaps

# The panel worked here: every store, the first five items. Fifty entities is
# enough that the grouped lags plainly matter (a by-position lag would bleed at
# 49 interior boundaries) while keeping the one-hot entity effects and the run
# small. The full 500-series file downloads; the subset is taken in-memory.
STORES = tuple(range(1, 11))
ITEMS = (1, 2, 3, 4, 5)

_TARGET = "sales"
_KEYS = ["store", "item"]  # the composite entity key

# Daily lags: yesterday (momentum), the same weekday last week (the dominant
# retail cycle) and two weeks back. sales_lag_1 and sales_lag_7 double as the
# naive-last and weekly-seasonal-naive one-step forecasts the model must beat.
_LAGS = [1, 7, 14]

# Hold out the final year: train on 2013-2016, forecast all of 2017. A single
# date cutoff splits every entity at the same instant — the shared calendar is
# exactly what lets one global boundary stand in for a per-entity split.
CUTOFF = pd.Timestamp("2017-01-01")

_WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def fetch_raw(settings: Settings) -> Path:
    """Download the store-item CSV into ``settings.raw_dir`` and verify its checksum.

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


def select_panel(
    df: pd.DataFrame, stores: Sequence[int] = STORES, items: Sequence[int] = ITEMS
) -> pd.DataFrame:
    """Keep only the ``(store, item)`` entities this analysis models.

    The raw file carries all 500 series; the analysis works a subset so the
    one-hot entity effects and the run stay small. Filtering is by value, so an
    empty result (a store or item that is not in the data) is a caller error the
    downstream row-count and uniqueness checks would surface.

    Args:
        df: Frame with integer ``store`` and ``item`` columns.
        stores: Store ids to keep.
        items: Item ids to keep.

    Returns:
        A new frame holding only the selected entities.
    """
    mask = df["store"].isin(list(stores)) & df["item"].isin(list(items))
    return df.loc[mask].copy()


def order_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the ``date`` axis and sort rows within each entity.

    A panel's lags are taken within each ``(store, item)`` group and *by row
    position inside the group*, so every entity's rows must be contiguous and in
    time order before :func:`ds.features.add_lagged_features` runs. The composite
    ``(store, item, date)`` key must also be unique — a duplicated day for one
    entity would corrupt the by-position lags, and sorting would silently
    interleave the duplicates. (The library's ``assert_unique`` guards a *single*
    column; a panel's key is composite, so the check is inline here.)

    Args:
        df: Frame with string ``date`` and integer ``store``/``item`` columns.

    Returns:
        A new frame with a parsed ``date`` datetime column, sorted by
        ``(store, item, date)`` with a reset index.

    Raises:
        DataValidationError: If any ``(store, item, date)`` triple repeats.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    if bool(out.duplicated(subset=[*_KEYS, "date"]).any()):
        raise DataValidationError("duplicate (store, item, date) rows in the panel")
    return out.sort_values([*_KEYS, "date"]).reset_index(drop=True)


def run(output_dir: Path, settings: Settings | None = None) -> dict[str, float]:
    """Run the store-item panel forecasting pipeline end to end.

    Args:
        output_dir: Directory to write artifacts (figures, reports) into.
        settings: Data-directory configuration; defaults to the shared one.

    Returns:
        One-step-ahead regression metrics on the held-out 2017 window
        (``mae``/``rmse``/``r2``), plus ``weekly_naive_``- and ``naive_last_``-
        prefixed counterparts from the two naive references the model must beat.
    """
    seed_everything()
    settings = settings or get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire — one idempotent, checksum-verified download (fetch_dataset's
    # fifth consumer, after air_quality, adult_income, sunspots and bbc_news).
    fetch_raw(settings)
    df = load_raw(RAW_NAME, settings=settings)

    # 2. Validate at the boundary, on the *full* file: the four columns we
    # depend on, integer codes and counts, no gaps, non-negative sales, and the
    # published row count (500 series x 1,826 days).
    df = standardize_column_names(df)
    require_columns(df, ["date", "store", "item", _TARGET])
    df = check_schema(df, {"store": "int64", "item": "int64", _TARGET: "int64"}, coerce=True)
    assert_no_nulls(df)
    assert_in_range(df, _TARGET, min_value=0)
    assert_row_count(df, EXPECTED_ROWS)

    # 3. Select the panel and order it within each entity — the two steps every
    # grouped operation below depends on (contiguous, time-sorted, unique keys).
    df = select_panel(df)
    df = order_panel(df)

    # 4. Explore — persist the shape every series shares and the modeling
    # choices rest on: the summary, mean sales per store and per item (the entity
    # effects the one-hot terms will carry), and the weekly profile (the strong
    # day-of-week cycle the lag_7 feature and the month dummies must capture).
    set_theme("notebook")
    summarize(df).to_csv(output_dir / "summary.csv")
    df.groupby("store")[_TARGET].mean().to_csv(output_dir / "store_means.csv")
    df.groupby("item")[_TARGET].mean().to_csv(output_dir / "item_means.csv")
    weekday_profile = (
        df.assign(weekday=df["date"].dt.day_name())
        .groupby("weekday")[_TARGET]
        .mean()
        .reindex(_WEEKDAY_NAMES)
    )
    weekday_profile.to_csv(output_dir / "weekday_profile.csv")
    logger.info(
        "Weekday sales range %.1f-%.1f (mean %.1f): the cycle lag_7 captures",
        float(weekday_profile.min()),
        float(weekday_profile.max()),
        float(df[_TARGET].mean()),
    )

    # 5. Features — all stateless, so safe before the split. Calendar terms
    # (day-of-week and month effects, an elapsed-months trend) via
    # add_datetime_features, then the autoregressive lags **within each entity**
    # via add_lagged_features(group=...) — the group-aware path this project
    # surfaced. Without group=, the first rows of every store/item after the
    # first would read the previous entity's tail as their history.
    df = add_datetime_features(df, "date", features=["dayofweek", "month", "elapsed_months"])
    df = add_lagged_features(df, _TARGET, _LAGS, group=_KEYS)

    # 6. Chronological split on the date cutoff — every entity is cut at
    # 2017-01-01 at once, so the test window is strictly each series' future.
    # (train_test_split_by_time splits one series by row fraction; a panel wants
    # a value cutoff, recorded as friction and done inline here.)
    train = df[df["date"] < CUTOFF].copy()
    test = df[df["date"] >= CUTOFF].copy()

    # 7. Fit the one-hot plan on train, apply to both. The only fitted state is
    # the entity + calendar vocabularies; the lags are stateless and the trend is
    # a plain numeric column OLS handles scale-free (no impute/clip/scale step —
    # a scope finding, matching flights and sunspots).
    plan = [
        FitStep(
            "one_hot_encode",
            lambda frame: fit_one_hot_categories(
                frame, columns=["store", "item", "date_dayofweek", "date_month"]
            ),
        ),
    ]
    scoring = fit_pipeline(train, plan)
    train_enc = scoring.apply(train)
    test_enc = scoring.apply(test)
    assert_no_nulls(train_enc)
    assert_no_nulls(test_enc)

    # 8. Persist the processed panel and the whole scoring pipeline.
    save_processed(
        pd.concat([train_enc, test_enc]), "store_sales_features.parquet", settings=settings
    )
    params_dir = settings.processed_dir / "params"
    save_params(scoring, params_dir / "store_sales_scoring.json")

    # 9. Model — one pooled linear model over every entity: entity + calendar
    # one-hots, the elapsed-months trend and the three lags. Fitted once,
    # persisted, and the held-out window scored from the *reloaded* copy. ``date``
    # orders the rows but is not a feature.
    x_train, y_train = split_features_target(train_enc.drop(columns=["date"]), _TARGET)
    x_test, y_test = split_features_target(test_enc.drop(columns=["date"]), _TARGET)
    save_model(LinearRegression().fit(x_train, y_train), params_dir / "store_sales_model.joblib")
    model = load_model(params_dir / "store_sales_model.joblib")
    linear_preds = [float(value) for value in model.predict(x_test)]

    # 10. Evaluate one-step-ahead against the two naive references. On a panel
    # the per-entity naive forecasts need no fit_baseline (which is single-series)
    # — each test row already carries them: sales_lag_1 is that entity's
    # naive-last, sales_lag_7 its same-weekday-last-week (weekly seasonal naive).
    naive_last_preds = [float(value) for value in test_enc[f"{_TARGET}_lag_1"]]
    weekly_preds = [float(value) for value in test_enc[f"{_TARGET}_lag_7"]]
    actual = y_test.tolist()
    comparison = compare_models(
        actual,
        {
            "linear_regression": linear_preds,
            "weekly_naive": weekly_preds,
            "naive_last": naive_last_preds,
        },
    )
    comparison.to_csv(output_dir / "model_comparison.csv")
    metrics = regression_metrics(actual, linear_preds)
    for name, preds in (("weekly_naive", weekly_preds), ("naive_last", naive_last_preds)):
        scores = regression_metrics(actual, preds)
        metrics.update({f"{name}_{key}": value for key, value in scores.items()})
    logger.info("Held-out 2017 metrics (pooled linear vs naive references): %s", metrics)

    # 11. Visualize — residuals and the model-comparison bars over the whole
    # panel, then one entity's forecast: store 1 / item 1's 2016 tail, its 2017
    # actual, and the model and weekly-naive forecasts overlaid. The encoded test
    # rows are 1:1 with the pre-encode ``test`` rows, so the entity mask indexes
    # both.
    fig, ax = plt.subplots()
    plot_residuals(actual, linear_preds, ax=ax)
    ax.set_title("Residuals vs predicted - held-out 2017, all entities")
    fig.savefig(output_dir / "residuals.png", bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots()
    plot_model_comparison(comparison, metric="mae", ax=ax2)
    fig2.savefig(output_dir / "model_comparison.png", bbox_inches="tight")
    plt.close(fig2)

    entity = (test["store"] == STORES[0]) & (test["item"] == ITEMS[0])
    entity_mask = entity.to_numpy()
    entity_dates = test.loc[entity, "date"]
    entity_actual = [actual[i] for i, keep in enumerate(entity_mask) if keep]
    entity_linear = [linear_preds[i] for i, keep in enumerate(entity_mask) if keep]
    entity_weekly = [weekly_preds[i] for i, keep in enumerate(entity_mask) if keep]
    history = train[(train["store"] == STORES[0]) & (train["item"] == ITEMS[0])].iloc[-56:]
    fig3, ax3 = plt.subplots()
    plot_series(history["date"], history[_TARGET], label="history", ax=ax3)
    plot_series(
        entity_dates,
        entity_actual,
        predictions={"linear regression": entity_linear, "weekly naive": entity_weekly},
        label="actual",
        ax=ax3,
    )
    ax3.set_xlabel("date")
    ax3.set_ylabel("units sold")
    ax3.set_title(f"Store {STORES[0]} / item {ITEMS[0]} - 2017 forecast vs actual")
    fig3.savefig(output_dir / "forecast.png", bbox_inches="tight")
    plt.close(fig3)

    return metrics


def main() -> None:
    settings = get_settings()
    metrics = run(settings.processed_dir / "store_sales", settings=settings)
    print("Pipeline finished. Held-out 2017 metrics (pooled linear vs naive references):")
    for name, value in metrics.items():
        print(f"  {name:>22}: {value:,.3f}")


if __name__ == "__main__":
    main()
