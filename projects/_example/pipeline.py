"""End-to-end example pipeline built entirely from the ``ds`` toolkit.

It exercises the library's full stage set on realistically dirty data to prove
the toolkit composes into a real workflow: generate (dirty) → save/load →
validate → clean → feature-engineer → time-split → model → evaluate →
visualize.

Run it with::

    uv run python projects/_example/pipeline.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ds import Settings, get_logger, seed_everything
from ds.evaluation import regression_metrics
from ds.features import add_datetime_features, one_hot_encode, scale_features
from ds.io import load_raw, save_processed, save_table
from ds.modeling.tabular import split_features_target
from ds.modeling.timeseries import train_test_split_by_time
from ds.preprocessing import (
    clip_outliers,
    coerce_dtypes,
    drop_constant_columns,
    drop_duplicate_rows,
    impute_missing,
    standardize_column_names,
)
from ds.validation import assert_no_nulls, check_schema, require_columns
from ds.viz import plot_outliers, set_theme

logger = get_logger(__name__)

_REGIONS = ("north", "south", "east", "west")


def make_synthetic_sales(n_days: int = 365) -> pd.DataFrame:
    """Create a realistically dirty synthetic daily-sales frame.

    Trend, weekly seasonality and noise drive the signal; a categorical
    ``Region`` column, sprinkled missing values, a handful of extreme
    outliers, and a few duplicate rows are layered on top so the cleaning and
    feature stages have genuine work to do.
    """
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    trend = np.linspace(100, 300, n_days)
    weekly = 20 * np.sin(2 * np.pi * dates.dayofweek / 7)
    noise = np.random.normal(0, 10, n_days)
    sales = trend + weekly + noise
    region = np.random.choice(_REGIONS, size=n_days)

    df = pd.DataFrame({"Date": dates, "Total Sales ($)": sales, "Region": region})

    # Missing values: a few blank sales figures and a few blank regions.
    dirty_idx = np.random.choice(n_days, size=13, replace=False)
    missing_sales_idx, missing_region_idx, outlier_idx = (
        dirty_idx[:5],
        dirty_idx[5:10],
        dirty_idx[10:13],
    )
    df.loc[missing_sales_idx, "Total Sales ($)"] = np.nan
    df.loc[missing_region_idx, "Region"] = np.nan

    # Outliers: a few implausible sales spikes, well outside any Tukey fence.
    df.loc[outlier_idx, "Total Sales ($)"] = df.loc[outlier_idx, "Total Sales ($)"] + 1000

    # Duplicate rows: append exact copies of a couple of clean rows.
    df = pd.concat([df, df.iloc[[20, 21]]], ignore_index=True)

    return df


def run(output_dir: Path) -> dict[str, float]:
    """Run the full pipeline, writing artifacts under ``output_dir``.

    Returns:
        The regression metrics on the held-out (future) test window.
    """
    seed_everything(42)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as data_dir:
        settings = Settings(data_dir=Path(data_dir))

        # 1. Acquire — seed the raw-data directory, then load through it like a
        # real project would (raw/ -> processed/), never the repo's data/.
        raw = make_synthetic_sales()
        save_table(raw, settings.raw_dir / "sales.parquet")
        df = load_raw("sales.parquet", settings=settings)

        # 2. Validate — the assumptions the rest of the pipeline depends on.
        df = standardize_column_names(df)
        require_columns(df, ["date", "total_sales", "region"])
        df = check_schema(
            df,
            {"date": "datetime64[ns]", "total_sales": "float64", "region": "object"},
        )

        # 3. Clean.
        dirty = df.copy()  # kept for the outlier plot, before clipping erases them
        df = coerce_dtypes(df, {"region": "category"})
        df = drop_duplicate_rows(df)
        df = drop_constant_columns(df)  # region now varies, so it survives this time
        df = clip_outliers(df, columns=["total_sales"])
        df = impute_missing(df, columns=["total_sales"], strategy="median")
        df = impute_missing(df, columns=["region"], strategy="most_frequent")
        assert_no_nulls(df)

        # 4. Feature engineering (keep `date` for ordering, drop it before modeling).
        df = add_datetime_features(df, "date", drop=False)
        df = one_hot_encode(df, columns=["region"])
        date_parts = ["date_year", "date_month", "date_day", "date_dayofweek"]
        df = scale_features(df, columns=date_parts)

        save_processed(df, "sales_features.parquet", settings=settings)

        # 5. Chronological split — the test set is strictly in the future.
        train, test = train_test_split_by_time(df, "date")
        x_train, y_train = split_features_target(train.drop(columns=["date"]), "total_sales")
        x_test, y_test = split_features_target(test.drop(columns=["date"]), "total_sales")

    # 6. Model + evaluate.
    model = LinearRegression().fit(x_train, y_train)
    preds = model.predict(x_test)
    metrics = regression_metrics(y_test.tolist(), preds.tolist())
    logger.info("Test metrics: %s", metrics)

    # 7. Visualize.
    set_theme("notebook")
    fig, ax = plt.subplots()
    ax.plot(range(len(y_test)), y_test.to_numpy(), label="actual")
    ax.plot(range(len(preds)), preds, label="predicted")
    ax.set_title("Sales forecast — held-out window")
    ax.legend()
    fig.savefig(output_dir / "forecast.png", bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots()
    plot_outliers(dirty, columns=["total_sales"], ax=ax2)
    ax2.set_title("Outliers detected before clipping")
    fig2.savefig(output_dir / "outliers.png", bbox_inches="tight")
    plt.close(fig2)

    return metrics


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        metrics = run(Path(tmp))
    print("Pipeline finished. Held-out metrics:")
    for name, value in metrics.items():
        print(f"  {name:>4}: {value:,.3f}")


if __name__ == "__main__":
    main()
