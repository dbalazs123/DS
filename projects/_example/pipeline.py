"""End-to-end example pipeline built entirely from the ``ds`` toolkit.

It exercises one function from every stage of the lifecycle to prove the library
composes into a real workflow: generate → save/load → validate → clean →
feature-engineer → time-split → model → evaluate → visualize.

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

from ds import get_logger, seed_everything
from ds.evaluation import regression_metrics
from ds.features import add_datetime_features
from ds.io import load_table, save_table
from ds.modeling.tabular import split_features_target
from ds.modeling.timeseries import train_test_split_by_time
from ds.preprocessing import drop_constant_columns, standardize_column_names
from ds.validation import require_columns
from ds.viz import set_theme

logger = get_logger(__name__)


def make_synthetic_sales(n_days: int = 365) -> pd.DataFrame:
    """Create a synthetic daily-sales series with trend, weekly seasonality, noise."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    trend = np.linspace(100, 300, n_days)
    weekly = 20 * np.sin(2 * np.pi * dates.dayofweek / 7)
    noise = np.random.normal(0, 10, n_days)
    return pd.DataFrame(
        {
            "Date": dates,
            "Total Sales ($)": trend + weekly + noise,
            "Region": "north",  # constant column, dropped during cleaning
        }
    )


def run(output_dir: Path) -> dict[str, float]:
    """Run the full pipeline, writing artifacts under ``output_dir``.

    Returns:
        The regression metrics on the held-out (future) test window.
    """
    seed_everything(42)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire + persist, then reload through the io layer.
    raw = make_synthetic_sales()
    raw_path = save_table(raw, output_dir / "sales.parquet")
    df = load_table(raw_path)

    # 2. Clean.
    df = standardize_column_names(df)
    df = drop_constant_columns(df)
    require_columns(df, ["date", "total_sales"])

    # 3. Feature engineering (keep `date` for ordering, drop it before modeling).
    df = add_datetime_features(df, "date", drop=False)

    # 4. Chronological split — the test set is strictly in the future.
    train, test = train_test_split_by_time(df, "date")
    x_train, y_train = split_features_target(train.drop(columns=["date"]), "total_sales")
    x_test, y_test = split_features_target(test.drop(columns=["date"]), "total_sales")

    # 5. Model + evaluate.
    model = LinearRegression().fit(x_train, y_train)
    preds = model.predict(x_test)
    metrics = regression_metrics(y_test.tolist(), preds.tolist())
    logger.info("Test metrics: %s", metrics)

    # 6. Visualize.
    set_theme("notebook")
    fig, ax = plt.subplots()
    ax.plot(range(len(y_test)), y_test.to_numpy(), label="actual")
    ax.plot(range(len(preds)), preds, label="predicted")
    ax.set_title("Sales forecast — held-out window")
    ax.legend()
    fig.savefig(output_dir / "forecast.png", bbox_inches="tight")
    plt.close(fig)

    return metrics


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        metrics = run(Path(tmp))
    print("Pipeline finished. Held-out metrics:")
    for name, value in metrics.items():
        print(f"  {name:>4}: {value:,.3f}")


if __name__ == "__main__":
    main()
