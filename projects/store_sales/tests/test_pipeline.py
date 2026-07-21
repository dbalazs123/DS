"""Tests for the store_sales pipeline.

Run from the repo root with::

    uv run pytest projects/store_sales --no-cov

The end-to-end test downloads the real dataset once into a temporary data
directory; it skips (rather than fails) when the network is unavailable.
"""

from __future__ import annotations

import importlib.util
import urllib.error
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from ds.validation import DataValidationError

PIPELINE_PATH = Path(__file__).resolve().parent.parent / "pipeline.py"


def _load_pipeline() -> ModuleType:
    """Load the sibling ``pipeline.py`` by path (no package import needed)."""
    spec = importlib.util.spec_from_file_location("store_sales_pipeline", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _toy_panel() -> pd.DataFrame:
    """Two entities, three days each, deliberately out of order."""
    return pd.DataFrame(
        {
            "date": ["2013-01-02", "2013-01-01", "2013-01-03"] * 2,
            "store": [1, 1, 1, 2, 2, 2],
            "item": [1, 1, 1, 1, 1, 1],
            "sales": [11, 10, 12, 21, 20, 22],
        }
    )


def test_select_panel_keeps_only_requested_entities() -> None:
    pipeline = _load_pipeline()
    df = pd.DataFrame(
        {"store": [1, 2, 3], "item": [1, 6, 1], "date": ["2013-01-01"] * 3, "sales": [1, 2, 3]}
    )
    out = pipeline.select_panel(df, stores=[1, 2], items=[1])
    # store 3 dropped (not requested) and store 2 dropped (item 6 not requested).
    assert out["store"].tolist() == [1]


def test_order_panel_sorts_within_entity_and_parses_date() -> None:
    pipeline = _load_pipeline()
    out = pipeline.order_panel(_toy_panel())
    assert (
        out["date"].tolist()
        == [
            pd.Timestamp("2013-01-01"),
            pd.Timestamp("2013-01-02"),
            pd.Timestamp("2013-01-03"),
        ]
        * 2
    )
    # Each entity's rows are contiguous and ascending; the target travels along.
    assert out["store"].tolist() == [1, 1, 1, 2, 2, 2]
    assert out["sales"].tolist() == [10, 11, 12, 20, 21, 22]


def test_order_panel_rejects_duplicate_entity_day() -> None:
    pipeline = _load_pipeline()
    dup = pd.DataFrame(
        {"date": ["2013-01-01", "2013-01-01"], "store": [1, 1], "item": [1, 1], "sales": [5, 6]}
    )
    with pytest.raises(DataValidationError, match="duplicate"):
        pipeline.order_panel(dup)


def test_grouped_lags_never_bleed_across_the_entity_boundary() -> None:
    # The whole reason this project exists: a by-position lag over the ordered
    # panel would make store 2's first row read store 1's last sale (12) as its
    # history. The grouped lag must instead leave store 2 to start from its own.
    from ds.features import add_lagged_features

    pipeline = _load_pipeline()
    ordered = pipeline.order_panel(_toy_panel())
    lagged = add_lagged_features(ordered, "sales", [1], group=["store", "item"], dropna=False)
    first_of_store_2 = lagged[(lagged["store"] == 2)].iloc[0]
    assert bool(pd.isna(first_of_store_2["sales_lag_1"]))  # not 12 bled from store 1


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    from ds import Settings

    pipeline = _load_pipeline()
    settings = Settings(data_dir=tmp_path / "data")
    try:
        pipeline.fetch_raw(settings)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        pytest.skip(f"dataset download unavailable: {exc}")

    metrics = pipeline.run(tmp_path / "out", settings=settings)

    # The pooled model — entity + calendar effects and within-entity lags — beats
    # both naive references comfortably on the held-out 2017 window.
    assert metrics["r2"] > 0.8
    assert metrics["mae"] < metrics["weekly_naive_mae"]
    assert metrics["mae"] < metrics["naive_last_mae"]
    # The weekly seasonal naive is itself a better reference than last-value,
    # because the day-of-week cycle dominates — an honest ordering of the two.
    assert metrics["weekly_naive_mae"] < metrics["naive_last_mae"]
