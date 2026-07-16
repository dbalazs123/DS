"""Smoke test for the worked example pipeline.

``projects/_example/pipeline.py`` is the repo's proof that the library composes
end to end against realistically dirty data. Run it here so a regression in any
stage fails CI instead of quietly breaking the flagship demo. The example lives
outside the ``ds`` package, so we load it by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "projects" / "_example" / "pipeline.py"


def _load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ds_example_pipeline", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_synthetic_sales_is_dirty() -> None:
    example = _load_example()
    raw = example.make_synthetic_sales()

    # Missing values in both the numeric and categorical columns.
    assert raw["Total Sales ($)"].isna().sum() > 0
    assert raw["Region"].isna().sum() > 0

    # A genuine, non-constant categorical column.
    assert raw["Region"].nunique(dropna=True) > 1

    # Duplicate rows present before cleaning.
    assert raw.duplicated().sum() > 0


def test_example_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    example = _load_example()
    metrics = example.run(tmp_path)

    # Every regression metric the evaluation stage promises should be present.
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert all(isinstance(value, float) for value in metrics.values())

    # The visualization stage should have written both figure artifacts.
    assert (tmp_path / "forecast.png").is_file()
    assert (tmp_path / "outliers.png").is_file()


def test_example_pipeline_cleans_and_encodes_split_safely(tmp_path: Path) -> None:
    example = _load_example()

    from ds import Settings, seed_everything
    from ds.features import (
        add_datetime_features,
        apply_one_hot_encode,
        apply_scale_features,
        fit_one_hot_categories,
        fit_scale_params,
    )
    from ds.io import load_params, load_raw, save_params, save_table
    from ds.modeling.timeseries import train_test_split_by_time
    from ds.pipeline import Pipeline, PipelineStep
    from ds.preprocessing import (
        ImputeValues,
        OutlierBounds,
        apply_clip_outliers,
        apply_impute_missing,
        coerce_dtypes,
        drop_constant_columns,
        drop_duplicate_rows,
        fit_impute_values,
        fit_outlier_bounds,
        standardize_column_names,
    )
    from ds.validation import check_schema, require_columns

    seed_everything(42)
    raw = example.make_synthetic_sales()
    assert raw.duplicated().sum() > 0

    settings = Settings(data_dir=tmp_path / "data")
    save_table(raw, settings.raw_dir / "sales.parquet")
    df = load_raw("sales.parquet", settings=settings)

    df = standardize_column_names(df)
    require_columns(df, ["date", "total_sales", "region"])
    df = check_schema(df, {"date": "datetime64[ns]", "total_sales": "float64", "region": "object"})

    df = coerce_dtypes(df, {"region": "category"})
    df = drop_duplicate_rows(df)
    df = drop_constant_columns(df)
    assert "region" in df.columns  # non-constant, so it survives cleaning

    # Split first: everything below fits on train only and applies to both.
    train, test = train_test_split_by_time(df, "date")

    bounds = fit_outlier_bounds(train, columns=["total_sales"])
    lower, upper = bounds.bounds["total_sales"]
    train = apply_clip_outliers(train, bounds)
    test = apply_clip_outliers(test, bounds)
    # Both splits are clipped to the *train*-fitted fence.
    assert train["total_sales"].max() <= upper
    assert test["total_sales"].max() <= upper
    assert test["total_sales"].min() >= lower

    sales_fill = fit_impute_values(train, columns=["total_sales"], strategy="median")
    region_fill = fit_impute_values(train, columns=["region"], strategy="most_frequent")
    # The learned fill is the training window's median, not the test window's.
    assert sales_fill.fill_values["total_sales"] == pytest.approx(
        float(train["total_sales"].median())
    )
    train = apply_impute_missing(apply_impute_missing(train, sales_fill), region_fill)
    test = apply_impute_missing(apply_impute_missing(test, sales_fill), region_fill)
    assert train.isna().sum().sum() == 0
    assert test.isna().sum().sum() == 0

    train = add_datetime_features(train, "date", drop=False)
    test = add_datetime_features(test, "date", drop=False)

    vocabulary = fit_one_hot_categories(train, columns=["region"])
    train = apply_one_hot_encode(train, vocabulary)
    test = apply_one_hot_encode(test, vocabulary)
    encoded_columns = [c for c in train.columns if c.startswith("region_")]
    assert encoded_columns
    assert "region" not in train.columns
    # The fixed vocabulary yields identical column sets on both splits.
    assert list(train.columns) == list(test.columns)

    date_parts = ["date_year", "date_month", "date_day", "date_dayofweek"]
    scaling = fit_scale_params(train, columns=date_parts)
    train = apply_scale_features(train, scaling)
    test = apply_scale_features(test, scaling)
    assert abs(train["date_month"].mean()) < 1e-6
    # The test window sits in train's future, so train-fitted scaling maps it
    # beyond train's standardized range rather than re-centering it at zero.
    assert test["date_month"].mean() > train["date_month"].mean()

    # Persist the fitted state as the example does: the scoring-time
    # transforms travel as ONE pipeline object (order included), while the
    # target-column fits — train-time-only, since scoring rows have no
    # `total_sales` — are saved individually for the next training run.
    scoring = Pipeline(
        steps=(
            PipelineStep("impute_missing", region_fill),
            PipelineStep("one_hot_encode", vocabulary),
            PipelineStep("scale_features", scaling),
        )
    )
    params_dir = tmp_path / "params"
    save_params(scoring, params_dir / "scoring_pipeline.json")
    save_params(bounds, params_dir / "outlier_bounds.json")
    save_params(sales_fill, params_dir / "sales_fill.json")
    assert load_params(params_dir / "outlier_bounds.json", OutlierBounds) == bounds
    assert load_params(params_dir / "sales_fill.json", ImputeValues) == sales_fill
    scoring_reloaded = load_params(params_dir / "scoring_pipeline.json", Pipeline)
    assert scoring_reloaded == scoring
    assert [step.kind for step in scoring_reloaded.steps] == [
        "impute_missing",
        "one_hot_encode",
        "scale_features",
    ]

    # Fresh rows that never existed at fit time flow through the reloaded
    # pipeline to exactly the training feature columns: the missing region
    # takes train's modal fill and the unseen "central" encodes as all zeros.
    fresh = pd.DataFrame(
        {
            "date": pd.date_range("2025-06-01", periods=3, freq="D"),
            "region": ["north", None, "central"],
        }
    )
    fresh = add_datetime_features(fresh, "date", drop=False)
    fresh = scoring_reloaded.apply(fresh)
    feature_columns = [c for c in train.columns if c not in ("date", "total_sales")]
    assert set(feature_columns) <= set(fresh.columns)
    region_columns = [c for c in fresh.columns if c.startswith("region_")]
    assert fresh.loc[1, region_columns].sum() == 1  # filled with the train mode
    assert fresh.loc[2, region_columns].sum() == 0  # unseen category -> all zeros
