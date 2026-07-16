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


def test_example_pipeline_cleans_and_encodes(tmp_path: Path) -> None:
    example = _load_example()

    from ds import Settings, seed_everything
    from ds.features import add_datetime_features, one_hot_encode, scale_features
    from ds.io import load_raw, save_table
    from ds.preprocessing import (
        clip_outliers,
        coerce_dtypes,
        drop_constant_columns,
        drop_duplicate_rows,
        impute_missing,
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

    lower, upper = df["total_sales"].min(), df["total_sales"].max()
    df = clip_outliers(df, columns=["total_sales"])
    # Clipping should have pulled the injected spikes back inside the original range.
    assert df["total_sales"].max() <= upper
    assert df["total_sales"].min() >= lower

    df = impute_missing(df, columns=["total_sales"], strategy="median")
    df = impute_missing(df, columns=["region"], strategy="most_frequent")
    assert df.isna().sum().sum() == 0

    df = add_datetime_features(df, "date", drop=False)
    df = one_hot_encode(df, columns=["region"])
    encoded_columns = [c for c in df.columns if c.startswith("region_")]
    assert encoded_columns
    assert "region" not in df.columns

    df = scale_features(df, columns=["date_year", "date_month", "date_day", "date_dayofweek"])
    assert abs(df["date_month"].mean()) < 1e-6
