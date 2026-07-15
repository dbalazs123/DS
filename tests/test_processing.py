"""Tests for validation, preprocessing, eda and features."""

from __future__ import annotations

import pandas as pd
import pytest

from ds.eda import missing_value_report, summarize, top_correlations
from ds.features import add_datetime_features
from ds.preprocessing import drop_constant_columns, standardize_column_names
from ds.validation import DataValidationError, assert_no_nulls, require_columns


def test_require_columns_ok(sample_df: pd.DataFrame) -> None:
    assert require_columns(sample_df, ["Date", "category"]) is sample_df


def test_require_columns_missing_raises(sample_df: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="missing_col"):
        require_columns(sample_df, ["missing_col"])


def test_assert_no_nulls_detects_nulls() -> None:
    df = pd.DataFrame({"a": [1, None]})
    with pytest.raises(DataValidationError):
        assert_no_nulls(df)


def test_standardize_column_names(sample_df: pd.DataFrame) -> None:
    out = standardize_column_names(sample_df)
    assert "total_sales" in out.columns
    assert "date" in out.columns


def test_drop_constant_columns(sample_df: pd.DataFrame) -> None:
    out = drop_constant_columns(sample_df)
    assert "constant" not in out.columns
    assert "category" in out.columns


def test_summarize_shape(sample_df: pd.DataFrame) -> None:
    summary = summarize(sample_df)
    assert summary.index.name == "column"
    assert {"dtype", "non_null", "null_frac", "n_unique", "mean"} <= set(summary.columns)


def test_missing_value_report_lists_only_missing() -> None:
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    report = missing_value_report(df)
    assert list(report.index) == ["a"]
    assert report.loc["a", "n_missing"] == 1
    assert report.loc["a", "frac_missing"] == pytest.approx(1 / 3)


def test_missing_value_report_empty_when_complete() -> None:
    report = missing_value_report(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    assert report.empty
    assert list(report.columns) == ["n_missing", "frac_missing"]


def test_top_correlations_ranks_by_absolute_value() -> None:
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "neg": [4, 3, 2, 1],  # perfectly anti-correlated with x
            "noise": [1, 0, 1, 0],
        }
    )
    top = top_correlations(df, n=1)
    assert list(top.loc[0, ["feature_a", "feature_b"]]) == ["x", "neg"]
    assert top.loc[0, "correlation"] == pytest.approx(-1.0)


def test_top_correlations_handles_too_few_numeric_columns() -> None:
    out = top_correlations(pd.DataFrame({"only": [1, 2, 3], "text": ["a", "b", "c"]}))
    assert out.empty
    assert list(out.columns) == ["feature_a", "feature_b", "correlation"]


def test_add_datetime_features(sample_df: pd.DataFrame) -> None:
    out = add_datetime_features(sample_df, "Date")
    assert out["Date_year"].iloc[0] == 2024
    # 2024-01-06 is a Saturday.
    assert bool(out["Date_is_weekend"].iloc[1]) is True


def test_add_datetime_features_missing_column(sample_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        add_datetime_features(sample_df, "nope")
