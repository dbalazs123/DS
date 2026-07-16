"""Tests for validation, preprocessing, eda and features."""

from __future__ import annotations

import pandas as pd
import pytest

from ds.eda import missing_value_report, summarize, top_correlations
from ds.features import (
    add_datetime_features,
    bin_column,
    one_hot_encode,
    ordinal_encode,
    scale_features,
)
from ds.preprocessing import (
    clip_outliers,
    coerce_dtypes,
    drop_constant_columns,
    drop_duplicate_rows,
    flag_outliers,
    impute_missing,
    standardize_column_names,
)
from ds.validation import (
    DataValidationError,
    assert_dtypes,
    assert_in_range,
    assert_in_set,
    assert_no_nulls,
    check_schema,
    require_columns,
)


def test_require_columns_ok(sample_df: pd.DataFrame) -> None:
    assert require_columns(sample_df, ["Date", "category"]) is sample_df


def test_require_columns_missing_raises(sample_df: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="missing_col"):
        require_columns(sample_df, ["missing_col"])


def test_assert_no_nulls_detects_nulls() -> None:
    df = pd.DataFrame({"a": [1, None]})
    with pytest.raises(DataValidationError):
        assert_no_nulls(df)


def test_assert_in_range_ok_ignores_nulls() -> None:
    df = pd.DataFrame({"x": [1.0, 5.0, None]})
    assert assert_in_range(df, "x", min_value=0, max_value=10) is df


def test_assert_in_range_flags_out_of_bounds() -> None:
    df = pd.DataFrame({"x": [1, 2, 11]})
    with pytest.raises(DataValidationError, match="11"):
        assert_in_range(df, "x", max_value=10)


def test_assert_in_range_inclusive_neither() -> None:
    df = pd.DataFrame({"x": [0, 5, 10]})
    with pytest.raises(DataValidationError):
        assert_in_range(df, "x", min_value=0, max_value=10, inclusive="neither")


def test_assert_in_set_ok_and_offenders() -> None:
    df = pd.DataFrame({"s": ["a", "b", None]})
    assert assert_in_set(df, "s", ["a", "b"]) is df
    with pytest.raises(DataValidationError, match="'c'"):
        assert_in_set(pd.DataFrame({"s": ["a", "c"]}), "s", ["a", "b"])


def test_assert_in_range_missing_column_raises() -> None:
    with pytest.raises(KeyError):
        assert_in_range(pd.DataFrame({"a": [1]}), "nope", min_value=0)


def test_assert_dtypes_ok_and_mismatch() -> None:
    df = pd.DataFrame({"n": [1, 2], "s": ["a", "b"]})
    assert assert_dtypes(df, {"n": "int64"}) is df
    with pytest.raises(DataValidationError, match="expected"):
        assert_dtypes(df, {"n": "float64"})


def test_assert_dtypes_missing_column() -> None:
    with pytest.raises(DataValidationError, match="Missing"):
        assert_dtypes(pd.DataFrame({"a": [1]}), {"b": "int64"})


def test_check_schema_validates_and_coerces() -> None:
    df = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    out = check_schema(df, {"a": "int64", "b": "str"}, coerce=True)
    assert out["a"].dtype == "int64"


def test_check_schema_raises_on_bad_dtype() -> None:
    df = pd.DataFrame({"a": ["not-an-int"]})
    with pytest.raises(DataValidationError):
        check_schema(df, {"a": "int64"})


def test_check_schema_strict_rejects_extra_columns() -> None:
    df = pd.DataFrame({"a": [1], "extra": [2]})
    with pytest.raises(DataValidationError):
        check_schema(df, {"a": "int64"}, strict=True)


def test_standardize_column_names(sample_df: pd.DataFrame) -> None:
    out = standardize_column_names(sample_df)
    assert "total_sales" in out.columns
    assert "date" in out.columns


def test_drop_constant_columns(sample_df: pd.DataFrame) -> None:
    out = drop_constant_columns(sample_df)
    assert "constant" not in out.columns
    assert "category" in out.columns


def test_drop_duplicate_rows_default_keeps_first() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out = drop_duplicate_rows(df)
    assert len(out) == 2
    assert list(out["a"]) == [1, 2]


def test_drop_duplicate_rows_subset_and_keep_false() -> None:
    df = pd.DataFrame({"id": [1, 1, 2], "val": [10, 99, 20]})
    out = drop_duplicate_rows(df, ["id"], keep=False)
    # Both rows sharing id==1 are dropped when keep=False.
    assert list(out["id"]) == [2]


def test_drop_duplicate_rows_unknown_subset_raises() -> None:
    with pytest.raises(KeyError):
        drop_duplicate_rows(pd.DataFrame({"a": [1]}), ["nope"])


def test_coerce_dtypes_casts_columns() -> None:
    df = pd.DataFrame({"n": ["1", "2"], "c": ["a", "b"]})
    out = coerce_dtypes(df, {"n": "int64", "c": "category"})
    assert out["n"].dtype == "int64"
    assert out["c"].dtype == "category"


def test_coerce_dtypes_errors_ignore_leaves_column() -> None:
    df = pd.DataFrame({"n": ["1", "oops"]})
    out = coerce_dtypes(df, {"n": "int64"}, errors="ignore")
    assert out["n"].dtype == object


def test_coerce_dtypes_unknown_column_raises() -> None:
    with pytest.raises(KeyError):
        coerce_dtypes(pd.DataFrame({"a": [1]}), {"missing": "int64"})


def test_flag_outliers_iqr_detects_extreme_value() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100], "label": list("abcdef")})
    flags = flag_outliers(df)
    # Only the numeric column is checked and only the 100 is flagged.
    assert list(flags.columns) == ["x"]
    assert flags["x"].tolist() == [False, False, False, False, False, True]


def test_flag_outliers_zscore_and_ignores_nulls() -> None:
    df = pd.DataFrame({"x": [10.0, 10.0, 10.0, 10.0, 10.0, None, 1000.0]})
    flags = flag_outliers(df, method="zscore", factor=2.0)
    assert bool(flags["x"].iloc[6]) is True  # the 1000 is far from the mean
    assert bool(flags["x"].iloc[5]) is False  # NaN is never an outlier


def test_flag_outliers_non_numeric_column_raises() -> None:
    with pytest.raises(ValueError, match="non-numeric"):
        flag_outliers(pd.DataFrame({"c": ["a", "b"]}), ["c"])


def test_clip_outliers_winsorizes_without_dropping_rows() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
    out = clip_outliers(df)
    assert len(out) == len(df)
    assert out["x"].max() < 100  # the extreme value was pulled in


def test_impute_missing_mean_fills_numeric_only() -> None:
    df = pd.DataFrame({"x": [1.0, None, 3.0], "c": ["a", None, "a"]})
    out = impute_missing(df)  # default mean, numeric columns only
    assert out["x"].iloc[1] == pytest.approx(2.0)
    assert pd.isna(out["c"].iloc[1])  # non-numeric left untouched


def test_impute_missing_most_frequent_on_any_dtype() -> None:
    df = pd.DataFrame({"c": ["a", "a", None, "b"]})
    out = impute_missing(df, strategy="most_frequent")
    assert out["c"].iloc[2] == "a"


def test_impute_missing_constant_uses_fill_value() -> None:
    df = pd.DataFrame({"c": ["a", None]})
    out = impute_missing(df, ["c"], strategy="constant", fill_value="?")
    assert out["c"].iloc[1] == "?"


def test_impute_missing_mean_on_non_numeric_raises() -> None:
    with pytest.raises(ValueError, match="numeric"):
        impute_missing(pd.DataFrame({"c": ["a", None]}), ["c"], strategy="mean")


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


def test_one_hot_encode_replaces_categorical() -> None:
    df = pd.DataFrame({"n": [1, 2], "c": ["a", "b"]})
    out = one_hot_encode(df)
    assert "c" not in out.columns
    assert {"c_a", "c_b"} <= set(out.columns)
    assert "n" in out.columns  # numeric column untouched


def test_one_hot_encode_drop_first_and_unknown_column() -> None:
    df = pd.DataFrame({"c": ["a", "b", "c"]})
    out = one_hot_encode(df, drop_first=True)
    assert "c_a" not in out.columns  # first level dropped
    with pytest.raises(KeyError):
        one_hot_encode(df, ["nope"])


def test_ordinal_encode_respects_explicit_order() -> None:
    df = pd.DataFrame({"size": ["M", "S", "L", None]})
    out = ordinal_encode(df, categories={"size": ["S", "M", "L"]})
    assert out["size"].tolist() == [1, 0, 2, -1]  # NaN -> -1


def test_ordinal_encode_defaults_to_sorted_order() -> None:
    df = pd.DataFrame({"c": ["b", "a", "b"]})
    out = ordinal_encode(df)
    assert out["c"].tolist() == [1, 0, 1]


def test_scale_features_standard_zero_mean() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]})
    out = scale_features(df)
    assert out["x"].mean() == pytest.approx(0.0)
    assert out["c"].tolist() == ["a", "b", "c"]  # non-numeric untouched


def test_scale_features_minmax_and_constant_column() -> None:
    df = pd.DataFrame({"x": [10.0, 20.0, 30.0], "k": [5.0, 5.0, 5.0]})
    out = scale_features(df, method="minmax")
    assert out["x"].tolist() == [0.0, 0.5, 1.0]
    assert out["k"].tolist() == [0.0, 0.0, 0.0]  # constant -> zeros, not NaN


def test_scale_features_non_numeric_raises() -> None:
    with pytest.raises(ValueError, match="scaled"):
        scale_features(pd.DataFrame({"c": ["a", "b"]}), ["c"])


def test_bin_column_equal_width_adds_bin() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    out = bin_column(df, "x", bins=2, labels=["low", "high"])
    assert out["x_bin"].tolist() == ["low", "low", "high", "high"]


def test_bin_column_quantile() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8]})
    out = bin_column(df, "x", bins=4, method="quantile", drop=True)
    assert "x" not in out.columns
    assert out["x_bin"].nunique() == 4


def test_bin_column_missing_column_raises() -> None:
    with pytest.raises(KeyError):
        bin_column(pd.DataFrame({"a": [1]}), "nope", bins=2)
