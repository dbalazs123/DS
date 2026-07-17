"""Tests for validation, preprocessing, eda and features."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from ds.eda import missing_value_report, summarize, top_correlations
from ds.features import (
    OneHotCategories,
    OrdinalCategories,
    ScaleParams,
    TopKCategories,
    add_datetime_features,
    apply_collapse_categories,
    apply_one_hot_encode,
    apply_ordinal_encode,
    apply_scale_features,
    bin_column,
    collapse_categories,
    fit_one_hot_categories,
    fit_ordinal_categories,
    fit_scale_params,
    fit_topk_categories,
    one_hot_encode,
    ordinal_encode,
    scale_features,
)
from ds.preprocessing import (
    ImputeValues,
    OutlierBounds,
    apply_clip_outliers,
    apply_flag_outliers,
    apply_impute_missing,
    clip_outliers,
    coerce_dtypes,
    drop_constant_columns,
    drop_duplicate_rows,
    fit_impute_values,
    fit_outlier_bounds,
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


def test_fit_outlier_bounds_carries_train_bounds_to_new_data() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    bounds = fit_outlier_bounds(train, ["x"])
    # 50 is unremarkable within the test frame alone, but far outside the
    # train-fitted fence — split-safe application must still catch it.
    test = pd.DataFrame({"x": [50.0, 50.0, 50.0, 3.0, None]})
    flags = apply_flag_outliers(test, bounds)
    assert flags["x"].tolist() == [True, True, True, False, False]
    clipped = apply_clip_outliers(test, bounds)
    assert clipped["x"].max() == pytest.approx(bounds.bounds["x"][1])
    assert pd.isna(clipped["x"].iloc[4])  # missing values pass through


def test_apply_outlier_bounds_missing_column_raises() -> None:
    bounds = fit_outlier_bounds(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    with pytest.raises(KeyError):
        apply_clip_outliers(pd.DataFrame({"y": [1.0]}), bounds)
    with pytest.raises(KeyError):
        apply_flag_outliers(pd.DataFrame({"y": [1.0]}), bounds)


def test_outlier_wrappers_match_fit_apply() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
    bounds = fit_outlier_bounds(df)
    assert clip_outliers(df).equals(apply_clip_outliers(df, bounds))
    assert flag_outliers(df).equals(apply_flag_outliers(df, bounds))


def test_fit_impute_values_uses_train_statistics_on_test() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    values = fit_impute_values(train, ["x"], strategy="median")
    test = pd.DataFrame({"x": [100.0, None]})
    out = apply_impute_missing(test, values)
    # The gap is filled with train's median (2.0), not test's own values.
    assert out["x"].iloc[1] == pytest.approx(2.0)


def test_fit_impute_values_most_frequent_and_constant() -> None:
    train = pd.DataFrame({"c": ["a", "a", "b"]})
    values = fit_impute_values(train, strategy="most_frequent")
    assert apply_impute_missing(pd.DataFrame({"c": ["b", None]}), values)["c"].iloc[1] == "a"
    constant = fit_impute_values(train, ["c"], strategy="constant", fill_value="?")
    assert apply_impute_missing(pd.DataFrame({"c": [None]}), constant)["c"].iloc[0] == "?"


def test_apply_impute_missing_unknown_column_raises() -> None:
    values = fit_impute_values(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(KeyError):
        apply_impute_missing(pd.DataFrame({"y": [1.0]}), values)


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
    # Date-only data expands to hour 0; intraday data keeps its hour.
    assert out["Date_hour"].iloc[0] == 0
    intraday = pd.DataFrame({"t": pd.to_datetime(["2024-01-01 17:45:00"])})
    assert add_datetime_features(intraday, "t")["t_hour"].iloc[0] == 17


def test_add_datetime_features_missing_column(sample_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        add_datetime_features(sample_df, "nope")


def test_add_datetime_features_emits_only_the_selected_subset(sample_df: pd.DataFrame) -> None:
    out = add_datetime_features(sample_df, "Date", features=["month", "year"])
    added = [col for col in out.columns if col not in sample_df.columns]
    # Emitted in the documented order, regardless of the order requested.
    assert added == ["Date_year", "Date_month"]


def test_add_datetime_features_elapsed_months_is_monotone_and_stateless() -> None:
    df = pd.DataFrame({"t": pd.to_datetime(["1949-12-01", "1950-01-15", "1951-01-01"])})
    out = add_datetime_features(df, "t", features=["elapsed_months"])
    elapsed = out["t_elapsed_months"]
    # +1 across the year boundary, +12 across a full year; the day is ignored.
    assert (elapsed - elapsed.iloc[0]).tolist() == [0, 1, 13]
    # The origin is a fixed epoch, not learned from the frame, so the same
    # month scores identically in a later, disjoint batch.
    later = pd.DataFrame({"t": pd.to_datetime(["1950-01-31"])})
    rescored = add_datetime_features(later, "t", features=["elapsed_months"])
    assert rescored["t_elapsed_months"].iloc[0] == elapsed.iloc[1]
    # Opt-in only: the default emission stays the calendar-position set.
    assert "t_elapsed_months" not in add_datetime_features(df, "t").columns


def test_add_datetime_features_rejects_a_bad_selection(sample_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown datetime features"):
        add_datetime_features(sample_df, "Date", features=["decade"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="at least one"):
        add_datetime_features(sample_df, "Date", features=[])


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


def test_fit_one_hot_categories_fixes_vocabulary_across_splits() -> None:
    train = pd.DataFrame({"c": ["a", "b", "c"]})
    vocabulary = fit_one_hot_categories(train, ["c"])
    # Test lacks "b" entirely and brings the unseen "z".
    test = pd.DataFrame({"c": ["a", "z"]})
    enc_train = apply_one_hot_encode(train, vocabulary)
    enc_test = apply_one_hot_encode(test, vocabulary)
    assert list(enc_train.columns) == list(enc_test.columns) == ["c_a", "c_b", "c_c"]
    # The unseen category encodes as all zeros rather than a new column.
    assert enc_test.iloc[1].tolist() == [False, False, False]


def test_apply_one_hot_encode_dummy_na_and_missing_column() -> None:
    train = pd.DataFrame({"c": ["a", None]})
    vocabulary = fit_one_hot_categories(train, ["c"], dummy_na=True)
    out = apply_one_hot_encode(pd.DataFrame({"c": [None, "a"]}), vocabulary)
    assert bool(out["c_nan"].iloc[0]) is True
    with pytest.raises(KeyError):
        apply_one_hot_encode(pd.DataFrame({"other": [1]}), vocabulary)


def test_one_hot_wrapper_matches_fit_apply_incl_category_dtype() -> None:
    df = pd.DataFrame({"c": pd.Categorical(["a", "b"], categories=["b", "a", "unused"])})
    vocabulary = fit_one_hot_categories(df, ["c"])
    # Declared categories (order and unused levels) are respected, matching
    # what pandas.get_dummies does for the single-call form.
    assert vocabulary.categories["c"] == ("b", "a", "unused")
    assert one_hot_encode(df, ["c"]).equals(apply_one_hot_encode(df, vocabulary))


def test_fit_ordinal_categories_unseen_encodes_minus_one() -> None:
    train = pd.DataFrame({"size": ["M", "S", "L"]})
    order = fit_ordinal_categories(train, categories={"size": ["S", "M", "L"]})
    out = apply_ordinal_encode(pd.DataFrame({"size": ["L", "XL", None]}), order)
    assert out["size"].tolist() == [2, -1, -1]
    with pytest.raises(KeyError):
        apply_ordinal_encode(pd.DataFrame({"other": ["x"]}), order)


def test_ordinal_wrapper_matches_fit_apply() -> None:
    df = pd.DataFrame({"c": ["b", "a", "b"]})
    assert ordinal_encode(df).equals(apply_ordinal_encode(df, fit_ordinal_categories(df)))


def test_fit_topk_categories_keeps_most_frequent_with_deterministic_ties() -> None:
    df = pd.DataFrame({"zone": ["a", "a", "a", "c", "c", "b", "b", "d", None]})
    params = fit_topk_categories(df, k=2)
    # "b" and "c" tie on count; ascending value breaks the tie, nulls don't count.
    assert params.categories == {"zone": ("a", "b")}
    assert params.other_label == "other"


def test_apply_collapse_categories_collapses_rare_and_unseen_keeps_nulls() -> None:
    train = pd.DataFrame({"zone": ["a", "a", "b", "b", "c", "d"]})
    params = fit_topk_categories(train, ["zone"], k=2)
    test = pd.DataFrame({"zone": ["a", "d", "never_seen", None]})
    out = apply_collapse_categories(test, params)
    # Rare-at-fit and unseen values both collapse; missing stays missing for
    # the imputation step to handle.
    assert out["zone"].tolist()[:3] == ["a", "other", "other"]
    assert pd.isna(out["zone"].iloc[3])
    with pytest.raises(KeyError):
        apply_collapse_categories(pd.DataFrame({"other": ["x"]}), params)


def test_collapsed_column_feeds_one_hot_with_a_closed_vocabulary() -> None:
    # The intended composition for high-cardinality columns: collapse to
    # top-k + "other", then one-hot the now-small vocabulary.
    train = pd.DataFrame({"zone": ["a", "a", "b", "c", "d", "e"]})
    params = fit_topk_categories(train, ["zone"], k=1)
    collapsed_train = apply_collapse_categories(train, params)
    vocabulary = fit_one_hot_categories(collapsed_train, ["zone"])
    assert vocabulary.categories["zone"] == ("a", "other")
    test = apply_collapse_categories(pd.DataFrame({"zone": ["z", "a"]}), params)
    encoded = apply_one_hot_encode(test, vocabulary)
    assert list(encoded.columns) == ["zone_a", "zone_other"]
    assert encoded["zone_other"].tolist() == [True, False]


def test_collapse_wrapper_matches_fit_apply_incl_category_dtype() -> None:
    df = pd.DataFrame({"c": pd.Categorical(["a", "a", "b", None])})
    params = fit_topk_categories(df, ["c"], k=1)
    applied = apply_collapse_categories(df, params)
    assert collapse_categories(df, ["c"], k=1).equals(applied)
    assert applied["c"].dtype == object  # category dtype recoded to plain object


def test_fit_topk_categories_small_column_and_bad_arguments() -> None:
    df = pd.DataFrame({"c": ["a", "b"]})
    # Fewer distinct values than k: everything is kept.
    assert fit_topk_categories(df, k=10).categories == {"c": ("a", "b")}
    with pytest.raises(ValueError, match="at least 1"):
        fit_topk_categories(df, k=0)
    with pytest.raises(ValueError, match="kept category"):
        fit_topk_categories(pd.DataFrame({"c": ["other", "x"]}), k=2)
    with pytest.raises(KeyError):
        fit_topk_categories(df, ["nope"], k=1)


def test_fit_scale_params_scales_test_with_train_statistics() -> None:
    train = pd.DataFrame({"x": [0.0, 10.0]})
    params = fit_scale_params(train, ["x"], method="minmax")
    out = apply_scale_features(pd.DataFrame({"x": [5.0, 20.0]}), params)
    # Test values scale against train's range: beyond it maps past 1 honestly.
    assert out["x"].tolist() == [0.5, 2.0]


def test_apply_scale_features_constant_fit_column_and_missing_column() -> None:
    params = fit_scale_params(pd.DataFrame({"k": [5.0, 5.0]}), ["k"])
    out = apply_scale_features(pd.DataFrame({"k": [7.0, 9.0]}), params)
    assert out["k"].tolist() == [0.0, 0.0]  # constant at fit time -> zeros
    with pytest.raises(KeyError):
        apply_scale_features(pd.DataFrame({"other": [1.0]}), params)


def test_scale_wrapper_matches_fit_apply() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    standard = fit_scale_params(df, method="standard")
    assert scale_features(df, method="standard").equals(apply_scale_features(df, standard))
    minmax = fit_scale_params(df, method="minmax")
    assert scale_features(df, method="minmax").equals(apply_scale_features(df, minmax))


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


# --- Persistable fit parameters: to_dict / from_dict round-trips ---


def _through_json(data: dict[str, object]) -> dict[str, object]:
    """Round-trip a payload through strict JSON, as ds.io.save_params does."""
    loaded = json.loads(json.dumps(data, allow_nan=False))
    assert isinstance(loaded, dict)
    return loaded


def test_outlier_bounds_round_trips_including_non_finite() -> None:
    # An all-null fit yields (-inf, inf) bounds, which strict JSON cannot hold
    # as literals — the tagged encoding must carry them through anyway.
    bounds = fit_outlier_bounds(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "empty": [np.nan] * 4}), method="zscore"
    )
    assert bounds.bounds["empty"] == (-np.inf, np.inf)
    restored = OutlierBounds.from_dict(_through_json(bounds.to_dict()))
    assert restored == bounds
    df = pd.DataFrame({"x": [0.0, 2.5, 99.0], "empty": [1.0, 2.0, 3.0]})
    assert apply_clip_outliers(df, restored).equals(apply_clip_outliers(df, bounds))


def test_impute_values_round_trips_numpy_scalar_fills() -> None:
    values = fit_impute_values(pd.DataFrame({"x": [1.0, 2.0, np.nan]}), strategy="median")
    assert isinstance(values.fill_values["x"], np.floating)  # the raw fit is a numpy scalar
    payload = _through_json(values.to_dict())
    restored = ImputeValues.from_dict(payload)
    assert isinstance(restored.fill_values["x"], float)  # reloaded as a plain float
    assert restored.fill_values["x"] == values.fill_values["x"]
    assert restored.strategy == values.strategy
    df = pd.DataFrame({"x": [np.nan, 5.0]})
    assert apply_impute_missing(df, restored).equals(apply_impute_missing(df, values))


def test_impute_values_round_trips_nan_fill() -> None:
    # A NaN fill (e.g. the mean of an all-null column) has no strict-JSON
    # literal either; it survives via the same tagged encoding.
    values = ImputeValues(fill_values={"x": float("nan")}, strategy="constant")
    restored = ImputeValues.from_dict(_through_json(values.to_dict()))
    fill = restored.fill_values["x"]
    assert isinstance(fill, float) and math.isnan(fill)


def test_scale_params_round_trips_both_methods() -> None:
    df = pd.DataFrame({"x": [10.0, 20.0, 30.0], "k": [5.0, 5.0, 5.0]})
    for method in ("standard", "minmax"):
        params = fit_scale_params(df, method=method)  # type: ignore[arg-type]
        restored = ScaleParams.from_dict(_through_json(params.to_dict()))
        assert restored == params
        assert apply_scale_features(df, restored).equals(apply_scale_features(df, params))


def test_one_hot_categories_round_trips_non_string_categories() -> None:
    # Vocabularies fitted from data hold numpy scalars and non-string values;
    # the reload must normalize them to plain Python and restore the tuples.
    categories = fit_one_hot_categories(
        pd.DataFrame({"c": ["b", "a"], "n": pd.array([2, 1], dtype="int64")}),
        columns=["c", "n"],
        drop_first=True,
        dummy_na=True,
    )
    restored = OneHotCategories.from_dict(_through_json(categories.to_dict()))
    assert restored == categories
    assert all(isinstance(cats, tuple) for cats in restored.categories.values())
    assert restored.categories["n"] == (1, 2)
    assert all(type(cat) is int for cat in restored.categories["n"])
    df = pd.DataFrame({"c": ["a", "zzz"], "n": [1, 3]})
    assert apply_one_hot_encode(df, restored).equals(apply_one_hot_encode(df, categories))


def test_topk_categories_round_trips_non_string_categories() -> None:
    params = fit_topk_categories(
        pd.DataFrame({"c": ["b", "a", "a"], "n": pd.array([2, 1, 1], dtype="int64")}),
        columns=["c", "n"],
        k=1,
    )
    restored = TopKCategories.from_dict(_through_json(params.to_dict()))
    assert restored == params
    assert all(isinstance(cats, tuple) for cats in restored.categories.values())
    assert type(restored.categories["n"][0]) is int
    df = pd.DataFrame({"c": ["a", "zzz"], "n": [1, 3]})
    assert apply_collapse_categories(df, restored).equals(apply_collapse_categories(df, params))


def test_ordinal_categories_round_trips() -> None:
    order = fit_ordinal_categories(
        pd.DataFrame({"size": ["M", "S"]}), categories={"size": ["S", "M", "L"]}
    )
    restored = OrdinalCategories.from_dict(_through_json(order.to_dict()))
    assert restored == order
    df = pd.DataFrame({"size": ["L", "S", "XL", None]})
    assert apply_ordinal_encode(df, restored).equals(apply_ordinal_encode(df, order))


def test_from_dict_rejects_malformed_payloads() -> None:
    good = OutlierBounds(bounds={"x": (0.0, 1.0)}, method="iqr", factor=1.5).to_dict()

    with pytest.raises(ValueError, match="must be a mapping"):
        OutlierBounds.from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected a 'OutlierBounds' payload"):
        OutlierBounds.from_dict({**good, "type": "ScaleParams"})
    with pytest.raises(ValueError, match="missing fields"):
        OutlierBounds.from_dict({k: v for k, v in good.items() if k != "factor"})
    with pytest.raises(ValueError, match="unexpected fields"):
        OutlierBounds.from_dict({**good, "stale_field": 1})
    with pytest.raises(ValueError, match="method"):
        OutlierBounds.from_dict({**good, "method": "mad"})
    with pytest.raises(ValueError, match="pair"):
        OutlierBounds.from_dict({**good, "bounds": {"x": [1.0]}})
    with pytest.raises(ValueError, match="number"):
        OutlierBounds.from_dict({**good, "factor": True})
    with pytest.raises(ValueError, match="tagged non-finite float"):
        OutlierBounds.from_dict({**good, "bounds": {"x": [{"__float__": "huge"}, 1.0]}})

    with pytest.raises(ValueError, match="strategy"):
        ImputeValues.from_dict({"type": "ImputeValues", "fill_values": {}, "strategy": "modal"})
    scale = ScaleParams(center={"x": 0.0}, spread={"x": 1.0}, method="standard").to_dict()
    with pytest.raises(ValueError, match="same columns"):
        ScaleParams.from_dict({**scale, "spread": {"y": 1.0}})
    one_hot = OneHotCategories(categories={"c": ("a",)}, drop_first=False, dummy_na=False)
    with pytest.raises(ValueError, match="bool"):
        OneHotCategories.from_dict({**one_hot.to_dict(), "drop_first": 1})
    with pytest.raises(ValueError, match="list of categories"):
        OrdinalCategories.from_dict({"type": "OrdinalCategories", "categories": {"c": "abc"}})
    topk = TopKCategories(categories={"c": ("a",)}, other_label="other").to_dict()
    with pytest.raises(ValueError, match="string"):
        TopKCategories.from_dict({**topk, "other_label": 3})
    with pytest.raises(ValueError, match="silently merge"):
        TopKCategories.from_dict({**topk, "other_label": "a"})
