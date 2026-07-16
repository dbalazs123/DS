"""Feature engineering across tabular, time-series and text data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from ds._serde import (
    as_bool,
    as_float,
    check_payload,
    check_str_mapping,
    decode_scalar,
    encode_scalar,
)

ScaleMethod = Literal["standard", "minmax"]
BinMethod = Literal["width", "quantile"]


@dataclass(frozen=True)
class ScaleParams:
    """Per-column centre/spread learned by :func:`fit_scale_params`.

    A column scales as ``(value - center) / spread``: for ``"standard"`` the
    centre is the mean and the spread the (sample) standard deviation, for
    ``"minmax"`` the centre is the minimum and the spread the value range.

    Attributes:
        center: Mapping of column name to the value subtracted before dividing.
        spread: Mapping of column name to the divisor (``0.0`` marks a constant
            column, which scales to all zeros).
        method: The method the parameters were fitted with.
    """

    center: Mapping[str, float]
    spread: Mapping[str, float]
    method: ScaleMethod

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation.

        Persist the result with :func:`ds.io.save_params` or rebuild the
        dataclass with :meth:`from_dict`.

        Returns:
            A dict that round-trips through :meth:`from_dict`.
        """
        return {
            "type": "ScaleParams",
            "center": {col: encode_scalar(value) for col, value in self.center.items()},
            "spread": {col: encode_scalar(value) for col, value in self.spread.items()},
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScaleParams:
        """Rebuild a :class:`ScaleParams` from :meth:`to_dict` output.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The reconstructed :class:`ScaleParams`.

        Raises:
            ValueError: If ``data`` is not a well-formed ``ScaleParams``
                payload (wrong type tag, missing/unexpected fields, an unknown
                method, or centre/spread naming different columns) — e.g. a
                stale or hand-edited file.
        """
        payload = check_payload(data, "ScaleParams", frozenset({"center", "spread", "method"}))
        method = payload["method"]
        if method not in ("standard", "minmax"):
            raise ValueError(f"ScaleParams.method must be 'standard' or 'minmax', got {method!r}")
        center = {
            col: as_float(decode_scalar(value), f"center[{col!r}]", "ScaleParams")
            for col, value in check_str_mapping(payload["center"], "center", "ScaleParams").items()
        }
        spread = {
            col: as_float(decode_scalar(value), f"spread[{col!r}]", "ScaleParams")
            for col, value in check_str_mapping(payload["spread"], "spread", "ScaleParams").items()
        }
        if set(center) != set(spread):
            raise ValueError("ScaleParams.center and .spread must name the same columns")
        return cls(center=center, spread=spread, method=method)


@dataclass(frozen=True)
class OneHotCategories:
    """Per-column category vocabulary learned by :func:`fit_one_hot_categories`.

    Attributes:
        categories: Mapping of column name to its ordered category vocabulary.
        drop_first: Whether the first level of each column is dropped.
        dummy_na: Whether an indicator column is added for missing values.
    """

    categories: Mapping[str, tuple[Any, ...]]
    drop_first: bool
    dummy_na: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation.

        Category values must be JSON-representable scalars
        (str/int/float/bool/``None``); numpy scalars are unwrapped and the
        tuples become lists (restored by :meth:`from_dict`). Persist the
        result with :func:`ds.io.save_params` or rebuild the dataclass with
        :meth:`from_dict`.

        Returns:
            A dict that round-trips through :meth:`from_dict`.
        """
        return {
            "type": "OneHotCategories",
            "categories": _encode_categories(self.categories),
            "drop_first": self.drop_first,
            "dummy_na": self.dummy_na,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OneHotCategories:
        """Rebuild a :class:`OneHotCategories` from :meth:`to_dict` output.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The reconstructed :class:`OneHotCategories`.

        Raises:
            ValueError: If ``data`` is not a well-formed ``OneHotCategories``
                payload (wrong type tag, missing/unexpected fields, non-bool
                flags, or a malformed vocabulary) — e.g. a stale or
                hand-edited file.
        """
        payload = check_payload(
            data, "OneHotCategories", frozenset({"categories", "drop_first", "dummy_na"})
        )
        return cls(
            categories=_decode_categories(payload["categories"], "OneHotCategories"),
            drop_first=as_bool(payload["drop_first"], "drop_first", "OneHotCategories"),
            dummy_na=as_bool(payload["dummy_na"], "dummy_na", "OneHotCategories"),
        )


@dataclass(frozen=True)
class OrdinalCategories:
    """Per-column category order learned by :func:`fit_ordinal_categories`.

    Attributes:
        categories: Mapping of column name to its ordered category vocabulary;
            a value's code is its position in this order.
    """

    categories: Mapping[str, tuple[Any, ...]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation.

        Category values must be JSON-representable scalars
        (str/int/float/bool/``None``); numpy scalars are unwrapped and the
        tuples become lists (restored by :meth:`from_dict`). Persist the
        result with :func:`ds.io.save_params` or rebuild the dataclass with
        :meth:`from_dict`.

        Returns:
            A dict that round-trips through :meth:`from_dict`.
        """
        return {"type": "OrdinalCategories", "categories": _encode_categories(self.categories)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OrdinalCategories:
        """Rebuild an :class:`OrdinalCategories` from :meth:`to_dict` output.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The reconstructed :class:`OrdinalCategories`.

        Raises:
            ValueError: If ``data`` is not a well-formed ``OrdinalCategories``
                payload (wrong type tag, missing/unexpected fields, or a
                malformed vocabulary) — e.g. a stale or hand-edited file.
        """
        payload = check_payload(data, "OrdinalCategories", frozenset({"categories"}))
        return cls(categories=_decode_categories(payload["categories"], "OrdinalCategories"))


def _encode_categories(categories: Mapping[str, tuple[Any, ...]]) -> dict[str, list[Any]]:
    """Encode a per-column vocabulary as JSON-safe lists of scalars."""
    return {col: [encode_scalar(cat) for cat in cats] for col, cats in categories.items()}


def _decode_categories(value: Any, type_name: str) -> dict[str, tuple[Any, ...]]:
    """Decode and validate a per-column vocabulary back into tuples."""
    decoded: dict[str, tuple[Any, ...]] = {}
    for col, cats in check_str_mapping(value, "categories", type_name).items():
        if not isinstance(cats, Sequence) or isinstance(cats, str):
            raise ValueError(f"{type_name}.categories[{col!r}] must be a list of categories")
        decoded[col] = tuple(decode_scalar(cat) for cat in cats)
    return decoded


def add_datetime_features(df: pd.DataFrame, column: str, *, drop: bool = False) -> pd.DataFrame:
    """Expand a datetime column into calendar features.

    Adds ``<column>_year``, ``_month``, ``_day``, ``_dayofweek``, ``_hour``
    and ``_is_weekend`` columns — the workhorse features for most time-series
    and tabular models with a temporal component. Hour of day is included
    because it carries the dominant signal for intraday data (a lesson from
    the real-data taxi-fare project); on date-only data it is constantly zero,
    where :func:`ds.preprocessing.drop_constant_columns` removes it.

    Args:
        df: The source DataFrame.
        column: Name of a datetime (or datetime-parseable) column.
        drop: If ``True``, drop the original column from the result.

    Returns:
        A new DataFrame with the added feature columns.

    Raises:
        KeyError: If ``column`` is not present.
    """
    if column not in df.columns:
        raise KeyError(column)
    out = df.copy()
    ts = pd.to_datetime(out[column])
    out[f"{column}_year"] = ts.dt.year
    out[f"{column}_month"] = ts.dt.month
    out[f"{column}_day"] = ts.dt.day
    out[f"{column}_dayofweek"] = ts.dt.dayofweek
    out[f"{column}_hour"] = ts.dt.hour
    out[f"{column}_is_weekend"] = ts.dt.dayofweek.isin((5, 6))
    if drop:
        out = out.drop(columns=[column])
    return out


def _categorical_columns(df: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    """Resolve which columns to encode, validating any explicit names."""
    if columns is None:
        return list(df.select_dtypes(include=["object", "category"]).columns)
    resolved = list(columns)
    missing = [col for col in resolved if col not in df.columns]
    if missing:
        raise KeyError(missing)
    return resolved


def fit_one_hot_categories(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    drop_first: bool = False,
    dummy_na: bool = False,
) -> OneHotCategories:
    """Learn the category vocabulary for one-hot encoding without applying it.

    Fixes each column's category set from one frame — typically the training
    split — so :func:`apply_one_hot_encode` produces the same indicator columns
    on every frame, even when a category is absent from (or new in) the data
    being encoded. A ``category``-dtype column contributes its declared
    categories (in their declared order); any other column contributes its
    sorted unique non-null values.

    Args:
        df: The DataFrame to learn categories from (typically the training
            split).
        columns: Columns to fit; ``None`` fits every ``object``/``category``
            column.
        drop_first: Drop the first level of each column to avoid collinearity.
        dummy_na: Add an indicator column for missing values.

    Returns:
        The learned :class:`OneHotCategories`.

    Raises:
        KeyError: If a name in ``columns`` is not a column of ``df``.
    """
    resolved = _categorical_columns(df, columns)
    categories = {
        col: (
            tuple(df[col].cat.categories)
            if isinstance(df[col].dtype, pd.CategoricalDtype)
            else tuple(sorted(df[col].dropna().unique()))
        )
        for col in resolved
    }
    return OneHotCategories(categories=categories, drop_first=drop_first, dummy_na=dummy_na)


def apply_one_hot_encode(df: pd.DataFrame, categories: OneHotCategories) -> pd.DataFrame:
    """One-hot encode with a previously learned category vocabulary.

    The split-safe counterpart of :func:`one_hot_encode`: the indicator column
    set is fixed by :func:`fit_one_hot_categories`, so train and test frames
    encode to identical columns. A category unseen at fit time produces
    all-zero indicators (it is treated as missing when ``dummy_na`` is set).

    Args:
        df: The DataFrame to encode.
        categories: Vocabulary learned by :func:`fit_one_hot_categories`.

    Returns:
        A new DataFrame with the fitted columns replaced by indicators.

    Raises:
        KeyError: If a fitted column is not a column of ``df``.
    """
    missing = [col for col in categories.categories if col not in df.columns]
    if missing:
        raise KeyError(missing)
    out = df.copy()
    for col, cats in categories.categories.items():
        out[col] = pd.Categorical(out[col], categories=list(cats))
    return pd.get_dummies(
        out,
        columns=list(categories.categories),
        drop_first=categories.drop_first,
        dummy_na=categories.dummy_na,
    )


def one_hot_encode(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    drop_first: bool = False,
    dummy_na: bool = False,
) -> pd.DataFrame:
    """One-hot encode categorical columns into indicator columns.

    Convenience wrapper that infers each column's categories from ``df`` and
    encodes the same frame — fine for exploration or pre-split use. For a
    train/test workflow use :func:`fit_one_hot_categories` +
    :func:`apply_one_hot_encode` so both sides share one vocabulary. New
    columns are named ``<column>_<value>``; numeric columns are untouched.

    Args:
        df: The source DataFrame.
        columns: Columns to encode; ``None`` encodes every ``object``/``category``
            column.
        drop_first: Drop the first level of each column to avoid collinearity.
        dummy_na: Add an indicator column for missing values.

    Returns:
        A new DataFrame with the selected columns replaced by indicators.

    Raises:
        KeyError: If a name in ``columns`` is not a column of ``df``.
    """
    return apply_one_hot_encode(
        df, fit_one_hot_categories(df, columns, drop_first=drop_first, dummy_na=dummy_na)
    )


def fit_ordinal_categories(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    categories: Mapping[str, Sequence[object]] | None = None,
) -> OrdinalCategories:
    """Learn the category order for ordinal encoding without applying it.

    Fixes each column's category order — the one supplied in ``categories`` or,
    by default, the sorted unique non-null values — from one frame, typically
    the training split, so :func:`apply_ordinal_encode` assigns the same codes
    on every frame.

    Args:
        df: The DataFrame to learn categories from (typically the training
            split).
        columns: Columns to fit; ``None`` fits every ``object``/``category``
            column.
        categories: Optional per-column ordering of the categories, giving the
            codes a meaningful rank (e.g. ``{"size": ["S", "M", "L"]}``).

    Returns:
        The learned :class:`OrdinalCategories`.

    Raises:
        KeyError: If a name in ``columns`` is not a column of ``df``.
    """
    resolved = _categorical_columns(df, columns)
    lookup = categories or {}
    fitted = {
        col: tuple(lookup[col]) if col in lookup else tuple(sorted(df[col].dropna().unique()))
        for col in resolved
    }
    return OrdinalCategories(categories=fitted)


def apply_ordinal_encode(df: pd.DataFrame, categories: OrdinalCategories) -> pd.DataFrame:
    """Ordinal encode with a previously learned category order.

    The split-safe counterpart of :func:`ordinal_encode`: each value maps to
    its position in the order fixed by :func:`fit_ordinal_categories`. Missing
    values — and categories unseen at fit time — encode as ``-1``, matching
    :class:`pandas.Categorical`.

    Args:
        df: The DataFrame to encode.
        categories: Category order learned by :func:`fit_ordinal_categories`.

    Returns:
        A new DataFrame with the fitted columns replaced by integer codes.

    Raises:
        KeyError: If a fitted column is not a column of ``df``.
    """
    missing = [col for col in categories.categories if col not in df.columns]
    if missing:
        raise KeyError(missing)
    out = df.copy()
    for col, order in categories.categories.items():
        codes = pd.Categorical(df[col], categories=list(order), ordered=True).codes
        out[col] = pd.Series(codes, index=df.index, dtype="int64")
    return out


def ordinal_encode(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    categories: Mapping[str, Sequence[object]] | None = None,
) -> pd.DataFrame:
    """Encode categorical columns as integer codes.

    Convenience wrapper that infers each column's category order from ``df``
    (or takes it from ``categories``) and encodes the same frame — fine for
    exploration or pre-split use. For a train/test workflow use
    :func:`fit_ordinal_categories` + :func:`apply_ordinal_encode` so both sides
    share one order. Missing or unseen values encode as ``-1``, matching
    :class:`pandas.Categorical`.

    Args:
        df: The source DataFrame.
        columns: Columns to encode; ``None`` encodes every ``object``/``category``
            column.
        categories: Optional per-column ordering of the categories, giving the
            codes a meaningful rank (e.g. ``{"size": ["S", "M", "L"]}``).

    Returns:
        A new DataFrame with the selected columns replaced by integer codes.

    Raises:
        KeyError: If a name in ``columns`` is not a column of ``df``.
    """
    return apply_ordinal_encode(df, fit_ordinal_categories(df, columns, categories=categories))


def _numeric_columns(df: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    """Resolve the numeric columns to scale, validating any explicit names."""
    if columns is None:
        return list(df.select_dtypes("number").columns)
    resolved = list(columns)
    missing = [col for col in resolved if col not in df.columns]
    if missing:
        raise KeyError(missing)
    non_numeric = [col for col in resolved if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"non-numeric columns cannot be scaled: {non_numeric}")
    return resolved


def fit_scale_params(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    method: ScaleMethod = "standard",
) -> ScaleParams:
    """Learn per-column centre and spread without applying them.

    Fit on the training split, then reuse the parameters on test data or new
    rows via :func:`apply_scale_features` so held-out data never influences
    the centre or spread. ``"standard"`` learns mean and (sample) standard
    deviation; ``"minmax"`` learns minimum and value range.

    Args:
        df: The DataFrame to learn parameters from (typically the training
            split).
        columns: Numeric columns to fit; ``None`` uses every numeric column.
        method: ``"standard"`` (z-score) or ``"minmax"``.

    Returns:
        The learned :class:`ScaleParams`.

    Raises:
        KeyError: If a name in ``columns`` is not a column of ``df``.
        ValueError: If a named column is not numeric.
    """
    resolved = _numeric_columns(df, columns)
    center: dict[str, float] = {}
    spread: dict[str, float] = {}
    for col in resolved:
        series = df[col].astype(float)
        if method == "standard":
            center[col] = float(series.mean())
            spread[col] = float(series.std())
        else:
            low = float(series.min())
            center[col] = low
            spread[col] = float(series.max()) - low
    return ScaleParams(center=center, spread=spread, method=method)


def apply_scale_features(df: pd.DataFrame, params: ScaleParams) -> pd.DataFrame:
    """Scale columns with previously learned centre and spread.

    The split-safe counterpart of :func:`scale_features`: each fitted column
    becomes ``(value - center) / spread`` using the parameters learned by
    :func:`fit_scale_params`, so test values beyond the training range scale
    honestly (e.g. above ``1`` for ``"minmax"``). A column that was constant at
    fit time maps to all zeros rather than producing ``inf``/``NaN``.

    Args:
        df: The DataFrame to scale.
        params: Parameters learned by :func:`fit_scale_params`.

    Returns:
        A new DataFrame with the fitted columns scaled.

    Raises:
        KeyError: If a fitted column is not a column of ``df``.
        ValueError: If a fitted column is not numeric in ``df``.
    """
    resolved = _numeric_columns(df, list(params.center))
    out = df.copy()
    for col in resolved:
        series = out[col].astype(float)
        spread = params.spread[col]
        out[col] = (series - params.center[col]) / spread if spread else 0.0
    return out


def scale_features(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    method: ScaleMethod = "standard",
) -> pd.DataFrame:
    """Scale numeric columns to a common range.

    Convenience wrapper that fits centre/spread on ``df`` and scales the same
    frame — fine for exploration or pre-split use. For a train/test workflow
    use :func:`fit_scale_params` + :func:`apply_scale_features` so the
    parameters come from the training split only. ``"standard"`` centres each
    column and divides by its (sample) standard deviation; ``"minmax"``
    rescales it to ``[0, 1]``. A constant column maps to all zeros rather than
    producing ``inf``/``NaN``.

    Args:
        df: The source DataFrame.
        columns: Numeric columns to scale; ``None`` uses every numeric column.
        method: ``"standard"`` (z-score) or ``"minmax"``.

    Returns:
        A new DataFrame with the selected columns scaled.

    Raises:
        KeyError: If a name in ``columns`` is not a column of ``df``.
        ValueError: If a named column is not numeric.
    """
    return apply_scale_features(df, fit_scale_params(df, columns, method=method))


def bin_column(
    df: pd.DataFrame,
    column: str,
    *,
    bins: int | Sequence[float],
    method: BinMethod = "width",
    labels: Sequence[str] | None = None,
    drop: bool = False,
) -> pd.DataFrame:
    """Discretize a numeric column into a categorical ``<column>_bin`` column.

    ``"width"`` cuts the value range into equal-width intervals
    (:func:`pandas.cut`); ``"quantile"`` cuts into equal-frequency intervals
    (:func:`pandas.qcut`), dropping duplicate edges when the data is degenerate.

    Args:
        df: The source DataFrame.
        column: Name of the numeric column to bin.
        bins: Number of bins, or explicit edges for ``"width"`` /
            quantile fractions for ``"quantile"``.
        method: ``"width"`` (equal-width) or ``"quantile"`` (equal-frequency).
        labels: Optional labels for the resulting bins.
        drop: If ``True``, drop the original column from the result.

    Returns:
        A new DataFrame with the added ``<column>_bin`` column.

    Raises:
        KeyError: If ``column`` is not present.
    """
    if column not in df.columns:
        raise KeyError(column)
    out = df.copy()
    bin_labels: Sequence[str] | None = list(labels) if labels is not None else None
    if method == "width":
        out[f"{column}_bin"] = pd.cut(out[column], bins=bins, labels=bin_labels)
    else:
        out[f"{column}_bin"] = pd.qcut(out[column], q=bins, labels=bin_labels, duplicates="drop")
    if drop:
        out = out.drop(columns=[column])
    return out


__all__ = [
    "OneHotCategories",
    "OrdinalCategories",
    "ScaleParams",
    "add_datetime_features",
    "apply_one_hot_encode",
    "apply_ordinal_encode",
    "apply_scale_features",
    "bin_column",
    "fit_one_hot_categories",
    "fit_ordinal_categories",
    "fit_scale_params",
    "one_hot_encode",
    "ordinal_encode",
    "scale_features",
]
