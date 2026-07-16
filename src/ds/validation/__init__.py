"""Data-quality and schema checks.

Lightweight guards for the assumptions a pipeline makes about its inputs. The
``assert_*`` helpers are dependency-free and cover the common inline cases;
:func:`check_schema` leans on ``pandera`` (a core dependency) for a declarative
column/dtype schema when you want one call to cover a whole frame.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

Inclusive = Literal["both", "neither", "left", "right"]


class DataValidationError(ValueError):
    """Raised when a DataFrame violates an expected constraint."""


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Assert that ``df`` contains every column in ``columns``.

    Args:
        df: The DataFrame to check.
        columns: Column names that must be present.

    Returns:
        The same DataFrame, to support fluent chaining.

    Raises:
        DataValidationError: If any required column is missing.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    return df


def assert_no_nulls(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Assert that the given columns (or the whole frame) contain no nulls.

    Args:
        df: The DataFrame to check.
        columns: Columns to check; ``None`` checks all columns.

    Returns:
        The same DataFrame, to support fluent chaining.

    Raises:
        DataValidationError: If nulls are found.
    """
    subset = df if columns is None else df[list(columns)]
    null_counts = subset.isna().sum()
    offenders = null_counts[null_counts > 0]
    if not offenders.empty:
        raise DataValidationError(f"Null values found in: {offenders.to_dict()}")
    return df


def assert_in_range(
    df: pd.DataFrame,
    column: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    inclusive: Inclusive = "both",
) -> pd.DataFrame:
    """Assert that a numeric column's values fall within a range.

    At least one of ``min_value`` / ``max_value`` should be given. Nulls are
    ignored (use :func:`assert_no_nulls` for those), so this composes with the
    other guards.

    Args:
        df: The DataFrame to check.
        column: Name of the column to validate.
        min_value: Lower bound, or ``None`` for no lower bound.
        max_value: Upper bound, or ``None`` for no upper bound.
        inclusive: Which endpoints are allowed — ``"both"``, ``"neither"``,
            ``"left"`` or ``"right"`` (matching :meth:`pandas.Series.between`).

    Returns:
        The same DataFrame, to support fluent chaining.

    Raises:
        KeyError: If ``column`` is not present.
        DataValidationError: If any non-null value falls outside the range.
    """
    if column not in df.columns:
        raise KeyError(column)
    series = df[column]
    present = series.notna()
    below = pd.Series(False, index=df.index)
    above = pd.Series(False, index=df.index)
    if min_value is not None:
        below = series < min_value if inclusive in ("both", "left") else series <= min_value
    if max_value is not None:
        above = series > max_value if inclusive in ("both", "right") else series >= max_value
    offenders = series[present & (below | above)]
    if not offenders.empty:
        raise DataValidationError(
            f"Values in {column!r} outside [{min_value}, {max_value}]: "
            f"{offenders.unique().tolist()}"
        )
    return df


def assert_in_set(df: pd.DataFrame, column: str, allowed: Iterable[object]) -> pd.DataFrame:
    """Assert that a column's values all belong to an allowed set.

    Nulls are ignored. Handy for categorical columns with a known vocabulary
    (status codes, country codes, …).

    Args:
        df: The DataFrame to check.
        column: Name of the column to validate.
        allowed: The permitted values.

    Returns:
        The same DataFrame, to support fluent chaining.

    Raises:
        KeyError: If ``column`` is not present.
        DataValidationError: If any non-null value is outside ``allowed``.
    """
    if column not in df.columns:
        raise KeyError(column)
    allowed_set = set(allowed)
    series = df[column]
    unexpected = series[series.notna() & ~series.isin(allowed_set)]
    if not unexpected.empty:
        raise DataValidationError(
            f"Unexpected values in {column!r}: {unexpected.unique().tolist()}"
        )
    return df


def _dtype_matches(actual: object, expected: str | type) -> bool:
    """Return whether a pandas dtype matches an expected dtype spec."""
    try:
        return bool(actual == pd.api.types.pandas_dtype(expected))
    except TypeError:
        return False


def assert_dtypes(df: pd.DataFrame, dtypes: Mapping[str, str | type]) -> pd.DataFrame:
    """Assert that columns have the expected dtypes.

    Args:
        df: The DataFrame to check.
        dtypes: Mapping of column name to the expected dtype (e.g. ``"int64"``,
            ``"category"``, ``float``).

    Returns:
        The same DataFrame, to support fluent chaining.

    Raises:
        DataValidationError: If a column is missing or has the wrong dtype.
    """
    missing = [col for col in dtypes if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    mismatches = {
        col: (str(df[col].dtype), str(expected))
        for col, expected in dtypes.items()
        if not _dtype_matches(df[col].dtype, expected)
    }
    if mismatches:
        raise DataValidationError(f"Unexpected dtypes (actual, expected): {mismatches}")
    return df


def check_schema(
    df: pd.DataFrame,
    columns: Mapping[str, str | type],
    *,
    nullable: bool = True,
    coerce: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """Validate ``df`` against a declarative column/dtype schema via ``pandera``.

    A lightweight front door to :class:`pandera.DataFrameSchema`: pass a
    ``{column: dtype}`` mapping and get back the validated frame (coerced to the
    requested dtypes when ``coerce`` is set). Any ``pandera`` failure is
    re-raised as :class:`DataValidationError` so callers catch one error type.

    Args:
        df: The DataFrame to validate.
        columns: Mapping of column name to expected dtype.
        nullable: Whether the columns may contain nulls.
        coerce: If ``True``, coerce columns to the schema dtypes rather than
            failing on a mismatch.
        strict: If ``True``, reject columns not named in ``columns``.

    Returns:
        The validated (and, when ``coerce`` is set, coerced) DataFrame.

    Raises:
        DataValidationError: If ``df`` does not satisfy the schema.
    """
    schema = pa.DataFrameSchema(
        {
            name: pa.Column(dtype, nullable=nullable, coerce=coerce)
            for name, dtype in columns.items()
        },
        strict=strict,
        coerce=coerce,
    )
    try:
        return schema.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise DataValidationError(str(exc)) from exc


__all__ = [
    "DataValidationError",
    "assert_dtypes",
    "assert_in_range",
    "assert_in_set",
    "assert_no_nulls",
    "check_schema",
    "require_columns",
]
