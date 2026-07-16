"""Private helpers for the fitted-parameter dict round-trips.

Shared by the ``to_dict``/``from_dict`` methods on the ``fit_*`` parameter
dataclasses in :mod:`ds.preprocessing` and :mod:`ds.features`. Not part of the
public API.

Scalars are kept JSON-safe: numpy scalars are unwrapped to their Python
equivalents, and non-finite floats — which strict JSON has no literal for —
are written as a tagged ``{"__float__": "inf" | "-inf" | "nan"}`` mapping so a
plain string like ``"inf"`` in a category vocabulary stays unambiguous.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

_FLOAT_TAG = "__float__"
_NON_FINITE = ("inf", "-inf", "nan")


def encode_scalar(value: Any) -> Any:
    """Return ``value`` in a strictly JSON-representable form.

    Numpy scalars become their Python equivalents and non-finite floats become
    a tagged mapping; everything else passes through unchanged (and, if it is
    not JSON-serializable, fails loudly at dump time).
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {_FLOAT_TAG: str(value)}
    return value


def decode_scalar(value: Any) -> Any:
    """Invert :func:`encode_scalar`, validating any tagged mapping."""
    if isinstance(value, Mapping):
        tagged = value.get(_FLOAT_TAG)
        if set(value) != {_FLOAT_TAG} or tagged not in _NON_FINITE:
            raise ValueError(f"not a tagged non-finite float: {dict(value)!r}")
        return float(tagged)
    return value


def check_payload(data: Any, type_name: str, fields: frozenset[str]) -> dict[str, Any]:
    """Validate the shape of a ``from_dict`` payload and return it as a dict.

    Ensures ``data`` is a mapping carrying the expected ``"type"`` tag and
    exactly the expected field names, so a stale or hand-edited file fails
    with a message naming what is wrong rather than a downstream ``KeyError``.

    Raises:
        ValueError: If ``data`` is not a mapping, is tagged with a different
            type, or has missing/unexpected fields.
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"{type_name} payload must be a mapping, got {type(data).__name__}")
    tag = data.get("type")
    if tag != type_name:
        raise ValueError(f"expected a {type_name!r} payload, got type {tag!r}")
    missing = fields - set(data)
    if missing:
        raise ValueError(f"{type_name} payload is missing fields: {sorted(missing)}")
    extra = set(data) - fields - {"type"}
    if extra:
        raise ValueError(f"{type_name} payload has unexpected fields: {sorted(extra)}")
    return dict(data)


def check_str_mapping(value: Any, field: str, type_name: str) -> dict[str, Any]:
    """Validate that a payload field is a mapping with string keys."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{type_name}.{field} must be a mapping with string keys")
    return dict(value)


def as_float(value: Any, field: str, type_name: str) -> float:
    """Validate that a decoded payload value is a real number and return it.

    ``bool`` is rejected explicitly — it is an ``int`` subclass, and a stray
    ``true`` in a hand-edited file should not silently become ``1.0``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{type_name}.{field} must be a number, got {value!r}")
    return float(value)


def as_bool(value: Any, field: str, type_name: str) -> bool:
    """Validate that a payload value is a bool and return it."""
    if not isinstance(value, bool):
        raise ValueError(f"{type_name}.{field} must be a bool, got {value!r}")
    return value
