"""Fit-once/apply-many pipelines over the ``fit_*``/``apply_*`` pairs.

A :class:`Pipeline` owns an ordered sequence of fitted-parameter steps and
applies the matching ``apply_*`` transforms in one call, so a scoring run no
longer re-strings the individual ``apply_*`` calls (and their order) by hand.
It persists through the existing :func:`ds.io.save_params` /
:func:`ds.io.load_params` machinery by delegating each step to its parameter
class's validated ``to_dict``/``from_dict`` round-trip.

The module lives at the top level rather than inside a stage because a
pipeline composes transforms from *two* stages — :mod:`ds.preprocessing` and
:mod:`ds.features` — and homing it in either would couple the stages to each
other. Imports run strictly pipeline → stages, so no cycle can form.

Two design points worth knowing:

- **A step records its apply form, not just its parameters.** Each step
  carries a ``kind`` naming the ``apply_*`` transform it means; this is what
  disambiguates :class:`~ds.preprocessing.OutlierBounds`, which serves both
  ``apply_clip_outliers`` and ``apply_flag_outliers``.
- **A pipeline holds scoring-time transforms only.** Include just the steps
  new rows should flow through. Parameters fitted on the target column (e.g.
  the worked example's sales bounds and sales fill — scoring rows have no
  target to transform) are train-time-only: keep them out of the pipeline and
  persist them individually if a later training run needs them. Stateless
  transforms (e.g. :func:`ds.features.add_datetime_features`) take no fitted
  parameters and run outside the pipeline as plain calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from ds._serde import check_payload
from ds.features import (
    OneHotCategories,
    OrdinalCategories,
    ScaleParams,
    TopKCategories,
    apply_collapse_categories,
    apply_one_hot_encode,
    apply_ordinal_encode,
    apply_scale_features,
)
from ds.preprocessing import (
    ImputeValues,
    OutlierBounds,
    apply_clip_outliers,
    apply_flag_outliers,
    apply_impute_missing,
)

StepKind = Literal[
    "clip_outliers",
    "collapse_categories",
    "flag_outliers",
    "impute_missing",
    "one_hot_encode",
    "ordinal_encode",
    "scale_features",
]
StepParams = (
    OutlierBounds
    | ImputeValues
    | TopKCategories
    | OneHotCategories
    | OrdinalCategories
    | ScaleParams
)

# The step-kind registry: maps each kind to the parameter class its apply form
# consumes. ``from_dict`` resolves classes through it (an unknown kind in a
# stale or hand-edited file fails here, by name), and construction validates
# against it so a step can never pair a kind with the wrong parameters. Two
# kinds share OutlierBounds — that is exactly why steps are tagged by kind
# rather than by the parameter class's own ``"type"`` tag.
_KIND_TO_CLASS: dict[str, type[StepParams]] = {
    "clip_outliers": OutlierBounds,
    "collapse_categories": TopKCategories,
    "flag_outliers": OutlierBounds,
    "impute_missing": ImputeValues,
    "one_hot_encode": OneHotCategories,
    "ordinal_encode": OrdinalCategories,
    "scale_features": ScaleParams,
}


@dataclass(frozen=True)
class PipelineStep:
    """One fitted transform in a :class:`Pipeline`.

    Attributes:
        kind: The ``apply_*`` transform this step means (e.g.
            ``"impute_missing"``). For :class:`~ds.preprocessing.OutlierBounds`
            parameters this is where clip-vs-flag is recorded:
            ``"clip_outliers"`` or ``"flag_outliers"``.
        params: The fitted parameters the transform consumes.

    Raises:
        ValueError: If ``kind`` is not a known step kind.
        TypeError: If ``params`` is not the parameter class ``kind`` consumes
            (e.g. an ``"impute_missing"`` step built with ``ScaleParams``).
    """

    kind: StepKind
    params: StepParams

    def __post_init__(self) -> None:
        expected = _KIND_TO_CLASS.get(self.kind)
        if expected is None:
            raise ValueError(
                f"unknown pipeline step kind {self.kind!r}; known kinds: {sorted(_KIND_TO_CLASS)}"
            )
        if not isinstance(self.params, expected):
            raise TypeError(
                f"step kind {self.kind!r} takes {expected.__name__} parameters, "
                f"got {type(self.params).__name__}"
            )

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply this step's transform and return the result as a new frame.

        A ``"flag_outliers"`` step augments the frame with one boolean
        ``<column>_outlier`` column per fitted column (flags are extra
        information, not a rewrite); every other kind transforms the fitted
        columns in place.

        Args:
            df: The DataFrame to transform. Never mutated.

        Returns:
            A new DataFrame with the transform applied.

        Raises:
            KeyError: If a fitted column is not a column of ``df``.
            ValueError: If a fitted column has the wrong dtype in ``df``, or a
                ``"flag_outliers"`` step would overwrite an existing
                ``<column>_outlier`` column.
        """
        params = self.params
        if isinstance(params, OutlierBounds):
            if self.kind == "flag_outliers":
                return _join_outlier_flags(df, params)
            return apply_clip_outliers(df, params)
        if isinstance(params, ImputeValues):
            return apply_impute_missing(df, params)
        if isinstance(params, TopKCategories):
            return apply_collapse_categories(df, params)
        if isinstance(params, OneHotCategories):
            return apply_one_hot_encode(df, params)
        if isinstance(params, OrdinalCategories):
            return apply_ordinal_encode(df, params)
        return apply_scale_features(df, params)


def _join_outlier_flags(df: pd.DataFrame, bounds: OutlierBounds) -> pd.DataFrame:
    """Add ``<column>_outlier`` flag columns to ``df``, refusing to overwrite."""
    flags = apply_flag_outliers(df, bounds)
    out = df.copy()
    for col in flags.columns:
        name = f"{col}_outlier"
        if name in out.columns:
            raise ValueError(f"flag_outliers step would overwrite existing column {name!r}")
        out[name] = flags[col]
    return out


@dataclass(frozen=True)
class Pipeline:
    """An ordered, persistable sequence of fitted transform steps.

    Fit the individual parameters once (on the training split), assemble them
    into a pipeline, and :meth:`apply` runs the matching ``apply_*``
    transforms in order on any frame — so a scoring run reloads one object
    instead of re-stringing the calls by hand. Steps of the same parameter
    type coexist (e.g. two ``"impute_missing"`` steps with different
    strategies); order is preserved by construction, application and the
    dict round-trip alike.

    Persist with :func:`ds.io.save_params` and reload with
    :func:`ds.io.load_params` — :class:`Pipeline` satisfies the same
    :class:`ds.io.FittedParams` protocol as the per-transform dataclasses.

    Attributes:
        steps: The steps, in application order.
    """

    steps: tuple[PipelineStep, ...]

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        for step in steps:
            if not isinstance(step, PipelineStep):
                raise TypeError(
                    f"Pipeline steps must be PipelineStep instances, got {type(step).__name__}"
                )
        object.__setattr__(self, "steps", steps)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run every step's transform on ``df``, in order.

        Args:
            df: The DataFrame to transform. Never mutated.

        Returns:
            A new DataFrame with all steps applied (a plain copy for an
            empty pipeline).

        Raises:
            KeyError: If a step's fitted column is not present when that step
                runs.
            ValueError: If a step's fitted column has the wrong dtype, or a
                ``"flag_outliers"`` step would overwrite an existing column.
        """
        if not self.steps:
            return df.copy()
        out = df
        for step in self.steps:
            out = step.apply(out)
        return out

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation.

        Each step contributes its ``kind`` and its parameter class's own
        ``to_dict`` payload, so everything the per-class round-trips guarantee
        (tagged non-finite floats, unwrapped numpy scalars, re-tupled
        vocabularies) holds for the pipeline too. Persist the result with
        :func:`ds.io.save_params` or rebuild with :meth:`from_dict`.

        Returns:
            A dict that round-trips through :meth:`from_dict`.
        """
        return {
            "type": "Pipeline",
            "steps": [{"kind": step.kind, "params": step.params.to_dict()} for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Pipeline:
        """Rebuild a :class:`Pipeline` from :meth:`to_dict` output.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The reconstructed :class:`Pipeline`, steps in their saved order.

        Raises:
            ValueError: If ``data`` is not a well-formed ``Pipeline`` payload —
                wrong type tag, missing/unexpected fields, a step with an
                unknown kind, or a step whose params payload fails its own
                class's validation (the error names the offending step).
        """
        payload = check_payload(data, "Pipeline", frozenset({"steps"}))
        raw_steps = payload["steps"]
        if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, str):
            raise ValueError("Pipeline.steps must be a list of step payloads")
        return cls(steps=tuple(_step_from_dict(item, i) for i, item in enumerate(raw_steps)))


def _step_from_dict(data: Any, index: int) -> PipelineStep:
    """Rebuild one step of a ``Pipeline`` payload, naming the step on failure."""
    label = f"Pipeline step {index}"
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} must be a mapping, got {type(data).__name__}")
    if set(data) != {"kind", "params"}:
        raise ValueError(f"{label} must have exactly the fields 'kind' and 'params'")
    kind = data["kind"]
    if kind not in _KIND_TO_CLASS:
        raise ValueError(
            f"{label} has unknown kind {kind!r}; known kinds: {sorted(_KIND_TO_CLASS)}"
        )
    try:
        params = _KIND_TO_CLASS[kind].from_dict(data["params"])
    except ValueError as exc:
        raise ValueError(f"{label} ({kind!r}) has a malformed params payload: {exc}") from exc
    return PipelineStep(kind=kind, params=params)


__all__ = [
    "Pipeline",
    "PipelineStep",
    "StepKind",
    "StepParams",
]
