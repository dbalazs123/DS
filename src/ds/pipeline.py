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

The fit side has a matching executor: :func:`fit_pipeline` runs a *plan* — an
ordered sequence of :class:`FitStep` entries pairing a step kind with a fit
callable — as the fit → apply → fit chain the per-pair API otherwise requires
by hand (each parameter set is fitted on the training frame as transformed by
the steps before it), and returns the assembled :class:`Pipeline`. The
:class:`Pipeline` itself stays pure composition: :func:`fit_pipeline` is a
convenience over the same ``fit_*``/``apply_*`` primitives, not a new fitting
contract.

Four design points worth knowing:

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
- **A fit plan is code, not data.** A :class:`FitStep` carries a callable
  (typically a lambda closing over ``columns=``, ``strategy=``, ``k=``, …),
  so the varying ``fit_*`` signatures need no declarative mirror and the plan
  is not persistable — persist the *fitted* :class:`Pipeline` it produces.
- **Model-side transforms live in the estimator, not the pipeline.** Every
  step kind maps named DataFrame columns to named DataFrame columns; a
  transform whose fitted state *manufactures* its column space — a text
  vectorizer turning one string column into thousands of learned sparse
  columns — has no honest home in that contract (materializing the sparse
  matrix as a dense frame would hide the real cost, and wrapping the
  scikit-learn object whole would smuggle a pickle into the strict-JSON
  ``save_params`` story). Keep such transforms inside the estimator (e.g. a
  ``TfidfVectorizer`` in a scikit-learn pipeline, as ``projects/sms_spam``
  does) and persist them with :func:`ds.modeling.persistence.save_model`;
  the ds pipeline carries the frame-shaped steps around it. The convention
  is the settled resolution of backlog item 18 for now — a second text
  project decides whether a first-class vectorize step earns a build.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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


@dataclass(frozen=True)
class FitStep:
    """One step of a fit plan: a step kind plus the callable that fits it.

    The fit-side counterpart of :class:`PipelineStep`: where a
    :class:`PipelineStep` carries *fitted* parameters, a :class:`FitStep`
    carries the function that will fit them. ``fit`` is typically a lambda
    over one of the stage ``fit_*`` helpers, closing over its keyword
    arguments — that is how the varying signatures (``columns=``,
    ``strategy=``, ``k=``, …) fit one plan type without a declarative mirror
    of each:

    >>> FitStep(
    ...     "impute_missing", lambda df: fit_impute_values(df, columns=["age"], strategy="median")
    ... )  # doctest: +SKIP

    Attributes:
        kind: The ``apply_*`` transform the fitted step will mean (same
            vocabulary as :attr:`PipelineStep.kind`).
        fit: A callable that learns this step's parameters from a frame.
            :func:`fit_pipeline` calls it with the training frame as
            transformed by the plan's earlier steps. It must return the
            parameter class ``kind`` consumes — checked when the fitted
            :class:`PipelineStep` is built.

    Raises:
        ValueError: If ``kind`` is not a known step kind.
    """

    kind: StepKind
    fit: Callable[[pd.DataFrame], StepParams]

    def __post_init__(self) -> None:
        if self.kind not in _KIND_TO_CLASS:
            raise ValueError(
                f"unknown pipeline step kind {self.kind!r}; known kinds: {sorted(_KIND_TO_CLASS)}"
            )


def fit_pipeline(df: pd.DataFrame, plan: Sequence[FitStep]) -> Pipeline:
    """Fit an ordered plan of transforms on one frame, into a :class:`Pipeline`.

    Executes the fit → apply → fit chain the per-pair API requires by hand:
    each :class:`FitStep`'s ``fit`` callable runs on the training frame *as
    transformed by the steps before it*, and the fitted parameters are
    assembled into a :class:`Pipeline` in plan order. One plan can therefore
    serve both a training run (fit once, persist the result) and per-fold
    re-fitting inside :func:`ds.evaluation.cross_validate_kfold` (via its
    ``make_pipeline`` argument).

    Write the plan as the *scoring* plan: only transforms new rows should
    flow through, nothing fitted on the target column. When ``df`` is a
    modeling frame (features plus target), pass explicit ``columns=`` to the
    ``fit_*`` callables — their ``columns=None`` defaults select columns by
    dtype and would sweep the target in.

    Args:
        df: The frame to fit on (typically the training split). Never
            mutated.
        plan: The :class:`FitStep` entries, in the order the fitted pipeline
            should apply them.

    Returns:
        A :class:`Pipeline` with one fitted step per plan entry (an empty
        pipeline for an empty plan).

    Raises:
        TypeError: If a step's ``fit`` returns a parameter class its ``kind``
            does not consume.
        KeyError: If a step's fit or apply needs a column the (transformed)
            frame does not have.
        ValueError: If a step's fit or apply rejects the (transformed) frame
            — e.g. a wrong dtype, or a ``"flag_outliers"`` step that would
            overwrite an existing ``<column>_outlier`` column.
    """
    steps: list[PipelineStep] = []
    out = df
    for entry in plan:
        step = PipelineStep(kind=entry.kind, params=entry.fit(out))
        out = step.apply(out)
        steps.append(step)
    return Pipeline(steps=tuple(steps))


__all__ = [
    "FitStep",
    "Pipeline",
    "PipelineStep",
    "StepKind",
    "StepParams",
    "fit_pipeline",
]
