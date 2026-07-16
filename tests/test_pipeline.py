"""Tests for the fit-once/apply-many ``ds.pipeline`` module."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ds.features import (
    fit_one_hot_categories,
    fit_ordinal_categories,
    fit_scale_params,
)
from ds.io import load_params, save_params
from ds.pipeline import Pipeline, PipelineStep
from ds.preprocessing import (
    apply_clip_outliers,
    apply_flag_outliers,
    apply_impute_missing,
    fit_impute_values,
    fit_outlier_bounds,
)


@pytest.fixture()
def train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [10.0, 12.0, 11.0, 500.0, None],
            "size": ["S", "M", "M", "L", None],
            "region": ["north", "south", "north", None, "south"],
        }
    )


def _through_json(data: dict[str, object]) -> dict[str, object]:
    """Round-trip a payload through strict JSON, as ds.io.save_params does."""
    loaded = json.loads(json.dumps(data, allow_nan=False))
    assert isinstance(loaded, dict)
    return loaded


# --- Construction ---


def test_step_rejects_unknown_kind(train_df: pd.DataFrame) -> None:
    bounds = fit_outlier_bounds(train_df, ["amount"])
    with pytest.raises(ValueError, match="unknown pipeline step kind 'winsorize'"):
        PipelineStep("winsorize", bounds)  # type: ignore[arg-type]


def test_step_rejects_mismatched_params(train_df: pd.DataFrame) -> None:
    bounds = fit_outlier_bounds(train_df, ["amount"])
    with pytest.raises(TypeError, match="'impute_missing' takes ImputeValues"):
        PipelineStep("impute_missing", bounds)


def test_pipeline_normalizes_steps_to_tuple_and_rejects_non_steps(
    train_df: pd.DataFrame,
) -> None:
    step = PipelineStep("clip_outliers", fit_outlier_bounds(train_df, ["amount"]))
    pipeline = Pipeline(steps=[step])  # type: ignore[arg-type]
    assert pipeline.steps == (step,)
    with pytest.raises(TypeError, match="must be PipelineStep instances"):
        Pipeline(steps=("not a step",))  # type: ignore[arg-type]


# --- Application ---


def test_apply_matches_hand_strung_calls(train_df: pd.DataFrame) -> None:
    bounds = fit_outlier_bounds(train_df, ["amount"])
    fills = fit_impute_values(train_df, ["amount"], strategy="median")
    pipeline = Pipeline(
        steps=(
            PipelineStep("clip_outliers", bounds),
            PipelineStep("impute_missing", fills),
        )
    )
    fresh = pd.DataFrame({"amount": [9999.0, None, 11.5]})
    expected = apply_impute_missing(apply_clip_outliers(fresh, bounds), fills)
    assert pipeline.apply(fresh).equals(expected)
    # The input frame is never mutated.
    assert fresh["amount"].isna().sum() == 1


def test_apply_runs_steps_in_order(train_df: pd.DataFrame) -> None:
    # Two imputers over the same column: whichever runs first fills the gap,
    # and the other finds nothing left to do. Same steps, different order,
    # observably different result.
    zero_fill = fit_impute_values(train_df, ["amount"], strategy="constant", fill_value=0.0)
    median_fill = fit_impute_values(train_df, ["amount"], strategy="median")
    zero_first = Pipeline(
        steps=(
            PipelineStep("impute_missing", zero_fill),
            PipelineStep("impute_missing", median_fill),
        )
    )
    median_first = Pipeline(
        steps=(
            PipelineStep("impute_missing", median_fill),
            PipelineStep("impute_missing", zero_fill),
        )
    )
    fresh = pd.DataFrame({"amount": [None]})
    assert zero_first.apply(fresh).loc[0, "amount"] == 0.0
    assert median_first.apply(fresh).loc[0, "amount"] == pytest.approx(
        float(train_df["amount"].median())
    )
    assert zero_first != median_first


def test_two_steps_of_the_same_parameter_type_coexist(train_df: pd.DataFrame) -> None:
    numeric_fill = fit_impute_values(train_df, ["amount"], strategy="median")
    modal_fill = fit_impute_values(train_df, ["region"], strategy="most_frequent")
    pipeline = Pipeline(
        steps=(
            PipelineStep("impute_missing", numeric_fill),
            PipelineStep("impute_missing", modal_fill),
        )
    )
    fresh = pd.DataFrame({"amount": [None], "region": [None]})
    out = pipeline.apply(fresh)
    assert out.loc[0, "amount"] == pytest.approx(float(train_df["amount"].median()))
    assert out.loc[0, "region"] == "north"

    restored = Pipeline.from_dict(_through_json(pipeline.to_dict()))
    assert restored == pipeline
    assert restored.apply(fresh).equals(out)


def test_clip_and_flag_are_distinct_kinds_of_the_same_bounds(train_df: pd.DataFrame) -> None:
    bounds = fit_outlier_bounds(train_df, ["amount"])
    clip = Pipeline(steps=(PipelineStep("clip_outliers", bounds),))
    flag = Pipeline(steps=(PipelineStep("flag_outliers", bounds),))
    assert clip != flag

    fresh = pd.DataFrame({"amount": [9999.0, 11.0]})
    clipped = clip.apply(fresh)
    assert clipped.equals(apply_clip_outliers(fresh, bounds))

    flagged = flag.apply(fresh)
    assert flagged["amount"].equals(fresh["amount"])  # flags never rewrite values
    assert (
        flagged["amount_outlier"].tolist() == apply_flag_outliers(fresh, bounds)["amount"].tolist()
    )

    # The kind survives the round-trip, not just the bounds.
    assert Pipeline.from_dict(_through_json(flag.to_dict())).apply(fresh).equals(flagged)


def test_flag_step_refuses_to_overwrite_existing_column(train_df: pd.DataFrame) -> None:
    bounds = fit_outlier_bounds(train_df, ["amount"])
    pipeline = Pipeline(steps=(PipelineStep("flag_outliers", bounds),))
    fresh = pd.DataFrame({"amount": [1.0], "amount_outlier": [False]})
    with pytest.raises(ValueError, match="would overwrite existing column 'amount_outlier'"):
        pipeline.apply(fresh)


def test_empty_pipeline_is_identity_on_a_copy() -> None:
    pipeline = Pipeline(steps=())
    df = pd.DataFrame({"x": [1, 2]})
    out = pipeline.apply(df)
    assert out.equals(df)
    assert out is not df


# --- Persistence ---


def test_round_trip_preserves_step_order_and_every_kind(train_df: pd.DataFrame) -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep("flag_outliers", fit_outlier_bounds(train_df, ["amount"])),
            PipelineStep("clip_outliers", fit_outlier_bounds(train_df, ["amount"], factor=3.0)),
            PipelineStep("impute_missing", fit_impute_values(train_df, ["amount"])),
            PipelineStep("one_hot_encode", fit_one_hot_categories(train_df, ["region"])),
            PipelineStep(
                "ordinal_encode",
                fit_ordinal_categories(train_df, ["size"], categories={"size": ["S", "M", "L"]}),
            ),
            PipelineStep("scale_features", fit_scale_params(train_df, ["amount"])),
        )
    )
    restored = Pipeline.from_dict(_through_json(pipeline.to_dict()))
    assert restored == pipeline
    assert [step.kind for step in restored.steps] == [step.kind for step in pipeline.steps]

    fresh = pd.DataFrame({"amount": [9999.0, None], "size": ["L", "XL"], "region": ["north", "x"]})
    assert restored.apply(fresh).equals(pipeline.apply(fresh))


def test_pipeline_saves_and_loads_through_ds_io(tmp_path: Path, train_df: pd.DataFrame) -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep("impute_missing", fit_impute_values(train_df, ["amount"])),
            PipelineStep("scale_features", fit_scale_params(train_df, ["amount"])),
        )
    )
    path = save_params(pipeline, tmp_path / "pipeline.json")
    assert load_params(path, Pipeline) == pipeline


def test_from_dict_rejects_malformed_payloads(train_df: pd.DataFrame) -> None:
    good = Pipeline(
        steps=(PipelineStep("impute_missing", fit_impute_values(train_df, ["amount"])),)
    ).to_dict()

    with pytest.raises(ValueError, match="must be a mapping"):
        Pipeline.from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected a 'Pipeline' payload"):
        Pipeline.from_dict({**good, "type": "ImputeValues"})
    with pytest.raises(ValueError, match="missing fields"):
        Pipeline.from_dict({"type": "Pipeline"})
    with pytest.raises(ValueError, match="unexpected fields"):
        Pipeline.from_dict({**good, "stale_field": 1})
    with pytest.raises(ValueError, match="steps must be a list"):
        Pipeline.from_dict({"type": "Pipeline", "steps": "impute_missing"})
    with pytest.raises(ValueError, match="Pipeline step 0 must be a mapping"):
        Pipeline.from_dict({"type": "Pipeline", "steps": ["impute_missing"]})
    with pytest.raises(ValueError, match="exactly the fields 'kind' and 'params'"):
        Pipeline.from_dict({"type": "Pipeline", "steps": [{"kind": "impute_missing"}]})
    with pytest.raises(ValueError, match="step 1 has unknown kind 'winsorize'"):
        Pipeline.from_dict({**good, "steps": [*good["steps"], {"kind": "winsorize", "params": {}}]})

    # A step whose nested params fail their own class's validation is named.
    step = good["steps"][0]
    with pytest.raises(ValueError, match=r"step 0 \('impute_missing'\) has a malformed params"):
        Pipeline.from_dict(
            {**good, "steps": [{**step, "params": {**step["params"], "strategy": "modal"}}]}
        )
    # ...including a params payload of the wrong class for the kind.
    scale = Pipeline(
        steps=(PipelineStep("scale_features", fit_scale_params(train_df, ["amount"])),)
    ).to_dict()
    with pytest.raises(
        ValueError, match="expected a 'ImputeValues' payload, got type 'ScaleParams'"
    ):
        Pipeline.from_dict(
            {**good, "steps": [{"kind": "impute_missing", "params": scale["steps"][0]["params"]}]}
        )
