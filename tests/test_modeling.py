"""Tests for modeling, evaluation and viz."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from ds.evaluation import (
    classification_metrics,
    compare_models,
    confusion_frame,
    cross_validate_by_time,
    cross_validate_kfold,
    per_class_metrics,
    regression_metrics,
)
from ds.features import fit_scale_params
from ds.modeling.baseline import fit_baseline
from ds.modeling.nlp import count_tokens
from ds.modeling.persistence import load_model, save_model
from ds.modeling.tabular import split_features_target, train_test_split_random
from ds.modeling.timeseries import train_test_split_by_time
from ds.pipeline import FitStep, Pipeline, fit_pipeline
from ds.preprocessing import ImputeValues, fit_impute_values
from ds.viz import set_theme


def test_split_features_target() -> None:
    df = pd.DataFrame({"x1": [1, 2], "x2": [3, 4], "y": [0, 1]})
    x, y = split_features_target(df, "y")
    assert list(x.columns) == ["x1", "x2"]
    assert list(y) == [0, 1]


def test_split_features_target_missing() -> None:
    with pytest.raises(KeyError):
        split_features_target(pd.DataFrame({"a": [1]}), "y")


def test_time_split_is_chronological() -> None:
    df = pd.DataFrame({"t": [3, 1, 2, 4, 5], "v": [30, 10, 20, 40, 50]})
    train, test = train_test_split_by_time(df, "t", test_size=0.4)
    assert list(train["t"]) == [1, 2, 3]
    assert list(test["t"]) == [4, 5]


def test_time_split_bad_size() -> None:
    with pytest.raises(ValueError, match="test_size"):
        train_test_split_by_time(pd.DataFrame({"t": [1]}), "t", test_size=1.5)


def _labeled_frame(n: int = 100) -> pd.DataFrame:
    # 70/30 binary target, so stratification is observable.
    return pd.DataFrame({"x": range(n), "y": [1 if i % 10 < 3 else 0 for i in range(n)]})


def test_random_split_sizes_and_columns() -> None:
    train, test = train_test_split_random(_labeled_frame(), test_size=0.2)
    assert len(train) == 80
    assert len(test) == 20
    assert list(train.columns) == list(test.columns) == ["x", "y"]
    assert sorted([*train["x"], *test["x"]]) == list(range(100))  # a partition, not a resample


def test_random_split_stratify_preserves_class_balance() -> None:
    train, test = train_test_split_random(_labeled_frame(), test_size=0.2, stratify="y")
    assert train["y"].mean() == pytest.approx(0.3)
    assert test["y"].mean() == pytest.approx(0.3)


def test_random_split_is_reproducible_under_seed() -> None:
    from ds import seed_everything

    seed_everything(7)
    first, _ = train_test_split_random(_labeled_frame(), stratify="y")
    seed_everything(7)
    second, _ = train_test_split_random(_labeled_frame(), stratify="y")
    assert first["x"].tolist() == second["x"].tolist()


def test_random_split_rejects_bad_inputs() -> None:
    df = _labeled_frame(10)
    with pytest.raises(ValueError, match="test_size"):
        train_test_split_random(df, test_size=1.0)
    with pytest.raises(KeyError):
        train_test_split_random(df, stratify="nope")
    with pytest.raises(ValueError, match="least populated class"):
        train_test_split_random(pd.DataFrame({"y": [0, 0, 1]}), stratify="y")


def test_regression_metrics_perfect() -> None:
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0


def test_classification_metrics_perfect() -> None:
    m = classification_metrics([0, 1, 1], [0, 1, 1])
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0


def test_confusion_frame_labels_axes() -> None:
    cm = confusion_frame([0, 1, 1, 0], [0, 1, 0, 0])
    assert cm.index.name == "true"
    assert cm.columns.name == "predicted"
    assert cm.loc[0, 0] == 2  # both true-0 samples predicted 0
    assert cm.loc[1, 0] == 1  # one true-1 sample predicted 0


def test_per_class_metrics_breaks_out_each_label() -> None:
    frame = per_class_metrics([0, 1, 1], [0, 1, 1])
    assert set(frame.columns) == {"precision", "recall", "f1", "support"}
    assert frame.index.name == "label"
    assert frame.loc[1, "support"] == 2
    assert frame.loc[1, "f1"] == 1.0


def test_fit_baseline_mean_ignores_nulls() -> None:
    y = pd.Series([1.0, 3.0, None])
    baseline = fit_baseline(y, strategy="mean")
    assert baseline.predict(3) == [2.0, 2.0, 2.0]


def test_fit_baseline_majority_predicts_modal_label() -> None:
    y = pd.Series([0, 1, 1, 0, 1, None])
    baseline = fit_baseline(y, strategy="majority")
    assert baseline.values == (1.0,)
    assert baseline.predict(3) == [1.0, 1.0, 1.0]


def test_fit_baseline_majority_breaks_ties_low() -> None:
    assert fit_baseline(pd.Series([1, 0, 0, 1]), strategy="majority").values == (0.0,)


def test_fit_baseline_majority_rejects_bad_targets() -> None:
    with pytest.raises(ValueError, match="all-null"):
        fit_baseline(pd.Series([None, None], dtype=float), strategy="majority")
    with pytest.raises(ValueError, match="numeric labels"):
        fit_baseline(pd.Series(["yes", "yes", "no"]), strategy="majority")
    with pytest.raises(ValueError, match="season_length"):
        fit_baseline(pd.Series([0, 1, 1]), strategy="majority", season_length=2)


def test_fit_baseline_naive_last_repeats_final_value() -> None:
    y = pd.Series([10.0, 20.0, 30.0])
    assert fit_baseline(y, strategy="naive_last").predict(2) == [30.0, 30.0]


def test_fit_baseline_seasonal_naive_cycles_last_season() -> None:
    y = pd.Series([1.0, 2.0, 3.0, 40.0, 50.0, 60.0])
    baseline = fit_baseline(y, strategy="seasonal_naive", season_length=3)
    assert baseline.values == (40.0, 50.0, 60.0)
    assert baseline.predict(5) == [40.0, 50.0, 60.0, 40.0, 50.0]


def test_fit_baseline_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_baseline(pd.Series([], dtype=float))
    with pytest.raises(ValueError, match="season_length"):
        fit_baseline(pd.Series([1.0, 2.0]), strategy="seasonal_naive")
    with pytest.raises(ValueError, match="season_length"):
        fit_baseline(pd.Series([1.0, 2.0]), strategy="seasonal_naive", season_length=3)
    with pytest.raises(ValueError, match="season_length"):
        fit_baseline(pd.Series([1.0, 2.0]), strategy="mean", season_length=2)
    with pytest.raises(ValueError, match="nulls"):
        fit_baseline(pd.Series([1.0, None]), strategy="naive_last")
    with pytest.raises(ValueError, match="non-negative"):
        fit_baseline(pd.Series([1.0])).predict(-1)


def test_model_round_trips_through_disk(tmp_path: Path) -> None:
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 1.0, 0.0, 1.0]})
    y = pd.Series([2.0, 4.5, 6.0, 8.5])
    model = LinearRegression().fit(x, y)

    saved_to = save_model(model, tmp_path / "artifacts" / "model.joblib")
    assert saved_to.exists()  # parent directory was created

    reloaded = load_model(saved_to)
    assert reloaded.predict(x).tolist() == model.predict(x).tolist()


def test_save_model_accepts_str_path(tmp_path: Path) -> None:
    path = save_model(LinearRegression(), str(tmp_path / "model.joblib"))
    assert isinstance(path, Path)
    assert path.exists()


def test_load_model_missing_file_names_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nope.joblib"):
        load_model(tmp_path / "nope.joblib")


def _linear_frame(n: int = 12) -> pd.DataFrame:
    # y is an exact linear function of x, so LinearRegression recovers it and
    # every fold's error is ~0 — which makes the fold plumbing observable.
    t = list(range(1, n + 1))
    return pd.DataFrame({"t": t, "x": [2.0 * v for v in t], "y": [3.0 * 2.0 * v for v in t]})


def test_cross_validate_by_time_expands_history_per_fold() -> None:
    result = cross_validate_by_time(
        _linear_frame(),
        time_column="t",
        target="y",
        make_model=lambda: LinearRegression(),
        n_splits=3,
    )
    assert result.index.name == "fold"
    assert list(result.index) == [1, 2, 3]
    # Rolling origin: each fold trains on one more block of history.
    assert result["train_size"].tolist() == [3.0, 6.0, 9.0]
    assert result["test_size"].tolist() == [3.0, 3.0, 3.0]
    assert (result["mae"] < 1e-8).all()


def test_cross_validate_by_time_rejects_bad_inputs() -> None:
    df = _linear_frame(4)
    with pytest.raises(KeyError):
        cross_validate_by_time(
            df, time_column="nope", target="y", make_model=lambda: LinearRegression()
        )
    with pytest.raises(ValueError, match="n_splits"):
        cross_validate_by_time(
            df, time_column="t", target="y", make_model=lambda: LinearRegression(), n_splits=0
        )
    with pytest.raises(ValueError, match="rows"):
        cross_validate_by_time(
            df, time_column="t", target="y", make_model=lambda: LinearRegression(), n_splits=4
        )


def test_cross_validate_kfold_scores_every_fold() -> None:
    df = _linear_frame().drop(columns=["t"])
    result = cross_validate_kfold(df, target="y", make_model=lambda: LinearRegression(), n_splits=4)
    assert result.index.name == "fold"
    assert len(result) == 4
    assert set(result.columns) == {"train_size", "test_size", "mae", "rmse", "r2"}
    assert (result["mae"] < 1e-8).all()


def test_cross_validate_kfold_rejects_bad_inputs() -> None:
    df = _linear_frame(4).drop(columns=["t"])
    with pytest.raises(KeyError):
        cross_validate_kfold(df, target="nope", make_model=lambda: LinearRegression())
    with pytest.raises(ValueError, match="n_splits"):
        cross_validate_kfold(df, target="y", make_model=lambda: LinearRegression(), n_splits=9)


def test_cross_validate_kfold_stratify_keeps_fold_class_balance() -> None:
    from sklearn.dummy import DummyClassifier

    df = _labeled_frame(50)  # 70/30 target; plain KFold lets fold balance drift
    counts: list[int] = []

    def record_and_score(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
        counts.append(int(sum(y_true)))
        return {"positives": float(sum(y_true))}

    result = cross_validate_kfold(
        df,
        target="y",
        make_model=lambda: DummyClassifier(strategy="most_frequent"),
        n_splits=5,
        stratify=True,
        metrics_fn=record_and_score,
    )
    assert len(result) == 5
    assert counts == [3, 3, 3, 3, 3]  # every 10-row fold carries exactly 30% positives


def test_cross_validate_kfold_stratify_warns_on_sparse_class() -> None:
    # scikit-learn warns (not raises) when a class has fewer members than
    # n_splits; such folds simply miss that class.
    from sklearn.dummy import DummyClassifier

    df = pd.DataFrame({"x": range(6), "y": [0, 0, 0, 0, 0, 1]})
    with pytest.warns(UserWarning, match="least populated class"):
        result = cross_validate_kfold(
            df,
            target="y",
            make_model=lambda: DummyClassifier(strategy="most_frequent"),
            n_splits=3,
            stratify=True,
        )
    assert len(result) == 3


def test_cross_validate_kfold_refits_pipeline_per_fold() -> None:
    # x carries NaNs, so the model would reject the frame unless each fold's
    # imputation actually runs — and shuffle=False makes the fold-train
    # medians differ from the whole-frame median, so re-fitting (rather than
    # reusing one whole-frame fit) is observable in the fitted parameters.
    df = pd.DataFrame(
        {
            "x": [None, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, None],
            "y": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
        }
    )
    plan = [
        FitStep("impute_missing", lambda d: fit_impute_values(d, ["x"], strategy="median")),
        FitStep("scale_features", lambda d: fit_scale_params(d, ["x"])),
    ]
    seen: list[pd.DataFrame] = []
    fitted: list[Pipeline] = []

    def make_pipeline(frame: pd.DataFrame) -> Pipeline:
        seen.append(frame)
        pipeline = fit_pipeline(frame, plan)
        fitted.append(pipeline)
        return pipeline

    result = cross_validate_kfold(
        df,
        target="y",
        make_model=lambda: LinearRegression(),
        make_pipeline=make_pipeline,
        n_splits=2,
        shuffle=False,
    )
    assert len(result) == 2
    assert result["mae"].notna().all()

    # One fresh fit per fold, on exactly that fold's training rows (the
    # target column rides along, as at final-training time).
    assert [len(frame) for frame in seen] == [5, 5]
    assert all("y" in frame.columns for frame in seen)
    assert seen[0].index.tolist() == [5, 6, 7, 8, 9]
    assert seen[1].index.tolist() == [0, 1, 2, 3, 4]

    # The fitted statistics are the fold's own, not the whole frame's.
    fold_fills: list[float] = []
    for pipeline in fitted:
        params = pipeline.steps[0].params
        assert isinstance(params, ImputeValues)
        fold_fills.append(float(params.fill_values["x"]))
    assert fold_fills == [6.5, 2.5]
    assert float(df["x"].median()) not in fold_fills


def test_cross_validate_kfold_pipeline_composes_with_stratify() -> None:
    from sklearn.dummy import DummyClassifier

    df = _labeled_frame(50)
    result = cross_validate_kfold(
        df,
        target="y",
        make_model=lambda: DummyClassifier(strategy="most_frequent"),
        make_pipeline=lambda frame: fit_pipeline(
            frame, [FitStep("scale_features", lambda d: fit_scale_params(d, ["x"]))]
        ),
        n_splits=5,
        stratify=True,
        metrics_fn=classification_metrics,
    )
    assert len(result) == 5
    # The majority class is 0 everywhere, so accuracy is the fold's negative
    # share — stratified to exactly 70% in every fold.
    assert result["accuracy"].tolist() == [0.7] * 5


def test_compare_models_scores_side_by_side() -> None:
    y_true = [1.0, 2.0, 3.0]
    frame = compare_models(y_true, {"model": [1.0, 2.0, 3.0], "baseline": [2.0, 2.0, 2.0]})
    assert frame.index.name == "model"
    assert list(frame.index) == ["model", "baseline"]  # mapping order preserved
    assert frame.loc["model", "mae"] == 0.0
    assert frame.loc["baseline", "mae"] == pytest.approx(2.0 / 3.0)


def test_compare_models_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        compare_models([1.0], {})
    with pytest.raises(ValueError, match="align"):
        compare_models([1.0, 2.0], {"short": [1.0]})


def test_plot_model_comparison_defaults_to_first_metric() -> None:
    from ds.viz import plot_model_comparison

    frame = compare_models([1.0, 2.0], {"a": [1.0, 2.0], "b": [2.0, 3.0]})
    ax = plot_model_comparison(frame)
    assert ax.get_xlabel() == "mae"
    with pytest.raises(KeyError):
        plot_model_comparison(frame, metric="nope")
    with pytest.raises(ValueError, match="no metric columns"):
        plot_model_comparison(pd.DataFrame(index=pd.Index(["a"], name="model")))


def test_count_tokens_fallback() -> None:
    # Works with or without tiktoken installed; count is always positive.
    assert count_tokens("hello world foo") >= 1


def test_count_tokens_matches_tiktoken() -> None:
    # Only runs when the `nlp` extra is installed and tiktoken's vocabulary is
    # reachable; verifies the accurate path (not just the whitespace fallback).
    tiktoken = pytest.importorskip("tiktoken")
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # vocab fetch needs network on first use
        pytest.skip(f"tiktoken vocabulary unavailable: {exc}")
    text = "Tokenization isn't the same as splitting on spaces."
    assert count_tokens(text) == len(encoding.encode(text))


def test_set_theme_applies_palette() -> None:
    set_theme("talk")
    assert mpl.rcParams["axes.spines.top"] is False
