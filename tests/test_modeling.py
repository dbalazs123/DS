"""Tests for modeling, evaluation and viz."""

from __future__ import annotations

import matplotlib as mpl
import pandas as pd
import pytest

from ds.evaluation import classification_metrics, regression_metrics
from ds.modeling.nlp import count_tokens
from ds.modeling.tabular import split_features_target
from ds.modeling.timeseries import train_test_split_by_time
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


def test_regression_metrics_perfect() -> None:
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0


def test_classification_metrics_perfect() -> None:
    m = classification_metrics([0, 1, 1], [0, 1, 1])
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0


def test_count_tokens_fallback() -> None:
    # Works with or without tiktoken installed; count is always positive.
    assert count_tokens("hello world foo") >= 1


def test_count_tokens_matches_tiktoken() -> None:
    # Only runs when the `nlp` extra is installed; verifies the accurate path
    # (not just the whitespace fallback) is used when tiktoken is available.
    tiktoken = pytest.importorskip("tiktoken")
    encoding = tiktoken.get_encoding("cl100k_base")
    text = "Tokenization isn't the same as splitting on spaces."
    assert count_tokens(text) == len(encoding.encode(text))


def test_set_theme_applies_palette() -> None:
    set_theme("talk")
    assert mpl.rcParams["axes.spines.top"] is False
