"""Tests for the io subpackage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ds.io import load_table, save_table


@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".json"])
def test_save_then_load_roundtrip(tmp_path: Path, suffix: str) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / f"data{suffix}"
    save_table(df, path)
    loaded = load_table(path)
    assert list(loaded.columns) == ["a", "b"]
    assert len(loaded) == 2


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1]})
    path = tmp_path / "nested" / "deep" / "out.csv"
    assert save_table(df, path) == path
    assert path.exists()


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_table(tmp_path / "nope.csv")


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="Unsupported"):
        save_table(df, tmp_path / "out.xlsx")
