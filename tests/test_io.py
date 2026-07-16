"""Tests for the io subpackage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ds.config import Settings
from ds.io import load_raw, load_table, save_processed, save_table


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".parquet", ".json", ".jsonl"])
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


def test_save_processed_writes_under_processed_dir(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    written = save_processed(df, "clean.parquet", settings=settings)
    assert written == settings.processed_dir / "clean.parquet"
    assert written.exists()
    # The file is a normal table, readable by the generic loader.
    assert list(load_table(written).columns) == ["a", "b"]


def test_load_raw_reads_from_raw_dir(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    save_table(pd.DataFrame({"a": [1]}), settings.raw_dir / "in.csv")
    assert len(load_raw("in.csv", settings=settings)) == 1


def test_data_helpers_reject_escaping_paths(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    with pytest.raises(ValueError, match="outside"):
        save_processed(pd.DataFrame({"a": [1]}), "/etc/passwd", settings=settings)
    with pytest.raises(ValueError, match="outside"):
        load_raw("../../secret.csv", settings=settings)
