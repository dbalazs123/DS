"""Tests for the io subpackage."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ds.config import Settings
from ds.features import OneHotCategories, ScaleParams
from ds.io import load_params, load_raw, load_table, save_params, save_processed, save_table
from ds.preprocessing import OutlierBounds


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


def test_save_load_params_round_trip(tmp_path: Path) -> None:
    bounds = OutlierBounds(bounds={"x": (-float("inf"), 9.5)}, method="iqr", factor=1.5)
    path = save_params(bounds, tmp_path / "nested" / "bounds.json")
    assert path == tmp_path / "nested" / "bounds.json"  # parents were created
    assert load_params(path, OutlierBounds) == bounds
    # The file is strict JSON: no bare Infinity/NaN literals.
    json.loads(path.read_text(), parse_constant=lambda name: pytest.fail(f"bare {name}"))


def test_params_helpers_require_json_suffix(tmp_path: Path) -> None:
    params = ScaleParams(center={"x": 0.0}, spread={"x": 1.0}, method="standard")
    with pytest.raises(ValueError, match="Unsupported"):
        save_params(params, tmp_path / "params.yaml")
    (tmp_path / "params.txt").write_text("{}")
    with pytest.raises(ValueError, match="Unsupported"):
        load_params(tmp_path / "params.txt", ScaleParams)


def test_load_params_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_params(tmp_path / "nope.json", ScaleParams)


def test_load_params_rejects_invalid_json_and_wrong_class(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_params(broken, ScaleParams)

    # A file holding one parameter type must not load as another.
    saved = save_params(
        OneHotCategories(categories={"c": ("a", "b")}, drop_first=False, dummy_na=False),
        tmp_path / "vocab.json",
    )
    with pytest.raises(ValueError, match="expected a 'ScaleParams' payload"):
        load_params(saved, ScaleParams)

    # A payload that is not even a mapping fails the same way.
    (tmp_path / "list.json").write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_params(tmp_path / "list.json", ScaleParams)
