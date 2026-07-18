"""Tests for the io subpackage."""

from __future__ import annotations

import hashlib
import json
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from ds.config import Settings
from ds.features import OneHotCategories, ScaleParams
from ds.io import (
    fetch_dataset,
    load_params,
    load_raw,
    load_table,
    save_params,
    save_processed,
    save_table,
)
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


# --- fetch_dataset --------------------------------------------------------


class _FakeResponse:
    """Minimal context-manager stand-in for ``urllib.request.urlopen``."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _patch_urlopen(
    monkeypatch: pytest.MonkeyPatch, responses: dict[str, bytes | Exception]
) -> list[str]:
    """Route ``urlopen`` to ``responses`` by URL; record the URLs it was called with.

    A mapped ``Exception`` value is raised (an unreachable mirror); ``bytes`` are
    served as a fake response. Returns a list the caller can assert against to
    prove which mirrors were hit (and that a valid cache short-circuits the net).
    """
    calls: list[str] = []

    def fake_urlopen(url: str) -> _FakeResponse:
        calls.append(url)
        outcome = responses[url]
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(outcome)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return calls


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_fetch_dataset_downloads_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_dir=tmp_path)
    payload = b"col\n1\n"
    url = "https://mirror.example/data.csv"
    calls = _patch_urlopen(monkeypatch, {url: payload})

    path = fetch_dataset("data.csv", [url], sha256=_digest(payload), settings=settings)

    assert path == settings.raw_dir / "data.csv"
    assert path.read_bytes() == payload
    assert calls == [url]


def test_fetch_dataset_uses_valid_cache_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_dir=tmp_path)
    payload = b"cached\n"
    settings.raw_dir.mkdir(parents=True)
    (settings.raw_dir / "data.csv").write_bytes(payload)
    # Any network call would raise, so returning proves the cache short-circuited.
    calls = _patch_urlopen(monkeypatch, {})

    path = fetch_dataset(
        "data.csv", ["https://mirror.example/data.csv"], sha256=_digest(payload), settings=settings
    )

    assert path.read_bytes() == payload
    assert calls == []


def test_fetch_dataset_reverifies_and_replaces_corrupt_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_dir=tmp_path)
    good = b"good bytes\n"
    settings.raw_dir.mkdir(parents=True)
    (settings.raw_dir / "data.csv").write_bytes(b"partial download")  # wrong checksum
    url = "https://mirror.example/data.csv"
    calls = _patch_urlopen(monkeypatch, {url: good})

    path = fetch_dataset("data.csv", [url], sha256=_digest(good), settings=settings)

    assert path.read_bytes() == good  # the poisoned cache was re-downloaded
    assert calls == [url]


def test_fetch_dataset_falls_back_across_mirrors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_dir=tmp_path)
    payload = b"served by second mirror\n"
    first, second = "https://down.example/data.csv", "https://up.example/data.csv"
    calls = _patch_urlopen(
        monkeypatch, {first: urllib.error.URLError("unreachable"), second: payload}
    )

    path = fetch_dataset("data.csv", [first, second], sha256=_digest(payload), settings=settings)

    assert path.read_bytes() == payload
    assert calls == [first, second]  # tried in order, stopped at the first that served


def test_fetch_dataset_rejects_wrong_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_dir=tmp_path)
    url = "https://mirror.example/data.csv"
    _patch_urlopen(monkeypatch, {url: b"drifted bytes\n"})

    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch_dataset("data.csv", [url], sha256=_digest(b"expected bytes\n"), settings=settings)
    assert not (settings.raw_dir / "data.csv").exists()  # bad bytes never written


def test_fetch_dataset_raises_when_all_mirrors_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_dir=tmp_path)
    urls = ["https://a.example/d.csv", "https://b.example/d.csv"]
    _patch_urlopen(monkeypatch, {u: urllib.error.URLError("down") for u in urls})

    with pytest.raises(urllib.error.URLError, match="no mirror reachable"):
        fetch_dataset("d.csv", urls, sha256=_digest(b"anything"), settings=settings)


def test_fetch_dataset_rejects_escaping_name(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    with pytest.raises(ValueError, match="outside"):
        fetch_dataset(
            "../escape.csv", ["https://mirror.example/x"], sha256="0" * 64, settings=settings
        )

    # A payload that is not even a mapping fails the same way.
    (tmp_path / "list.json").write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_params(tmp_path / "list.json", ScaleParams)
