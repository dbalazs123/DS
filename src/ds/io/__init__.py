"""Data acquisition: loading and saving tabular data and fitted parameters.

A thin, format-aware layer over pandas that infers the format from the file
extension so calling code never branches on ``.csv`` vs ``.parquet``. The
:func:`load_raw` / :func:`save_processed` pair builds on it to resolve paths
against the project's ``data/`` layout (see :mod:`ds.config`), and the
:func:`save_params` / :func:`load_params` pair persists the ``fit_*``
parameter dataclasses as JSON so a pipeline's fitted state can be saved
alongside its model and reloaded in a later run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, Self, TypeVar

import pandas as pd

from ds.config import Settings, get_settings

SUPPORTED_SUFFIXES = (".csv", ".tsv", ".parquet", ".json", ".jsonl")


class FittedParams(Protocol):
    """Any fitted-parameter dataclass with a validated dict round-trip.

    The ``fit_*`` dataclasses (:class:`ds.preprocessing.OutlierBounds`,
    :class:`ds.preprocessing.ImputeValues`, :class:`ds.features.ScaleParams`,
    :class:`ds.features.OneHotCategories` and
    :class:`ds.features.OrdinalCategories`) all satisfy this protocol.
    """

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        ...  # pragma: no cover - protocol stub

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Rebuild the dataclass from :meth:`to_dict` output, validating it."""
        ...  # pragma: no cover - protocol stub


P = TypeVar("P", bound=FittedParams)


def save_params(params: FittedParams, path: str | Path) -> Path:
    """Save a fitted-parameter dataclass as JSON, creating parent directories.

    The file is strict JSON (``allow_nan=False``): non-finite values are
    tag-encoded by the dataclass's ``to_dict``, never written as the invalid
    ``Infinity``/``NaN`` literals. Reload with :func:`load_params`.

    Args:
        params: A ``fit_*`` result, e.g. :class:`ds.preprocessing.OutlierBounds`
            or :class:`ds.features.ScaleParams`.
        path: Destination path; must end in ``.json``.

    Returns:
        The path written to.

    Raises:
        ValueError: If ``path`` does not have a ``.json`` extension.
    """
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported extension {path.suffix!r}; parameters are saved as .json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params.to_dict(), indent=2, allow_nan=False) + "\n")
    return path


def load_params(path: str | Path, cls: type[P]) -> P:
    """Load a fitted-parameter dataclass saved by :func:`save_params`.

    The payload is validated by the class's ``from_dict``, so loading a file
    that holds a different parameter type — or a stale/hand-edited one —
    fails with a clear error instead of building a broken object.

    Args:
        path: Path to a ``.json`` file written by :func:`save_params`.
        cls: The expected dataclass, e.g. ``OutlierBounds``.

    Returns:
        The reconstructed parameters, as an instance of ``cls``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the extension is not ``.json``, the file is not valid
            JSON, or the payload is not a well-formed ``cls`` representation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported extension {path.suffix!r}; parameters are saved as .json")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    return cls.from_dict(payload)


def load_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a tabular file into a DataFrame, inferring the format.

    Args:
        path: Path to a ``.csv``, ``.tsv``, ``.parquet``, ``.json`` or
            ``.jsonl`` (JSON Lines) file.
        **kwargs: Passed through to the underlying pandas reader.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        csv_df: pd.DataFrame = pd.read_csv(path, **kwargs)
        return csv_df
    if suffix == ".tsv":
        tsv_df: pd.DataFrame = pd.read_csv(path, sep="\t", **kwargs)
        return tsv_df
    if suffix == ".parquet":
        parquet_df: pd.DataFrame = pd.read_parquet(path, **kwargs)
        return parquet_df
    if suffix == ".json":
        json_df: pd.DataFrame = pd.read_json(path, **kwargs)
        return json_df
    if suffix == ".jsonl":
        jsonl_df: pd.DataFrame = pd.read_json(path, lines=True, **kwargs)
        return jsonl_df
    raise ValueError(f"Unsupported extension {suffix!r}; expected one of {SUPPORTED_SUFFIXES}")


def save_table(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Save a DataFrame, inferring the format and creating parent directories.

    Args:
        df: The DataFrame to write.
        path: Destination path; format is taken from the extension.
        **kwargs: Passed through to the underlying pandas writer.

    Returns:
        The path written to.

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False, **kwargs)
    elif suffix == ".tsv":
        df.to_csv(path, sep="\t", index=False, **kwargs)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False, **kwargs)
    elif suffix == ".json":
        df.to_json(path, **kwargs)
    elif suffix == ".jsonl":
        df.to_json(path, orient="records", lines=True, **kwargs)
    else:
        raise ValueError(f"Unsupported extension {suffix!r}; expected one of {SUPPORTED_SUFFIXES}")
    return path


def _resolve_within(base: Path, name: str | Path) -> Path:
    """Resolve ``name`` under ``base``, refusing to escape the base directory.

    Guards the ``data/`` helpers against absolute paths and ``..`` traversal so
    a caller-supplied name can never write outside the configured tree.
    """
    base_resolved = base.resolve()
    candidate = (base_resolved / name).resolve()
    if not candidate.is_relative_to(base_resolved):
        raise ValueError(f"{name!r} resolves outside {base_resolved}")
    return candidate


def load_raw(name: str | Path, *, settings: Settings | None = None, **kwargs: Any) -> pd.DataFrame:
    """Load a table from the project's raw-data directory.

    Resolves ``name`` against ``settings.raw_dir`` (see :mod:`ds.config`) and
    then defers to :func:`load_table`, so the same code path works on a laptop,
    in CI or in a notebook without hard-coded paths.

    Args:
        name: File name (or relative path) under the raw-data directory.
        settings: Settings to resolve paths from; defaults to
            :func:`ds.config.get_settings`.
        **kwargs: Passed through to :func:`load_table`.

    Returns:
        The loaded DataFrame.

    Raises:
        ValueError: If ``name`` resolves outside the raw-data directory.
        FileNotFoundError: If the resolved file does not exist.
    """
    resolved = settings or get_settings()
    return load_table(_resolve_within(resolved.raw_dir, name), **kwargs)


def save_processed(
    df: pd.DataFrame, name: str | Path, *, settings: Settings | None = None, **kwargs: Any
) -> Path:
    """Save a table into the project's processed-data directory.

    Resolves ``name`` against ``settings.processed_dir`` (see :mod:`ds.config`)
    and then defers to :func:`save_table`, creating the directory if needed.

    Args:
        df: The DataFrame to write.
        name: File name (or relative path) under the processed-data directory.
        settings: Settings to resolve paths from; defaults to
            :func:`ds.config.get_settings`.
        **kwargs: Passed through to :func:`save_table`.

    Returns:
        The path written to.

    Raises:
        ValueError: If ``name`` resolves outside the processed-data directory.
    """
    resolved = settings or get_settings()
    return save_table(df, _resolve_within(resolved.processed_dir, name), **kwargs)


__all__ = [
    "SUPPORTED_SUFFIXES",
    "FittedParams",
    "load_params",
    "load_raw",
    "load_table",
    "save_params",
    "save_processed",
    "save_table",
]
