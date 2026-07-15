"""Data acquisition: loading and saving tabular data.

A thin, format-aware layer over pandas that infers the format from the file
extension so calling code never branches on ``.csv`` vs ``.parquet``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_SUFFIXES = (".csv", ".parquet", ".json")


def load_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a tabular file into a DataFrame, inferring the format.

    Args:
        path: Path to a ``.csv``, ``.parquet`` or ``.json`` file.
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
    if suffix == ".parquet":
        parquet_df: pd.DataFrame = pd.read_parquet(path, **kwargs)
        return parquet_df
    if suffix == ".json":
        json_df: pd.DataFrame = pd.read_json(path, **kwargs)
        return json_df
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
    elif suffix == ".parquet":
        df.to_parquet(path, index=False, **kwargs)
    elif suffix == ".json":
        df.to_json(path, **kwargs)
    else:
        raise ValueError(f"Unsupported extension {suffix!r}; expected one of {SUPPORTED_SUFFIXES}")
    return path


__all__ = ["SUPPORTED_SUFFIXES", "load_table", "save_table"]
