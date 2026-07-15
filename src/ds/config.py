"""Project configuration and canonical filesystem paths.

Settings are read from environment variables (prefixed ``DS_``) or a local
``.env`` file, so the same code runs unchanged across a laptop, CI and a
notebook. Access the singleton via :func:`get_settings`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: this file is src/ds/config.py, so three parents up.
_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent.parent


class Settings(BaseSettings):
    """Runtime configuration for the toolkit.

    Attributes:
        data_dir: Root directory for datasets (kept out of version control).
        random_seed: Default seed used across the toolkit for reproducibility.
        log_level: Logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.
    """

    model_config = SettingsConfigDict(
        env_prefix="DS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=_REPO_ROOT / "data")
    random_seed: int = Field(default=42)
    log_level: str = Field(default="INFO")

    @property
    def raw_dir(self) -> Path:
        """Directory for immutable, original source data."""
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        """Directory for intermediate, in-progress transformations."""
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        """Directory for final, analysis-ready datasets."""
        return self.data_dir / "processed"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached :class:`Settings` singleton."""
    return Settings()
