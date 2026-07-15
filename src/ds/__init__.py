"""DS — a data science toolkit for every situation.

The package is organized by data-science *process* rather than by data type.
Each subpackage owns one stage of the lifecycle:

- :mod:`ds.io` — data acquisition, loading and saving
- :mod:`ds.validation` — schema and data-quality checks
- :mod:`ds.preprocessing` — cleaning and reshaping
- :mod:`ds.eda` — exploratory summaries
- :mod:`ds.features` — feature engineering
- :mod:`ds.modeling` — training and tuning helpers
- :mod:`ds.evaluation` — metrics and reporting
- :mod:`ds.viz` — plotting with a consistent theme

Cross-cutting concerns live at the top level: :mod:`ds.config`,
:mod:`ds.logging` and :mod:`ds.reproducibility`.
"""

from __future__ import annotations

from ds.config import Settings, get_settings
from ds.logging import get_logger
from ds.reproducibility import seed_everything

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "__version__",
    "get_logger",
    "get_settings",
    "seed_everything",
]
