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
:mod:`ds.logging`, :mod:`ds.reproducibility`, and :mod:`ds.pipeline`, which
composes the stages' ``fit_*``/``apply_*`` pairs into a persistable
fit-once/apply-many pipeline.

**Import convention.** Stage functions are imported from their stage
(``from ds.eda import summarize``, ``from ds.pipeline import Pipeline``); the
stage name is part of the API, telling you which lifecycle stage a helper
belongs to. Only the stage-independent infrastructure below is re-exported
into the top-level ``ds`` namespace — deliberately, so ``import ds`` stays
cheap (it never pulls in matplotlib via :mod:`ds.viz` or the modeling stacks)
and no two stages can collide in one flat namespace. See the "Importing from
DS" section of the Guide for the rationale.
"""

from __future__ import annotations

from ds.config import Settings, get_settings
from ds.logging import get_logger
from ds.reproducibility import seed_everything

# Single source of truth for the package version; Hatch reads it at build time
# (see [tool.hatch.version] in pyproject.toml).
__version__ = "0.2.0"

# The top-level surface is deliberately just the cross-cutting infrastructure
# every project reaches for regardless of lifecycle stage. Stage helpers (and
# ``ds.pipeline.Pipeline``) stay import-by-stage — see the module docstring and
# ``tests/test_public_api.py``, which pins this list.
__all__ = [
    "Settings",
    "__version__",
    "get_logger",
    "get_settings",
    "seed_everything",
]
