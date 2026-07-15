"""Model training and tuning helpers, grouped by domain.

Submodules:
    - :mod:`ds.modeling.tabular` — classical tabular ML
    - :mod:`ds.modeling.timeseries` — temporal splitting and forecasting
    - :mod:`ds.modeling.nlp` — text and LLM helpers
"""

from __future__ import annotations

from ds.modeling.tabular import split_features_target
from ds.modeling.timeseries import train_test_split_by_time

__all__ = ["split_features_target", "train_test_split_by_time"]
