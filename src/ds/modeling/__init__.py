"""Model training and tuning helpers, grouped by domain.

Submodules:
    - :mod:`ds.modeling.tabular` — classical tabular ML
    - :mod:`ds.modeling.timeseries` — temporal splitting and forecasting
    - :mod:`ds.modeling.baseline` — naive baselines every model must beat
    - :mod:`ds.modeling.nlp` — text and LLM helpers
    - :mod:`ds.modeling.persistence` — save/reload fitted models
"""

from __future__ import annotations

from ds.modeling.baseline import Baseline, fit_baseline
from ds.modeling.persistence import load_model, save_model
from ds.modeling.tabular import split_features_target, train_test_split_random
from ds.modeling.timeseries import train_test_split_by_time

__all__ = [
    "Baseline",
    "fit_baseline",
    "load_model",
    "save_model",
    "split_features_target",
    "train_test_split_by_time",
    "train_test_split_random",
]
