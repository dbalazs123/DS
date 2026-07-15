"""Utilities for making analyses reproducible."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> int:
    """Seed Python, NumPy and the ``PYTHONHASHSEED`` environment variable.

    Optional deep-learning backends (PyTorch, TensorFlow) are seeded too when
    they happen to be importable, but they are never a hard dependency.

    Args:
        seed: The seed value to apply everywhere.

    Returns:
        The seed that was applied, for convenient logging.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # pragma: no cover - optional dependency
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch not installed by default
        pass

    return seed
