"""Persist and reload fitted models.

Completes the fit-once/score-later story: fitted *parameters* travel as
validated JSON (``ds.io.save_params``/``load_params``), and the fitted
*estimator* travels through these helpers, so a later run — or another
process — can reload both and score new rows without refitting.

Models are serialized with ``joblib`` (the format scikit-learn's own docs
recommend for estimators). That is a pickle under the hood, which is exactly
why it is used **only** for the model: unpickling executes arbitrary code
from the file, so :func:`load_model` must only ever be pointed at files
written by you or a process you trust. The transform parameters deliberately
stay in strict, validated JSON instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def save_model(model: object, path: str | Path) -> Path:
    """Serialize a fitted model to disk with ``joblib``.

    Creates parent directories as needed. Any picklable object works; the
    conventional suffix is ``.joblib``.

    Args:
        model: The fitted model (e.g. a scikit-learn estimator) to persist.
        path: Destination file path.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path) -> Any:
    """Reload a model persisted by :func:`save_model`.

    Warning:
        Loading unpickles the file, which can execute arbitrary code. Only
        load model files written by you or a process you trust — never one
        from an untrusted source. (Fitted transform parameters don't carry
        this risk: they persist as validated JSON via
        :func:`ds.io.save_params`/:func:`ds.io.load_params`.)

    Args:
        path: Path to a file written by :func:`save_model`.

    Returns:
        The deserialized model.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model file at {path}")
    return joblib.load(path)


__all__ = ["load_model", "save_model"]
