"""Pin the public top-level surface of the ``ds`` package.

DS keeps a strict *import-by-stage* convention: stage helpers are imported from
their stage (``from ds.eda import summarize``) and only stage-independent
infrastructure is re-exported into the top-level ``ds`` namespace. These tests
enforce that decision so a stray re-export (or a dropped one) fails CI rather
than silently widening or narrowing the promised surface.
"""

from __future__ import annotations

import subprocess
import sys

import ds

# The complete, curated top-level surface. Adding a name here is a deliberate
# API decision, not an accident — update this set in the same change.
EXPECTED_PUBLIC_NAMES = {
    "Settings",
    "__version__",
    "get_logger",
    "get_settings",
    "seed_everything",
}

# A representative slice of stage helpers (and the pipeline composer) that must
# stay import-by-stage — i.e. must NOT leak into the ``ds`` namespace.
STAGE_ONLY_NAMES = [
    "summarize",  # ds.eda
    "load_table",  # ds.io
    "require_columns",  # ds.validation
    "standardize_column_names",  # ds.preprocessing
    "one_hot_encode",  # ds.features
    "split_features_target",  # ds.modeling
    "regression_metrics",  # ds.evaluation
    "set_theme",  # ds.viz
    "Pipeline",  # ds.pipeline
    "PipelineStep",  # ds.pipeline
]


def test_all_is_exactly_the_curated_surface() -> None:
    assert set(ds.__all__) == EXPECTED_PUBLIC_NAMES


def test_every_all_name_resolves() -> None:
    for name in ds.__all__:
        assert hasattr(ds, name), f"{name!r} is in __all__ but not importable from ds"


def test_stage_helpers_are_not_re_exported() -> None:
    leaked = [name for name in STAGE_ONLY_NAMES if hasattr(ds, name)]
    assert not leaked, (
        f"stage-only names leaked into the top-level ds namespace: {leaked}; "
        "import them by stage instead"
    )


def test_importing_ds_stays_cheap() -> None:
    """``import ds`` must not eagerly pull in a stage's heavy dependencies.

    Run in a fresh interpreter so other tests (which do import matplotlib) can't
    pre-populate ``sys.modules``. This is the load-bearing half of the
    import-by-stage decision: the top level stays light precisely because it
    does not re-export the stages.
    """
    check = (
        "import sys, ds; "
        "heavy = [m for m in ('matplotlib', 'sklearn', 'ds.viz', 'ds.modeling') "
        "if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    result = subprocess.run(
        [sys.executable, "-c", check],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"importing ds eagerly loaded heavy modules: {result.stdout}{result.stderr}"
