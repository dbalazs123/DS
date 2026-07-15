"""Smoke test for the worked example pipeline.

``projects/_example/pipeline.py`` is the repo's proof that the library composes
end to end. Run it here so a regression in any stage fails CI instead of quietly
breaking the flagship demo. The example lives outside the ``ds`` package, so we
load it by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "projects" / "_example" / "pipeline.py"


def _load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ds_example_pipeline", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    example = _load_example()
    metrics = example.run(tmp_path)

    # Every regression metric the evaluation stage promises should be present.
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert all(isinstance(value, float) for value in metrics.values())

    # The visualization stage should have written its figure artifact.
    assert (tmp_path / "forecast.png").is_file()
