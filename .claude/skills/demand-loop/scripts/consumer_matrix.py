"""Map the ds public surface to its real-project consumers.

The demand loop picks the next project by finding which library surfaces are
*thinnest by real consumers* — a stage whose helpers only one project (or none)
has ever reached for is a candidate the next dataset should stress. This script
does the mechanical part: parse every ``__all__`` under ``src/ds/``, then grep
each ``projects/<name>/`` (excluding ``_example``) for uses of each name.

Run it from the repo root with
``python .claude/skills/demand-loop/scripts/consumer_matrix.py``.

Read the output with two caveats the raw grep cannot know:

- **Pipeline indirection.** The ``apply_*`` transforms are consumed *through*
  ``fit_pipeline``/``Pipeline`` (the plan executor calls them), and
  ``load_table``/``save_table`` through ``load_raw``/``save_processed``. They
  show as zero direct consumers but are not genuinely unconsumed — discount them.
- **Thin != unbuilt.** A surface with one consumer is a candidate to stress a
  second time, not proof of a gap. The judgment — which *data shape* stresses the
  thinnest cluster — is yours; this only surfaces the counts.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# Stage-independent infrastructure names that every project imports as plumbing;
# their consumer counts carry no signal for choosing the next data shape.
_INFRA = {
    "Settings",
    "get_settings",
    "get_logger",
    "seed_everything",
    "__version__",
    "timer",
    "PALETTE",
    "SUPPORTED_SUFFIXES",
    "MetricsFunction",
    "StepKind",
    "StepParams",
    "PipelineStep",
    "DataValidationError",
}


def public_names(src: Path) -> dict[str, str]:
    """Every ``__all__`` entry under ``src`` mapped to the module it lives in."""
    names: dict[str, str] = {}
    for path in sorted(src.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                continue
            try:
                values = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
            module = (
                str(path.relative_to(src.parent)).replace("/__init__.py", "").replace(".py", "")
            )
            for name in values:
                names.setdefault(name, module)
    return names


def main() -> None:
    root = Path.cwd()
    names = public_names(root / "src")
    projects = sorted(
        p.name
        for p in (root / "projects").iterdir()
        if p.is_dir() and p.name != "_example" and not p.name.startswith(".")
    )
    text = {
        pr: "".join(f.read_text() for f in (root / "projects" / pr).rglob("*.py"))
        for pr in projects
    }

    rows: list[tuple[int, str, str, list[str]]] = []
    for name in sorted(n for n in names if n not in _INFRA):
        users = [pr for pr in projects if re.search(rf"\b{re.escape(name)}\b", text[pr])]
        rows.append((len(users), name, names[name], users))

    print(f"{len(projects)} real-data projects: {', '.join(projects)}\n")
    print(f"{'count':>5}  {'function':<28}  consumers")
    for count, name, _module, users in sorted(rows):
        print(f"{count:>5}  {name:<28}  {', '.join(users) or '-'}")

    thin = [(c, n) for c, n, _m, _u in rows if c <= 1]
    print("\nThinnest surfaces (<=1 direct consumer) — candidates to stress next:")
    for count, name in sorted(thin):
        print(f"  [{count}] {name}")
    print(
        "\nReminder: discount apply_* (used via fit_pipeline) and "
        "load_table/save_table (via load_raw/save_processed) — not truly unconsumed."
    )


if __name__ == "__main__":
    main()
