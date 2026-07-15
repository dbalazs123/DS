# CLAUDE.md

Guidance for AI assistants (Claude Code and others) working in this repository.

## Project

**DS** — "Data science toolkit for every situation."

The project is at its earliest stage. As of this writing the repository contains
only `README.md`; there is no source code, package layout, tests, or build
tooling committed yet. Treat this file as the seed of the codebase conventions,
and **keep it updated as real structure lands** — when you add the first package,
dependency manifest, or test suite, revise the relevant sections here in the same
change.

## Current state of the repository

```
.
├── README.md      # One-line project description
└── CLAUDE.md      # This file
```

There is no `pyproject.toml`, `requirements.txt`, `setup.py`, `src/` tree, test
directory, CI config, or linter config yet. Do not reference or assume any of
these exist until they do. If a task depends on one of them, create it as part
of the task rather than pretending it is already there.

## Establishing conventions (when adding the first code)

The name signals a Python data science library. Nothing is locked in, so when you
create the initial project scaffolding, prefer these widely-used defaults unless
the user asks otherwise, and record what you chose back in this file:

- **Packaging:** a `pyproject.toml` (PEP 621) as the single source of project
  metadata and dependencies. Pick one build/dependency tool (e.g. `uv`, `poetry`,
  or plain `pip` + `venv`) and note the choice here so it stays consistent.
- **Layout:** a `src/`-based package (`src/ds/`) to avoid import-path surprises,
  with `__init__.py` exposing the public API.
- **Tests:** `pytest`, with tests under `tests/` mirroring the package tree.
- **Formatting & linting:** `ruff` (lint + format) and `mypy` for type checking.
  Add configuration to `pyproject.toml`.
- **Notebooks:** if example or exploratory notebooks are added, keep them in a
  `notebooks/` or `examples/` directory, not mixed into the package source.

Once these exist, replace this section with the actual, verified commands
(install, test, lint, typecheck) that a contributor runs.

## Development workflow

### Branching

Active development for this task happens on the branch **`claude/claude-md-docs-ent70v`**.

- Develop changes on the designated feature branch; create it locally from the
  latest default branch if needed.
- Do **not** push to `master` (the default branch) directly.
- Commit with clear, descriptive messages; push with `git push -u origin <branch>`.
- Do not open a pull request unless the user explicitly asks for one.

### Commits

- Keep commits focused and descriptive.
- Do not commit data files, credentials, model checkpoints, or large binaries.
  Add a `.gitignore` covering `__pycache__/`, `.venv/`, `*.pyc`, `.ipynb_checkpoints/`,
  `.pytest_cache/`, `.mypy_cache/`, and common data/artifact directories before the
  first substantive code commit.

## Conventions for AI assistants

- **Do not invent structure.** This repository is nearly empty; describe and act
  on what actually exists. Verify with `git ls-files` / `find` before claiming a
  file or module is present.
- **Keep this file honest.** When you add code, tooling, or workflows, update the
  matching section here in the same change so the documentation never drifts ahead
  of (or behind) reality.
- **Prefer small, verifiable steps.** After adding tooling, run it and record the
  exact working commands rather than aspirational ones.
- **Data science hygiene:** keep raw data out of version control, make analyses
  reproducible (pin dependencies, seed randomness where relevant), and separate
  reusable library code from one-off exploratory notebooks.
