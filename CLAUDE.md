# CLAUDE.md

Guidance for AI assistants (Claude Code and others) working in this repository.

## Project

**DS** — "Data science toolkit for every situation."

DS is a **hybrid workspace**: a reusable library (`src/ds/`) organized by
data-science *process*, a `projects/` area for individual analyses that consume
the library, and a `templates/` scaffold for new projects. Keep this file honest
— when structure, tooling, or commands change, update the matching section here
in the **same** change.

## Current state of the repository

```
.
├── pyproject.toml         # PEP 621 metadata + deps + tool config (uv, ruff, mypy, pytest)
├── Makefile               # thin task wrappers (make help)
├── src/ds/                # reusable library, organized by DS process
│   ├── config.py logging.py reproducibility.py   # cross-cutting
│   ├── io/ validation/ preprocessing/ eda/ features/
│   ├── modeling/          # tabular.py timeseries.py nlp.py
│   ├── evaluation/ viz/ utils/
│   └── py.typed
├── projects/              # analyses/experiments (see _example/pipeline.py)
├── templates/project/     # copier template for new projects
├── notebooks/             # exploratory notebooks
├── data/                  # git-ignored: raw/ interim/ processed/
├── docs/                  # mkdocs-material + mkdocstrings
├── tests/                 # mirrors src/ds/
└── .github/workflows/ci.yml
```

## Tooling & commands (verified)

- **Packaging / env:** `uv` (build backend: hatchling). Lean core deps; heavy
  domain stacks live in optional extras `nlp` and `timeseries`. Dev tools are in
  the `dev` dependency group (installed by `uv sync`).
- **Layout:** `src/ds/` package with `py.typed`; public API re-exported from
  `src/ds/__init__.py`.
- **Tests:** `pytest` under `tests/`, with coverage on `ds`.
- **Lint/format:** `ruff`. **Types:** `mypy` in `--strict` mode.
- **Notebooks:** in `notebooks/`, never mixed into `src/ds/`.

Commands a contributor runs:

```bash
uv sync            # create env, install ds + dev tools
make check         # lint + typecheck + test (what CI runs)
make lint          # uv run ruff check .
make format        # uv run ruff format . && ruff check --fix .
make typecheck     # uv run mypy
make test          # uv run pytest
make docs          # uv run mkdocs build
```

When adding a utility, place it in the module for its lifecycle stage, give it a
Google-style docstring and full type hints (the codebase is `mypy --strict`), add
a mirroring test, and export it from the module's `__all__`. Heavy dependencies
go in the right extra and must degrade gracefully when absent (see
`src/ds/modeling/nlp.py`).

## Development workflow

### Branching

Do work on a per-task feature branch, not on a branch named here (a hard-coded
name only goes stale). If the task specifies a branch, use that one; otherwise
create a descriptive branch from the latest default branch.

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

- **Do not invent structure.** Describe and act on what actually exists. Verify
  with `git ls-files` / `find` before claiming a file or module is present.
- **Keep this file honest.** When you add code, tooling, or workflows, update the
  matching section here in the same change so the documentation never drifts ahead
  of (or behind) reality.
- **Prefer small, verifiable steps.** After adding tooling, run it and record the
  exact working commands rather than aspirational ones.
- **Data science hygiene:** keep raw data out of version control, make analyses
  reproducible (pin dependencies, seed randomness where relevant), and separate
  reusable library code from one-off exploratory notebooks.
