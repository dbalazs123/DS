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
│   ├── config.py logging.py reproducibility.py cli.py   # cross-cutting + `ds` CLI
│   ├── pipeline.py        # fit-once/apply-many Pipeline + fit_pipeline plan executor
│   ├── _serde.py          # private: fitted-parameter dict/JSON round-trip helpers
│   ├── io/ validation/ preprocessing/ eda/ features/
│   ├── modeling/          # tabular.py timeseries.py nlp.py
│   ├── evaluation/ viz/ utils/
│   └── py.typed
├── projects/              # analyses/experiments (_example = synthetic teaching
│                          #   reference; nyc_taxis = real-data fare prediction;
│                          #   titanic = real-data survival classification;
│                          #   flights = real-data passenger forecasting;
│                          #   diamonds = real-data cut grading, multiclass;
│                          #   sms_spam = real-data spam detection, text;
│                          #   air_quality = real-data sensor gap-filling,
│                          #   hourly-axis regression with real missingness;
│                          #   adult_income = real-data income classification,
│                          #   heavily categorical / high-cardinality)
├── templates/project/     # copier template for new projects
├── notebooks/             # exploratory notebooks
├── data/                  # git-ignored: raw/ interim/ processed/
├── docs/                  # mkdocs-material + mkdocstrings
├── tests/                 # mirrors src/ds/
└── .github/workflows/ci.yml
```

## Tooling & commands (verified)

- **Packaging / env:** `uv` (build backend: hatchling). Lean core deps; heavier
  dependencies live in optional extras (currently just `nlp` = tiktoken), and an
  extra only carries deps that library code actually consumes — add the dep in
  the same change as its first consumer. Dev tools are in the `dev` dependency
  group (installed by `uv sync`).
- **Layout:** `src/ds/` package with `py.typed`; public API re-exported from
  `src/ds/__init__.py`.
- **Tests:** `pytest` under `tests/`, with coverage on `ds`.
- **Lint/format:** `ruff`. **Types:** `mypy` in `--strict` mode.
- **Notebooks:** in `notebooks/`, never mixed into `src/ds/`.

Commands a contributor runs:

```bash
uv sync            # create env, install ds + dev tools
ds new "<name>"    # scaffold a project under projects/ (wraps copier)
ds run "<name>"    # run projects/<slug>/pipeline.py, resolved by name/slug
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
`src/ds/modeling/nlp.py`). [`CONTRIBUTING.md`](CONTRIBUTING.md) has the full
contributor workflow (setup, the one-command `make check`, and this recipe in
more detail).

## Roadmap

[`ROADMAP.md`](ROADMAP.md) carries the plan of record (P1–P13 all done; P12
ran the sixth demand loop — `projects/air_quality`, the first project against
real instrument-outage missingness on a gapped hourly axis — and **P13 served
its backlog** (items 22–26): built `cross_validate_by_time(make_pipeline=...)`
(item 22, item 9's parked question — the rolling-origin twin of the k-fold
factory; the per-fold refit measurably moves the CV numbers here where titanic
saw no change) and the `ds.validation` guards `assert_unique` / `assert_row_count`
(items 24/25, two second-project triggers fired twice), and resolved items 23
and 26 by documentation. The demand queue is now **empty** — the next step is a
seventh demand loop: a new real-data project whose friction regenerates the
backlog. The doc also carries a goal evaluation of the whole toolkit, the
friction backlogs from the real-data projects, and the settled-decision
rationales this file's notes point to. Read it before starting new library
work — and note its ordering rule: new library work should trace to a friction
item from a real project, not a brainstormed candidate list.

## Engineering notes (hard-won gotchas)

Things that cost a round-trip to discover; save yourself the CI failure:

- **Run `make format` before committing, not just `make lint`.** CI's `lint`
  job runs the full pre-commit suite, which includes `ruff-format`. `ruff check`
  passing does **not** mean the formatter is satisfied — e.g. a two-line
  function signature ruff wants collapsed will fail the lint job.
- **`git add` new files *before* running `pre-commit run --all-files`.**
  pre-commit only sees git-tracked files, so a pre-flight run over freshly
  created (still-untracked) files silently checks nothing. Also note the
  pinned pre-commit ruff (`.pre-commit-config.yaml` rev) trails the dev-env
  ruff, so a rule retired in the newer version (e.g. UP027) can still fail CI
  even when `make lint`/`make format` are clean.
- **`mypy --strict` + pandas-stubs quirks:** `DataFrame.corr(method=...)` wants a
  `Literal["pearson","kendall","spearman"]`, not `str`; and `df.loc[a, b]`
  returns a broad scalar union that `float()` rejects — index positionally via
  `.to_numpy()` instead. matplotlib is type-checked (`py.typed`), so annotate
  plot helpers `-> Axes` and resolve the optional `ax` explicitly.
- **Coverage gate is enforced at 85%** (`--cov-fail-under` in `pyproject.toml`).
  Running a single project's tests needs `uv run pytest projects/<name> --no-cov`,
  since the gate measures the whole `ds` package.
- **The `test-extras` CI job installs `--extra all` and re-runs the suite.**
  Extras only carry deps that code actually consumes (today: `tiktoken`), so the
  job stays cheap by construction. If you add code behind a new extra, declare
  the dep in that extra and this job exercises it automatically.
- **CI enforces the lock: every job runs `uv sync --locked`.** After editing any
  dependency in `pyproject.toml`, run `uv lock` and commit the refreshed
  `uv.lock` in the same change, or CI fails loudly (drift no longer silently
  re-resolves). Bump Python deps deliberately with `uv lock --upgrade`.
  Dependabot is scoped to **GitHub Actions only** (`.github/dependabot.yml`): it
  can't regenerate a uv lockfile, so a pip PR would be un-mergeable against the
  `--locked` gate — that's why Python-dep bumps are a manual `uv lock` step.
- **Public-API convention (settled — import by stage):** stage functions are
  imported by stage (`from ds.eda import ...`), and `ds.pipeline.Pipeline` /
  `PipelineStep` likewise from `ds.pipeline`; only the stage-independent
  infrastructure (`Settings`, `get_settings`, `get_logger`, `seed_everything`)
  is re-exported from `src/ds/__init__.py`. This was the deliberate resolution
  of the "API discoverability" question (see ROADMAP.md): a flat top-level
  re-export was rejected because the stage name is the teaching tool, and
  because re-exporting the stages would force `import ds` to eagerly load
  matplotlib/scikit-learn and risk cross-stage name collisions.
  `tests/test_public_api.py` pins the exact top-level surface, so widening or
  narrowing it fails CI. The package version is single-sourced in
  `src/ds/__init__.py` via Hatch's dynamic version (`[tool.hatch.version]`).
- **CLI scope is deliberately narrow (settled).** The `ds` CLI carries
  `version`, `new`, and `run` — commands that are *project-aware* (they know the
  `projects/` layout and the `ds new` slug convention) and so add something the
  raw command doesn't. `ds check` was considered and **rejected**: it would only
  shell out to the same tools `make check` already orchestrates, and `make` is
  the canonical dev entry point (README/CLAUDE/CONTRIBUTING all assume it), so a
  second entry point would either duplicate the sequence (drift risk against the
  Makefile) or just call `make` (adding nothing). Don't re-add it — see
  ROADMAP.md for the full rationale. `ds run` resolves a name against existing
  `projects/` directories rather than building a path from it, keeping the same
  path-traversal discipline as `ds new`'s slug.
- **Reuse across stages** rather than duplicating (e.g. `ds.viz` plots call
  `ds.eda` / `ds.evaluation` functions).
- **Verified raw fetch lives in `ds.io.fetch_dataset`** (checksum required,
  keyword-only; cache re-verify is inside the helper). Real-data projects whose
  canonical host isn't reachable everywhere download through it against pinned
  personal mirrors — don't re-hand-roll the multi-mirror/sha256/cache-reverify
  dance in a project. The seaborn-mirror projects (`titanic`/`nyc_taxis`/
  `diamonds`) deliberately keep their plain un-pinned "download if absent" (a few
  inline lines, below the aliasing bar); folding them in is parked until one
  actually pulls an optional-checksum widening (see ROADMAP item 27).
- **Sanitize user-facing input paths.** `ds new`'s slug collapses any
  non-`[a-z0-9]` run to `_` precisely so a name like `../x` can't escape
  `projects/` — keep that discipline for anything that builds a filesystem path.
- **Stacked PRs merge with merge-commits, not squash**, to keep the stack
  conflict-free (squash/rebase rewrites SHAs the child branches don't recognize).

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
