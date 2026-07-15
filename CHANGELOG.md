# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ds.eda`: `missing_value_report` (columns with gaps, ranked) and
  `top_correlations` (most-correlated numeric pairs, for redundancy/leakage
  screening).
- `ds.evaluation`: `confusion_frame` (labeled confusion matrix) and
  `per_class_metrics` (per-class precision/recall/f1/support).
- `ds` command-line interface (`ds version`, `ds new <name>`); `ds new`
  scaffolds a project from the Copier template.
- CI now runs a `test-extras` job that adds the lightweight `tiktoken`
  dependency and re-runs the suite, exercising the accurate token-counting path
  instead of only its fallback (kept lean — it avoids the heavier `nlp`/`all`
  extras that no code exercises yet).
- Documentation `Guide` page: a per-stage cookbook and the new-project workflow.
- Project template now scaffolds a stage-by-stage `pipeline.py` skeleton, a
  per-project `notebooks/` folder, and a `tests/` folder with a starter test.
- `tests/test_example.py` runs the worked `_example` pipeline end to end so a
  regression in any lifecycle stage fails CI.
- `nbstripout` pre-commit hook to strip notebook outputs before they reach git.

### Changed
- Package version is now single-sourced from `__version__` in
  `src/ds/__init__.py` via Hatch's dynamic version, instead of being duplicated
  in `pyproject.toml`.
- CLAUDE.md branching guidance no longer hard-codes a (stale) branch name.

### Fixed
- `ds.modeling.nlp.count_tokens` now falls back to its whitespace estimate when
  `tiktoken` is installed but its encoding cannot be loaded (e.g. the vocabulary
  download fails offline), instead of raising — honoring the module's
  "always callable" contract.
- Release workflow no longer attaches a stray `dist/.gitignore` asset; it now
  uploads only the built wheel and sdist.

## [0.1.0] - 2026-07-15

### Added
- Initial toolkit scaffold: `uv` packaging, `ruff` + `mypy --strict` + `pytest`
  tooling, and GitHub Actions CI.
- `ds` library organized by data-science process: `io`, `validation`,
  `preprocessing`, `eda`, `features`, `modeling` (tabular / timeseries / nlp),
  `evaluation`, `viz`, plus cross-cutting `config`, `logging`, `reproducibility`.
- Hybrid workspace: `projects/` (with a worked `_example` pipeline), a `copier`
  project template under `templates/`, and `notebooks/` + git-ignored `data/`.
- Documentation site (`mkdocs-material` + `mkdocstrings`) and contributor docs.
- Continuous-improvement config: pre-commit hooks, Dependabot, issue/PR templates.
- CI hardening: `pre-commit`-driven lint/type job, an enforced coverage gate
  (`--cov-fail-under`), a strict docs build that validates on PRs and deploys to
  GitHub Pages on `master`, and a release workflow (tag push or manual
  `workflow_dispatch`) that builds the package and publishes a GitHub Release.
