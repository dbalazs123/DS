# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Persistable fit parameters: every `fit_*` dataclass (`OutlierBounds`,
  `ImputeValues`, `ScaleParams`, `OneHotCategories`, `OrdinalCategories`) now
  has a validated `to_dict`/`from_dict` round-trip, and `ds.io` gained
  `save_params`/`load_params` (plus the `FittedParams` protocol) to persist
  them as strict JSON — so a pipeline can save its fitted state alongside the
  model and score new rows in a later run or another process. Numpy scalar
  fills are stored as plain numbers, non-finite bounds as tagged mappings
  (JSON has no `inf`/`nan` literal), and vocabulary tuples round-trip through
  lists; `from_dict` validates the payload so a stale, hand-edited or
  wrong-type file fails with a clear error. Shared encode/decode plumbing
  lives in the private `ds._serde` module.
- Split-safe `fit_*`/`apply_*` pairs for every statistic-learning transform,
  each returning/consuming a small frozen parameters dataclass so statistics
  can be fitted on the training split and reused on test data or new rows:
  `ds.preprocessing.fit_outlier_bounds` (`OutlierBounds`) with
  `apply_clip_outliers`/`apply_flag_outliers`, `fit_impute_values`
  (`ImputeValues`) with `apply_impute_missing`, and
  `ds.features.fit_scale_params` (`ScaleParams`) with `apply_scale_features`,
  `fit_one_hot_categories` (`OneHotCategories`) with `apply_one_hot_encode`,
  `fit_ordinal_categories` (`OrdinalCategories`) with `apply_ordinal_encode`.
  The fixed category vocabulary guarantees identical encoded columns on every
  frame (unseen categories → all-zero indicators / `-1` codes). The existing
  single-call forms are now thin fit-and-apply-on-the-same-frame wrappers with
  unchanged behavior. `docs/guide.md` gained a "Fit on train, apply to test"
  cookbook section.
- `ROADMAP.md` and an "Engineering notes" section in `CLAUDE.md` capturing the
  planned stage build-out and hard-won tooling gotchas.
- `ds.eda`: `missing_value_report` (columns with gaps, ranked) and
  `top_correlations` (most-correlated numeric pairs, for redundancy/leakage
  screening).
- `ds.evaluation`: `confusion_frame` (labeled confusion matrix) and
  `per_class_metrics` (per-class precision/recall/f1/support).
- `ds.viz` plotting helpers that return a matplotlib `Axes`: `plot_missingness`,
  `plot_confusion_matrix` and `plot_residuals` (the first two visualize
  `missing_value_report` and `confusion_frame`).
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
- `projects/_example/pipeline.py` now saves all five fitted parameter objects
  next to its processed data with `save_params` and rebuilds them from disk
  with `load_params` to score fresh rows that did not exist at fit time
  (including a missing and an unseen region), closing the "new incoming rows"
  loop; `tests/test_example.py` asserts the reloaded state matches the fitted
  state and the fresh rows encode to the training feature columns.
  `docs/guide.md`'s split-safe cookbook section documents the pattern.
- `projects/_example/pipeline.py` now splits chronologically *before* any
  statistic-learning transform and runs clip → impute → one-hot → scale as
  fit-on-train/apply-to-both via the new `fit_*`/`apply_*` pairs, so the
  held-out window no longer leaks into the learned statistics;
  `tests/test_example.py` asserts the split-safe behavior.
- Package version is now single-sourced from `__version__` in
  `src/ds/__init__.py` via Hatch's dynamic version, instead of being duplicated
  in `pyproject.toml`.
- CLAUDE.md branching guidance no longer hard-codes a (stale) branch name.
- `projects/_example/pipeline.py` now generates realistically dirty synthetic
  data (missing values, outliers, a categorical column, duplicate rows) and
  runs it through `ds.io.load_raw`/`save_processed`, `ds.validation.check_schema`,
  `ds.preprocessing.coerce_dtypes`/`drop_duplicate_rows`/`clip_outliers`/
  `impute_missing`, and `ds.features.one_hot_encode`/`scale_features`, so the
  worked example actually exercises the stage functions fleshed out above
  instead of data too clean to need them. `docs/guide.md` and the project
  template's `pipeline.py.jinja` comment now point at `load_raw` instead of
  `load_table`/`settings.raw_dir`.

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
