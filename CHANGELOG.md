# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Model/Evaluate build-out (P3 of `ROADMAP.md`'s plan of record, re-ranked
  against the `nyc_taxis` friction backlog first):
  - `ds.features.add_datetime_features` now also emits ``<column>_hour``
    (friction item 2 — hour of day was the taxi data's strongest temporal
    signal and had to be hand-rolled; on date-only data the column is
    constantly zero, where `drop_constant_columns` removes it).
  - `ds.modeling.baseline`: `fit_baseline` (``mean`` / ``naive_last`` /
    ``seasonal_naive``) returns a frozen `Baseline` whose `predict(n)` feeds
    `ds.evaluation` directly — the reference point every first metric needs
    (friction item 3), deliberately not a scikit-learn estimator since a
    baseline needs no feature matrix.
  - `ds.evaluation.cross_validate_by_time` (rolling-origin folds — every
    test window strictly in its training data's future, the repeated-fold
    counterpart to `train_test_split_by_time`) and `cross_validate_kfold`
    (order-free data), both building a fresh model per fold via a
    `make_model` factory and scoring with a `metrics_fn` that defaults to
    `regression_metrics` (pass `classification_metrics` or a custom scorer
    to compose instead of forking the API).
  - `ds.evaluation.compare_models` (one row of metrics per named model) and
    its paired `ds.viz.plot_model_comparison`, per the stage↔viz
    convention.
  - `projects/nyc_taxis` dogfoods it all: library `pickup_hour`,
    `fit_baseline` and a persisted comparison frame/plot replace its
    hand-rolled versions with identical metrics.
- `ds.modeling.persistence`: `save_model`/`load_model` persist a fitted
  estimator with `joblib` (now a declared core dependency — it was already
  present transitively via scikit-learn), completing the fit-once/score-later
  story: fitted transform parameters travel as validated JSON
  (`save_params`/`load_params`) and the model as a joblib file, so a later
  run reloads both and scores with no refitting or in-memory carryover.
  `load_model` documents the unpickling trust boundary (only load files you
  or a trusted process wrote — the reason parameters deliberately stay in
  JSON). Both worked projects now run the reload loop end to end, closing the
  top item of the `nyc_taxis` friction backlog; the guide's Model section
  documents the pattern. Next up per `ROADMAP.md`: the Model/Evaluate
  build-out (P3).
- `projects/nyc_taxis`: the first project on **real** data — fare prediction
  over the 6,433-ride March-2019 NYC taxis sample (seaborn `taxis`, mirrored
  from the NYC TLC trip records; downloaded once into git-ignored
  `data/raw/`). It runs the full lifecycle on `ds` + scikit-learn alone, with
  the split-safe transforms persisted as one scoring `Pipeline` and the model
  evaluated against a naive train-mean baseline, and exists to drive the
  hybrid workspace's demand loop: the library friction it surfaced (no `hour`
  from `add_datetime_features`, no high-cardinality encoder, no model
  persistence, no baseline estimators) is recorded as the friction backlog in
  `ROADMAP.md`.
- `docs/guide.md` cookbook gained three cross-stage recipes for combinations
  the per-stage walkthrough didn't cover: validating right after
  `load_raw`/`load_table` with `check_schema` (acquire + validate), using
  `top_correlations` to screen out redundant features before
  `scale_features` (explore + feature), and fitting a scikit-learn estimator
  and closing the loop with `regression_metrics` and `plot_residuals` (model +
  evaluate + visualize). Closes the "Docs cookbook" item in `ROADMAP.md`.
- `ds run <name>`: run a project's `pipeline.py` by name. It resolves the name
  against the real directories under `projects/`, matching the literal name or
  the same slug `ds new` derives (`"Customer Churn"`, `customer_churn` and
  `customer-churn` all reach `projects/customer_churn/`; `_example` is reachable
  as `example`), then runs that project with the current interpreter and
  propagates its exit code. A miss lists the runnable projects. Because it
  selects an existing `projects/` entry rather than building a path from the
  name, a traversal attempt like `../evil` matches nothing — the same
  path-traversal discipline as `ds new`'s slug. The decision to add `ds run`
  (and to *reject* `ds check` as a redundant shell-out to `make check`) is
  recorded in `ROADMAP.md` and `CLAUDE.md`; `docs/guide.md` and `README.md`
  document the command.
- `tests/test_public_api.py` pins the curated top-level `ds` surface
  (`Settings`, `get_settings`, `get_logger`, `seed_everything`, `__version__`),
  asserts stage helpers and `ds.pipeline.Pipeline` are *not* re-exported, and
  checks that `import ds` stays cheap (loads no matplotlib/scikit-learn).
- `ds.pipeline`: a composable fit-once/apply-many `Pipeline` over the
  `fit_*`/`apply_*` pairs. A `Pipeline` holds an ordered tuple of
  `PipelineStep`s — each pairing fitted parameters with the `apply_*`
  transform it means via a `kind` tag (which is how one `OutlierBounds`
  serves both `"clip_outliers"` and `"flag_outliers"`; the flag form adds
  boolean `<column>_outlier` columns) — applies them all with one
  `apply(df)` call, and persists through the existing
  `ds.io.save_params`/`load_params` machinery by delegating to the per-class
  `to_dict`/`from_dict` round-trips. Step order and same-type duplicate
  steps survive the round-trip; `from_dict` fails with an error naming the
  offending step on unknown kinds or malformed/stale payloads. Train-time-only
  parameters (target-column fits) deliberately stay out of a pipeline —
  see the new "Compose the applies into one pipeline" cookbook section in
  `docs/guide.md`.
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
- `ROADMAP.md` restructured around a goal evaluation (2026-07): per-goal
  verdicts, a P1–P4 plan of record (next up: model persistence, then
  Model/Evaluate build-out), the `nyc_taxis` friction backlog, and the settled
  decisions kept with their rationale. A "demand first" rule joins the working
  agreement: new library work traces to real-project friction, not a
  brainstormed candidate list.
- The `test-extras` CI job now installs `--extra all` as declared instead of
  hand-injecting `tiktoken` — cheap by construction now that extras only carry
  consumed dependencies.
- **API discoverability settled: DS keeps the strict import-by-stage
  convention** — no flat top-level re-export of stage helpers, and
  `ds.pipeline.Pipeline` stays `from ds.pipeline import Pipeline`. The top-level
  `ds` namespace re-exports only stage-independent infrastructure, so `import
  ds` stays cheap and stages can't collide in one flat namespace. Documented in
  `docs/guide.md` ("Importing from DS"), `docs/index.md`, `README.md` and
  `CLAUDE.md`; recorded with its rationale in `ROADMAP.md`. No behavior change —
  every existing import keeps working.
- `projects/_example/pipeline.py`'s fresh-rows step now saves and reloads one
  `ds.pipeline.Pipeline` (impute region → encode region → scale calendar
  features) instead of three separate parameter files, keeping the
  train-time-only target-column bounds/fill persisted individually;
  `tests/test_example.py` mirrors the new flow.
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

### Removed
- Unused dependency pins, so the install surface matches what code consumes:
  `polars` from the core dependencies, the `timeseries` extra entirely
  (`statsmodels`, `sktime` had zero importers), and
  `sentence-transformers`/`anthropic` from the `nlp` extra — which is now
  exactly `tiktoken`, the one extra dependency existing code exercises. A
  dependency is (re-)added in the same change as its first consumer; intended
  future extras live in `ROADMAP.md` until then.

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
