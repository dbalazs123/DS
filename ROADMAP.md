# Roadmap

Planned direction for the DS toolkit. This is a living document — update it as
work lands. For *how* to add a function, see [CONTRIBUTING.md](CONTRIBUTING.md);
for hard-won gotchas, see the "Engineering notes" section of
[CLAUDE.md](CLAUDE.md).

## Where things stand

The library is organized by data-science process and every stage carries a
working set of the most-reached-for helpers. Built out so far:

| Stage | Module | Status |
|-------|--------|--------|
| Acquire | `ds.io` | `load_table`, `save_table` (csv/tsv/parquet/json/jsonl), `load_raw`, `save_processed`, `save_params`/`load_params` (fitted-parameter JSON) |
| Validate | `ds.validation` | `require_columns`, `assert_no_nulls`, `assert_in_range`, `assert_in_set`, `assert_dtypes`, `check_schema` |
| Clean | `ds.preprocessing` | `standardize_column_names`, `drop_constant_columns`, `drop_duplicate_rows`, `coerce_dtypes`, `flag_outliers`, `clip_outliers`, `impute_missing` + split-safe pairs `fit_outlier_bounds`/`apply_flag_outliers`/`apply_clip_outliers`, `fit_impute_values`/`apply_impute_missing` |
| Explore | `ds.eda` | `summarize`, `missing_value_report`, `top_correlations` |
| Feature | `ds.features` | `add_datetime_features` (incl. `_hour`), `one_hot_encode`, `ordinal_encode`, `collapse_categories` (top-k + "other"), `scale_features`, `bin_column` + split-safe pairs `fit_one_hot_categories`/`apply_one_hot_encode`, `fit_ordinal_categories`/`apply_ordinal_encode`, `fit_topk_categories`/`apply_collapse_categories`, `fit_scale_params`/`apply_scale_features` |
| Model | `ds.modeling` | `split_features_target`, `train_test_split_by_time`, `train_test_split_random` (shuffled, optionally stratified), `fit_baseline` (mean / majority / naive-last / seasonal-naive), `save_model`/`load_model` (joblib persistence), `count_tokens` |
| Evaluate | `ds.evaluation` | `regression_metrics`, `classification_metrics`, `confusion_frame`, `per_class_metrics`, `cross_validate_by_time` (rolling origin), `cross_validate_kfold`, `compare_models` |
| Visualize | `ds.viz` | `set_theme`, `plot_missingness`, `plot_outliers`, `plot_confusion_matrix`, `plot_residuals`, `plot_model_comparison` |

Supporting: `ds.pipeline` (a persistable fit-once/apply-many `Pipeline` over
the `fit_*`/`apply_*` pairs), `ds` CLI (`ds version`, `ds new`, `ds run`), a
per-stage docs Guide with cross-stage recipes, a `test-extras` CI job,
single-sourced version, and an extended project template. `projects/` holds the
synthetic worked example (`_example`) and two **real-data** projects:
`nyc_taxis` (regression) and `titanic` (classification).

## Goal evaluation (2026-07)

A deliberate stop to work backward from the project's stated goals instead of
extending recent momentum. Verdicts, and what they imply:

- **Hybrid workspace (library + `projects/` that consume it)** — *was the
  most under-served goal.* Until `nyc_taxis`, `projects/` held only the
  synthetic `_example`, whose data is generated to be exactly as dirty as the
  library can clean; the promotion loop ("friction in a project becomes
  library work") had never run, while eight consecutive PRs invested in
  supply-side library polish. **Consequence:** every library addition should
  now be pulled by a project need, not pushed from a candidate list. The
  friction backlogs below are the queue (currently the `titanic` one — the
  `nyc_taxis` list is fully served).
- **Fit-once / score-later** — *stopped one step short of its own goal;
  since closed (P2).* Fitted parameters and the `Pipeline` persist as strict
  JSON, but at evaluation time the fitted **model** could not be persisted at
  all, so "score new rows in a later run or another process" broke at the
  estimator. `ds.modeling.persistence.save_model`/`load_model` closed this;
  both worked projects now score from reloaded state only.
- **Every stage carries working helpers** — *was sharply uneven; since
  rebalanced (P3).* At evaluation time Model was two split helpers plus
  `count_tokens` and Evaluate four point-metric functions — no baselines, no
  cross-validation, no model comparison. P3 filled exactly those gaps
  (`fit_baseline`, rolling-origin + k-fold cross-validation,
  `compare_models` + its paired plot), each traced to the friction backlog
  rather than a candidate list.
- **"Toolkit for every situation"** — *was dishonest at the packaging layer,
  now fixed.* The `timeseries` extra (statsmodels, sktime) had zero importers,
  most of the `nlp` extra was unused, and `polars` sat unused in the core
  dependencies. Extras now carry only dependencies code actually consumes
  (the rule is recorded in `pyproject.toml` and CLAUDE.md). In-scope reality
  today: tabular regression/classification on pandas — widen it by building,
  not by declaring.
- **Engineering discipline** — *well-served* (strict typing, mirrored tests,
  coverage gate, honest docs). No change; point it at the gaps above.

Completed work that mattered less against the goals (kept, but the lesson is
recorded): `ds run` and the cross-stage cookbook recipes are good polish that
consumed cycles while the demand side stayed empty; `count_tokens` is an
orphaned NLP toe-dip. The lesson is the ordering rule above: demand first.

## Plan of record

- **P1 — run the demand loop on real data: DONE.** `projects/nyc_taxis`
  predicts cab fares from the real March-2019 NYC rides sample (seaborn
  `taxis`, mirrored from the NYC TLC records; downloaded once into
  git-ignored `data/raw/`). Full lifecycle on `ds` + scikit-learn, split-safe
  transforms persisted as one scoring `Pipeline`, evaluated against a naive
  baseline (r² 0.73 vs baseline mae 7.2 → 2.6). Its friction list *is* the
  backlog below.
- **P2 — model persistence: DONE.** `ds.modeling.persistence` provides
  `save_model`/`load_model`, joblib under the hood (the format scikit-learn's
  own docs recommend; now a declared core dependency per the
  first-consumer rule). Deliberate line, recorded: joblib/pickle is used
  **only** for the estimator — unpickling executes arbitrary code, so
  `load_model` documents the trust boundary (only load files you or a trusted
  process wrote), while transform parameters stay in strict validated JSON.
  Both worked projects now persist pipeline + model and score from the
  reloaded state with no in-memory carryover; the guide's Model section
  documents the pattern and the warning.
- **P3 — bring Model/Evaluate up to the Clean/Feature standard: DONE.**
  Re-ranked against the friction backlog first, so the batch opened with the
  smallest demand-traced win (`add_datetime_features` now emits `_hour` —
  friction item 2), then, core deps only and to the standard recipe:
  - `ds.modeling.baseline.fit_baseline` (mean / naive-last / seasonal-naive)
    returning a frozen `Baseline` with `predict(n)` — deliberately *not* a
    scikit-learn estimator, because a baseline needs no feature matrix; the
    training target is the whole input, and the output feeds
    `ds.evaluation` directly (friction item 3).
  - `ds.evaluation.cross_validate_by_time` — rolling-origin folds, the
    repeated-fold counterpart to `train_test_split_by_time` (a shuffled
    k-fold on temporal data trains on the future) — plus
    `cross_validate_kfold` for order-free data. Both take a `make_model`
    factory (fresh estimator per fold, no cross-fold state) and a
    `metrics_fn` defaulting to `regression_metrics`, so classification (or a
    custom scorer) composes instead of forking the API.
  - `ds.evaluation.compare_models` + `ds.viz.plot_model_comparison`,
    following the settled stage↔viz pairing convention.
  `nyc_taxis` dogfoods the batch: library `pickup_hour`, `fit_baseline` and
  a persisted comparison frame/plot replaced its hand-rolled versions, with
  identical metrics.
- **P4 — honest packaging: DONE.** Unused pins removed (`polars` from core;
  `sentence-transformers`/`anthropic`/`statsmodels`/`sktime` from extras —
  `nlp` is now exactly `tiktoken`). A dependency is added in the same change
  as its first consumer. Intended future extras (e.g. a statsmodels-backed
  `timeseries`) live here until that code exists.

- **P5 — regenerate demand with a second real-data project: DONE.**
  `projects/titanic` classifies passenger survival on the classic 891-row
  manifest (seaborn-data mirror; real missingness at three severities, the
  target respelled as a feature, derived duplicate columns). Full lifecycle
  on `ds` + scikit-learn — validated leakage drops, stratified split, a
  five-step persisted scoring `Pipeline`, held-out split scored from the
  reloaded model — and the first real exercise of the untouched
  classification surface: `classification_metrics`, `confusion_frame`,
  `per_class_metrics`, `plot_confusion_matrix`, and the first composition of
  `cross_validate_kfold` with `metrics_fn=classification_metrics`. Held-out
  accuracy 0.799 / F1 0.731 vs the sex-only rule (0.777 / 0.692) and the
  majority class (0.615 / 0.0). Per the demand-first rule the project
  promotes nothing itself; its friction list is the new backlog below.

**Next up:** work the `titanic` friction backlog below, demand-first — top
observed-pain item first, one promotion per change, each consumed by the
project that demanded it. Deprioritized until a project pulls them: more EDA
helpers, more viz, more cookbook recipes, more CLI.

## Friction backlog (from `projects/nyc_taxis`)

Demand-driven candidates, in observed-pain order:

1. ~~**Model persistence**~~ — **resolved by P2**:
   `ds.modeling.persistence.save_model`/`load_model`; the project now scores
   the held-out window from the reloaded model.
2. ~~**`add_datetime_features` has no `hour`**~~ — **resolved in P3**: the
   helper now emits `<column>_hour` (constantly zero on date-only data, where
   `drop_constant_columns` removes it); the project consumes it.
3. ~~**No baseline estimators**~~ — **resolved in P3**:
   `ds.modeling.baseline.fit_baseline`; the project's hand-rolled train-mean
   baseline is gone.
4. ~~**No high-cardinality strategy**~~ — **resolved**:
   `ds.features.fit_topk_categories`/`apply_collapse_categories` keep a
   column's top-k levels and collapse the rest (and anything unseen at
   scoring time) to `"other"`, so the existing one-hot/ordinal encoders take
   it from there. Top-k+"other" was chosen over frequency encoding because it
   preserves level identity (what a linear fare model needs) and composes
   with the existing encoders instead of adding a parallel numeric path. The
   project now consumes the zone columns it originally dropped, and they earn
   their place: vs a boroughs-only variant on the same held-out window, MAE
   2.62 → 2.26 (−14%), r² 0.729 → 0.765 (k=15, asserted in the project's
   end-to-end test).
5. **Pipeline fit-side observation** (for the settled pure-composition
   decision below): assembling the scoring pipeline required manually fitting
   each parameter set on a progressively transformed train frame
   (fit → apply → fit next). Pure composition stays settled for now, but if a
   second project repeats this dance, a declarative fit-a-plan convenience
   earns a fresh look. **Trigger fired:** `titanic` repeated the dance
   verbatim (five fit/apply pairs) — see item 9 below.

## Friction backlog (from `projects/titanic`)

The second run of the demand loop. Numbering continues from the `nyc_taxis`
list so item references stay unambiguous; in observed-pain order:

6. ~~**No classification-shaped baseline.**~~ — **resolved**:
   `fit_baseline` now takes `strategy="majority"` (predict the modal training
   label, ties to the smallest label), the classification twin of `"mean"`.
   Scoped to the observed demand: labels must be numeric (the int-coded 0/1
   target that raised the item) — string labels stay out until a project
   demands them, because the frozen `Baseline` contract is
   `tuple[float, ...]`. The project's hand-rolled `y_train.mode()` reference
   is gone, with identical held-out metrics (majority accuracy 0.615 / F1 0.0).
7. ~~**No split helper for order-free data.**~~ — **resolved**:
   `ds.modeling.tabular.train_test_split_random` is the order-free twin of
   `train_test_split_by_time` — shuffled, with an optional `stratify` column
   whose class balance both halves preserve, seeded through numpy's global
   generator like the rest of the stage (so `seed_everything` reproduces
   it). The project's raw `sklearn.model_selection.train_test_split` call is
   gone; the wrapper makes the identical scikit-learn call, so the split and
   the held-out metrics are byte-identical (accuracy 0.799 / F1 0.731).
8. **`cross_validate_kfold` cannot stratify.** It wraps plain `KFold`, so on
   the 62/38 target the fold class balance drifts (observed per-fold recall
   0.64–0.79 across otherwise stable folds). A stratified option
   (`StratifiedKFold` under a flag) would compose naturally with
   `metrics_fn=classification_metrics`.
9. **Cross-validation cannot re-fit the transform chain per fold.**
   `make_model` rebuilds the estimator per fold, but the frame handed to
   `cross_validate_kfold` already carries transforms fitted on the *whole*
   training split — every fold's test rows influenced the imputation and
   scaling statistics its train rows were transformed with. Doing it
   properly means re-fitting the fit → apply → fit chain inside each fold,
   which nothing in `ds.pipeline` can express: the same fit-side gap as
   item 5, whose "second project repeats this dance" trigger has now fired.

Where the library did *not* fight: the classification metric/plot surface
itself (`classification_metrics`, `confusion_frame`, `per_class_metrics`,
`plot_confusion_matrix`, `compare_models` with a swapped `metrics_fn`)
composed first-try with no workarounds.

## Settled decisions (recorded rationale)

Kept for the record — CLAUDE.md's engineering notes point here. Each was
re-checked in the 2026-07 evaluation; verdicts inline.

### The four thin stages fleshed out *(stands)*

Each stage every analysis touches carries its most-reached-for helpers, built
to the standard recipe: right stage module → Google-style docstring + full
type hints (`mypy --strict`) → mirroring test → export from `__all__`,
favouring the core deps (pandas, numpy, scikit-learn, matplotlib). When adding
more, keep pairing stage functions with `ds.viz` plots where it helps (as
`plot_outliers` visualizes `flag_outliers`).

### The worked example dogfoods the stages *(stands, superseded as proof)*

`projects/_example/pipeline.py` runs realistically dirty **synthetic** data
through the full lifecycle and `tests/test_example.py` asserts the split-safe
behavior. It remains the teaching reference; `projects/nyc_taxis` is now the
proof on data the library didn't design.

### Fit/apply (split-safe) transforms *(stands)*

The six statistic-learning transforms (`impute_missing`, `scale_features`,
`clip_outliers`/`flag_outliers`, `one_hot_encode`, `ordinal_encode`,
`collapse_categories`) each have a paired `fit_*`/`apply_*` form: `fit_*`
learns parameters from one frame and returns a small frozen dataclass,
`apply_*` applies them to any frame. The single-call forms remain as
fit-and-apply-on-the-same-frame conveniences and are implemented as exactly
that, so the two forms can't drift. Category vocabularies are fixed at fit
time (unseen categories → all-zero indicators / `-1` codes / the `"other"`
label).

### Persistable fit parameters *(revisited: scope was too narrow)*

The six `fit_*` dataclasses carry validated `to_dict`/`from_dict` round-trips
and `ds.io.save_params`/`load_params` persist them as strict JSON. Decisions
that stand: per-class methods rather than a generic `asdict` mechanism (honest
types under `mypy --strict`, per-class edge-case handling next to each
definition, shared plumbing in private `ds._serde`, `ds.io` typed against the
`FittedParams` protocol); strict JSON on disk (tagged non-finite floats, numpy
scalars unwrapped, tuples re-tupled, `from_dict` validates type tag + exact
field set). **Revisit resolved:** the cited goal — "score new rows in a later
run or another process" — was unmet without persisting the *model* too; P2
extended the story to the estimator via `ds.modeling.persistence` (JSON stays
the format for parameters, joblib is used only for the model).

### Composable fit/apply pipeline *(stands; one observation logged)*

`ds.pipeline.Pipeline` holds an ordered tuple of `PipelineStep`s (fitted
parameters + the `apply_*` kind they mean), applies them in order, and
persists through `save_params`/`load_params`. Decisions that stand: a
top-level `ds.pipeline` module (composes two stages; imports run strictly
pipeline → stages); a closed `StepParams` union + `StepKind` literal under
`mypy --strict`; steps tagged by *kind* because `OutlierBounds` serves two
apply forms; train-time-only parameters stay out (scoring rows have no
target). The per-pair API stays the primitive; the pipeline is pure
composition — see friction item 5 for the observation that could reopen the
fit side.

### API discoverability: import by stage *(stands)*

Stage helpers are imported from their stage (`from ds.eda import summarize`),
`Pipeline` from `ds.pipeline`; the top-level `ds` namespace re-exports only
stage-independent infrastructure (`Settings`, `get_settings`, `get_logger`,
`seed_everything`). The stage name is the teaching tool; a flat re-export
would force `import ds` to eagerly load matplotlib/scikit-learn and pile all
stages' names into one namespace. `tests/test_public_api.py` pins the exact
top-level surface (and that `import ds` stays cheap). `Pipeline` earns no
top-level re-export — a pipeline *composes* stage transforms, so flattening
the composer while its building blocks stay stage-scoped would be the one
inconsistent case.

### The `ds` CLI: `run` added, `check` rejected *(stands)*

`ds run <name>` cleared the bar as a *project-aware default*: it resolves
names against the real directories under `projects/` (literal or `ds new`
slug), lists the runnable projects on a miss, and never builds a path from
the name (same traversal discipline as `ds new`). `ds check` stays rejected:
it would either duplicate `make check`'s sequence (drift risk) or just call
`make` (adding nothing) — `make` is the canonical dev entry point. Don't
re-add it.

### Docs cookbook: cross-stage recipes *(stands)*

The highest-value cross-stage recipes are in `docs/guide.md` (validate at the
acquire boundary; screen redundant features before scaling; fit/evaluate/
diagnose with a real estimator). Add a recipe if and when a new combination
comes up in practice — pre-building a catalog of hypothetical ones was
considered and skipped.

## Working agreement

- Branch from the latest default branch per task; never push to `master`
  directly; open a PR only when asked.
- `make check` (lint + typecheck + tests) and, for doc changes,
  `mkdocs build --strict` must pass before committing.
- Keep `README.md`, `CLAUDE.md`, the docs, and `CHANGELOG.md` honest in the same
  change that alters structure, tooling, or the public API.
- **Demand first:** new library work should trace to a friction item from a
  real project (or a P2/P3 plan-of-record item), not to a brainstormed
  candidate list.
