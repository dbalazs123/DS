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
| Feature | `ds.features` | `add_datetime_features`, `one_hot_encode`, `ordinal_encode`, `scale_features`, `bin_column` + split-safe pairs `fit_one_hot_categories`/`apply_one_hot_encode`, `fit_ordinal_categories`/`apply_ordinal_encode`, `fit_scale_params`/`apply_scale_features` |
| Model | `ds.modeling` | `split_features_target`, `train_test_split_by_time`, `count_tokens` — **the thinnest stage; see the evaluation below** |
| Evaluate | `ds.evaluation` | `regression_metrics`, `classification_metrics`, `confusion_frame`, `per_class_metrics` |
| Visualize | `ds.viz` | `set_theme`, `plot_missingness`, `plot_outliers`, `plot_confusion_matrix`, `plot_residuals` |

Supporting: `ds.pipeline` (a persistable fit-once/apply-many `Pipeline` over
the `fit_*`/`apply_*` pairs), `ds` CLI (`ds version`, `ds new`, `ds run`), a
per-stage docs Guide with cross-stage recipes, a `test-extras` CI job,
single-sourced version, and an extended project template. `projects/` holds the
synthetic worked example (`_example`) and the first **real-data** project
(`nyc_taxis`).

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
  friction backlog below is the current queue.
- **Fit-once / score-later** — *stopped one step short of its own goal.*
  Fitted parameters and the `Pipeline` persist as strict JSON, but the fitted
  **model** cannot be persisted at all, so "score new rows in a later run or
  another process" breaks at the estimator. This is the top library gap (P2).
- **Every stage carries working helpers** — *sharply uneven.* Clean/Feature
  are rich; Model is two split helpers plus `count_tokens` and Evaluate is
  four point-metric functions — no baselines, no cross-validation (not even
  rolling-origin, though the flagship example is a forecast), no model
  comparison. The stages where the science happens are the thinnest (P3).
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
- **P2 — model persistence (next up).** `ds.modeling` gains
  `save_model`/`load_model` (joblib under the hood — already present
  transitively via scikit-learn; document the unpickling trust boundary).
  Then the worked example and `nyc_taxis` reload **pipeline + model** with no
  in-memory carryover, and the fit-once/score-later goal is actually met.
- **P3 — bring Model/Evaluate up to the Clean/Feature standard.** Core deps
  only, standard recipe: baseline estimators (`fit_baseline`: mean /
  naive-last / seasonal-naive) so every first metric has a reference point;
  `cross_validate_by_time` (rolling-origin) plus a plain k-fold wrapper; a
  small model-comparison frame paired with a `ds.viz` plot. Re-rank against
  the friction backlog before building.
- **P4 — honest packaging: DONE.** Unused pins removed (`polars` from core;
  `sentence-transformers`/`anthropic`/`statsmodels`/`sktime` from extras —
  `nlp` is now exactly `tiktoken`). A dependency is added in the same change
  as its first consumer. Intended future extras (e.g. a statsmodels-backed
  `timeseries`) live here until that code exists.

Deprioritized until a project pulls them: more EDA helpers, more viz, more
cookbook recipes, more CLI.

## Friction backlog (from `projects/nyc_taxis`)

Demand-driven candidates, in observed-pain order:

1. **Model persistence** — the scoring `Pipeline` round-trips as JSON but the
   fitted estimator cannot be saved (= P2).
2. **`add_datetime_features` has no `hour`** — hour of day was the strongest
   temporal signal in the data and had to be hand-rolled. Add an `_hour`
   column (and consider opting parts in/out) — smallest, clearest win.
3. **No baseline estimators** — the train-mean reference model was
   hand-rolled (= first slice of P3).
4. **No high-cardinality strategy** — the ~200-level zone columns can't be
   one-hot or ordinal encoded; the project fell back to boroughs. Candidate:
   a `fit_*/apply_*` frequency or top-k("other") encoder in `ds.features`.
5. **Pipeline fit-side observation** (for the settled pure-composition
   decision below): assembling the scoring pipeline required manually fitting
   each parameter set on a progressively transformed train frame
   (fit → apply → fit next). Pure composition stays settled for now, but if a
   second project repeats this dance, a declarative fit-a-plan convenience
   earns a fresh look.

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

The five statistic-learning transforms (`impute_missing`, `scale_features`,
`clip_outliers`/`flag_outliers`, `one_hot_encode`, `ordinal_encode`) each have
a paired `fit_*`/`apply_*` form: `fit_*` learns parameters from one frame and
returns a small frozen dataclass, `apply_*` applies them to any frame. The
single-call forms remain as fit-and-apply-on-the-same-frame conveniences and
are implemented as exactly that, so the two forms can't drift. Category
vocabularies are fixed at fit time (unseen categories → all-zero indicators /
`-1` codes).

### Persistable fit parameters *(revisited: scope was too narrow)*

The five `fit_*` dataclasses carry validated `to_dict`/`from_dict` round-trips
and `ds.io.save_params`/`load_params` persist them as strict JSON. Decisions
that stand: per-class methods rather than a generic `asdict` mechanism (honest
types under `mypy --strict`, per-class edge-case handling next to each
definition, shared plumbing in private `ds._serde`, `ds.io` typed against the
`FittedParams` protocol); strict JSON on disk (tagged non-finite floats, numpy
scalars unwrapped, tuples re-tupled, `from_dict` validates type tag + exact
field set). **Revisit recorded:** the cited goal — "score new rows in a later
run or another process" — is unmet without persisting the *model* too; P2
extends the story to the estimator.

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
