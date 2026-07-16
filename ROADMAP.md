# Roadmap

Planned direction for the DS toolkit. This is a living document — update it as
work lands. For *how* to add a function, see [CONTRIBUTING.md](CONTRIBUTING.md);
for hard-won gotchas, see the "Engineering notes" section of
[CLAUDE.md](CLAUDE.md).

## Where things stand

The library is organized by data-science process and every stage now carries a
working set of the most-reached-for helpers. Built out so far:

| Stage | Module | Status |
|-------|--------|--------|
| Acquire | `ds.io` | `load_table`, `save_table` (csv/tsv/parquet/json/jsonl), `load_raw`, `save_processed`, `save_params`/`load_params` (fitted-parameter JSON) |
| Validate | `ds.validation` | `require_columns`, `assert_no_nulls`, `assert_in_range`, `assert_in_set`, `assert_dtypes`, `check_schema` |
| Clean | `ds.preprocessing` | `standardize_column_names`, `drop_constant_columns`, `drop_duplicate_rows`, `coerce_dtypes`, `flag_outliers`, `clip_outliers`, `impute_missing` + split-safe pairs `fit_outlier_bounds`/`apply_flag_outliers`/`apply_clip_outliers`, `fit_impute_values`/`apply_impute_missing` |
| Explore | `ds.eda` | `summarize`, `missing_value_report`, `top_correlations` |
| Feature | `ds.features` | `add_datetime_features`, `one_hot_encode`, `ordinal_encode`, `scale_features`, `bin_column` + split-safe pairs `fit_one_hot_categories`/`apply_one_hot_encode`, `fit_ordinal_categories`/`apply_ordinal_encode`, `fit_scale_params`/`apply_scale_features` |
| Model | `ds.modeling` | `split_features_target`, `train_test_split_by_time`, `count_tokens` |
| Evaluate | `ds.evaluation` | `regression_metrics`, `classification_metrics`, `confusion_frame`, `per_class_metrics` |
| Visualize | `ds.viz` | `set_theme`, `plot_missingness`, `plot_outliers`, `plot_confusion_matrix`, `plot_residuals` |

Supporting: `ds.pipeline` (a persistable fit-once/apply-many `Pipeline` over
the `fit_*`/`apply_*` pairs), `ds` CLI (`ds version`, `ds new`, `ds run`), a per-stage
docs Guide, a `test-extras` CI job, single-sourced version, and an extended
project template.

## Done — the four thin stages are fleshed out

Each stage every analysis touches now has its most-reached-for helpers, built to
the standard recipe: right stage module → Google-style docstring + full type
hints (`mypy --strict`) → mirroring test → export from `__all__`, favouring the
core deps (pandas, numpy, scikit-learn, matplotlib).

- **`ds.preprocessing`** — `coerce_dtypes`, `flag_outliers` / `clip_outliers`,
  `drop_duplicate_rows`, `impute_missing`, paired with `ds.viz.plot_outliers`.
- **`ds.features`** — `one_hot_encode`, `ordinal_encode`, `scale_features`,
  `bin_column`.
- **`ds.validation`** — `assert_in_range`, `assert_in_set`, `assert_dtypes`, and
  a `pandera`-backed `check_schema`.
- **`ds.io`** — `.tsv`/`.jsonl` formats plus a `data/`-aware `load_raw` /
  `save_processed` pair resolving paths via `ds.config`.

When adding more, keep pairing stage functions with `ds.viz` plots where it
helps (as `plot_outliers` visualizes `flag_outliers`, mirroring `plot_missingness`
and `plot_confusion_matrix`).

## Done — the worked example dogfoods the new stages

`projects/_example/pipeline.py` generates realistically dirty synthetic data
(missing values, outliers, a genuine categorical column, duplicate rows) and
runs it through the full lifecycle: `load_raw`/`save_processed` (acquire),
`check_schema`/`require_columns` (validate), `coerce_dtypes`,
`drop_duplicate_rows` alongside the existing
`standardize_column_names`/`drop_constant_columns` (structural clean), a
chronological split, then split-safe clip → impute → encode → scale via the
`fit_*`/`apply_*` pairs (fitted on train, applied to both — see the fit/apply
section below), and the existing model → evaluate → visualize flow, with
`ds.viz.plot_outliers` alongside the forecast figure. `tests/test_example.py`
asserts the new-stage behavior (no nulls after cleaning, identical encoded
columns on both splits, outliers clipped to train-fitted bounds) rather than
just the metric keys.

## Done — fit/apply (split-safe) transforms

The five statistic-learning transforms (`impute_missing`, `scale_features`,
`clip_outliers`/`flag_outliers`, `one_hot_encode`, `ordinal_encode`) now each
have a paired `fit_*`/`apply_*` form: `fit_*` learns the parameters from one
frame and returns them as a small frozen dataclass, `apply_*` takes them and
transforms any frame. The single-call forms remain as convenience wrappers
(`fit` + `apply` on the same frame) for exploratory/pre-split use, and are
implemented as exactly that, so the two forms can't drift.

- `ds.preprocessing`: `fit_outlier_bounds` → `OutlierBounds` →
  `apply_clip_outliers`/`apply_flag_outliers` (one fit serves both, since they
  share bounds); `fit_impute_values` → `ImputeValues` → `apply_impute_missing`.
- `ds.features`: `fit_scale_params` → `ScaleParams` → `apply_scale_features`;
  `fit_one_hot_categories` → `OneHotCategories` → `apply_one_hot_encode`;
  `fit_ordinal_categories` → `OrdinalCategories` → `apply_ordinal_encode`.
- The dogfooding friction that motivated this is resolved: the category
  vocabulary is fixed once at fit time, so train and test always encode to the
  same column set (unseen categories → all-zero indicators / `-1` codes), and
  learned fills/bounds/centre+spread can be captured, inspected and reused on
  new rows.
- `projects/_example/pipeline.py` now splits chronologically *first* and runs
  every statistic-learning transform as fit-on-train/apply-to-both, closing the
  leakage the previous example knowingly demonstrated; `tests/test_example.py`
  asserts the split-safe behavior (train-fitted bounds/fills applied to test,
  identical column sets on both splits) and `docs/guide.md` documents the
  pattern in a dedicated cookbook section.

## Done — persistable fit parameters

The five `fit_*` dataclasses (`OutlierBounds`, `ImputeValues`, `ScaleParams`,
`OneHotCategories`, `OrdinalCategories`) each carry a `to_dict`/`from_dict`
pair, and `ds.io.save_params`/`load_params` persist them as JSON — so a
pipeline can save its fitted state alongside the model and score new incoming
rows in a later run or another process. The worked example and the guide's
split-safe cookbook section demonstrate the save-then-reload loop end to end.

- **Per-class methods, not a generic mechanism** — chosen deliberately for
  `mypy --strict`: a `dataclasses.asdict`/`fields`-driven round-trip types as
  `dict[str, Any] → T` with `cast`s for the `Literal` fields and no place to
  validate per-field, whereas explicit methods give concrete return types and
  put each class's edge-case handling (non-finite bounds, numpy scalar fills,
  tuple vocabularies) next to its definition. The shared plumbing (scalar
  encode/decode, payload-shape checks) lives in the private `ds._serde`
  module, and `ds.io` stays decoupled by typing `save_params`/`load_params`
  against a `FittedParams` protocol instead of importing the dataclasses.
- **Strict JSON on disk** — non-finite floats (JSON has no `inf`/`nan`
  literal) are written as a tagged `{"__float__": "inf"}` mapping rather than
  Python's non-standard bare literals (`allow_nan=False` enforces it); numpy
  scalars are unwrapped via `.item()`; vocabulary tuples round-trip through
  lists and are re-tupled on load. Every `from_dict` validates the type tag,
  the exact field set and the field shapes, so a stale or hand-edited file
  fails with a message naming what is wrong.

## Done — composable fit/apply pipeline

The open question is closed: `ds.pipeline.Pipeline` is a small frozen object
holding an ordered `tuple` of `PipelineStep`s, each pairing fitted parameters
with the `apply_*` transform it means; `apply(df)` runs the steps in order,
and `to_dict`/`from_dict` delegate to the per-class round-trips so a whole
pipeline persists through `ds.io.save_params`/`load_params` (it satisfies the
same `FittedParams` protocol). A scoring run reloads one file instead of
re-stringing the `apply_*` calls by hand — the worked example's fresh-rows
step now does exactly that. The per-pair API stays the primitive; the
pipeline is pure composition.

Decisions made deliberately, for the record:

- **Where it lives: a new top-level `ds.pipeline` module** (like `ds.config`
  and `ds.reproducibility`), not inside a stage — a pipeline composes
  transforms from *two* stages (`ds.preprocessing` and `ds.features`), so
  homing it in either would couple the stages to each other. Imports run
  strictly pipeline → stages, so no cycle can form. It is imported from
  `ds.pipeline`, not re-exported from `ds/__init__.py` — the settled
  import-by-stage convention (see "API discoverability" below).
- **Typing under `mypy --strict`: a closed union, not a protocol.**
  `StepParams` is the union of the five parameter dataclasses and `StepKind`
  a `Literal` of the six apply forms; dispatch is exhaustive `isinstance`
  narrowing with no `cast`s, and a step validates kind ↔ parameter-class
  agreement at construction. The set of fit/apply pairs is small and closed,
  so the union is honest; a new pair extends the union, the kind literal and
  the registry in one place.
- **Steps are tagged by *kind* (the apply form), not by the parameter class's
  `"type"` tag** — necessary because `OutlierBounds` serves two apply forms
  (`"clip_outliers"` winsorizes; `"flag_outliers"` adds boolean
  `<column>_outlier` columns). The kind→class registry resolves classes in
  `from_dict`; the nested payloads are still validated by each class's own
  `from_dict`, so unknown kinds, wrong-class payloads and malformed fields
  all fail with an error naming the offending step. Step order and same-type
  duplicates (e.g. two `ImputeValues` with different strategies) survive the
  round-trip.
- **Train-time-only parameters stay out by design.** A pipeline holds
  scoring-time transforms only; anything fitted on the target column (the
  example's sales bounds/fill — scoring rows have no target) is persisted
  individually instead. Stateless transforms (`add_datetime_features`) take
  no fitted parameters and run outside the pipeline as plain calls.

`tests/test_pipeline.py` covers construction, application order, persistence
and every edge case above; `docs/guide.md`'s split-safe cookbook section
documents the pattern.

## Done — API discoverability: import by stage

**Decision: keep the strict import-by-stage convention; do not add a flat
top-level re-export.** Stage helpers are imported from their stage
(`from ds.eda import summarize`), and `ds.pipeline.Pipeline` / `PipelineStep`
from `ds.pipeline`; the top-level `ds` namespace re-exports only the
stage-independent infrastructure (`Settings`, `get_settings`, `get_logger`,
`seed_everything`). `Pipeline` does **not** earn a top-level re-export.

Grounded in the codebase's own conventions rather than a fresh preference:

- **The `__all__` policy already drew this line** — `src/ds/__init__.py` has
  only ever re-exported cross-cutting helpers, and CLAUDE.md's "stage functions
  are imported by stage" note codified it. The Guide's cookbook and the worked
  example both teach `from ds.<stage> import ...` throughout. A flat re-export
  would unwind a convention every existing doc and file already follows.
- **The stage name is the teaching tool.** `from ds.features import
  one_hot_encode` says *this is a feature step*; flattening to
  `ds.one_hot_encode` discards the process organization that is the library's
  whole reason for its shape.
- **Cost and collisions (the flagged edge cases).** Re-exporting the stages
  would force `ds/__init__.py` to import every stage at `import ds` time —
  including matplotlib via `ds.viz`, scikit-learn via `ds.modeling`, and the
  optional-extra guards in `ds.modeling.nlp` — and would pile all stages'
  names into one namespace to be disambiguated. Import-by-stage keeps `import
  ds` light (verified: no matplotlib/sklearn loaded) and each stage's names
  contained. `from ds import Settings, get_logger, seed_everything` (relied on
  by the example and template) is unchanged.
- **`Pipeline` re-export, specifically:** rejected for symmetry — a pipeline
  *composes* stage transforms, so re-exporting the composer while its
  `fit_*`/`apply_*` building blocks stay stage-scoped would be the one
  inconsistent case. It stays `from ds.pipeline import Pipeline`.

Documented prominently in `docs/guide.md` ("Importing from DS"), `docs/index.md`
and `README.md`; the convention note in `CLAUDE.md` is expanded; and
`tests/test_public_api.py` pins the exact top-level surface (and asserts
`import ds` stays cheap) so it can't drift.

## Done — the `ds` CLI grows to `run`; `check` rejected

**Decision: add `ds run`, reject `ds check`.** The gate was whether a subcommand
earns its keep over what already exists — `make check` (lint + typecheck + test)
and `uv run python projects/<name>/pipeline.py` — by adding real value
(discoverability, a project-aware default, cross-platform reach) rather than
just shelling out.

- **`ds run <name>` — added.** It clears the bar as a *project-aware default*:
  it resolves the name against the real directories under `projects/`, matching
  either the literal directory name or the same slug `ds new` derives (so
  `"Customer Churn"`, `customer_churn` and `customer-churn` all reach
  `projects/customer_churn/`, and `_example` is reachable as `example`), then
  runs that project's `pipeline.py` with the current interpreter. It adds
  *discoverability* — a miss lists the runnable projects instead of failing
  blankly — and is symmetric with `ds new` (new creates the project, run runs
  it). Crucially it never builds a path out of the name: it selects an existing
  entry from the `projects/` listing, so a traversal attempt like `../evil`
  matches nothing rather than escaping the directory — the same slug discipline
  that keeps `ds new` safe. This is more than a wrapper over the raw
  `python projects/<name>/pipeline.py`: that form requires knowing the exact
  layout, offers no fuzzy-name resolution, and lists nothing on a typo.
- **`ds check` — rejected.** It would be a pure shell-out to the same tools
  `make check` already orchestrates, and `make` is the project's canonical dev
  entry point (README, CLAUDE.md and CONTRIBUTING.md all assume it). A second
  entry point either reimplements the lint→typecheck→test sequence (a drift
  risk against the Makefile — two sources of truth) or just calls `make` (adding
  nothing, and still failing on a make-less Windows). Its one plausible unique
  benefit — cross-platform reach where `make` is absent — doesn't apply to a
  workflow whose whole documented surface is `uv` + `make`, so it never earns
  its keep. Recorded in CLAUDE.md's engineering notes so it isn't re-added.

`tests/test_cli.py` covers name resolution (literal/slug/underscored),
traversal rejection, exit-code propagation, and the not-found listing;
`docs/guide.md`, `README.md` and `CLAUDE.md` document `ds run`.

## Later / bigger bets

The remaining open item:

- **Docs cookbook** — mostly covered now: `docs/guide.md` already walks every
  stage with copy-pasteable recipes, kept in sync with the worked example.
  What's left is smaller — recipes for less-common combinations as they come
  up — rather than a first pass.

## Working agreement

- Branch from the latest default branch per task; never push to `master`
  directly; open a PR only when asked.
- `make check` (lint + typecheck + tests) and, for doc changes,
  `mkdocs build --strict` must pass before committing.
- Keep `README.md`, `CLAUDE.md`, the docs, and `CHANGELOG.md` honest in the same
  change that alters structure, tooling, or the public API.
