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
| Feature | `ds.features` | `add_datetime_features` (selectable `features=` subset; opt-in `_elapsed_months` trend counter), `one_hot_encode`, `ordinal_encode`, `collapse_categories` (top-k + "other"), `scale_features`, `bin_column` + split-safe pairs `fit_one_hot_categories`/`apply_one_hot_encode`, `fit_ordinal_categories`/`apply_ordinal_encode`, `fit_topk_categories`/`apply_collapse_categories`, `fit_scale_params`/`apply_scale_features` |
| Model | `ds.modeling` | `split_features_target`, `train_test_split_by_time`, `train_test_split_random` (shuffled, optionally stratified), `fit_baseline` (mean / majority / naive-last / seasonal-naive), `save_model`/`load_model` (joblib persistence), `count_tokens` |
| Evaluate | `ds.evaluation` | `regression_metrics`, `classification_metrics`, `confusion_frame`, `per_class_metrics`, `cross_validate_by_time` (rolling origin), `cross_validate_kfold` (optionally stratified; re-fits a transform pipeline per fold via `make_pipeline`), `compare_models` |
| Visualize | `ds.viz` | `set_theme`, `plot_missingness`, `plot_outliers`, `plot_confusion_matrix`, `plot_residuals`, `plot_model_comparison`, `plot_series` (composable series/forecast plot) |

Supporting: `ds.pipeline` (a persistable fit-once/apply-many `Pipeline` over
the `fit_*`/`apply_*` pairs, fitted in one call from a `FitStep` plan via
`fit_pipeline`), `ds` CLI (`ds version`, `ds new`, `ds run`), a
per-stage docs Guide with cross-stage recipes, a `test-extras` CI job,
single-sourced version, and an extended project template. `projects/` holds the
synthetic worked example (`_example`) and four **real-data** projects:
`nyc_taxis` (regression), `titanic` (binary classification), `flights`
(forecasting) and `diamonds` (multiclass classification).

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
  friction backlogs below are the queue (the `nyc_taxis`, `titanic` and
  `flights` lists are fully served; the fourth demand loop — `diamonds` —
  regenerated the queue with items 14–17).
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

- **P6 — regenerate demand with a third real-data project (forecasting):
  DONE.** `projects/flights` forecasts the 144 monthly international-airline-
  passenger totals, 1949–1960 (the classic Box & Jenkins series, seaborn-data
  mirror) — chosen for a genuine time axis with strong yearly seasonality,
  and the first project to stress the time-series surface:
  `train_test_split_by_time` gains its second consumer, and
  `cross_validate_by_time` (rolling-origin folds) and `fit_baseline`'s
  `"naive_last"`/`"seasonal_naive"` strategies their first real ones. Full
  lifecycle on `ds` + scikit-learn: hand-assembled time axis, calendar
  features with the monthly-resolution noise dropped, a hand-rolled
  `month_index` trend, a one-step fit plan (the month one-hot vocabulary)
  persisted as the scoring `Pipeline`, the model persisted and the held-out
  window scored from reloaded state, and a linear trend + month-effects
  model evaluated against both naive references on the strictly future
  29-month window (MAE 34.3 vs seasonal-naive 64.8 and naive-last 81.4;
  r² 0.63 — honest about an additive model under multiplicative
  seasonality). Per the demand-first rule the project promotes nothing
  itself; its friction list is the new backlog below.

- **P7 — serve the `flights` backlog: DONE.** Items 10–13 in observed-pain
  order, each dogfooded by `projects/flights` in the same change (held-out
  metrics equivalent throughout — the trend column is the hand-rolled
  counter shifted by a constant the intercept absorbs):
  - `ds.viz.plot_series` (item 10) — one composable series plot: a solid
    observed line plus optional dashed, named prediction overlays, colours
    drawn from the Axes' cycle so repeated calls on one `ax` compose. One
    helper covers both of the project's hand-rolled figures — the raw
    series *and* the history + forecast-vs-actual view — rather than two
    single-purpose ones.
  - `add_datetime_features(features=...)` (item 11) — an explicit selection
    parameter, chosen over a resolution-aware default because inferring
    resolution from the frame is fitted state in disguise (a later scoring
    batch can be too small or too regular to infer from) and misfires
    silently; an explicit list is stateless and self-documenting. Default
    unchanged (the full calendar set).
  - `"elapsed_months"` (item 12) — the trend counter lives *inside*
    `add_datetime_features` as an opt-in selectable feature (same source
    column, same expansion mechanism, and item 11's parameter already
    provides opt-in) rather than as a second helper. Origin: a fixed
    calendar epoch (whole months since January of year 0), so scoring later
    rows is stateless; only differences matter for a trend term. Kept out
    of the default set (a modeling device, near-collinear with `_year`); a
    days/finer variant stays unbuilt until a project pulls it.
  - Item 13 — **struck, not built** (see the backlog).

- **P8 — regenerate demand with a fourth real-data project (multiclass):
  DONE.** `projects/diamonds` grades the cut of the 53,940 classic ggplot2
  diamonds (seaborn-data mirror) into the five ordered classes Fair < Good <
  Very Good < Premium < Ideal — chosen, after grepping which helpers still
  had no real consumer, for quirks that pull several untouched surfaces at
  once. First real consumers earned: the ordinal-encoding pair *with the
  explicit `categories=` domain ordering* (color J→D, clarity I1→IF — the
  sorted-unique default would rank both wrongly), `bin_column` (cut mix per
  carat quantile band), `plot_outliers` on non-synthetic data, and the
  multiclass metric surface (`confusion_frame` / `per_class_metrics` /
  `plot_confusion_matrix` at 5×5, `classification_metrics(average="macro")`).
  Full lifecycle on `ds` + scikit-learn: boundary validation with the three
  grade vocabularies, the physically impossible zero-dimension rows dropped
  by a hand-rolled mask (validation asserts, nothing filters), exact
  duplicates dropped as split-leaking re-entries (the deliberate opposite of
  titanic's keep), a three-step fit plan (clip the measurement-error
  dimension columns only — depth/table extremes *are* the Fair-cut signal —
  ordinal-encode, scale), stratified 5-fold CV with the plan re-fitted per
  fold, pipeline + model persisted and the held-out split scored from
  reloaded state. Held-out: accuracy 0.655 / macro F1 0.551 vs the
  proportions-only grading rule (0.546 / 0.357) and the majority class
  (0.400 / 0.114); CV macro F1 0.549 ± 0.009. The confusion structure is the
  honest headline: errors sit almost entirely between adjacent grades, with
  `Good` collapsing into `Very Good` (recall 0.12). Per the demand-first
  rule the project promotes nothing itself; its friction list is the new
  backlog below.

**Next up:** serve the `diamonds` backlog (items 14–17 below), in
observed-pain order. Item 9's parked question —
`cross_validate_by_time(make_pipeline=...)` — stays parked: the trigger has
still not fired. Deprioritized until a project pulls them: more EDA helpers,
more cookbook recipes, more CLI.

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
5. ~~**Pipeline fit-side observation**~~ — **resolved with item 9** (the
   "second project repeats this dance" trigger fired: `titanic` repeated the
   fit → apply → fit chain verbatim, five fit/apply pairs):
   `ds.pipeline.fit_pipeline` executes an ordered plan of `FitStep` entries
   as exactly that chain and returns the assembled `Pipeline`. Both projects
   replaced their hand-strung dance with a plan; persisted scoring pipelines
   and held-out metrics came out byte-identical. The amended
   pure-composition rationale is recorded under settled decisions below.

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
8. ~~**`cross_validate_kfold` cannot stratify.**~~ — **resolved, with one
   honest correction**: a `stratify` flag (`StratifiedKFold` under the hood,
   same global-generator seeding) keeps every fold at the frame's class
   balance, composing with `metrics_fn=classification_metrics`; the project
   passes it. It fixes exactly what it controls — per-fold positive counts
   went from a ~15-row spread to the ±1 rounding minimum — but the recall
   drift this item blamed on that imbalance did **not** shrink (measured
   across 30 seeds on the project's frame: mean per-fold recall spread ~0.13
   with and without stratification). The drift is sampling variance in
   *which* positives land in a fold, not in how many, so the project's CV
   assertions were deliberately not tightened.
9. ~~**Cross-validation cannot re-fit the transform chain per fold.**~~ —
   **resolved, with one honest finding**: `cross_validate_kfold` takes a
   `make_pipeline` factory (the `make_model` twin — typically
   `lambda frame: fit_pipeline(frame, plan)` with the training run's own
   plan); each fold fits a fresh pipeline on its training rows only and
   applies it to both halves. `titanic` now cross-validates the *raw*
   training split with the same five-step plan it fits the scoring pipeline
   from. The finding: the leak was real in protocol but its measured effect
   here rounds to zero — the per-fold statistics genuinely change (fold age
   medians 28.0–29.0 vs 28.5 whole-train, fare fences 63.3–66.6 vs 65.7) yet
   not one of the 712 fold predictions flips, because logistic regression
   absorbs small affine shifts in imputation/scaling, so the per-fold CV
   metrics came out identical. As with item 8, the honest result is recorded
   rather than a manufactured delta; the protocol is now sound either way.
   `cross_validate_by_time` deliberately does *not* grow the parameter until
   a project pulls it — it currently has no consumer (demand first).

Where the library did *not* fight: the classification metric/plot surface
itself (`classification_metrics`, `confusion_frame`, `per_class_metrics`,
`plot_confusion_matrix`, `compare_models` with a swapped `metrics_fn`)
composed first-try with no workarounds.

## Friction backlog (from `projects/flights`)

The third run of the demand loop — the first to stress the time-series
surface. Numbering continues from the `titanic` list; in observed-pain
order:

10. ~~**No time-series plot in `ds.viz`.**~~ — **resolved in P7**:
    `ds.viz.plot_series` — one solid observed line, optional dashed named
    prediction overlays, colours from the Axes' cycle so calls compose on
    one `ax`. One helper replaced both of the project's hand-rolled figures:
    `series.png` is a single call, `forecast.png` two composed calls
    (training tail, then the held-out window with the model and
    seasonal-naive overlays). The API-shape question resolved to *one*
    composable plot, not a separate forecast helper.
11. ~~**`add_datetime_features` is all-or-nothing.**~~ — **resolved in P7**:
    a `features=` selection parameter scopes the emission (default: the
    full calendar set, unchanged). The deliberately-open shape question was
    decided for explicit selection over a resolution-aware default:
    inferring resolution from the frame is fitted state in disguise — a
    later scoring batch can be too small or too regular to infer the same
    answer from — and it misfires silently, while an explicit list is
    stateless, self-documenting, and exactly matches the observed pain
    (the project knew precisely which columns were noise). The project's
    hand-drop of `date_dayofweek`/`date_is_weekend` is gone, and so is its
    reliance on `drop_constant_columns` catching `_day`/`_hour` — the
    scoped call never emits them.
12. ~~**No elapsed-time/trend feature.**~~ — **resolved in P7**:
    `"elapsed_months"`, an opt-in member of item 11's selection (same
    source column and expansion mechanism, so no second helper) emitting
    whole months since a *fixed* calendar epoch (January of year 0, i.e.
    `year * 12 + month - 1`). The fixed epoch is the stateless-origin
    answer: nothing is learned from the frame, so a later scoring run maps
    the same timestamp to the same value, and for a trend term only
    differences matter. Excluded from the default set (a modeling device,
    near-collinear with `_year`). The project's hand-rolled `month_index`
    is gone with equivalent held-out metrics (the counter differs by a
    constant the intercept absorbs). A days/finer-grained variant stays
    unbuilt until a project demands one.
13. ~~**The time axis was assembled by hand.**~~ — **struck in P7, not
    built**, as the backlog itself anticipated: the pain is one
    `pd.to_datetime(..., format=)` call plus a project-specific uniqueness
    check, observed once. A helper would wrap a single well-documented
    pandas call behind a new name to learn without removing meaningful
    code — below the helper bar. Won't build until a second project
    repeats the pain (and shows which shape recurs: two-column splits,
    format strings, or the uniqueness check).

Notes from the same run, for the record:

- **Item 9's parked question stays parked.** This project's rolling-origin
  CV does consume an already-transformed frame — exactly the situation
  `cross_validate_kfold(make_pipeline=...)` exists for — but the demand
  trigger did **not** fire: the only fitted state is the month one-hot
  vocabulary, and re-fitting it per fold was measured to produce the
  identical 12 calendar months on every fold (every training window spans at
  least 20 months). `cross_validate_by_time` still has no consumer that
  needs per-fold re-fitting.
- **`fit_pipeline` scope finding, not a gap:** the fit plan has exactly one
  step. A complete series whose extremes are signal, scored by a scale-free
  model, needs no imputation, clipping or scaling — on clean time-series
  data most of the fit-based transform surface has nothing to do, and the
  executor's value reduces to building the persistable scoring `Pipeline`.

Where the library did *not* fight: `train_test_split_by_time`,
`fit_baseline`'s `"naive_last"`/`"seasonal_naive"` (whose positional
alignment is correct by construction when the scored window starts right
after training), `cross_validate_by_time` (first real consumer — composed
first-try, time column and target excluded from the features as documented),
`regression_metrics`/`compare_models`/`plot_residuals`/
`plot_model_comparison`, and `drop_constant_columns` catching the constant
calendar columns.

## Friction backlog (from `projects/diamonds`)

The fourth run of the demand loop — the first multiclass one. Numbering
continues from the `flights` list; in observed-pain order:

14. **The classification metric/plot surface is label-blind.** Everything is
    typed `Sequence[int]`, so the five string cut grades had to be int-coded
    before any metric ran — fine in itself (the ordinal encoder did it in
    one call) — but then *every consumer-facing artifact needed the codes
    hand-mapped back to names*: the project carries a `_named()` helper to
    relabel `confusion_frame`/`per_class_metrics` output before persisting,
    and a `_relabel_confusion_axes()` helper to re-set
    `plot_confusion_matrix`'s integer tick labels on the returned Axes. At
    two classes (titanic) 0/1 was readable; at five, `3` for Premium is not.
    Three hand-rolled mapping sites in one pipeline is the largest observed
    pain of the run. Candidate shape: an optional `labels=` mapping on
    `confusion_frame`/`per_class_metrics`/`plot_confusion_matrix` (display
    names only — the metric math staying on int codes is fine); accepting
    string labels outright would also touch `fit_baseline`'s deliberately
    numeric contract (item 6), which this project did *not* need changed.
15. **Validation asserts, nothing filters.** The 20 physically impossible
    zero-dimension rows must be *removed*, and no helper does that — the
    project hand-rolls a boolean mask (`drop_impossible_dimensions`) and
    then re-states the same bound in `assert_in_range(min_value=0,
    inclusive="right")` for the surviving rows, so the rule lives in two
    places that can drift. A `drop_out_of_range` (or a `filter=` mode on the
    range check) would collapse the pair. Observed once; the mask is three
    lines, so this may not clear the helper bar until a second project
    repeats it — recorded for the trigger.
16. **The project template scaffolds a shape no real project keeps.** First
    real `ds new` dogfooding (it ran cleanly and the slug/layout were
    right). Three divergences, each rewritten within minutes of scaffolding:
    the stub `run(output_dir)` lacks the `settings: Settings | None`
    parameter all four real pipelines need (it is how the end-to-end test
    injects a temporary data directory — the scaffolded test calls `run()`
    with no settings, which for a real pipeline would write into the shared
    data tree); the scaffolded README says `uv run python
    projects/<slug>/pipeline.py` while `ds new` itself prints "run it with
    `ds run <slug>`"; and an empty `description` (the default) leaves a
    dangling "`<Name> — `" in the module docstring and a blank README
    section. Small, concrete, template-only fixes.
17. **`classification_metrics`' binary default forces a wrapper at every
    multiclass call site.** `average="binary"` raises beyond two classes,
    and the `metrics_fn` hooks (`cross_validate_kfold`, `compare_models`)
    take a two-argument callable, so the project defines
    `macro_classification_metrics` and threads it through five call sites.
    The wrapper is four lines and `functools.partial` would do; recorded
    because the *next* multiclass project will write the identical wrapper —
    may not clear the helper bar, but the recurrence trigger is precise.

Notes from the same run, for the record:

- **`plot_outliers` ranks by count, and count is not severity.** The plot
  put price/depth/carat first (thousands of honestly skewed values) while
  the physically impossible measurements — the actual data errors, a 58.9 mm
  `y` on a 2-carat stone — were near-invisible at 2–3 flagged values each;
  the clipping decision had to be made against `summarize()`'s max column
  instead. A magnitude-aware view would have shown it directly, but the
  count view is honest about what it claims and one consultation of an
  existing report is not much pain — a note, not an item, until it recurs.
- **Imputation stays unexercised at severity beyond titanic.** Diamonds has
  zero missing values, so this run adds nothing on that surface — recorded
  so the gap isn't mistaken for coverage.

Where the library did *not* fight: `fit_ordinal_categories(categories=...)` /
`apply_ordinal_encode` composed first-try — including the JSON round-trip of
the explicit worst-to-best orders through the persisted `Pipeline` (asserted
in the project's end-to-end test) and unseen-category behaviour never
triggering thanks to the vocabulary validation upstream; the stratified
five-class `cross_validate_kfold(make_pipeline=..., stratify=True)`
composition; `fit_baseline("majority")` on the int-coded target (item 6's
numeric-label scoping held exactly); `bin_column`'s quantile bins as an
EDA device; `drop_duplicate_rows`; `train_test_split_random(stratify=)` at
five classes; and the whole persistence story (`fit_pipeline` →
`save_params`/`load_params`, `save_model`/`load_model`).

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

### Composable fit/apply pipeline *(amended: the fit side gained an executor)*

`ds.pipeline.Pipeline` holds an ordered tuple of `PipelineStep`s (fitted
parameters + the `apply_*` kind they mean), applies them in order, and
persists through `save_params`/`load_params`. Decisions that stand: a
top-level `ds.pipeline` module (composes two stages; imports run strictly
pipeline → stages); a closed `StepParams` union + `StepKind` literal under
`mypy --strict`; steps tagged by *kind* because `OutlierBounds` serves two
apply forms; train-time-only parameters stay out (scoring rows have no
target). The per-pair API stays the primitive.

**Amendment (friction items 5 + 9, the promised design pass):** "the
pipeline is pure composition" was the fit-side half of the original
decision, and its stated trigger — a second project repeating the manual
fit → apply → fit dance — fired. Resolution: `Pipeline` *remains* pure
composition (construction, application and persistence are unchanged — a
pipeline still holds only fitted parameters), and the dance moved into
`fit_pipeline`, an executor over the same per-pair primitives: it runs an
ordered plan of `FitStep` entries (a step kind plus a fit callable) and
returns the assembled `Pipeline`. Two alternatives were weighed and
rejected: a fully declarative fit-spec (one dataclass per `fit_*` form,
making plans persistable data) would mirror every fit signature and drift
with them, and nothing demanded persisting *plans* — only fitted pipelines;
solving only the CV leak (a factory parameter alone) would have left item
5's dance in both projects. A `FitStep` carries a callable closing over the
varying keyword arguments (`columns=`, `strategy=`, `k=`), so the closed
unions stay closed and `mypy --strict` types the plan without a parallel
spec hierarchy. The same mechanism is what `cross_validate_kfold`'s
`make_pipeline` factory re-runs inside each fold, so one plan serves the
training run and leak-free cross-validation.

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
