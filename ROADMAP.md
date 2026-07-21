# Roadmap

Planned direction for the DS toolkit — the **live** state and the demand queue.
This is a living document; keep it small so it stays cheap to read in full at the
start of a session. For *how* to add a function, see
[CONTRIBUTING.md](CONTRIBUTING.md); for hard-won gotchas, see the "Engineering
notes" section of [CLAUDE.md](CLAUDE.md).

The historical detail — the goal evaluation, the completed plan of record
(P1–P17), the per-project friction backlogs (items 1–31), and the
settled-decision rationales — lives in [`ROADMAP_ARCHIVE.md`](ROADMAP_ARCHIVE.md).
Consult it (grep by item number or decision name) only when you need the "why"
behind a resolved item; you should not need to read it in full to start new work.

## Where things stand

The library is organized by data-science process and every stage carries a
working set of the most-reached-for helpers. Built out so far:

| Stage | Module | Status |
|-------|--------|--------|
| Acquire | `ds.io` | `load_table`, `save_table` (csv/tsv/parquet/json/jsonl), `load_raw`, `save_processed`, `fetch_dataset` (checksum-verified multi-mirror download), `save_params`/`load_params` (fitted-parameter JSON) |
| Validate | `ds.validation` | `require_columns`, `assert_row_count`, `assert_no_nulls`, `assert_in_range`, `assert_in_set`, `assert_unique`, `assert_dtypes`, `check_schema` |
| Clean | `ds.preprocessing` | `standardize_column_names`, `drop_constant_columns`, `drop_duplicate_rows`, `coerce_dtypes`, `flag_outliers`, `clip_outliers`, `impute_missing` + split-safe pairs `fit_outlier_bounds`/`apply_flag_outliers`/`apply_clip_outliers`, `fit_impute_values`/`apply_impute_missing` |
| Explore | `ds.eda` | `summarize`, `missing_value_report`, `top_correlations`, `target_rate_by_category` (per-level grouped target rate, the categorical read on the target) |
| Feature | `ds.features` | `add_datetime_features` (selectable `features=` subset; opt-in `_elapsed_months` trend counter), `add_lagged_features` (autoregressive lag columns; per-entity via `group=` on a panel), `text_features` (char/word/word-length columns), `one_hot_encode`, `ordinal_encode`, `collapse_categories` (top-k + "other"), `scale_features`, `bin_column` + split-safe pairs `fit_one_hot_categories`/`apply_one_hot_encode`, `fit_ordinal_categories`/`apply_ordinal_encode`, `fit_topk_categories`/`apply_collapse_categories`, `fit_scale_params`/`apply_scale_features` |
| Model | `ds.modeling` | `split_features_target`, `train_test_split_by_time`, `train_test_split_random` (shuffled, optionally stratified), `fit_baseline` (mean / majority / naive-last / seasonal-naive), `forecast_recursive` (recursive multi-step forecast from a lag-feature model), `save_model`/`load_model` (joblib persistence), `count_tokens` |
| Evaluate | `ds.evaluation` | `regression_metrics`, `classification_metrics`, `probability_metrics` (threshold-free ROC-AUC / average precision / Brier for a rare-event target), `choose_threshold` (PR-curve sweep for an F1-optimal or target-precision/recall operating point), `confusion_frame`, `per_class_metrics`, `cross_validate_by_time` (rolling origin; optionally re-fits a transform pipeline per fold via `make_pipeline`), `cross_validate_kfold` (optionally stratified; same `make_pipeline` re-fit), `compare_models` |
| Visualize | `ds.viz` | `set_theme`, `plot_missingness`, `plot_outliers`, `plot_target_rate` (per-level target rate + baseline line), `plot_confusion_matrix`, `plot_pr_curve`/`plot_roc_curve` (operating-point curves + no-skill baseline), `plot_residuals`, `plot_model_comparison`, `plot_series` (composable series/forecast plot) |

Supporting: `ds.pipeline` (a persistable fit-once/apply-many `Pipeline` over
the `fit_*`/`apply_*` pairs, fitted in one call from a `FitStep` plan via
`fit_pipeline`), `ds` CLI (`ds version`, `ds new`, `ds run`), a
per-stage docs Guide with cross-stage recipes, a `test-extras` CI job,
single-sourced version, and an extended project template. `projects/` holds the
synthetic worked example (`_example`) and twelve **real-data** projects:
`nyc_taxis` (regression), `titanic` (binary classification), `flights`
(forecasting), `diamonds` (multiclass classification), `sms_spam` (text /
binary spam classification), `air_quality` (sensor gap-filling regression
on an hourly time axis), `adult_income` (heavily-categorical binary income
classification), `sunspots` (autoregressive forecasting — the second
forecasting project, on a non-calendar solar cycle), `bbc_news` (multiclass
text topic classification — the second text project), `store_sales`
(store × item daily-sales **panel** — the first multi-entity project, which
pulled group-aware lags), `bank_marketing` (term-deposit subscription —
the first **imbalanced / rare-event** project, which pulled probabilistic
evaluation metrics) and `mammography` (rare calcification screening — the
second imbalanced project, which tunes the operating point and pulled
`choose_threshold` + the ROC/PR curve plots).

## Demand queue (next up)

The demand queue is **empty** — every friction item raised so far (items 1–41,
in `ROADMAP_ARCHIVE.md`) is resolved, struck, or parked with a recorded revisit
trigger. The twelfth demand loop (`mammography`, the **second** imbalanced /
rare-event project) is done: it gave `probability_metrics` its second consumer
and, by *tuning the operating point* rather than reweighting, fired the two items
`bank_marketing` parked — pulling `ds.evaluation.choose_threshold` (item 37) and
`ds.viz.plot_pr_curve` / `plot_roc_curve` (item 38), and recording the rest
(items 39–41). The next step is an ordinary **thirteenth demand loop**: a new
real-data project chosen by the grep-driven rule (grep which library surfaces
still have no real consumer and pick the data shape that stresses the thinnest
cluster by absence), whose friction regenerates the backlog. Deprioritized until
a project pulls them: a **panel-aware split / rolling-origin backtest** in
`ds.modeling`/`ds.evaluation` (items 33/35 — a *second* panel project is the
build trigger), an **out-of-fold threshold calibration** helper (item 40 — a
second project needing the operating point chosen under CV, not on train scores),
an `apply_threshold` convenience (item 41 — a second consumer hand-rolling the
`scores >= t` comprehension), a Cramér's-V / mutual-information categorical
*ranker* (item 29's unbuilt sibling shape), a first-class `ds.pipeline` vectorize
step (item 18 — only if a text project shows the model-side convention genuinely
fails, which two now have not), more cookbook recipes, more CLI.

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
- **Keep this file live-only; history goes to the archive.** When a demand loop
  completes, append its friction backlog and the completed plan-of-record entry
  to [`ROADMAP_ARCHIVE.md`](ROADMAP_ARCHIVE.md), continuing the item numbering —
  **not** here. `ROADMAP.md` holds only "Where things stand", the demand queue,
  and this working agreement, so it stays cheap to read in full every session. A
  size-budget test (`tests/test_roadmap_size.py`) fails CI if this file grows
  past its line budget, so re-bloat can't slip in silently.
