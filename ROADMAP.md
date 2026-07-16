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
| Acquire | `ds.io` | `load_table`, `save_table` (csv/tsv/parquet/json/jsonl), `load_raw`, `save_processed` |
| Validate | `ds.validation` | `require_columns`, `assert_no_nulls`, `assert_in_range`, `assert_in_set`, `assert_dtypes`, `check_schema` |
| Clean | `ds.preprocessing` | `standardize_column_names`, `drop_constant_columns`, `drop_duplicate_rows`, `coerce_dtypes`, `flag_outliers`, `clip_outliers`, `impute_missing` |
| Explore | `ds.eda` | `summarize`, `missing_value_report`, `top_correlations` |
| Feature | `ds.features` | `add_datetime_features`, `one_hot_encode`, `ordinal_encode`, `scale_features`, `bin_column` |
| Model | `ds.modeling` | `split_features_target`, `train_test_split_by_time`, `count_tokens` |
| Evaluate | `ds.evaluation` | `regression_metrics`, `classification_metrics`, `confusion_frame`, `per_class_metrics` |
| Visualize | `ds.viz` | `set_theme`, `plot_missingness`, `plot_outliers`, `plot_confusion_matrix`, `plot_residuals` |

Supporting: `ds` CLI (`ds version`, `ds new`), a per-stage docs Guide, a
`test-extras` CI job, single-sourced version, and an extended project template.

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
`drop_duplicate_rows`, `clip_outliers`, `impute_missing` alongside the existing
`standardize_column_names`/`drop_constant_columns` (clean), `one_hot_encode` +
`scale_features` alongside `add_datetime_features` (feature), then the
existing split → model → evaluate → visualize flow, now with
`ds.viz.plot_outliers` alongside the forecast figure. `tests/test_example.py`
asserts the new-stage behavior (no nulls after cleaning, encoded columns
present, outliers clipped) rather than just the metric keys.

## Later / bigger bets

- **API discoverability** — decide deliberately whether to curate a flat
  re-export of the most-used functions at the top level, or keep the
  import-by-stage convention (and document it prominently either way).
- **Docs cookbook** — mostly covered now: `docs/guide.md` already walks every
  stage with copy-pasteable recipes, kept in sync with the worked example.
  What's left is smaller — recipes for less-common combinations as they come
  up — rather than a first pass.
- **`ds` CLI** — grow beyond `new` (e.g. `ds check`, `ds run`) if it earns its
  keep.
- **Fit/apply (split-safe) transforms — agreed next step.** `impute_missing`,
  `scale_features`, `clip_outliers`, `one_hot_encode` and `ordinal_encode` all
  fit their statistics (means, bounds, categories) on whichever frame they're
  given, so a pipeline can only apply them *before* the train/test split —
  applying them split-safely (fit on train, transform test with train's
  statistics) isn't possible today. Rebuilding the worked example surfaced the
  real friction this causes, to inform that design:
  - Every one of these five functions had to run pre-split in the example, so
    the test window's imputation median, scaling mean/std, clip bounds and
    one-hot categories are all leaked from the future. Harmless for a demo,
    disqualifying for a real evaluation.
  - `one_hot_encode`/`ordinal_encode` infer categories from whatever data they
    see; a category present only in train (or only in test) silently produces
    a different column set on each side today — a fit/apply split would need
    to fix the category vocabulary once, from train, and apply it to both.
  - `impute_missing`'s per-column fill values (mean/median/mode) and
    `scale_features`'s per-column center/spread are computed and discarded
    inline — there's no way to capture and reuse them, so a caller who wants
    train-only statistics has to reimplement the strategy by hand outside the
    helper today.
  - `clip_outliers`/`flag_outliers` recompute IQR/z-score bounds from
    whatever frame they're given; bounds learned on train can't currently be
    carried over and applied to test data or to new incoming rows.
  - Shape: each function likely wants a paired `fit_*`/`apply_*` (or a small
    stateful transformer object) that returns learned parameters from `fit`
    and takes them as input to `apply`, while the existing single-call form
    stays as a convenience wrapper (`fit` + `apply` on the same frame) for
    exploratory/pre-split use — matching how the worked example uses them
    today.

## Working agreement

- Branch from the latest default branch per task; never push to `master`
  directly; open a PR only when asked.
- `make check` (lint + typecheck + tests) and, for doc changes,
  `mkdocs build --strict` must pass before committing.
- Keep `README.md`, `CLAUDE.md`, the docs, and `CHANGELOG.md` honest in the same
  change that alters structure, tooling, or the public API.
