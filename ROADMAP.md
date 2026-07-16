# Roadmap

Planned direction for the DS toolkit. This is a living document — update it as
work lands. For *how* to add a function, see [CONTRIBUTING.md](CONTRIBUTING.md);
for hard-won gotchas, see the "Engineering notes" section of
[CLAUDE.md](CLAUDE.md).

## Where things stand

The library is organized by data-science process and each stage is real but
some are still thin. Built out so far:

| Stage | Module | Status |
|-------|--------|--------|
| Acquire | `ds.io` | thin — `load_table`, `save_table` |
| Validate | `ds.validation` | thin — `require_columns`, `assert_no_nulls` |
| Clean | `ds.preprocessing` | `standardize_column_names`, `drop_constant_columns`, `drop_duplicate_rows`, `coerce_dtypes`, `flag_outliers`, `clip_outliers`, `impute_missing` |
| Explore | `ds.eda` | `summarize`, `missing_value_report`, `top_correlations` |
| Feature | `ds.features` | thin — `add_datetime_features` |
| Model | `ds.modeling` | `split_features_target`, `train_test_split_by_time`, `count_tokens` |
| Evaluate | `ds.evaluation` | `regression_metrics`, `classification_metrics`, `confusion_frame`, `per_class_metrics` |
| Visualize | `ds.viz` | `set_theme`, `plot_missingness`, `plot_outliers`, `plot_confusion_matrix`, `plot_residuals` |

Supporting: `ds` CLI (`ds version`, `ds new`), a per-stage docs Guide, a
`test-extras` CI job, single-sourced version, and an extended project template.

## Next up — flesh out the thin stages

Add the most-reached-for helpers to the stages every analysis touches. Follow
the standard recipe: right stage module → Google-style docstring + full type
hints (`mypy --strict`) → mirroring test → export from `__all__`. Prefer the
core deps (pandas, numpy, scikit-learn, matplotlib); anything heavier goes in an
extra and must degrade gracefully.

- **`ds.preprocessing`** — ✅ done (`coerce_dtypes`, `flag_outliers` /
  `clip_outliers`, `drop_duplicate_rows`, `impute_missing`, paired with
  `ds.viz.plot_outliers`).
- **`ds.features`** — one-hot / ordinal categorical encoding; numeric scaling
  wrappers; equal-width / quantile binning.
- **`ds.validation`** — assert a column's values fall in a range or an allowed
  set; assert expected dtypes; a lightweight expected-schema check (there's
  already a `pandera` dependency worth leaning on here).
- **`ds.io`** — support more formats; add a `data/`-aware pair
  (`load_raw` / `save_processed`) that resolves paths via `ds.config` settings.

Pair new stage functions with `ds.viz` plots where it helps (e.g. a
distribution / outlier plot for `preprocessing`, mirroring how `plot_missingness`
and `plot_confusion_matrix` visualize `eda` / `evaluation` outputs).

## Later / bigger bets

- **API discoverability** — decide deliberately whether to curate a flat
  re-export of the most-used functions at the top level, or keep the
  import-by-stage convention (and document it prominently either way).
- **Docs cookbook** — expand the Guide with worked recipes as stages grow.
- **`ds` CLI** — grow beyond `new` (e.g. `ds check`, `ds run`) if it earns its
  keep.

## Working agreement

- Branch from the latest default branch per task; never push to `master`
  directly; open a PR only when asked.
- `make check` (lint + typecheck + tests) and, for doc changes,
  `mkdocs build --strict` must pass before committing.
- Keep `README.md`, `CLAUDE.md`, the docs, and `CHANGELOG.md` honest in the same
  change that alters structure, tooling, or the public API.
