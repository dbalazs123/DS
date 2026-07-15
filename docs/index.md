# DS Toolkit

**Data science toolkit for every situation.**

DS is a hybrid workspace: a reusable library (`src/ds/`) organized by data-science
*process*, plus a `projects/` area for individual analyses and `templates/` for
scaffolding new work.

## Quickstart

```bash
uv sync                       # create the env, install ds + dev tools
uv run pytest                 # run the tests
uv run python projects/_example/pipeline.py   # run the worked example
```

## The library, by process

| Stage | Module | Example |
|-------|--------|---------|
| Acquire | `ds.io` | `load_table`, `save_table` |
| Validate | `ds.validation` | `require_columns`, `assert_no_nulls` |
| Clean | `ds.preprocessing` | `standardize_column_names`, `drop_constant_columns` |
| Explore | `ds.eda` | `summarize` |
| Feature | `ds.features` | `add_datetime_features` |
| Model | `ds.modeling` | `split_features_target`, `train_test_split_by_time` |
| Evaluate | `ds.evaluation` | `regression_metrics`, `classification_metrics` |
| Visualize | `ds.viz` | `set_theme` |

Cross-cutting: `ds.config`, `ds.logging`, `ds.reproducibility`.

See the [API Reference](api.md) for the full, auto-generated documentation.
