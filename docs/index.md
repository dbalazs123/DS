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
| Explore | `ds.eda` | `summarize`, `missing_value_report`, `top_correlations` |
| Feature | `ds.features` | `add_datetime_features` |
| Model | `ds.modeling` | `split_features_target`, `train_test_split_by_time` |
| Evaluate | `ds.evaluation` | `regression_metrics`, `classification_metrics`, `confusion_frame` |
| Visualize | `ds.viz` | `set_theme`, `plot_missingness`, `plot_confusion_matrix`, `plot_residuals` |

Cross-cutting: `ds.config`, `ds.logging`, `ds.reproducibility`.

## Next steps

- The [Guide](guide.md) walks through each stage with copy-pasteable recipes and
  how to scaffold a new project.
- The [API Reference](api.md) has the full, auto-generated documentation.
