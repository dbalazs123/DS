# DS

**Data science toolkit for every situation.**

DS is a single home for data science work: a reusable library organized by
*process*, a workspace for individual projects, and tooling that keeps it aligned
with current best practices.

## Quickstart

```bash
uv sync                                        # env + library + dev tools
uv run pre-commit install                      # install git hooks (optional)
make check                                      # lint + typecheck + tests
uv run python projects/_example/pipeline.py     # run the worked example
```

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11.

## Layout

```
src/ds/        Reusable library, organized by DS process
projects/      Individual analyses/experiments (see _example/)
templates/     Copier template to scaffold new projects
notebooks/     Exploratory notebooks
data/          Datasets (git-ignored: raw/ interim/ processed/)
docs/          mkdocs-material documentation
tests/         Test suite mirroring src/ds/
```

## The library, by process

| Stage | Module | Examples |
|-------|--------|----------|
| Acquire | `ds.io` | `load_table`, `save_table` |
| Validate | `ds.validation` | `require_columns`, `assert_no_nulls` |
| Clean | `ds.preprocessing` | `standardize_column_names`, `drop_constant_columns` |
| Explore | `ds.eda` | `summarize`, `missing_value_report`, `top_correlations` |
| Feature | `ds.features` | `add_datetime_features` |
| Model | `ds.modeling` | `split_features_target`, `train_test_split_by_time`, `count_tokens` |
| Evaluate | `ds.evaluation` | `regression_metrics`, `classification_metrics`, `confusion_frame`, `per_class_metrics` |
| Visualize | `ds.viz` | `set_theme`, `plot_missingness`, `plot_confusion_matrix`, `plot_residuals` |

Cross-cutting: `ds.config`, `ds.logging`, `ds.reproducibility`.

**Import each helper from its stage** (`from ds.features import one_hot_encode`,
`from ds.pipeline import Pipeline`); the stage name is part of the API. Only the
stage-independent infrastructure is re-exported at the top level
(`from ds import Settings, get_settings, get_logger, seed_everything`). This
keeps `import ds` cheap and avoids cross-stage name collisions — see
[Importing from DS](docs/guide.md#importing-from-ds).

## Starting a project

```bash
ds new "Customer Churn"     # scaffold projects/customer_churn/ (wraps copier)
ds run "Customer Churn"     # run projects/customer_churn/pipeline.py by name
```

`ds run` resolves the name against the directories under `projects/` (literal
name or slug — `"Customer Churn"`, `customer_churn` and `customer-churn` all
work) and runs that project's `pipeline.py`; run it with no match to see the
available projects listed. See the [Guide](docs/guide.md) for a per-stage
cookbook and the full workflow.

## Optional extras

The core install is lean; domain-heavy stacks live behind extras:

```bash
uv sync --extra nlp          # tiktoken, sentence-transformers, anthropic
uv sync --extra timeseries   # statsmodels, sktime
uv sync --extra all          # everything
```

## Development

`make help` lists all commands. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
workflow and conventions. CI runs pre-commit (ruff + mypy) and the test suite
with an enforced coverage gate on every push and PR.

**Docs** are built with `make docs` locally. CI builds them with `--strict` on
every push/PR and deploys them to GitHub Pages on every push to `master`.

**Releases** are cut by pushing a SemVer tag; CI builds the package and creates
the GitHub Release:

```bash
git tag v0.1.0 && git push origin v0.1.0
```
