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
| Explore | `ds.eda` | `summarize` |
| Feature | `ds.features` | `add_datetime_features` |
| Model | `ds.modeling` | `split_features_target`, `train_test_split_by_time`, `count_tokens` |
| Evaluate | `ds.evaluation` | `regression_metrics`, `classification_metrics` |
| Visualize | `ds.viz` | `set_theme` |

Cross-cutting: `ds.config`, `ds.logging`, `ds.reproducibility`.

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

**Docs** are built with `make docs` and published to GitHub Pages on every push
to `master`.

**Releases** are cut by pushing a SemVer tag; CI builds the package and creates
the GitHub Release:

```bash
git tag v0.1.0 && git push origin v0.1.0
```
