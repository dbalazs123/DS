# Guide

A practical tour of the toolkit: install it, run the worked example, learn the
one function you need from each lifecycle stage, and scaffold your own project.
For exhaustive signatures see the [API Reference](api.md).

## Getting started

```bash
uv sync                                       # env + library + dev tools
uv run python projects/_example/pipeline.py   # run the worked example
```

The example under `projects/_example/pipeline.py` chains one function from every
stage — generate → save/load → validate → clean → feature → time-split → model →
evaluate → visualize — and is the best single reference for how the pieces fit.

## Cookbook

Short, copy-pasteable recipes, in lifecycle order. Every function has full type
hints and a Google-style docstring.

### Acquire — `ds.io`

Format is inferred from the file suffix (CSV, Parquet, …), so the same two calls
handle any supported table.

```python
from ds.io import load_table, save_table

df = load_table("data/raw/sales.csv")
save_table(df, "data/processed/sales.parquet")
```

### Validate — `ds.validation`

Fail fast on the assumptions a pipeline depends on.

```python
from ds.validation import assert_no_nulls, require_columns

require_columns(df, ["date", "amount"])   # raises if a column is missing
assert_no_nulls(df, ["amount"])           # raises on nulls in `amount`
```

### Clean — `ds.preprocessing`

```python
from ds.preprocessing import drop_constant_columns, standardize_column_names

df = standardize_column_names(df)   # "Total Sales ($)" -> "total_sales"
df = drop_constant_columns(df)      # drop columns with a single value
```

### Explore — `ds.eda`

```python
from ds.eda import summarize

summary = summarize(df)   # per-column dtype, null counts, cardinality, stats
```

### Feature — `ds.features`

```python
from ds.features import add_datetime_features

df = add_datetime_features(df, "date")   # year, month, dayofweek, ...
```

### Model — `ds.modeling`

Split tabular data into features and target, or split a time series
chronologically so the test window is strictly in the future.

```python
from ds.modeling.tabular import split_features_target
from ds.modeling.timeseries import train_test_split_by_time

train, test = train_test_split_by_time(df, "date", test_size=0.2)
x_train, y_train = split_features_target(train, "amount")
```

Text work lives here too (needs the `nlp` extra for an accurate count, otherwise
falls back to a whitespace estimate):

```python
from ds.modeling.nlp import count_tokens

count_tokens("how many tokens is this?")
```

### Evaluate — `ds.evaluation`

```python
from ds.evaluation import classification_metrics, regression_metrics

regression_metrics(y_true, y_pred)       # mae, rmse, r2
classification_metrics(y_true, y_pred)   # accuracy, precision, recall, f1
```

### Visualize — `ds.viz`

```python
from ds.viz import set_theme

set_theme("notebook")   # consistent matplotlib theme + palette
```

### Cross-cutting

```python
from ds import get_logger, get_settings, seed_everything

seed_everything(42)                 # reproducible randomness
settings = get_settings()           # paths: settings.raw_dir, .processed_dir
logger = get_logger(__name__)
```

## Starting a new project

Projects live under `projects/`, one directory each, and *consume* the library
rather than re-implementing it. Scaffold one with the `ds` CLI:

```bash
ds new "Customer Churn"                 # creates projects/customer_churn/
```

or with Copier directly:

```bash
uv run copier copy templates/project projects/customer_churn
```

Either way you get a `pipeline.py` skeleton (one section per lifecycle stage), a
`notebooks/` folder, and a `tests/` folder. Run and test it with:

```bash
uv run python projects/customer_churn/pipeline.py
uv run pytest projects/customer_churn --no-cov
```

If you find yourself copy-pasting a helper between projects, that's the signal it
belongs back in `src/ds/`.
