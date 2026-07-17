# Guide

A practical tour of the toolkit: install it, run the worked example, learn the
one function you need from each lifecycle stage, and scaffold your own project.
For exhaustive signatures see the [API Reference](api.md).

## Getting started

```bash
uv sync                                       # env + library + dev tools
uv run python projects/_example/pipeline.py   # run the worked example
```

The example under `projects/_example/pipeline.py` runs realistically dirty
synthetic data (missing values, outliers, a categorical column, duplicate rows)
through every stage — generate → save/load → validate → clean → feature →
time-split → model → evaluate → visualize — and is the best single reference
for how the pieces fit.

## Importing from DS

DS is organized by data-science *process*, and the imports follow suit:
**import each helper from its stage.** The stage name is part of the public
API — it tells you which lifecycle stage a function belongs to.

```python
from ds.io import load_table                 # acquire
from ds.validation import require_columns    # validate
from ds.preprocessing import impute_missing  # clean
from ds.eda import summarize                 # explore
from ds.features import one_hot_encode       # feature
from ds.evaluation import regression_metrics # evaluate
from ds.viz import plot_residuals            # visualize
from ds.pipeline import Pipeline             # compose fitted transforms
```

The top-level `ds` namespace re-exports **only** the stage-independent
infrastructure every project reaches for, no matter which stage it's in:

```python
from ds import Settings, get_settings, get_logger, seed_everything
```

That is the whole flat surface — there is no `from ds import summarize`. The
choice is deliberate:

- **The stage is the teaching tool.** `from ds.features import one_hot_encode`
  says *this is a feature step*; a flat `ds.one_hot_encode` throws that away.
- **`import ds` stays cheap.** Because the top level doesn't re-export the
  stages, importing `ds` never drags in matplotlib (via `ds.viz`), scikit-learn
  (via `ds.modeling`), or the optional NLP/timeseries stacks. You pay for a
  stage only when you import it.
- **No cross-stage collisions.** Each stage owns its own namespace, so names
  never have to be disambiguated in one flat pile.

`ds.pipeline.Pipeline` (and `PipelineStep`) follow the same rule — imported from
`ds.pipeline`, not the top level — since a pipeline *composes* stage transforms
rather than being infrastructure of its own. `tests/test_public_api.py` pins
this surface so it can't drift.

## Cookbook

Short, copy-pasteable recipes, in lifecycle order. Every function has full type
hints and a Google-style docstring.

### Acquire — `ds.io`

Format is inferred from the file suffix (`.csv`, `.tsv`, `.parquet`, `.json`,
`.jsonl`), so the same two calls handle any supported table.

```python
from ds.io import load_table, save_table

df = load_table("data/raw/sales.csv")
save_table(df, "data/processed/sales.parquet")
```

For the standard `data/` layout, `load_raw` / `save_processed` resolve names
against `ds.config` settings (no hard-coded paths, and names can't escape the
data tree):

```python
from ds.io import load_raw, save_processed

df = load_raw("sales.csv")            # reads <data_dir>/raw/sales.csv
save_processed(df, "sales.parquet")   # writes <data_dir>/processed/sales.parquet
```

### Validate — `ds.validation`

Fail fast on the assumptions a pipeline depends on.

```python
from ds.validation import (
    assert_dtypes,
    assert_in_range,
    assert_in_set,
    assert_no_nulls,
    check_schema,
    require_columns,
)

require_columns(df, ["date", "amount"])        # raises if a column is missing
assert_no_nulls(df, ["amount"])                # raises on nulls in `amount`
assert_in_range(df, "amount", min_value=0)     # raises on negative amounts
assert_in_set(df, "status", ["open", "closed"])  # raises on unknown values
assert_dtypes(df, {"amount": "float64"})       # raises on the wrong dtype
```

For a whole-frame declarative check, `check_schema` leans on `pandera` and can
coerce dtypes as it validates:

```python
df = check_schema(df, {"amount": "float64", "status": "str"}, coerce=True)
```

### Clean — `ds.preprocessing`

```python
from ds.preprocessing import (
    clip_outliers,
    coerce_dtypes,
    drop_constant_columns,
    drop_duplicate_rows,
    flag_outliers,
    impute_missing,
    standardize_column_names,
)

df = standardize_column_names(df)          # "Total Sales ($)" -> "total_sales"
df = drop_constant_columns(df)             # drop columns with a single value
df = drop_duplicate_rows(df)               # de-duplicate rows (optional subset=)
df = coerce_dtypes(df, {"amount": "float64"})  # pin loader-inferred dtypes
df = impute_missing(df, strategy="median")     # fill gaps per column
flags = flag_outliers(df, method="iqr")        # boolean mask of extreme values
df = clip_outliers(df, method="iqr")           # winsorize instead of dropping
```

`impute_missing`, `flag_outliers` and `clip_outliers` learn their statistics
from the frame they're given — fine for exploration, but leaky across a
train/test split. For split-safe cleaning see
[Fit on train, apply to test](#fit-on-train-apply-to-test-split-safe-transforms).

### Explore — `ds.eda`

```python
from ds.eda import missing_value_report, summarize, top_correlations

summarize(df)              # per-column dtype, null counts, cardinality, stats
missing_value_report(df)   # just the columns with gaps, worst first
top_correlations(df, n=5)  # most correlated numeric pairs (redundancy / leakage)
```

### Feature — `ds.features`

```python
from ds.features import (
    add_datetime_features,
    bin_column,
    collapse_categories,
    one_hot_encode,
    ordinal_encode,
    scale_features,
)

df = add_datetime_features(df, "date")               # year, month, dayofweek, hour, ...
df = add_datetime_features(df, "date", features=["month", "elapsed_months"])  # scoped subset
df = collapse_categories(df, ["zone"], k=15)         # top-15 levels + "other"
df = one_hot_encode(df, ["category", "zone"])        # indicator columns
df = ordinal_encode(df, categories={"size": ["S", "M", "L"]})  # ranked codes
df = scale_features(df, ["amount"], method="minmax")  # rescale to [0, 1]
df = bin_column(df, "amount", bins=4, method="quantile")  # -> amount_bin
```

`add_datetime_features` emits the full calendar set by default; on coarse
data pass `features=` to scope it to what actually applies (on a monthly
series `_dayofweek`/`_is_weekend` would just encode the weekday each month's
first day lands on). The opt-in `"elapsed_months"` feature is the trend term
a linear forecaster needs: a monotone count of whole months since a fixed
calendar epoch, so scoring later rows involves no learned origin.

`collapse_categories` is the high-cardinality strategy: a column with hundreds
of levels (the taxi data's ~200 pickup/dropoff zones) keeps its `k` most
frequent values and everything else — including values first seen at scoring
time — becomes `"other"`, so the ordinary encoders can take it from there.

The encoders and scaler learn their vocabulary/parameters from the frame
they're given; for a train/test workflow use their `fit_*`/`apply_*` pairs —
see [Fit on train, apply to test](#fit-on-train-apply-to-test-split-safe-transforms).

### Fit on train, apply to test — split-safe transforms

Every statistic-learning transform has a `fit_*`/`apply_*` pair: `fit_*` learns
the parameters from one frame (means/medians, clip bounds, scale
centre/spread, category vocabulary) and returns them as a small frozen
dataclass; `apply_*` applies them to any frame. Fit on the training split and
apply to both, so the test window (or new incoming rows) never leaks into the
statistics — and both splits get identical encoded columns even when a
category is missing from one side:

```python
from ds.features import (
    apply_one_hot_encode,
    apply_scale_features,
    fit_one_hot_categories,
    fit_scale_params,
)
from ds.modeling.timeseries import train_test_split_by_time
from ds.preprocessing import (
    apply_clip_outliers,
    apply_impute_missing,
    fit_impute_values,
    fit_outlier_bounds,
)

train, test = train_test_split_by_time(df, "date")   # split FIRST

bounds = fit_outlier_bounds(train, ["amount"])       # train-only clip bounds
train = apply_clip_outliers(train, bounds)
test = apply_clip_outliers(test, bounds)

fills = fit_impute_values(train, ["amount"], strategy="median")
train = apply_impute_missing(train, fills)
test = apply_impute_missing(test, fills)             # filled with TRAIN's median

vocab = fit_one_hot_categories(train, ["category"])  # one vocabulary for both
train = apply_one_hot_encode(train, vocab)
test = apply_one_hot_encode(test, vocab)             # same columns, always

scaling = fit_scale_params(train, ["amount"])
train = apply_scale_features(train, scaling)
test = apply_scale_features(test, scaling)           # train's centre/spread
```

`apply_flag_outliers`, `apply_ordinal_encode` (paired with
`fit_ordinal_categories`) and `apply_collapse_categories` (paired with
`fit_topk_categories`, for high-cardinality columns) follow the same pattern.
Unseen categories one-hot encode as all zeros, ordinal-encode as `-1` and
collapse to the `"other"` label; the single-call forms (`impute_missing`,
`clip_outliers`, `scale_features`, ...) remain as
fit-and-apply-on-the-same-frame conveniences for exploratory, pre-split work.
The worked example (`projects/_example/pipeline.py`) runs this exact pattern
end to end.

#### Persist the fitted parameters

Every parameter dataclass has a validated `to_dict`/`from_dict` round-trip,
and `ds.io.save_params`/`load_params` write and read them as JSON — so a
pipeline can save its fitted state alongside its model and score new incoming
rows in a later run or another process, without refitting:

```python
from ds.io import load_params, save_params
from ds.features import OneHotCategories

save_params(fills, "artifacts/fills.json")           # training run
save_params(vocab, "artifacts/vocab.json")

vocab = load_params("artifacts/vocab.json", OneHotCategories)  # scoring run
new_rows = apply_one_hot_encode(new_rows, vocab)     # same columns as training
```

The files are strict JSON: numpy scalars (e.g. a `np.float64` median fill) are
stored as plain numbers, non-finite floats (an all-null column fits `±inf`
bounds) as a tagged `{"__float__": "inf"}` mapping, and category vocabularies
keep non-string values (ints, bools) intact. `load_params` validates the
payload against the class you ask for, so a stale, hand-edited or
wrong-type file fails with a clear error instead of building broken
parameters.

Pair them with `ds.modeling.persistence.save_model`/`load_model` (see
[Model](#model-dsmodeling)) and the scoring run reloads *everything* — fitted
transforms **and** estimator — from disk, with no refitting and no in-memory
carryover. All three worked projects (`projects/_example`,
`projects/nyc_taxis`, `projects/titanic`) run exactly that loop.

#### Compose the applies into one pipeline

Once several parameters are fitted, a scoring run shouldn't have to re-string
the `apply_*` calls — and their order — by hand. `ds.pipeline.Pipeline` owns
an ordered sequence of fitted steps, applies them all in one call, and
persists through the same `save_params`/`load_params` machinery:

```python
from ds.io import load_params, save_params
from ds.pipeline import Pipeline, PipelineStep

scoring = Pipeline(                                   # training run
    steps=(
        PipelineStep("impute_missing", fills),
        PipelineStep("one_hot_encode", vocab),
        PipelineStep("scale_features", scaling),
    )
)
save_params(scoring, "artifacts/scoring_pipeline.json")

scoring = load_params("artifacts/scoring_pipeline.json", Pipeline)  # scoring run
new_rows = scoring.apply(new_rows)                    # every step, in order
```

Each step names the `apply_*` transform it means via its *kind* — that is how
one `OutlierBounds` serves both forms: a `"clip_outliers"` step winsorizes the
fitted columns, while a `"flag_outliers"` step adds boolean
`<column>_outlier` columns instead. Steps of the same parameter type coexist
(e.g. a median fill for numerics and a modal fill for categoricals), and step
order survives the round-trip. Loading a stale or hand-edited file — an
unknown step kind, a malformed nested payload — fails with an error naming
the offending step.

Two things deliberately stay **out** of a pipeline: train-time-only
parameters (anything fitted on the target column — scoring rows have no
target to clip or fill; save those individually for the next training run)
and stateless transforms like `add_datetime_features`, which take no fitted
parameters and run as plain calls before or after `apply`. The worked example
(`projects/_example/pipeline.py`) draws exactly this line: one saved
`Pipeline` (impute region → encode region → scale calendar features) scores
rows that did not exist at fit time, while the target-column bounds and fill
are persisted separately.

#### Fit the whole plan in one call

The fit side has a matching executor. Fitting several dependent transforms by
hand means the fit → apply → fit dance: fit the bounds, apply them, fit the
fills on the clipped frame, apply those, fit the vocabulary on the imputed
frame… `ds.pipeline.fit_pipeline` runs that chain from a *plan* — an ordered
list of `FitStep` entries, each pairing a step kind with the callable that
fits it — and returns the assembled `Pipeline`:

```python
from ds.pipeline import FitStep, fit_pipeline

plan = [
    FitStep("clip_outliers", lambda df: fit_outlier_bounds(df, columns=["fare"])),
    FitStep("impute_missing", lambda df: fit_impute_values(df, columns=["age"], strategy="median")),
    FitStep("one_hot_encode", lambda df: fit_one_hot_categories(df, columns=["sex", "embarked"])),
    FitStep("scale_features", lambda df: fit_scale_params(df, columns=numeric_features)),
]
scoring = fit_pipeline(train, plan)                   # fit → apply → fit, in order
```

Each step's callable receives the training frame *as transformed by the
steps before it*, exactly as the manual chain would. The lambdas close over
the varying `fit_*` keyword arguments (`columns=`, `strategy=`, `k=`), so
one plan type covers every transform without mirroring each signature —
which also means a plan is code, not data: persist the fitted `Pipeline` it
returns, not the plan. Write the plan as the *scoring* plan (nothing fitted
on the target column), and pass explicit `columns=` when fitting on a
modeling frame — the `columns=None` defaults select by dtype and would sweep
the target in. The real-data projects fit their scoring pipelines this way
(five steps in `projects/nyc_taxis` and `projects/titanic`, three in
`projects/diamonds`, one in `projects/flights`), and the same plan re-fits
inside every cross-validation fold (see
[Evaluate](#evaluate-dsevaluation)).

### Model — `ds.modeling`

Split tabular data into features and target, and hold out a test set: split
a time series chronologically so the test window is strictly in the future,
or order-free data randomly — stratified on the target if a class imbalance
must survive the split. The random split draws from numpy's global
generator, so `seed_everything` makes it reproducible.

```python
from ds.modeling.tabular import split_features_target, train_test_split_random
from ds.modeling.timeseries import train_test_split_by_time

train, test = train_test_split_by_time(df, "date", test_size=0.2)   # temporal
train, test = train_test_split_random(df, test_size=0.2, stratify="label")  # order-free
x_train, y_train = split_features_target(train, "amount")
```

Every first metric needs a reference point: fit a naive baseline on the
training target and score it alongside the model — an r² only means something
relative to the naive floor:

```python
from ds.modeling.baseline import fit_baseline

baseline = fit_baseline(y_train, strategy="mean")  # or "naive_last",
baseline_preds = baseline.predict(len(y_test))     # or "seasonal_naive" with season_length=
```

For classification the same call takes `strategy="majority"` — predict the
modal training label (numeric, e.g. an int-coded 0/1 target), the reference
every classifier must beat.

Once an estimator is fitted, persist it so a later run (or another process)
scores without refitting — the model-side counterpart to
`save_params`/`load_params`:

```python
from ds.modeling.persistence import load_model, save_model

save_model(model, "artifacts/model.joblib")   # training run
model = load_model("artifacts/model.joblib")  # scoring run
preds = model.predict(x_new)
```

**Trust boundary:** `load_model` unpickles the file (via `joblib`), which can
execute arbitrary code — only load model files written by you or a process you
trust. Fitted *transform* parameters don't carry this risk; they persist as
validated JSON via `save_params`/`load_params`.

Text work lives here too (needs the `nlp` extra for an accurate count, otherwise
falls back to a whitespace estimate):

```python
from ds.modeling.nlp import count_tokens

count_tokens("how many tokens is this?")
```

### Evaluate — `ds.evaluation`

```python
from ds.evaluation import (
    classification_metrics,
    confusion_frame,
    per_class_metrics,
    regression_metrics,
)

regression_metrics(y_true, y_pred)       # mae, rmse, r2
classification_metrics(y_true, y_pred)   # accuracy, precision, recall, f1 (averaged)
confusion_frame(y_true, y_pred)          # labeled confusion matrix (true x predicted)
per_class_metrics(y_true, y_pred)        # precision/recall/f1/support per class
```

One held-out score can flatter (or slander) a model; cross-validate to see the
spread. For time-ordered rows use the rolling-origin form — every test window
is strictly in its training data's future, the repeated-fold counterpart to
`train_test_split_by_time` (a shuffled k-fold on temporal data trains on the
future); `cross_validate_kfold` is for order-free data. Both build a fresh
model per fold via the factory you pass and return one row of metrics per
fold:

```python
from sklearn.linear_model import LinearRegression

from ds.evaluation import cross_validate_by_time, cross_validate_kfold

cross_validate_by_time(          # fold i trains on blocks 1..i, tests on i+1
    df, time_column="date", target="amount", make_model=lambda: LinearRegression()
)
cross_validate_kfold(            # plain shuffled k-fold, order-free data only
    df.drop(columns=["date"]), target="amount", make_model=lambda: LinearRegression()
)
```

Both default to `regression_metrics` per fold; pass
`metrics_fn=classification_metrics` (or your own scorer) for classifiers —
and on a classification target give `cross_validate_kfold` `stratify=True`,
so every fold keeps the frame's class balance instead of letting it drift.

Watch what frame you hand it: if the features already carry fit-based
transforms fitted on the whole training split, every fold's test rows have
influenced the statistics (imputation fills, scale parameters, …) its
training rows were transformed with. `cross_validate_kfold` closes that leak
with `make_pipeline` — pass the raw frame plus the same fit plan the training
run uses (see [Fit the whole plan in one
call](#fit-the-whole-plan-in-one-call)), and the transform chain is re-fitted
on each fold's training rows only:

```python
from ds.pipeline import fit_pipeline

cross_validate_kfold(
    raw_train,                     # before the fit-based transforms
    target="survived",
    make_model=lambda: LogisticRegression(max_iter=1000),
    make_pipeline=lambda frame: fit_pipeline(frame, plan),
    stratify=True,
    metrics_fn=classification_metrics,
)
```

Finally, score candidates side by side — including the baseline — with
`compare_models`, whose frame feeds `ds.viz.plot_model_comparison`:

```python
from ds.evaluation import compare_models

comparison = compare_models(
    y_test, {"linear": y_pred, "train_mean": baseline_preds}
)
```

### Visualize — `ds.viz`

```python
from ds.viz import (
    plot_confusion_matrix,
    plot_missingness,
    plot_model_comparison,
    plot_outliers,
    plot_residuals,
    plot_series,
    set_theme,
)

set_theme("notebook")                    # consistent matplotlib theme + palette
plot_missingness(df)                     # bar chart of missing fractions
plot_outliers(df)                        # bar chart of outlier counts per column
plot_confusion_matrix(y_true, y_pred)    # annotated heatmap
plot_residuals(y_true, y_pred)           # residual-vs-predicted diagnostic
plot_model_comparison(comparison)        # one metric across models, as bars
plot_series(df["date"], df["amount"])    # one series over a time axis
```

Each plot returns a matplotlib `Axes` and accepts an existing `ax=`, so they
compose into multi-panel figures. They pair with the `ds.eda`,
`ds.preprocessing` and `ds.evaluation` helpers (`plot_missingness` visualizes
`missing_value_report`, `plot_outliers` visualizes `flag_outliers`,
`plot_confusion_matrix` visualizes `confusion_frame`, and
`plot_model_comparison` visualizes `compare_models`).

`plot_series` is the time-series workhorse: its `predictions=` mapping
overlays dashed, named forecast lines over the same time axis, and its
colours come from the Axes' colour cycle, so repeated calls on one `ax`
compose. The standard forecast-vs-actual figure is two calls — the training
tail first, then the held-out window with the model's and a
`fit_baseline` reference's predictions overlaid:

```python
ax = plot_series(history["date"], history["amount"], label="history")
plot_series(
    test["date"],
    y_test,
    predictions={"model": y_pred, "seasonal naive": baseline.predict(len(y_test))},
    label="actual",
    ax=ax,
)
```

### Cross-cutting

```python
from ds import get_logger, get_settings, seed_everything

seed_everything(42)                 # reproducible randomness
settings = get_settings()           # paths: settings.raw_dir, .processed_dir
logger = get_logger(__name__)
```

### Validate what you just loaded

`load_raw`/`load_table` don't validate — pair them with `check_schema` right
at the acquire boundary so a malformed file fails on read, not three stages
later inside a transform that assumes clean input:

```python
from ds.io import load_raw
from ds.validation import check_schema

df = check_schema(
    load_raw("sales.csv"),
    {"date": "datetime64[ns]", "amount": "float64", "category": "str"},
    coerce=True,
)
```

### Screen for redundant features before scaling

`top_correlations` isn't just an exploratory printout — run it right before
feature scaling to catch near-duplicate numeric columns (a copy-pasted unit
conversion, a feature and its rolling average) so you drop one side instead of
feeding a model two collinear inputs:

```python
from ds.eda import top_correlations
from ds.features import scale_features

redundant = top_correlations(df, n=5)
to_drop = redundant.loc[redundant["correlation"].abs() > 0.95, "feature_b"]
df = df.drop(columns=list(to_drop))
df = scale_features(df, ["amount", "amount_rolling_7d"], method="standard")
```

### Fit, evaluate and diagnose a model in one pass

`ds.modeling` only splits data — pair it with `ds.evaluation` and `ds.viz` to
close the loop from features to a diagnostic plot, using any scikit-learn-style
estimator:

```python
from sklearn.linear_model import LinearRegression

from ds.evaluation import regression_metrics
from ds.modeling.tabular import split_features_target
from ds.modeling.timeseries import train_test_split_by_time
from ds.viz import plot_residuals

train, test = train_test_split_by_time(df, "date", test_size=0.2)
x_train, y_train = split_features_target(train.drop(columns=["date"]), "amount")
x_test, y_test = split_features_target(test.drop(columns=["date"]), "amount")

model = LinearRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)

regression_metrics(y_test, y_pred)   # mae, rmse, r2
plot_residuals(y_test, y_pred)       # residual-vs-predicted diagnostic
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
ds run "Customer Churn"                          # runs projects/customer_churn/pipeline.py
uv run pytest projects/customer_churn --no-cov
```

`ds run <name>` is the project-aware counterpart to `ds new`: it matches the
name against the directories under `projects/` — literally or by the same slug
`ds new` derives, so `"Customer Churn"`, `customer_churn` and `customer-churn`
all find `projects/customer_churn/` — and runs that project's `pipeline.py` with
the current interpreter. Run it with a name it can't match (or a typo) and it
lists the runnable projects instead. It never builds a path out of the name, so
a stray `../` matches nothing rather than escaping `projects/`. The explicit
form still works if you prefer it:

```bash
uv run python projects/customer_churn/pipeline.py
```

`ds run` runs under whatever interpreter invoked it, so run it inside the
project environment (`uv run ds run "Customer Churn"` if `ds` isn't already on
your active `PATH`).

If you find yourself copy-pasting a helper between projects, that's the signal it
belongs back in `src/ds/`.
