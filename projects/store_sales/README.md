# Store sales

Forecast **daily units sold across a store × item panel** — the tenth project on
real data, and the first on a **panel**: not one flat table and not one
univariate series, but many series stacked in one frame. It was picked by the
demand-loop rule (stress the data *shape* no project had): every project before
it was single-entity, so the toolkit's time-series surface had only ever seen one
series at a time.

**Dataset:** the training file of the classic "Store Item Demand Forecasting
Challenge" — 500 daily series (10 stores × 50 items), 2013–2017 — from a *live*
third-party GitHub mirror
([DharitShah13/Kaggle-Store-Item-Demand-Forecasting-Challenge](https://github.com/DharitShah13/Kaggle-Store-Item-Demand-Forecasting-Challenge)).
Because that is a live repo (not a byte-frozen release), the fetch pins its
sha256 through `ds.io.fetch_dataset` and verifies the download — the helper's
fifth consumer. The analysis works a **subset** — every store, the first five
items = 50 entities — enough that the grouped operations plainly matter while the
one-hot entity effects and the run stay small.

## The friction it surfaced (the deliverable)

A panel breaks the single-series assumption at exactly one place, and finding it
was the point:

> `ds.features.add_lagged_features` lagged **by row position over the whole
> frame**. Stacking store 2 beneath store 1 made store 2's first rows read *store
> 1's tail* as their history — history bled across every entity boundary.

The fix this project pulled is the helper's new **`group=` parameter**: lags
taken independently *within* each `(store, item)` via a grouped `shift`, so no
value ever crosses an entity edge. That is the one library change of the loop.

The rest of the panel's friction is recorded in
[`ROADMAP_ARCHIVE.md`](../../ROADMAP_ARCHIVE.md) and handled **inline**, because on
a *shared-calendar* panel each single-series helper has a clean workaround worth
only a few lines — below the bar for a library alias:

- **`train_test_split_by_time` splits one series by row fraction.** A panel wants
  a *value* cutoff, so the split is a `date < 2017-01-01` mask — one global
  boundary that cuts every entity at the same instant.
- **`fit_baseline` is single-series.** Each entity's naive forecast is simply its
  own lag column: `sales_lag_1` *is* its naive-last, `sales_lag_7` *is* its
  same-weekday-last-week (weekly seasonal naive). No per-group fitting needed.
- **`assert_unique` guards a single column.** The panel key is the composite
  `(store, item, date)`, checked inline.
- **A per-entity rolling-origin backtest** (the parked `ds.evaluation` harness)
  is what a panel most wants next; this project evaluates one-step-ahead and
  leaves the backtest as the recorded next pull.

## Modeling decisions worth knowing

- **Order the panel first.** Rows are sorted by `(store, item, date)` so every
  entity is contiguous and time-ordered — the precondition the grouped lags (and
  everything downstream) depend on.
- **A pooled linear model with fixed effects.** One `LinearRegression` over every
  entity: store and item one-hots (entity effects), day-of-week and month
  one-hots (the weekly + yearly cycle), an elapsed-months trend, and the three
  within-entity lags (1, 7, 14 days). The only fitted state is the one-hot
  vocabularies (`ds.pipeline.fit_pipeline`); the lags are stateless and OLS is
  scale-free, so there is no impute/clip/scale step — a scope finding matching
  `flights` and `sunspots`.
- **Held out the final year.** Train on 2013–2016, forecast all of 2017 — a
  strictly-future window for every entity at once.
- **Results are honest and ordered.** One-step-ahead on 2017: pooled model
  **MAE ≈ 5.27, r² ≈ 0.89**, beating the **weekly-naive** (`sales_lag_7`,
  MAE ≈ 6.70) and **naive-last** (`sales_lag_1`, MAE ≈ 7.99) references. The
  weekly naive beating last-value is itself the read that the day-of-week cycle
  dominates — exactly the signal the calendar dummies and `lag_7` carry.

## Run

```bash
uv run ds run store_sales
```

## Test

```bash
uv run pytest projects/store_sales --no-cov
```

(`--no-cov` skips the library's repo-wide coverage gate, which doesn't apply to
a single project's tests. The end-to-end test downloads the dataset on first run
and skips itself when the network is unavailable.)

## Layout

- `pipeline.py` — the analysis, stage by stage.
- `notebooks/` — exploratory notebooks for this project.
- `tests/` — unit tests for the panel helpers (including the no-bleed guard on
  the grouped lags) and an end-to-end run.

Built on the shared [`ds`](../../src/ds) toolkit. Reads inputs from `data/raw/`,
writes outputs to `data/processed/`.
