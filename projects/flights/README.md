# Flights

Forecast **monthly airline passengers** — the third project on data that was
*not* generated to fit the toolkit, and the first **forecasting** one.

**Dataset:** the seaborn `flights` dataset — the 144 monthly totals of
international airline passengers, 1949–1960, i.e. the classic Box & Jenkins
"AirPassengers" series, mirrored in the
[seaborn-data](https://github.com/mwaskom/seaborn-data) repository (the same
fetch pattern as `nyc_taxis` and `titanic`). It was chosen deliberately:

- **A genuine time axis with visible seasonality.** A strong upward trend
  plus a pronounced yearly cycle (summer peaks) whose amplitude grows with
  the level — exactly the structure the untouched time-series surface exists
  for. `train_test_split_by_time` had one consumer before this project;
  `cross_validate_by_time` and `fit_baseline`'s `"naive_last"` /
  `"seasonal_naive"` strategies had none.
- **Small but sufficient.** 144 rows is tiny, but at monthly resolution it is
  twelve full seasonal cycles — enough to fit a 12-level month vocabulary, hold
  out a 29-month window (~2.5 cycles) and still leave every rolling-origin
  fold a training window longer than a year. The honest conclusions here are
  about *protocol* (does the model beat the naive references on a strictly
  future window), not about small metric deltas.
- **Its quirks are temporal, not tabular.** No missing values, no outliers to
  clip (the extremes *are* the seasonal peaks), the time axis split across two
  columns (`year` + a spelled-out `month`), and multiplicative seasonality an
  additive linear model can only approximate — so the pressure lands on the
  time-series helpers rather than on the cleaning stage a third time.

The pipeline downloads the CSV once into `data/raw/` (git-ignored) and runs
fetch → validate → assemble the time axis → explore (incl. the seasonal
profile) → stateless calendar/trend features (`add_datetime_features` + a
hand-rolled `month_index` trend) → chronological split → fit the one-step
transform plan (`ds.pipeline.fit_pipeline`: the month one-hot vocabulary) →
rolling-origin cross-validation (`cross_validate_by_time`) → persist the
scoring `Pipeline` and the fitted model → score the held-out window from the
*reloaded* model → evaluate against the naive-last and seasonal-naive
baselines → visualize (residuals, model comparison, and a hand-rolled
forecast-vs-actual plot).

This project exists to run the workspace's demand loop a third time. Friction
it surfaced in the library is recorded as the new backlog in
[`ROADMAP.md`](../../ROADMAP.md); nothing is promoted in the same change
(demand first, one step per change).

## Modeling decisions worth knowing

- **The time axis is assembled by hand.** The raw file splits it into `year`
  + a month name; `build_time_axis` parses them into one sorted `date`
  column and rejects duplicated months (corrupted input would otherwise
  silently interleave).
- **Half the calendar features don't apply to a monthly series.**
  `add_datetime_features` emits the full set; `date_day`/`date_hour` are
  constant (left for `drop_constant_columns` — that composition worked) while
  `date_dayofweek`/`date_is_weekend` are non-constant *noise* (the weekday of
  the 1st of each month) and are dropped by hand — recorded as friction.
- **Trend enters as `month_index`** (`year * 12 + month`, hand-rolled — no
  library helper emits an elapsed-time feature), and the seasonal shape as
  the one-hot month vocabulary; a numeric month would wrongly order December
  next to nothing.
- **The fit plan has exactly one step.** The series is complete (no
  imputation), its extremes are signal (no clipping), and OLS is scale-free
  (no scaling) — a *scope finding* about `fit_pipeline` on clean time-series
  data, not a failure: the executor still earns its keep as the thing that
  builds the persistable scoring `Pipeline`.
- **Rolling-origin CV consumes an already-transformed frame.**
  `cross_validate_by_time` has no `make_pipeline` (the question parked in
  ROADMAP item 9). Here that is provably harmless: the only fitted state is
  the month vocabulary, and every fold's training window spans more than a
  year, so per-fold re-fitting would re-learn the identical 12 calendar
  months. The demand trigger did *not* fire — recorded honestly in the
  backlog notes.
- **Two naive references.** `naive_last` (repeat the last training value) and
  `seasonal_naive` (repeat the last training year) — the latter is the
  reference any seasonal forecaster must beat. Because the held-out window
  starts right after the training window ends, the seasonal cycle stays
  aligned with the test months by construction. Held-out MAE: linear
  regression ≈ 34.3 vs seasonal-naive ≈ 64.8 and naive-last ≈ 81.4 (in
  thousands of passengers); r² ≈ 0.63 — modest, because an additive model
  underfits the growing seasonal amplitude, and honest, because both naive
  references degrade over a 29-month horizon while the trend term
  extrapolates.

## Run

```bash
uv run ds run flights
```

## Test

```bash
uv run pytest projects/flights --no-cov
```

(`--no-cov` skips the library's repo-wide coverage gate, which doesn't apply
to a single project's tests. The end-to-end test downloads the dataset on
first run and skips itself when the network is unavailable.)

## Layout

- `pipeline.py` — the analysis, stage by stage.
- `notebooks/` — exploratory notebooks for this project.
- `tests/` — unit tests for the hand-rolled time-axis/feature helpers and an
  end-to-end run.

Built on the shared [`ds`](../../src/ds) toolkit. Reads inputs from
`data/raw/`, writes outputs to `data/processed/`.
