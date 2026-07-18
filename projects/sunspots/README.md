# Sunspots

Forecast the **monthly sunspot number** — the eighth project on real data, and
the **second forecasting** one. It was picked, after `flights`, precisely for a
series the calendar-feature + naive approach handles *badly*, so the friction
would pull the toolkit's forecasting surface deeper.

**Dataset:** the monthly mean Zurich/SILSO sunspot number, 1749–1983 (2,820
months), from the widely-mirrored
[jbrownlee/Datasets](https://github.com/jbrownlee/Datasets) copy. Because that
is a *live* third-party repo (not a byte-frozen release), the fetch pins its
sha256 through `ds.io.fetch_dataset` and verifies the download — the helper's
third consumer after `air_quality` and `adult_income`. It was chosen
deliberately:

- **A cycle aligned to nothing on the calendar.** The ~11-year solar cycle
  wanders between roughly 9 and 14 years, so month-of-year carries essentially
  no signal (the Explore step shows the by-month means are flat within ±3% of
  the overall mean) and a `seasonal_naive` of period 12 is a poor guide. The
  calendar features `flights` leaned on — a month one-hot, an elapsed-months
  trend — are useless here *by construction*.
- **A series whose signal is its own history.** What predicts next month is the
  recent past — momentum plus the slow cycle — so this is the first project to
  need **autoregressive** (lag) features and, to forecast past the edge of the
  data, a **recursive multi-step** forecast. Those are the two library gaps it
  surfaced.
- **Long enough to be honest about horizons.** 2,820 months is ~21 cycles —
  plenty to fit an AR model, hold out a decade, and still run rolling-origin
  folds. The honest conclusion is about *horizon*: one-step forecasting is
  strong, multi-step is hard.

The pipeline downloads the CSV once into `data/raw/` (git-ignored) and runs
fetch → validate → assemble the time axis → explore (the flat by-month profile
and the series plot, `ds.viz.plot_series`) → autoregressive lag features
(`ds.features.add_lagged_features`) → chronological split → rolling-origin
one-step cross-validation (`cross_validate_by_time`) → persist the fitted model
→ from the *reloaded* model, both a one-step-ahead forecast and a recursive
multi-step forecast (`ds.modeling.timeseries.forecast_recursive`) → evaluate
against the naive references → visualize (residuals, model comparison, and a
forecast-vs-actual figure composed from two `plot_series` calls).

This project runs the workspace's demand loop an eighth time and its
forecasting rotation a second time. Its friction — no autoregressive-feature
helper, no recursive-forecast helper — is recorded in
[`ROADMAP.md`](../../ROADMAP.md), and, because forecasting is a committed
capability whose first project (`flights`) already delegated its model to raw
scikit-learn, that friction was served in the same demand loop: the pipeline
consumes `add_lagged_features` and `forecast_recursive` directly.

## Modeling decisions worth knowing

- **Autoregressive features, not calendar features.** `add_lagged_features`
  adds the value 1/2/3/6/12 months back. It is a *stateless* transform (a row's
  lags are the rows already beside it), so it is applied before the split; the
  warm-up rows with no complete history are dropped.
- **The model is a pure autoregression.** `LinearRegression` on the lag columns
  alone. There is deliberately **no `ds.pipeline` scoring `Pipeline`** here: the
  lags are stateless, the series is complete (no impute), its swings are the
  signal (no clip), and OLS is scale-free (no scale), so there is no fit-based
  frame transform to persist — a scope finding, not a gap (`flights` already had
  a one-step plan; this AR model has none). Only the model is persisted, and
  both forecasts score from the reloaded copy.
- **Two forecasts, one honest gap between them.**
  - *One-step-ahead* reads the true recent values at each step (what you have
    when forecasting *next* month): held-out **MAE ≈ 15.3, r² ≈ 0.88** — strong.
  - *Recursive multi-step* forecasts the whole decade from the end of training,
    feeding each prediction back as later steps' lags (`forecast_recursive`):
    **MAE ≈ 52.5, r² ≈ −0.35**. Error compounds over 120 steps and the forecast
    decays toward the mean, because the cycle is far longer than any lag — the
    genuine, well-known difficulty of long-range solar forecasting, recorded
    rather than hidden.
- **Even the decayed recursive forecast beats the calendar-naive references.**
  `seasonal_naive` (repeat the value 12 months back) MAE ≈ 58.1 and `naive_last`
  (repeat one arbitrary cycle phase) MAE ≈ 63.1 are both worse than the
  recursive AR — the point that autoregression, even extrapolated, carries more
  than the calendar approach does on this series.

## Run

```bash
uv run ds run sunspots
```

## Test

```bash
uv run pytest projects/sunspots --no-cov
```

(`--no-cov` skips the library's repo-wide coverage gate, which doesn't apply to
a single project's tests. The end-to-end test downloads the dataset on first
run and skips itself when the network is unavailable.)

## Layout

- `pipeline.py` — the analysis, stage by stage.
- `notebooks/` — exploratory notebooks for this project.
- `tests/` — unit tests for the hand-rolled time-axis helper and an end-to-end
  run.

Built on the shared [`ds`](../../src/ds) toolkit. Reads inputs from `data/raw/`,
writes outputs to `data/processed/`.
