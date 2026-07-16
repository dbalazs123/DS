# NYC Taxis

Predict a ride's **fare** from real trip records — the first project on data
that was *not* generated to fit the toolkit.

**Dataset:** the seaborn `taxis` dataset — 6,433 NYC yellow/green-cab rides
from March 2019, sampled from the NYC TLC trip records and mirrored in the
[seaborn-data](https://github.com/mwaskom/seaborn-data) repository. It brings
real quirks: missing payment types and boroughs, ~200-level pickup/dropoff
zone columns (too many to one-hot — the model uses boroughs instead), and
post-ride columns (`tip`, `tolls`, `total`) that would leak the target.

The pipeline downloads the CSV once into `data/raw/` (git-ignored) and runs
fetch → validate → explore → clean → chronological split →
fit-on-train/apply-to-both → persist the scoring `Pipeline` and the fitted
model (`ds.modeling.persistence`) → score the held-out window from the
*reloaded* model → evaluate against a naive train-mean baseline → residual
plot.

This project exists to run the workspace's demand loop: friction it surfaced
in the library is recorded in [`ROADMAP.md`](../../ROADMAP.md). Its first
item — no model persistence — has since been closed by
`ds.modeling.persistence.save_model`/`load_model`, which this pipeline now
uses; still open: no `hour` from `add_datetime_features`, no high-cardinality
encoder, no baseline estimators.

## Run

```bash
uv run ds run nyc_taxis
```

## Test

```bash
uv run pytest projects/nyc_taxis --no-cov
```

(`--no-cov` skips the library's repo-wide coverage gate, which doesn't apply
to a single project's tests. The end-to-end test downloads the dataset on
first run and skips itself when the network is unavailable.)

## Layout

- `pipeline.py` — the analysis, stage by stage.
- `notebooks/` — exploratory notebooks for this project.
- `tests/` — a unit test for the hand-rolled features and an end-to-end run.

Built on the shared [`ds`](../../src/ds) toolkit. Reads inputs from
`data/raw/`, writes outputs to `data/processed/`.
