# Air Quality

Reconstruct a **reference CO measurement** from the rest of a monitoring
station — the sixth project on data that was *not* generated to fit the
toolkit, and the first against **instrument-outage missingness** on a
**gapped hourly time axis**.

**Dataset:** the [UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality)
— 9,357 hourly records from a multi-sensor device co-located with certified
reference analyzers at road level in an Italian city, March 2004 to April
2005. The certified CO analyzer (`co_gt`, mg/m³) was down for 18% of those
hours; predicting its reading from what the rest of the station saw that hour
— the five metal-oxide sensor channels, temperature/humidity, and the NOx/NO2
reference analyzers — is the task, so the held-out evaluation mimics
back-filling an instrument outage from its neighbours. It was chosen
deliberately, by first grepping which library surfaces still had no real
consumer, and weighting the choice toward the open watch-list in `ROADMAP.md`:

- **Imputation at real severity, at last.** The biggest recorded coverage gap
  — imputation exercised only at titanic's severity, with diamonds and
  sms_spam both contributing zero missing values. Here −200 sentinels decode
  to genuine missingness: one column 90.2% gone, the target 18.0%, the
  NOx/NO2 feature channels partially and *independently* gapped. Median fills
  fitted on the training window flow through the persisted pipeline over
  those cell-level gaps.
- **`assert_dtypes`' first real consumer.** The raw file is semicolon-
  separated with **decimal commas** (`2,6` not `2.6`); `sep=";"` alone parses
  every measurement column as strings and nothing downstream complains. A
  dtype pin over the measurement columns makes that silent misparse loud —
  exactly the failure the guard exists for.
- **A rolling-origin CV whose per-fold state genuinely varies.** `ROADMAP.md`
  item 9's parked question — `cross_validate_by_time(make_pipeline=...)` —
  had been waiting for a consumer whose per-fold fitted state actually
  changes. This project is it: the impute medians and scale parameters drift
  with the seasons (`nox_gt`'s impute median swings ~28% across the folds),
  where the earlier projects' single-vocabulary plans re-fit to the same
  values every fold. That pulled the parameter into being — served as item
  22 — and the CV now re-fits the transform plan on each fold's own window.
- **A silently-wrong boundary parse and a hand-assembled time axis.** The
  file carries two empty trailing columns and 114 all-empty trailing rows
  (a plausible read keeps them), caught by a row-count check against the
  published size; and the time axis arrives in two pieces (`date` +
  dot-separated `time`), assembled by hand. Both re-fire earlier
  second-project triggers (items 20 and 13).

The pipeline downloads the CSV once into `data/raw/` (git-ignored) and runs
fetch → trim the file's structural junk and pin the parse (`assert_dtypes`) →
decode the −200 sentinels to NaN → the missing-value triage (drop the 90%
column, drop 366 device-offline hours, drop 1,647 unlabeled hours, impute the
partial remainder) → hand-assemble the hourly time axis → calendar features
(hour, is_weekend, the elapsed-months drift term) → chronological split
(`train_test_split_by_time`) → fit the three-step transform plan
(`ds.pipeline.fit_pipeline`: median impute / 24-level hour one-hot /
standardize) on the training window → rolling-origin cross-validation
(`cross_validate_by_time`, re-fitting that plan on each fold's own window via
`make_pipeline`) → persist the scoring
`Pipeline` and the fitted model → score the held-out window from the
*reloaded* model → evaluate against two references → visualize.

This project exists to run the workspace's demand loop a sixth time. Friction
it surfaced in the library is recorded as the backlog in
[`ROADMAP.md`](../../ROADMAP.md); nothing is promoted in the same change
(demand first, one step per change).

## Where the data comes from

The UCI archive is not reachable from every network, so `fetch_raw` pins two
**byte-identical GitHub mirrors** of the original `AirQualityUCI.csv` and
verifies the download's sha256
(`13277ae5…b567cc6407d0`) before trusting either — the cached copy included, so
a partial earlier download cannot poison a later run. The mirrors are personal
repositories; a silently drifted copy fails the checksum loudly rather than
parsing strangely. This is the first project to need a verified fetch; the
rationale is recorded in `ROADMAP.md`.

## Modeling decisions worth knowing

- **Most of the missingness is not imputation's job.** The report → triage →
  impute-the-remainder sequence is the real pattern: 90% missing means drop
  the column (`nmhc_gt`); all-channels-offline means drop the row (nothing to
  predict *from*); a missing *target* means the row is the deployment
  condition, not training data. Only the genuine partial gaps that survive —
  the independently-gapped NOx/NO2 channels — reach the median-impute step.
  `missing_value_report` and `plot_missingness` carry the triage.
- **`c6h6_gt` is excluded as a near-identity.** The correlation report shows
  it at 0.98 with `pt08_s2_nmhc` (it is published as a transform of that
  channel), so it is dropped rather than fed as a redundant feature.
- **The seasonal reference is hand-rolled, not `fit_baseline`.**
  `fit_baseline("seasonal_naive")` aligns *positionally*, and on this axis —
  with the unlabeled and offline hours removed — position `i − 24` is usually
  not the same hour yesterday. So the "same hour yesterday" reference is a
  time-*indexed* lookup (`same_hour_yesterday_reference`), falling back to the
  training mean where the lag hour is also unlabeled. Recorded as item 23.
- **The model beats both references.** Held-out (last ~20% of labeled hours):
  MAE 0.305 / RMSE 0.458 / r² 0.876, versus the same-hour-yesterday reading
  (MAE 0.799, r² 0.223) and the training mean (MAE 1.068, r² −0.030). A Ridge
  regression (the channels sit on wildly different scales — sensor
  resistances in the thousands, absolute humidity around one — and the
  penalty is scale-sensitive, hence the standardize step). Rolling-origin CV
  MAE 0.406 ± 0.111 agrees the held-out number is not a lucky window.

## Run

```bash
uv run ds run air_quality
```

## Test

```bash
uv run pytest projects/air_quality --no-cov
```

(`--no-cov` skips the library's repo-wide coverage gate, which doesn't apply
to a single project's tests. The end-to-end test downloads the dataset on
first run and skips itself when the network is unavailable.)

## Layout

- `pipeline.py` — the analysis, stage by stage.
- `notebooks/` — exploratory notebooks for this project.
- `tests/` — unit tests for the hand-rolled trim/sentinel/time-axis/reference
  helpers and an end-to-end run.

Built on the shared [`ds`](../../src/ds) toolkit. Reads inputs from
`data/raw/`, writes outputs to `data/processed/`.
