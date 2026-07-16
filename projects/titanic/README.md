# Titanic

Predict a passenger's **survival** from real manifest records — the second
project on data that was *not* generated to fit the toolkit, and the first
**classification** one.

**Dataset:** the seaborn `titanic` dataset — the classic 891-passenger RMS
Titanic manifest, mirrored in the
[seaborn-data](https://github.com/mwaskom/seaborn-data) repository. It brings
real quirks: missing values at three very different severities (`age` ~20%,
`deck` ~77%, `embarked` two rows), the target respelled as a feature
(`alive`), derived duplicate columns (`class`, `who`, `adult_male`,
`embark_town`, `alone`), and a heavily skewed `fare`.

The pipeline downloads the CSV once into `data/raw/` (git-ignored) and runs
fetch → validate → explore → clean → stratified split →
fit-on-train/apply-to-both → persist the scoring `Pipeline` and the fitted
model (`ds.modeling.persistence`) → score the held-out split from the
*reloaded* model → evaluate with the classification stack → confusion-matrix
and comparison plots.

This project exists to run the workspace's demand loop a second time — and to
exercise the evaluation surface no real project had touched:
`classification_metrics`, `confusion_frame`, `per_class_metrics`,
`plot_confusion_matrix`, and the first composition of `cross_validate_kfold`
with `metrics_fn=classification_metrics`. Friction it surfaced in the library
is recorded as the new backlog in [`ROADMAP.md`](../../ROADMAP.md); nothing
is promoted in the same change (demand first, one step per change).

## Modeling decisions worth knowing

- **`alive` and `class` are verified, then dropped.** They are the target and
  `pclass` respelled; the pipeline asserts the one-to-one mapping before
  dropping so a changed upstream file can't silently turn a "redundant"
  column into discarded signal. `who`/`adult_male`/`embark_town`/`alone` are
  deterministic functions of retained columns and go with them.
- **`deck` becomes a `deck_known` indicator.** At ~77% missing it is too
  sparse to impute a level into, but *whether* a deck was recorded is itself
  informative (cabins were recorded mostly for first class).
- **No `drop_duplicate_rows`.** 107 rows are exact duplicates yet are
  distinct passengers (the manifest has no identifier column); deduplicating
  would silently delete real people.
- **Stratified shuffled split.** No time axis, so the chronological splitter
  doesn't apply; scikit-learn's `train_test_split(stratify=...)` keeps the
  62/38 class balance in both halves (a gap recorded as friction —
  `ds.modeling` only ships the chronological splitter).
- **Two baselines.** The majority-class reference (hand-rolled:
  `fit_baseline` is regression-shaped — friction, recorded not built) and
  the classic sex-only rule (predict survival iff female), which any model
  must beat to justify its other features.

## Run

```bash
uv run ds run titanic
```

## Test

```bash
uv run pytest projects/titanic --no-cov
```

(`--no-cov` skips the library's repo-wide coverage gate, which doesn't apply
to a single project's tests. The end-to-end test downloads the dataset on
first run and skips itself when the network is unavailable.)

## Layout

- `pipeline.py` — the analysis, stage by stage.
- `notebooks/` — exploratory notebooks for this project.
- `tests/` — unit tests for the hand-rolled cleaning/features and an
  end-to-end run.

Built on the shared [`ds`](../../src/ds) toolkit. Reads inputs from
`data/raw/`, writes outputs to `data/processed/`.
