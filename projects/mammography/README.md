# Mammography

Flag the rare **calcification** in a mammography screen — the twelfth project on
real data and the **second** deliberately imbalanced / rare-event one after
`bank_marketing`. Only **2.3%** of the 11,183 screened regions in the Woods
mammography dataset are positive (rarer than `bank_marketing`'s 11%). It was
picked by the demand-loop rule to give `ds.evaluation.probability_metrics` its
*second* consumer and, in doing so, to fire the two friction items that first
project parked.

**The friction it surfaced (the deliverable):** `bank_marketing` handled its
imbalance by *reweighting* the loss (`class_weight="balanced"`), which keeps the
0.5 threshold meaningful. A screening programme instead has an **operating
point** — "catch at least this fraction of the calcifications" — that
reweighting can't express, so this project fits a plain logistic regression and
**tunes the decision threshold**. That pulled two surfaces into the library:

- `ds.evaluation.choose_threshold` — sweep the precision–recall curve for the
  F1-optimal threshold, or the cheapest one meeting a recall budget (the
  fiddly-but-reusable part is the `precision_recall_curve` off-by-one it hides).
- `ds.viz.plot_pr_curve` / `plot_roc_curve` — the operating-point *curve*, where
  the whole sweep is the finding, with its no-skill baseline.

On the held-out split the model **ranks** calcifications well (ROC-AUC 0.97,
average precision 0.65 vs the 0.02 prevalence floor), but at the naive 0.5 cut it
misses ~64% of them (recall 0.37). Tuning to an 80% recall budget lifts recall to
0.92 at a read-off precision cost (0.28) — the trade a screen actually has to
make, made explicit. Two data facts are load-bearing: the severity label ships
quoted (`'-1'` / `'1'`) and is stripped before encoding, and ~30% of rows are
exact duplicates that are **kept** (with six coarse standardized attributes and
no patient id, identical vectors are expected distinct screens — the
titanic / bank_marketing precedent).

**Dataset:** downloaded from a pinned third-party GitHub mirror and verified by
sha256 through `ds.io.fetch_dataset`.

## Run

```bash
uv run ds run mammography
```

## Test

```bash
uv run pytest projects/mammography --no-cov
```

(`--no-cov` skips the library's repo-wide coverage gate, which doesn't apply
to a single project's tests.)

## Layout

- `pipeline.py` — the analysis, stage by stage.
- `notebooks/` — exploratory notebooks for this project.
- `tests/` — checks for the pipeline.

Built on the shared [`ds`](../../src/ds) toolkit. Read inputs from `data/raw/`,
write outputs to `data/processed/`.
