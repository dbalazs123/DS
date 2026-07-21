# Bank Marketing

Predict who subscribes to a bank **term deposit** — the eleventh project on real
data, and the first deliberately **imbalanced / rare-event** one: only **11.3%**
of the 41,188 phone contacts in the UCI Bank Marketing dataset (the
`bank-additional-full` variant) said yes. It was picked by the demand-loop rule
(stress the data *shape* no project had): every prior classification project
worked a roughly-balanced target, so the toolkit had never met a rare positive
class where **accuracy is a trap** — a model that predicts "no" for everyone
scores 0.887 accuracy while finding *not one* subscriber.

**The friction it surfaced (the deliverable):** there was no way to score a
classifier's *probabilities* — its threshold-free ranking of who is likeliest to
say yes. That pulled the new `ds.evaluation.probability_metrics` (ROC-AUC /
average precision / Brier) into the library. On the held-out split the model's
accuracy (0.835) is actually *below* the majority floor (0.887), yet its ROC-AUC
(0.80 vs the 0.50 floor) and average precision (0.46 vs the 0.11 prevalence
floor) show it genuinely ranks subscribers — the exact case the probabilistic
metrics exist to reveal. Two boundary calls are load-bearing: `duration` is
dropped as **leakage** (known only after the call it predicts), and the
`pdays == 999` "never contacted" sentinel becomes a binary flag.

**Dataset:** downloaded from a pinned third-party GitHub mirror and verified by
sha256 through `ds.io.fetch_dataset`. The imbalance itself is handled
idiomatically with `class_weight="balanced"` on the estimator (a scikit-learn
argument, not a library gap).

## Run

```bash
uv run ds run bank_marketing
```

## Test

```bash
uv run pytest projects/bank_marketing --no-cov
```

(`--no-cov` skips the library's repo-wide coverage gate, which doesn't apply
to a single project's tests.)

## Layout

- `pipeline.py` — the analysis, stage by stage.
- `notebooks/` — exploratory notebooks for this project.
- `tests/` — checks for the pipeline.

Built on the shared [`ds`](../../src/ds) toolkit. Read inputs from `data/raw/`,
write outputs to `data/processed/`.
