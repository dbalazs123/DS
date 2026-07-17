# SMS Spam

Flag a **text message as spam** from its content — the fifth project on data
that was *not* generated to fit the toolkit, and the first **text** one.

**Dataset:** the SMS Spam Collection — 5,574 real SMS messages labelled
ham/spam (~13% spam), mirrored as a single headerless TSV in the
[pycon-2016-tutorial](https://github.com/justmarkham/pycon-2016-tutorial)
repository (the same one-file fetch pattern as the other four projects). It
was chosen deliberately, by first grepping which surfaces still had no real
consumer:

- **The `nlp` extra's first real consumer.** `ds.modeling.nlp.count_tokens`
  (and with it the tiktoken extra) was the library's only entirely-unconsumed
  module. This pipeline consumes it for the per-message `token_count` — and
  its graceful-degradation contract (whitespace count without tiktoken or its
  vocabulary) turns out to *dictate* where the feature may be used; see the
  determinism decision below.
- **Text stresses the gaps by absence.** The features stage has no text
  helpers (`char_count`/`token_count` are hand-rolled project code), and
  `ds.pipeline`'s closed step vocabulary has no vectorization step — so the
  TF-IDF vectorizer, the fitted heart of a text pipeline, cannot live in the
  persisted `Pipeline` and rides inside the scikit-learn model object
  instead. Observing exactly where that hurts is the point.
- **A second consumer for the `labels=` display mapping.** Binary
  `{0: "ham", 1: "spam"}` on `confusion_frame` / `per_class_metrics` /
  `plot_confusion_matrix`, right after diamonds' five-grade first use — the
  recurrence is recorded in `ROADMAP.md`.
- **A quote-laden headerless file at the boundary.** The raw TSV's double
  quotes are message text, not CSV quoting — under pandas' defaults two rows
  silently vanish into a neighbour. The fix (`header=None, names=...,
  quoting=csv.QUOTE_NONE`) rides entirely on `load_raw` forwarding pandas
  keywords.

The pipeline downloads the TSV once into `data/raw/` (git-ignored) and runs
fetch → validate at the boundary (columns, no nulls, the exact ham/spam
vocabulary) → drop exact duplicate messages → engineer the length features →
explore (length profiles by label, the label mix per length band via
`bin_column`, the length outlier profile) → encode the target → stratified
split (`train_test_split_random`) → fit the one-step transform plan
(`ds.pipeline.fit_pipeline`: standardize `char_count`) → stratified 5-fold
cross-validation with the plan re-fitted per fold
(`cross_validate_kfold(make_pipeline=...)`) → persist the scoring `Pipeline`
and the fitted model → score the held-out split from the *reloaded* model →
evaluate against the majority class and a keyword rule → visualize.

This project exists to run the workspace's demand loop a fifth time. Friction
it surfaced in the library is recorded as the backlog in
[`ROADMAP.md`](../../ROADMAP.md); nothing is promoted in the same change
(demand first, one step per change).

## Modeling decisions worth knowing

- **`token_count` is descriptive, never a model input.** CI runs the suite
  both without extras and with `--extra all`, and `count_tokens` returns real
  BPE counts only when tiktoken *and* its vocabulary are available, degrading
  to a whitespace count otherwise. Feeding it to the model would make the
  fitted coefficients depend on the installed extras, so it feeds only the
  exploration artifacts; the model reads the TF-IDF text features plus
  `char_count`, which is deterministic everywhere. Tests assert structure and
  baselines-beaten, never token-dependent values. One more wrinkle the first
  real consumer surfaced: the degradation is *per call* — with tiktoken
  installed but its vocabulary unreachable, every call re-attempts the
  download (~0.4 s), a ~35-minute stall over 5,000 messages — so the
  pipeline probes the accurate path once (`_resolve_token_counter`) and
  falls back wholesale. Recorded as `ROADMAP.md` item 19.
- **The fitted state is split across two artifacts.** The persisted ds
  `Pipeline` holds the one step the closed vocabulary can express — the
  `char_count` scaler — while the TF-IDF vocabulary persists inside the model
  joblib. Reloading the scoring pipeline alone cannot score a message; this
  is the headline friction item, recorded not patched.
- **Duplicates are dropped here, kept in titanic.** ~400 exact repeats
  (chain-letter spam sent verbatim, stock phrases like "Sorry, I'll call
  later") with no sender or timestamp column to tell a repeat from a
  re-entry — and an identical message straddling the split would hand the
  vectorizer the test answer verbatim, the text analogue of the diamonds
  leak.
- **The target is int-coded through the ordinal encoder.** `encode_label`
  maps ham/spam to 0/1 with the explicit order — stateless, so safe
  pre-split — landing the labels in the integer form the metric surface and
  `fit_baseline("majority")` are typed for, with spam = 1 as the positive
  class the binary metrics score. Artifacts and plots show the names via
  `labels=`; the metric math stays on the codes.
- **Two references.** The majority class (predict ham for everything:
  accuracy 0.873, spam F1 0.000 — why accuracy alone is the wrong lens at
  13% positives) and a hand-written keyword rule over the classic prize-claim
  vocabulary ("free", "win", "prize", "txt", …: accuracy 0.916, F1 0.659).
  Held-out, TF-IDF + logistic regression reaches accuracy 0.968 / spam F1
  0.864 (5-fold CV F1 0.853 ± 0.009 agrees).
- **Precision-heavy errors are the honest headline.** Held-out precision
  0.938 vs recall 0.802: the model rarely cries spam on ham but still waves
  a fifth of spam through — the asymmetry `per_class_metrics` and the 2×2
  matrix show and the accuracy headline (0.968) would bury.

## Run

```bash
uv run ds run sms_spam
```

## Test

```bash
uv run pytest projects/sms_spam --no-cov
```

(`--no-cov` skips the library's repo-wide coverage gate, which doesn't apply
to a single project's tests. The end-to-end test downloads the dataset on
first run and skips itself when the network is unavailable.)

## Layout

- `pipeline.py` — the analysis, stage by stage.
- `notebooks/` — exploratory notebooks for this project.
- `tests/` — unit tests for the hand-rolled encoding/feature/rule helpers and
  an end-to-end run.

Built on the shared [`ds`](../../src/ds) toolkit. Reads inputs from
`data/raw/`, writes outputs to `data/processed/`.
