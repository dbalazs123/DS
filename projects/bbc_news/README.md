# BBC News

Classify a news article into one of **five topics** — business, entertainment,
politics, sport, tech — the ninth project on real data, and the **second text**
one. It was picked, after `sms_spam`, to stress the text surface a second time
and decide the triggers that first project parked.

**Dataset:** the classic BBC News topic set — 2,225 articles, one label each,
from a widely-mirrored [tutorial repo](https://github.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial).
Because that is a *live* third-party repo, the fetch pins its sha256 through
`ds.io.fetch_dataset` and verifies the download (the helper's fourth consumer).
It was chosen deliberately:

- **Multiclass, not binary.** Five roughly-balanced topics (386–511 articles
  each), so the evaluation is macro-averaged and the confusion structure — which
  topics bleed into which — is part of the story, unlike `sms_spam`'s binary
  spam/ham.
- **A TF-IDF task with real auxiliary length signal.** The learned vocabulary is
  the heart (a length-only reference scores macro-F1 ≈ 0.33 against the full
  model's ≈ 0.96), but document length genuinely varies by topic (tech and
  politics run long, sport short), so the length features earn a place beside
  the vectors — exactly the shape that fires the parked text-feature trigger.

The pipeline downloads the CSV once into `data/raw/` (git-ignored) and runs
fetch → validate → drop verbatim-duplicate articles → text features → explore
(length by topic) → encode the target → stratified split → fit the one-step
scale plan (`ds.pipeline.fit_pipeline`) → stratified 5-fold cross-validation with
the plan re-fitted per fold → persist the scoring `Pipeline` and the fitted model
→ score the held-out split from the *reloaded* model → evaluate (macro-averaged)
against a length-only model and the majority class → visualize.

## Modeling decisions worth knowing

- **`ds.features.text_features` is this project's friction, served.** `sms_spam`
  hand-rolled its length columns because the features stage had none;
  hand-rolling the same family a second time is the trigger backlog item 21
  recorded, so the helper was built and this project consumes it —
  `char_count`, `word_count` and `avg_word_length`, all encoding-independent and
  model-safe.
- **`count_tokens` finally has a *modeling* consumer.** `sms_spam` kept its
  `token_count` descriptive-only, wary of the extras-dependent value. Here it is
  one coarse length feature beside thousands of TF-IDF terms, and the classifier
  is robust to which counting path runs, so it feeds the model — a documented
  evolution of the "descriptive-only" verdict, with the tests asserting
  path-independent macro-F1 bounds rather than exact values.
- **The TF-IDF vectorizer lives inside the sklearn estimator.** The
  model-side-transform convention (backlog item 18): `ds.pipeline`'s step
  vocabulary has no vectorization step, so the fitted vocabulary persists in the
  model joblib while the `ds` scoring `Pipeline` carries the frame-shaped scale
  step. This second text consumer confirms the convention suffices; no
  first-class vectorize step is built (it would smuggle a pickle into the
  strict-JSON `save_params` story).
- **Verbatim duplicates are dropped.** 99 articles appear more than once; with no
  per-article identifier a repeat is indistinguishable from a re-entry, and an
  identical article straddling the split would hand the vectorizer the test
  answer word for word (the diamonds/sms_spam leak).
- **The numbers.** Held-out macro-F1 ≈ 0.96 / accuracy ≈ 0.96 vs the length-only
  model (≈ 0.33) and the majority class (≈ 0.08); 5-fold macro-F1 ≈ 0.96 ± 0.01.
  The wide gap over the length-only reference is the honest headline: the topic
  signal is in the words, and the length features are a modest supplement.

## Run

```bash
uv run ds run bbc_news
```

## Test

```bash
uv run pytest projects/bbc_news --no-cov
```

(`--no-cov` skips the library's repo-wide coverage gate, which doesn't apply to
a single project's tests. The end-to-end test downloads the dataset on first run
and skips itself when the network is unavailable.)

## Layout

- `pipeline.py` — the analysis, stage by stage.
- `notebooks/` — exploratory notebooks for this project.
- `tests/` — unit tests for the text-feature/target helpers and an end-to-end
  run.

Built on the shared [`ds`](../../src/ds) toolkit. Reads inputs from `data/raw/`,
writes outputs to `data/processed/`.
