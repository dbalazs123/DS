# Roadmap archive

Historical detail split out of [`ROADMAP.md`](ROADMAP.md) to keep the active
roadmap small (it was read into context in full on every session). Nothing here
is live plan — the current state and demand queue live in `ROADMAP.md`. This
file is the durable record: the goal evaluation, the completed plan of record
(P1–P20), the per-project friction backlogs (**items 1–41**, referenced by
number from project code and `CHANGELOG.md`), and the settled-decision
rationales that `CLAUDE.md`'s engineering notes point to. Grep it by item number
or decision name when you need the "why" behind a resolved item.

## Goal evaluation (2026-07, refreshed post-P15)

A deliberate stop to work backward from the project's stated goals instead of
extending recent momentum. The per-goal verdicts below were written at the
original 2026-07 stop; the **post-P15 refresh** that supersedes them:

- **The hybrid workspace / promotion-loop goal — the most under-served at the
  original stop — is now well-served.** The demand loop has run seven times
  (`nyc_taxis`, `titanic`, `flights`, `diamonds`, `sms_spam`, `air_quality`,
  `adult_income`), every backlog has been dispatched, and *every* library
  addition since P1 traces to real-project friction. Release 0.2.0 shipped. The
  promotion loop is the settled way this repo grows.
- **The live gaps are forecasting/text depth (and, until this pass, `eda`
  depth).** The `eda` categorical↔target gap (item 29) is now **closed** —
  `target_rate_by_category` + `plot_target_rate` give the stage the categorical
  read on the target it lacked. The remaining depth gaps are forecasting and
  text: while `flights` (forecasting) and `sms_spam` (text) each have one
  project, both delegated their modeling heart to raw sklearn, so the
  forecasting/backtest and vectorization/text-feature surfaces are still
  shallow. The next direction (the "Next up" entry) targets exactly these: run a
  second forecasting and a second text project chosen to *pull* that depth.

The original verdicts, and what they implied at the time:

- **Hybrid workspace (library + `projects/` that consume it)** — *was the
  most under-served goal.* Until `nyc_taxis`, `projects/` held only the
  synthetic `_example`, whose data is generated to be exactly as dirty as the
  library can clean; the promotion loop ("friction in a project becomes
  library work") had never run, while eight consecutive PRs invested in
  supply-side library polish. **Consequence:** every library addition should
  now be pulled by a project need, not pushed from a candidate list. The
  friction backlogs below are the queue (the first six lists — `nyc_taxis`,
  `titanic`, `flights`, `diamonds`, `sms_spam` and `air_quality` — are fully
  dispatched; P14 ran the seventh demand loop (`adult_income`) and refilled the
  queue with items 27–29, and P15 served them, so the queue is empty again).
- **Fit-once / score-later** — *stopped one step short of its own goal;
  since closed (P2).* Fitted parameters and the `Pipeline` persist as strict
  JSON, but at evaluation time the fitted **model** could not be persisted at
  all, so "score new rows in a later run or another process" broke at the
  estimator. `ds.modeling.persistence.save_model`/`load_model` closed this;
  both worked projects now score from reloaded state only.
- **Every stage carries working helpers** — *was sharply uneven; since
  rebalanced (P3).* At evaluation time Model was two split helpers plus
  `count_tokens` and Evaluate four point-metric functions — no baselines, no
  cross-validation, no model comparison. P3 filled exactly those gaps
  (`fit_baseline`, rolling-origin + k-fold cross-validation,
  `compare_models` + its paired plot), each traced to the friction backlog
  rather than a candidate list.
- **"Toolkit for every situation"** — *was dishonest at the packaging layer,
  now fixed.* The `timeseries` extra (statsmodels, sktime) had zero importers,
  most of the `nlp` extra was unused, and `polars` sat unused in the core
  dependencies. Extras now carry only dependencies code actually consumes
  (the rule is recorded in `pyproject.toml` and CLAUDE.md). In-scope reality
  today is **committed and growing**: tabular regression/classification on
  pandas is the deepest surface, and forecasting and NLP/text are equally
  in-scope commitments being widened by demand loops (the `flights` forecasting
  project and the `sms_spam` text project each pulled real surface —
  `plot_series`, `train_test_split_by_time`, `count_tokens`, the naive-baseline
  strategies). The direction is to widen all three by *building* against real
  projects, not to trim the claims to tabular-only — the modeling docstrings
  that name forecasting and text stay as directional commitments.
- **Engineering discipline** — *well-served* (strict typing, mirrored tests,
  coverage gate, honest docs). No change; point it at the gaps above.

Completed work that mattered less against the goals (kept, but the lesson is
recorded): `ds run` and the cross-stage cookbook recipes are good polish that
consumed cycles while the demand side stayed empty. `count_tokens` began as an
NLP toe-dip and is now a **kept, half-earned** helper: `sms_spam` (P10) made it
a real — if descriptive-only — consumer, and it stays as an in-scope
commitment. The way it fully earns its keep is a *second* text project whose
modeling path genuinely needs it (see the "Next up" direction), not removal.
The lesson is the ordering rule above: demand first.

## Plan of record

- **P1 — run the demand loop on real data: DONE.** `projects/nyc_taxis`
  predicts cab fares from the real March-2019 NYC rides sample (seaborn
  `taxis`, mirrored from the NYC TLC records; downloaded once into
  git-ignored `data/raw/`). Full lifecycle on `ds` + scikit-learn, split-safe
  transforms persisted as one scoring `Pipeline`, evaluated against a naive
  baseline (r² 0.73 vs baseline mae 7.2 → 2.6). Its friction list *is* the
  backlog below.
- **P2 — model persistence: DONE.** `ds.modeling.persistence` provides
  `save_model`/`load_model`, joblib under the hood (the format scikit-learn's
  own docs recommend; now a declared core dependency per the
  first-consumer rule). Deliberate line, recorded: joblib/pickle is used
  **only** for the estimator — unpickling executes arbitrary code, so
  `load_model` documents the trust boundary (only load files you or a trusted
  process wrote), while transform parameters stay in strict validated JSON.
  Both worked projects now persist pipeline + model and score from the
  reloaded state with no in-memory carryover; the guide's Model section
  documents the pattern and the warning.
- **P3 — bring Model/Evaluate up to the Clean/Feature standard: DONE.**
  Re-ranked against the friction backlog first, so the batch opened with the
  smallest demand-traced win (`add_datetime_features` now emits `_hour` —
  friction item 2), then, core deps only and to the standard recipe:
  - `ds.modeling.baseline.fit_baseline` (mean / naive-last / seasonal-naive)
    returning a frozen `Baseline` with `predict(n)` — deliberately *not* a
    scikit-learn estimator, because a baseline needs no feature matrix; the
    training target is the whole input, and the output feeds
    `ds.evaluation` directly (friction item 3).
  - `ds.evaluation.cross_validate_by_time` — rolling-origin folds, the
    repeated-fold counterpart to `train_test_split_by_time` (a shuffled
    k-fold on temporal data trains on the future) — plus
    `cross_validate_kfold` for order-free data. Both take a `make_model`
    factory (fresh estimator per fold, no cross-fold state) and a
    `metrics_fn` defaulting to `regression_metrics`, so classification (or a
    custom scorer) composes instead of forking the API.
  - `ds.evaluation.compare_models` + `ds.viz.plot_model_comparison`,
    following the settled stage↔viz pairing convention.
  `nyc_taxis` dogfoods the batch: library `pickup_hour`, `fit_baseline` and
  a persisted comparison frame/plot replaced its hand-rolled versions, with
  identical metrics.
- **P4 — honest packaging: DONE.** Unused pins removed (`polars` from core;
  `sentence-transformers`/`anthropic`/`statsmodels`/`sktime` from extras —
  `nlp` is now exactly `tiktoken`). A dependency is added in the same change
  as its first consumer. Intended future extras (e.g. a statsmodels-backed
  `timeseries`) live here until that code exists.

- **P5 — regenerate demand with a second real-data project: DONE.**
  `projects/titanic` classifies passenger survival on the classic 891-row
  manifest (seaborn-data mirror; real missingness at three severities, the
  target respelled as a feature, derived duplicate columns). Full lifecycle
  on `ds` + scikit-learn — validated leakage drops, stratified split, a
  five-step persisted scoring `Pipeline`, held-out split scored from the
  reloaded model — and the first real exercise of the untouched
  classification surface: `classification_metrics`, `confusion_frame`,
  `per_class_metrics`, `plot_confusion_matrix`, and the first composition of
  `cross_validate_kfold` with `metrics_fn=classification_metrics`. Held-out
  accuracy 0.799 / F1 0.731 vs the sex-only rule (0.777 / 0.692) and the
  majority class (0.615 / 0.0). Per the demand-first rule the project
  promotes nothing itself; its friction list is the new backlog below.

- **P6 — regenerate demand with a third real-data project (forecasting):
  DONE.** `projects/flights` forecasts the 144 monthly international-airline-
  passenger totals, 1949–1960 (the classic Box & Jenkins series, seaborn-data
  mirror) — chosen for a genuine time axis with strong yearly seasonality,
  and the first project to stress the time-series surface:
  `train_test_split_by_time` gains its second consumer, and
  `cross_validate_by_time` (rolling-origin folds) and `fit_baseline`'s
  `"naive_last"`/`"seasonal_naive"` strategies their first real ones. Full
  lifecycle on `ds` + scikit-learn: hand-assembled time axis, calendar
  features with the monthly-resolution noise dropped, a hand-rolled
  `month_index` trend, a one-step fit plan (the month one-hot vocabulary)
  persisted as the scoring `Pipeline`, the model persisted and the held-out
  window scored from reloaded state, and a linear trend + month-effects
  model evaluated against both naive references on the strictly future
  29-month window (MAE 34.3 vs seasonal-naive 64.8 and naive-last 81.4;
  r² 0.63 — honest about an additive model under multiplicative
  seasonality). Per the demand-first rule the project promotes nothing
  itself; its friction list is the new backlog below.

- **P7 — serve the `flights` backlog: DONE.** Items 10–13 in observed-pain
  order, each dogfooded by `projects/flights` in the same change (held-out
  metrics equivalent throughout — the trend column is the hand-rolled
  counter shifted by a constant the intercept absorbs):
  - `ds.viz.plot_series` (item 10) — one composable series plot: a solid
    observed line plus optional dashed, named prediction overlays, colours
    drawn from the Axes' cycle so repeated calls on one `ax` compose. One
    helper covers both of the project's hand-rolled figures — the raw
    series *and* the history + forecast-vs-actual view — rather than two
    single-purpose ones.
  - `add_datetime_features(features=...)` (item 11) — an explicit selection
    parameter, chosen over a resolution-aware default because inferring
    resolution from the frame is fitted state in disguise (a later scoring
    batch can be too small or too regular to infer from) and misfires
    silently; an explicit list is stateless and self-documenting. Default
    unchanged (the full calendar set).
  - `"elapsed_months"` (item 12) — the trend counter lives *inside*
    `add_datetime_features` as an opt-in selectable feature (same source
    column, same expansion mechanism, and item 11's parameter already
    provides opt-in) rather than as a second helper. Origin: a fixed
    calendar epoch (whole months since January of year 0), so scoring later
    rows is stateless; only differences matter for a trend term. Kept out
    of the default set (a modeling device, near-collinear with `_year`); a
    days/finer variant stays unbuilt until a project pulls it.
  - Item 13 — **struck, not built** (see the backlog).

- **P8 — regenerate demand with a fourth real-data project (multiclass):
  DONE.** `projects/diamonds` grades the cut of the 53,940 classic ggplot2
  diamonds (seaborn-data mirror) into the five ordered classes Fair < Good <
  Very Good < Premium < Ideal — chosen, after grepping which helpers still
  had no real consumer, for quirks that pull several untouched surfaces at
  once. First real consumers earned: the ordinal-encoding pair *with the
  explicit `categories=` domain ordering* (color J→D, clarity I1→IF — the
  sorted-unique default would rank both wrongly), `bin_column` (cut mix per
  carat quantile band), `plot_outliers` on non-synthetic data, and the
  multiclass metric surface (`confusion_frame` / `per_class_metrics` /
  `plot_confusion_matrix` at 5×5, `classification_metrics(average="macro")`).
  Full lifecycle on `ds` + scikit-learn: boundary validation with the three
  grade vocabularies, the physically impossible zero-dimension rows dropped
  by a hand-rolled mask (validation asserts, nothing filters), exact
  duplicates dropped as split-leaking re-entries (the deliberate opposite of
  titanic's keep), a three-step fit plan (clip the measurement-error
  dimension columns only — depth/table extremes *are* the Fair-cut signal —
  ordinal-encode, scale), stratified 5-fold CV with the plan re-fitted per
  fold, pipeline + model persisted and the held-out split scored from
  reloaded state. Held-out: accuracy 0.655 / macro F1 0.551 vs the
  proportions-only grading rule (0.546 / 0.357) and the majority class
  (0.400 / 0.114); CV macro F1 0.549 ± 0.009. The confusion structure is the
  honest headline: errors sit almost entirely between adjacent grades, with
  `Good` collapsing into `Very Good` (recall 0.12). Per the demand-first
  rule the project promotes nothing itself; its friction list is the new
  backlog below.

- **P9 — serve the `diamonds` backlog: DONE.** Items 14–17 in observed-pain
  order — two served, one struck, one resolved by documentation (each
  rationale inline in the backlog below):
  - `labels=` display mapping (item 14) — an optional `Mapping[int, str]` on
    `confusion_frame` / `per_class_metrics` / `plot_confusion_matrix` puts
    class *names* on the consumer-facing axes while the metric math stays on
    the int codes. The project deleted `_named()` and
    `_relabel_confusion_axes()`; every persisted artifact (CSVs *and* PNGs)
    verified byte-identical (sha256) to the pre-change run — held-out
    accuracy 0.655 / macro F1 0.551 untouched.
  - Item 15 (a row-filtering counterpart to the range assert) — **struck,
    not built** (see the backlog).
  - Template fixes (item 16) — the stub now scaffolds the shape all four
    real pipelines actually keep: an injectable `settings` parameter, a test
    that injects a temporary data directory, `ds run <slug>` as the run
    instruction, and clean rendering of an empty description.
  - Item 17 (the multiclass metrics wrapper) — **resolved by documenting
    the `functools.partial` idiom**, not a helper (see the backlog); the
    project's wrapper `def` is now that one-liner.

- **P10 — regenerate demand with a fifth real-data project (text): DONE.**
  `projects/sms_spam` flags the 5,574 labelled messages of the SMS Spam
  Collection (~13% spam; single-file mirror in the pycon-2016-tutorial repo)
  as ham or spam — chosen, by the P8 rule of grepping which surfaces still
  had no real consumer, because `ds.modeling.nlp.count_tokens` and the
  `nlp`/tiktoken extra were the library's only entirely-unconsumed module,
  and a text pipeline predictably stresses the text gaps by absence (no text
  helpers in the features stage, no vectorization step kind in
  `ds.pipeline`). First real consumers earned: `count_tokens` (per-message
  `token_count` — descriptive only, because its graceful degradation makes
  the values depend on the installed extras; see the backlog notes) and the
  headerless-TSV path through `load_raw`'s pandas-kwargs forwarding
  (`header=None, names=..., quoting=csv.QUOTE_NONE` — the file's quotes are
  message text, and default quoting silently swallows two rows). Second
  consumers: the `labels=` display mapping (binary `{0: "ham", 1: "spam"}`)
  and `bin_column` (label mix per message-length quantile band). Full
  lifecycle on `ds` + scikit-learn: boundary validation, exact duplicates
  dropped as split-leaking verbatim repeats (the diamonds call, for the
  vectorizer-memorizes-text reason), hand-rolled length features, a one-step
  fit plan (standardize `char_count` — the *only* step of this pipeline the
  closed vocabulary can express; the TF-IDF vectorizer lives inside the
  sklearn model object), stratified 5-fold CV with the plan re-fitted per
  fold, pipeline + model persisted and the held-out split scored from
  reloaded state. Held-out (spam = positive): accuracy 0.968 / F1 0.864 vs
  the keyword rule (0.916 / 0.659) and the majority class (0.873 / 0.000);
  CV F1 0.853 ± 0.009. Precision 0.938 vs recall 0.802 is the honest
  headline: the model rarely cries spam on ham but waves a fifth of spam
  through. Per the demand-first rule the project promotes nothing itself;
  its friction list is the new backlog below.

- **P11 — serve the `sms_spam` backlog: DONE.** Items 18–21 in observed-pain
  order — one served, one resolved by documentation, two struck (each
  rationale inline in the backlog below):
  - `count_tokens` path memoization (item 19, served first as the backlog's
    recorded strongest candidate) — the encoding probe is resolved once per
    process per `model` and the outcome cached, success *and* failure, so
    the tiktoken-installed/vocabulary-unreachable case pays one failed
    download attempt instead of one per message (~35 minutes over the
    project's 5,171 messages → one ~0.4 s probe) and a run never mixes
    counting paths. The other recorded candidate shape — exposing the probe
    — was deliberately not built: with the stall gone no consumer needs to
    hold a counter callable or see which path ran. Dogfood proof by
    deletion: `projects/sms_spam` dropped `_resolve_token_counter` and calls
    the library directly, no-extras artifacts byte-identical (sha256) and
    the vocabulary-blocked `--extra all` run (the live repro) down from a
    ~35-minute stall to seconds. Both paths pinned by deterministic
    fake-module tests that need no network and hold in both CI jobs.
  - Item 18 (no vectorization step kind) — **resolved by documenting the
    "model-side transforms live in the estimator" convention**, not
    building from one consumer (see the backlog): now `ds.pipeline`'s
    fourth module-docstring design point and a Guide paragraph.
  - Items 20 (boundary row-count check) and 21 (text feature helpers) —
    **struck, not built** (see the backlog); both second-project triggers
    recorded.

- **P12 — regenerate demand with a sixth real-data project (gap-filling on
  a gapped hourly axis): DONE.** `projects/air_quality` reconstructs the
  reference CO analyzer's reading (`co_gt`, mg/m³) from the rest of a
  road-level monitoring station — the UCI Air Quality dataset, 9,357 hourly
  rows from an Italian city, March 2004 to April 2005; the analyzer was down
  for 18% of those hours, and back-filling such an outage from the
  co-located instruments is the task. Chosen by the P8 rule after grepping
  which surfaces still had no real consumer, and deliberately weighted
  toward the open watch-list: real missingness at last (−200 sentinels:
  one column 90.2% missing, the target 18.0%, the NOx/NO2 feature channels
  17.5% overall and ~5% within the labeled rows), a raw file whose parse
  fails silently (semicolons, decimal commas, trailing junk columns and 114
  all-empty trailing rows), a two-piece dotted-time axis, and a
  rolling-origin CV whose per-fold fitted state genuinely varies. First
  real consumers earned: `assert_dtypes` (the parse pin that makes the
  decimal-comma misparse loud — with `sep=";"` alone every measurement
  column arrives as strings) and the impute surface at real severity
  (median fills through the persisted pipeline over genuinely cell-level
  gaps — nyc_taxis/titanic exercised the pair, but never against an
  independent-outage structure). Full lifecycle on `ds` + scikit-learn:
  checksum-verified fetch (the UCI archive is not reachable from every
  network, so two byte-identical GitHub mirrors are pinned by sha256),
  expected-row-count check, three-way missingness triage driven by
  `missing_value_report` (drop the 90% column; drop 366 device-offline
  hours — the device's gaps are all-or-nothing, so there is nothing to
  predict from; drop 1,647 unlabeled hours), hand-assembled hourly time
  axis, a three-step fit plan (median impute / 24-level hour one-hot /
  standardize) fitted on the training window, rolling-origin CV (MAE
  0.406 ± 0.111) with a companion table measuring the per-fold fitted state
  the single up-front transform cannot re-fit, pipeline + model persisted
  and the held-out window scored from reloaded state. Held-out (last ~20%
  of labeled hours): MAE 0.305 / RMSE 0.458 / r² 0.876 vs the station's own
  same-hour-yesterday reading (MAE 0.799, r² 0.223) and the training mean
  (MAE 1.068, r² −0.030). Per the demand-first rule the project promotes
  nothing itself; its friction list is the new backlog below.

- **P13 — serve the `air_quality` backlog: DONE.** Items 22–26 in
  observed-pain order — three served, two resolved by documentation (each
  rationale inline in the backlog below), all dogfooded by `projects/air_quality`
  (and, for the shared guard, `projects/flights`) in the same change:
  - `cross_validate_by_time(make_pipeline=...)` (item 22, the headline —
    item 9's parked question) — the rolling-origin twin of
    `cross_validate_kfold`'s factory: the same `FitStep`-plan re-fitted per
    fold on each fold's expanding window only. air_quality deleted its
    hand-rolled `_fold_fit_state` boundary reproduction (the dogfood proof).
    Unlike item 9's titanic finding, the effect is **real on this data**: the
    per-fold impute medians swing ~28%, so the leak-free protocol measurably
    moves the CV numbers (mean 0.406 → 0.409) while held-out metrics and every
    persisted artifact stay byte-identical.
  - `assert_unique` (item 24, item 13's second-project trigger) and
    `assert_row_count` (item 25, item 20's second-project trigger) — two
    fluent guards in `ds.validation`. Both triggers had fired twice with an
    agreed shape; each adds the stage-consistent `DataValidationError` (and,
    for `assert_unique`, a correctness check raw `to_datetime` doesn't do).
    Item 24 resolved to the validation-guard shape over the recorded
    parse-wrapper — the reasoning is inline in the backlog.
  - Items 23 (`fit_baseline`'s positional/gapped-axis mismatch) and 26
    (sentinel-coded missingness invisibility) — **resolved by documentation**,
    each one consumer: item 23 in `fit_baseline`'s docstring (the completeness
    assumption the flights note quietly made), item 26 as a Guide Acquire-
    section gotcha (the `na_values=` decimal-comma trap and the load-bearing
    ordering).

- **P14 — regenerate demand with a seventh real-data project (heavily
  categorical): DONE.** `projects/adult_income` predicts whether a 1994 US
  census respondent earns over $50K — the 32,560-row training split of the UCI
  Adult / Census Income dataset (a GitHub mirror of the original `adult.data`,
  downloaded once into git-ignored `data/raw/` and verified by pinned sha256
  because the UCI archive is not reachable from every network). Chosen by the
  P8 rule after grepping which surfaces still had no real consumer, and
  deliberately weighted toward the thinnest *cluster* — categorical /
  high-cardinality feature handling: `collapse_categories`/`fit_topk_categories`
  had a single consumer (nyc_taxis), one-hot `drop_first`/`dummy_na` and the
  `flag_outliers` flag path had none, and "heavily-categorical / wide" was an
  unmet data shape. First real consumers earned: one-hot `drop_first=True` (the
  wide indicator matrix's dummy-trap guard, which a full-rank linear model
  needs) and the `flag_outliers` *flag*-not-clip path (the ~92%-zero
  `capital_gain`/`capital_loss` are reported but kept — clipping would erase the
  large-gain signal). Second consumers: `fit_topk_categories`/
  `collapse_categories` (the 41-country `native_country` and 14-trade
  `occupation` tails → `"other"`) and the `labels=` binary display mapping.
  Full lifecycle on `ds` + scikit-learn: checksum-verified fetch, boundary
  validation (`assert_row_count` + a `check_schema` numeric-dtype pin), the
  `"?"` sentinels decoded to NaN and the string target encoded 0/1, a stratified
  split, a four-step fit plan (collapse tails / mode-impute the sentinels / wide
  one-hot with `drop_first` / scale) fitted on the training split, stratified
  5-fold CV with the plan re-fitted per fold (the fitted state — occupation
  modes, kept top-k countries, scale centres — genuinely varies fold to fold),
  pipeline + model persisted and the held-out split scored from reloaded state.
  Held-out (>50K = positive): accuracy 0.861 / F1 0.679 / precision 0.763 /
  recall 0.611 vs an interpretable marital-status rule (0.713 / 0.590) and the
  majority class (0.759 / 0.000); CV F1 0.653 ± 0.007. The honest headline is
  what the library did *not* fight: the categorical cluster it was picked to
  stress by absence existed and composed first-try, so the project validates
  that surface more than it extends it — the two genuinely new gaps (items 27
  and 28) both have triggers that were already recorded and now fire. Per the
  demand-first rule the project promotes nothing itself; its friction list is
  the new backlog below.

- **P15 — serve the `adult_income` backlog: DONE.** Items 27–29 in observed-pain
  order — one served, two resolved without a build (each rationale inline in the
  backlog below), the serve dogfooded by `projects/air_quality` **and**
  `projects/adult_income` in the same change:
  - `ds.io.fetch_dataset` (item 27, the headline — the fetch-helper trigger fired
    on its second verbatim consumer) — `fetch_dataset(name, urls, *, sha256,
    settings=None)` downloads a raw file into `settings.raw_dir` trying each mirror
    in order, verifies the payload's sha256 before writing, and re-verifies (not
    trusts) a cached copy so a partial earlier download can't poison later runs —
    exactly the ~25-line dance both projects had hand-rolled, now one library call.
    The two recorded open questions were resolved with evidence, not a guess: the
    checksum is **required** (keyword-only, no default) because the trigger fired on
    the *checksum-verified multi-mirror* shape specifically — the seaborn-mirror
    projects (`titanic`/`nyc_taxis`/`diamonds`) fetch with a plain, un-pinned
    "download if absent" that is a few inline lines below the aliasing bar, so they
    were deliberately **not** folded in (making the checksum optional purely to
    absorb them would have built surface with no demand pulling it, and pinning a
    live upstream repo is a maintenance commitment nothing asked for); and the cache
    re-verify lives **inside** the helper, because re-hashing a cached copy and
    unlinking on mismatch is part of what "verified fetch" means and leaving it to
    the caller would force every caller to re-implement the guarantee. Stdlib only
    (`urllib`/`hashlib`), so no dependency was added; a `ds.io` addition, so
    `tests/test_public_api.py` (the top-level surface) is untouched. It also hardens
    the dance it absorbed: the destination is resolved through `ds.io`'s
    `_resolve_within` path guard, so a traversing `name` is refused (the projects
    joined the raw dir directly). Dogfood proof by deletion: both projects deleted
    their hand-rolled `fetch_raw` body — each now keeps only its data (mirror tuple,
    filename, pinned digest) and a one-line `fetch_raw` that binds them to
    `fetch_dataset` — and both end-to-end tests pass unchanged (held-out metrics and
    every persisted artifact equivalent, because the verified bytes are identical).
    The seaborn projects' un-checksummed fetch is recorded as the natural future
    consumer that would justify an *optional*-checksum widening only when a project
    actually pulls it.
  - Item 28 (a string `"?"` sentinel decoded by hand — air_quality item 26's
    second-sentinel trigger, string flavor) — **resolved as a documented one-liner,
    not built**, now that two differently-typed sentinels are on record. The shared
    part across −200 (numeric) and `"?"` (string) is exactly the one-line
    `df[columns].replace(sentinel, np.nan)`, which carries no logic to share, while
    the load-bearing *lesson* (the sentinel is invisible to
    `missing_value_report`/`assert_in_range`/`summarize` until decoded, and the
    decode must run before any of them sees the frame) is already the Guide's
    Acquire-section gotcha. A `mask_sentinels(df, columns, sentinel)` would alias
    that one-liner behind a new name — the item-13/15/20/26 precedent — so it stays
    documented; `adult_income` keeps its one-line `decode_sentinels`. A **third**
    differently-typed sentinel would force the thin helper.
  - Item 29 (`ds.eda` has no categorical↔target association view) — parked in
    P15, **served in the follow-on goal-alignment pass**: built as
    `target_rate_by_category` (the positive-rate-by-level group table) plus
    `plot_target_rate`, adopted by `adult_income`. Full rationale — why the group
    table over the Cramér's-V ranker, and why descriptive-not-fitted — is inline
    in the friction backlog entry below.

- **P16 — deepen forecasting with a second forecasting project (autoregression):
  DONE.** `projects/sunspots` forecasts the monthly Zurich/SILSO sunspot number,
  1749–1983 (2,820 months), chosen after `flights` precisely for a series the
  calendar-feature + naive approach handles *badly*: the ~11-year solar cycle is
  aligned to nothing on the calendar (the by-month means are flat within ±3% of
  the overall mean, and `seasonal_naive` of period 12 is a poor guide), so the
  signal is the series' own recent history — the first project to need
  **autoregressive features** and a **recursive multi-step forecast**. Because
  forecasting is a committed capability whose first project (`flights`) already
  delegated its model to raw scikit-learn, this second forecasting project's
  friction was served in the same demand loop (items 30–31):
  - `ds.features.add_lagged_features(df, column, lags, *, dropna=True)` (item 30)
    — the autoregressive counterpart to `add_datetime_features`: adds
    `<column>_lag_<k>` columns taken by row position (sort by the time axis
    first), dropping the warm-up rows by default. Stateless (a row's lags are the
    rows beside it), so it is applied before the split like the datetime
    features. It is the primitive an entire model class — autoregression —
    cannot exist without, which is why it is built on its first consumer's
    strength (the `plot_series` precedent), not parked as a one-liner: a single
    `.shift(k)` is a line, but the named multi-lag set + warm-up handling +
    ordering is the reusable shape.
  - `ds.modeling.timeseries.forecast_recursive(model, history, *, lags, steps)`
    (item 31) — forecast past the edge of the data by feeding each prediction
    back as later steps' lags, the multi-step forecast a single `model.predict`
    cannot produce. Clearly above the aliasing bar: it is a fiddly, error-prone
    recursion (buffer management, feature-order alignment, the optional
    `feature_names_in_` frame to avoid sklearn's name warning), not a wrapped
    one-liner. Pure-AR only by contract (an exogenous feature would need a future
    value it cannot supply) — recorded in the docstring.

  Full lifecycle on `ds` + scikit-learn: checksum-verified fetch (`fetch_dataset`'s
  third consumer — the mirror is a *live* upstream repo, so the sha256 pin is
  correct here, unlike the seaborn projects), boundary validation, hand-assembled
  time axis (`assert_unique`, its third consumer), an Explore step that *shows*
  the modeling choice (the flat by-month profile), lag features, a chronological
  hold-out of the last decade, rolling-origin one-step CV
  (`cross_validate_by_time` again), the model persisted and both forecasts scored
  from the reloaded copy. The honest headline is the horizon split: one-step-ahead
  MAE 15.3 / r² 0.88 (strong), recursive multi-step over 120 months MAE 52.5 /
  r² −0.35 (error compounds and the forecast decays toward the mean — the genuine
  difficulty of long-range solar forecasting) — yet even the decayed recursive
  forecast beats both calendar-naive references (seasonal-naive MAE 58.1,
  naive-last MAE 63.1). Scope finding, recorded not hidden: a pure-AR forecaster
  needs **no** `ds.pipeline` scoring `Pipeline` (lags stateless, series complete,
  swings are signal, OLS scale-free), so only the model is persisted — `flights`
  had a one-step plan, this has none.

- **P17 — deepen text with a second text project (multiclass topic
  classification): DONE.** `projects/bbc_news` classifies the 2,225 BBC News
  articles into five topics (business/entertainment/politics/sport/tech), chosen
  after `sms_spam` to stress the text surface a second time and decide the
  triggers that first project parked. Served in the same demand loop (item 21;
  items 18 and count_tokens' verdict re-decided by a second consumer):
  - `ds.features.text_features(df, column, *, features=None)` (item 21) — a
    stateless one-call expansion of a string column into `<column>_char_count`,
    `_word_count` and `_avg_word_length`, the text counterpart to
    `add_datetime_features`. `sms_spam` hand-rolled these because the stage had
    none; a second text project hand-rolling the same family is exactly the
    trigger item 21 recorded, and the shape it decides is the *frame helper*
    (emit the set in one call) over single-column counters. Deliberately
    encoding-independent (pure string ops), which is what keeps them model-safe —
    unlike `count_tokens`.
  - **`count_tokens` earns a modeling consumer.** `sms_spam` kept `token_count`
    descriptive-only, wary of its extras-dependent value (the P11 verdict). This
    project re-decides that verdict with evidence: as *one coarse length feature
    beside thousands of TF-IDF terms* the classifier is robust to which counting
    path runs, so it feeds the model here — the accurate-count path's first
    modeling consumer, with the tests asserting path-independent macro-F1 bounds.
    The "half-earned" verdict is upgraded, not contradicted: the caveat still
    stands where a model is *sensitive* to the exact count; it lifts where the
    model is robust.
  - **Item 18 (vectorization step kind) re-checked, convention reaffirmed.** The
    TF-IDF vectorizer is again the fitted heart and again lives inside the
    sklearn `ColumnTransformer` while the `ds` scoring `Pipeline` carries the
    frame-shaped scale step. This second text consumer *confirms* the
    model-side-transform convention suffices (the exact question the item parked
    for "a second text project"), so no first-class vectorize step is built — it
    would smuggle a pickle into the strict-JSON `save_params` story, the reason
    P11 rejected it. Struck-by-reaffirmation, not built.

  Full lifecycle on `ds` + scikit-learn: checksum-verified fetch
  (`fetch_dataset`'s fourth consumer), boundary validation, verbatim-duplicate
  drop (99 articles — the diamonds/sms_spam leak guard), the text features,
  ordinal-coded target, stratified split, a one-step scale plan
  (`fit_pipeline`), stratified 5-fold macro CV with the plan re-fitted per fold,
  pipeline + model persisted and the held-out split scored from the reloaded
  model. Held-out macro-F1 0.964 / accuracy 0.965 vs a length-only model (0.329 —
  the honest headline that the topic signal is in the *words*, length a modest
  supplement) and the majority class (0.077); CV macro-F1 0.957 ± 0.009.

- **P18 — stress the first data *shape* no project had: a panel (multi-entity):
  DONE.** `projects/store_sales` forecasts daily units sold across a store × item
  panel — 50 entities (10 stores × the first 5 items) of the classic "Store Item
  Demand" dataset, 2013–2017. Chosen by the grep-driven demand rule: every prior
  project was single-entity, so the time-series surface had only ever seen one
  series at a time. A panel breaks that assumption at exactly one place, and the
  loop's single library change served it in the same batch (item 32):
  - `ds.features.add_lagged_features(..., *, group=None)` (item 32) — lags taken
    *within* each entity via a grouped `shift`, so history never bleeds across an
    entity boundary. The ungrouped path lagged by row position over the whole
    frame; stacking store 2 beneath store 1 made store 2's first rows read store
    1's tail as their history — a silent correctness bug, not a convenience gap,
    which is why it was fixed rather than parked. Clearly above the aliasing bar
    (the no-bleed guarantee, warm-up handling *per group*, multi-key support).
    `store_sales` is its first consumer, with a test asserting the boundary
    never bleeds.

  Full lifecycle on `ds` + scikit-learn: checksum-verified fetch
  (`fetch_dataset`'s fifth consumer), boundary validation on the full 913k-row
  file, panel selection + within-entity ordering, calendar features
  (`add_datetime_features` day-of-week/month/elapsed-months) + the grouped lags,
  a date-cutoff split (train 2013–2016, forecast 2017), a one-hot plan over entity
  + calendar effects (`fit_pipeline`), a pooled `LinearRegression`, pipeline +
  model persisted and the held-out window scored from the reloaded model. One-step
  held-out 2017: pooled model MAE 5.27 / r² 0.894, beating the weekly-seasonal
  naive (`sales_lag_7`, MAE 6.70) and naive-last (`sales_lag_1`, MAE 7.99)
  references — the naive ordering itself the read that the day-of-week cycle
  dominates. The rest of the panel's single-series friction (items 33–35) was
  handled inline and recorded, not built — see the backlog below.

- **P19 — stress the first data *shape* no project had: an imbalanced /
  rare-event target: DONE.** `projects/bank_marketing` predicts term-deposit
  subscription on the 41,188-row UCI Bank Marketing dataset where only **11.3%**
  say yes. Chosen by the grep-driven demand rule: every prior classification
  project (titanic, adult_income, diamonds, sms_spam, bbc_news) worked a
  roughly-balanced target scored on *hard labels*, so the evaluation surface had
  never met a rare positive class where **accuracy is a trap** — a majority-class
  predictor scores 0.887 accuracy while finding not one subscriber. The loop's
  single library change served that gap (item 36):
  - `ds.evaluation.probability_metrics(y_true, y_score)` (item 36) — ROC-AUC,
    average precision (PR-AUC) and Brier score from a classifier's predicted
    *probabilities*, the threshold-free read on ranking quality that
    `classification_metrics` (hard labels only) structurally could not give.
    Clearly above the aliasing bar: a whole capability class the library lacked,
    plus a correctness guarantee — it raises on single-class `y_true` where
    scikit-learn silently warns and returns `nan`. Same two-argument shape as the
    other `*_metrics` helpers, so it drops straight into `compare_models`
    (scoring the model against a prevalence floor that ranks at chance). Its
    first consumer is `bank_marketing`, with tests pinning the no-skill floors
    (ROC-AUC 0.5, AP = positive rate) and the single-class guard.

  Full lifecycle on `ds` + scikit-learn: checksum-verified fetch
  (`fetch_dataset`'s sixth consumer), boundary validation with the numeric
  dtypes pinned, `duration` dropped as **leakage** (known only after the call it
  predicts) and the `pdays == 999` "never contacted" sentinel folded into a
  binary flag, the categorical read via `target_rate_by_category`, stratified
  split, a two-step one-hot + scale plan (`fit_pipeline`) re-fitted per fold,
  stratified 5-fold CV, pipeline + a `class_weight="balanced"` model persisted
  and the held-out split scored from the reloaded model — evaluated
  *probabilistically* (the new metric) **and** on hard labels. The honest
  headline is the whole point: held-out accuracy 0.835 sits *below* the majority
  floor's 0.887, yet ROC-AUC 0.80 (vs the 0.50 floor) and average precision 0.46
  (vs the 0.11 prevalence floor) show the model genuinely ranks subscribers, and
  recall 0.645 recovers real positives the majority floor (recall 0.0) finds
  none of — the exact case probabilistic metrics exist to reveal. The rest of the
  rare-event friction (items 37–38) was handled inline and recorded, not built —
  see the backlog below.

- **P20 — a *second* imbalanced project that tunes the operating point, firing
  the two items P19 parked: DONE.** `projects/mammography` flags the rare
  calcification (2.3% positive) in 11,183 screened regions of the Woods
  mammography dataset. Chosen by the grep-driven rule to give
  `probability_metrics` its *second* consumer; because a screening programme has
  an **operating point** reweighting can't express (`bank_marketing` used
  `class_weight="balanced"` and kept 0.5), this project fits a plain logistic
  regression and **tunes the threshold** — the exact shape items 37–38 said would
  justify building the tools they parked. Two library additions, both now above
  the aliasing bar on first-consumer strength:
  - `ds.evaluation.choose_threshold(y_true, y_score, *, criterion, target)` (item
    37) — sweeps the precision–recall curve for the F1-optimal threshold, or the
    cheapest threshold meeting a `target_precision` / `target_recall` floor.
    Above the bar as fiddly reusable logic *with* a correctness guarantee: it
    drops the trailing `(precision=1, recall=0)` point `precision_recall_curve`
    returns with no matching threshold (the classic off-by-one), and raises when
    a target is unreachable rather than silently returning that phantom point.
  - `ds.viz.plot_pr_curve` / `plot_roc_curve` (item 38) — the operating-point
    *curve* with its no-skill baseline (prevalence line / chance diagonal), the
    view that pairs with `probability_metrics` the way `plot_confusion_matrix`
    pairs with `confusion_frame`. The pipeline scatters the three tuned operating
    points onto the PR curve.

  Full lifecycle on `ds` + scikit-learn: checksum-verified fetch
  (`fetch_dataset`'s seventh consumer, first no-header CSV via `names=`), boundary
  validation with the six attributes' dtypes pinned, the quoted `'-1'`/`'1'` label
  stripped and encoded, a one-step scale plan (`fit_pipeline`) re-fitted per fold,
  stratified 5-fold CV, pipeline + a plain (un-reweighted) model persisted and the
  held-out split scored from the reloaded model — thresholds chosen on the *train*
  scores, never the test set. Honest headline: the model **ranks** calcifications
  well (ROC-AUC 0.97, average precision 0.65 vs the 0.02 prevalence floor), but at
  the naive 0.5 cut it misses ~64% of them (recall 0.37); tuning to an 80% recall
  budget lifts recall to 0.92 at a read-off precision cost (0.28) — the screening
  trade made explicit, the operating point reweighting cannot target. The
  remaining friction (items 39–41) was handled inline and recorded, not built —
  see the backlog below.

## Friction backlog (from `projects/nyc_taxis`)

Demand-driven candidates, in observed-pain order:

1. ~~**Model persistence**~~ — **resolved by P2**:
   `ds.modeling.persistence.save_model`/`load_model`; the project now scores
   the held-out window from the reloaded model.
2. ~~**`add_datetime_features` has no `hour`**~~ — **resolved in P3**: the
   helper now emits `<column>_hour` (constantly zero on date-only data, where
   `drop_constant_columns` removes it); the project consumes it.
3. ~~**No baseline estimators**~~ — **resolved in P3**:
   `ds.modeling.baseline.fit_baseline`; the project's hand-rolled train-mean
   baseline is gone.
4. ~~**No high-cardinality strategy**~~ — **resolved**:
   `ds.features.fit_topk_categories`/`apply_collapse_categories` keep a
   column's top-k levels and collapse the rest (and anything unseen at
   scoring time) to `"other"`, so the existing one-hot/ordinal encoders take
   it from there. Top-k+"other" was chosen over frequency encoding because it
   preserves level identity (what a linear fare model needs) and composes
   with the existing encoders instead of adding a parallel numeric path. The
   project now consumes the zone columns it originally dropped, and they earn
   their place: vs a boroughs-only variant on the same held-out window, MAE
   2.62 → 2.26 (−14%), r² 0.729 → 0.765 (k=15, asserted in the project's
   end-to-end test).
5. ~~**Pipeline fit-side observation**~~ — **resolved with item 9** (the
   "second project repeats this dance" trigger fired: `titanic` repeated the
   fit → apply → fit chain verbatim, five fit/apply pairs):
   `ds.pipeline.fit_pipeline` executes an ordered plan of `FitStep` entries
   as exactly that chain and returns the assembled `Pipeline`. Both projects
   replaced their hand-strung dance with a plan; persisted scoring pipelines
   and held-out metrics came out byte-identical. The amended
   pure-composition rationale is recorded under settled decisions below.

## Friction backlog (from `projects/titanic`)

The second run of the demand loop. Numbering continues from the `nyc_taxis`
list so item references stay unambiguous; in observed-pain order:

6. ~~**No classification-shaped baseline.**~~ — **resolved**:
   `fit_baseline` now takes `strategy="majority"` (predict the modal training
   label, ties to the smallest label), the classification twin of `"mean"`.
   Scoped to the observed demand: labels must be numeric (the int-coded 0/1
   target that raised the item) — string labels stay out until a project
   demands them, because the frozen `Baseline` contract is
   `tuple[float, ...]`. The project's hand-rolled `y_train.mode()` reference
   is gone, with identical held-out metrics (majority accuracy 0.615 / F1 0.0).
7. ~~**No split helper for order-free data.**~~ — **resolved**:
   `ds.modeling.tabular.train_test_split_random` is the order-free twin of
   `train_test_split_by_time` — shuffled, with an optional `stratify` column
   whose class balance both halves preserve, seeded through numpy's global
   generator like the rest of the stage (so `seed_everything` reproduces
   it). The project's raw `sklearn.model_selection.train_test_split` call is
   gone; the wrapper makes the identical scikit-learn call, so the split and
   the held-out metrics are byte-identical (accuracy 0.799 / F1 0.731).
8. ~~**`cross_validate_kfold` cannot stratify.**~~ — **resolved, with one
   honest correction**: a `stratify` flag (`StratifiedKFold` under the hood,
   same global-generator seeding) keeps every fold at the frame's class
   balance, composing with `metrics_fn=classification_metrics`; the project
   passes it. It fixes exactly what it controls — per-fold positive counts
   went from a ~15-row spread to the ±1 rounding minimum — but the recall
   drift this item blamed on that imbalance did **not** shrink (measured
   across 30 seeds on the project's frame: mean per-fold recall spread ~0.13
   with and without stratification). The drift is sampling variance in
   *which* positives land in a fold, not in how many, so the project's CV
   assertions were deliberately not tightened.
9. ~~**Cross-validation cannot re-fit the transform chain per fold.**~~ —
   **resolved, with one honest finding**: `cross_validate_kfold` takes a
   `make_pipeline` factory (the `make_model` twin — typically
   `lambda frame: fit_pipeline(frame, plan)` with the training run's own
   plan); each fold fits a fresh pipeline on its training rows only and
   applies it to both halves. `titanic` now cross-validates the *raw*
   training split with the same five-step plan it fits the scoring pipeline
   from. The finding: the leak was real in protocol but its measured effect
   here rounds to zero — the per-fold statistics genuinely change (fold age
   medians 28.0–29.0 vs 28.5 whole-train, fare fences 63.3–66.6 vs 65.7) yet
   not one of the 712 fold predictions flips, because logistic regression
   absorbs small affine shifts in imputation/scaling, so the per-fold CV
   metrics came out identical. As with item 8, the honest result is recorded
   rather than a manufactured delta; the protocol is now sound either way.
   `cross_validate_by_time` deliberately does *not* grow the parameter until
   a project pulls it — it currently has no consumer (demand first).

Where the library did *not* fight: the classification metric/plot surface
itself (`classification_metrics`, `confusion_frame`, `per_class_metrics`,
`plot_confusion_matrix`, `compare_models` with a swapped `metrics_fn`)
composed first-try with no workarounds.

## Friction backlog (from `projects/flights`)

The third run of the demand loop — the first to stress the time-series
surface. Numbering continues from the `titanic` list; in observed-pain
order:

10. ~~**No time-series plot in `ds.viz`.**~~ — **resolved in P7**:
    `ds.viz.plot_series` — one solid observed line, optional dashed named
    prediction overlays, colours from the Axes' cycle so calls compose on
    one `ax`. One helper replaced both of the project's hand-rolled figures:
    `series.png` is a single call, `forecast.png` two composed calls
    (training tail, then the held-out window with the model and
    seasonal-naive overlays). The API-shape question resolved to *one*
    composable plot, not a separate forecast helper.
11. ~~**`add_datetime_features` is all-or-nothing.**~~ — **resolved in P7**:
    a `features=` selection parameter scopes the emission (default: the
    full calendar set, unchanged). The deliberately-open shape question was
    decided for explicit selection over a resolution-aware default:
    inferring resolution from the frame is fitted state in disguise — a
    later scoring batch can be too small or too regular to infer the same
    answer from — and it misfires silently, while an explicit list is
    stateless, self-documenting, and exactly matches the observed pain
    (the project knew precisely which columns were noise). The project's
    hand-drop of `date_dayofweek`/`date_is_weekend` is gone, and so is its
    reliance on `drop_constant_columns` catching `_day`/`_hour` — the
    scoped call never emits them.
12. ~~**No elapsed-time/trend feature.**~~ — **resolved in P7**:
    `"elapsed_months"`, an opt-in member of item 11's selection (same
    source column and expansion mechanism, so no second helper) emitting
    whole months since a *fixed* calendar epoch (January of year 0, i.e.
    `year * 12 + month - 1`). The fixed epoch is the stateless-origin
    answer: nothing is learned from the frame, so a later scoring run maps
    the same timestamp to the same value, and for a trend term only
    differences matter. Excluded from the default set (a modeling device,
    near-collinear with `_year`). The project's hand-rolled `month_index`
    is gone with equivalent held-out metrics (the counter differs by a
    constant the intercept absorbs). A days/finer-grained variant stays
    unbuilt until a project demands one.
13. ~~**The time axis was assembled by hand.**~~ — **struck in P7, not
    built**, as the backlog itself anticipated: the pain is one
    `pd.to_datetime(..., format=)` call plus a project-specific uniqueness
    check, observed once. A helper would wrap a single well-documented
    pandas call behind a new name to learn without removing meaningful
    code — below the helper bar. Won't build until a second project
    repeats the pain (and shows which shape recurs: two-column splits,
    format strings, or the uniqueness check).

Notes from the same run, for the record:

- **Item 9's parked question stays parked.** This project's rolling-origin
  CV does consume an already-transformed frame — exactly the situation
  `cross_validate_kfold(make_pipeline=...)` exists for — but the demand
  trigger did **not** fire: the only fitted state is the month one-hot
  vocabulary, and re-fitting it per fold was measured to produce the
  identical 12 calendar months on every fold (every training window spans at
  least 20 months). `cross_validate_by_time` still has no consumer that
  needs per-fold re-fitting.
- **`fit_pipeline` scope finding, not a gap:** the fit plan has exactly one
  step. A complete series whose extremes are signal, scored by a scale-free
  model, needs no imputation, clipping or scaling — on clean time-series
  data most of the fit-based transform surface has nothing to do, and the
  executor's value reduces to building the persistable scoring `Pipeline`.

Where the library did *not* fight: `train_test_split_by_time`,
`fit_baseline`'s `"naive_last"`/`"seasonal_naive"` (whose positional
alignment is correct by construction when the scored window starts right
after training), `cross_validate_by_time` (first real consumer — composed
first-try, time column and target excluded from the features as documented),
`regression_metrics`/`compare_models`/`plot_residuals`/
`plot_model_comparison`, and `drop_constant_columns` catching the constant
calendar columns.

## Friction backlog (from `projects/diamonds`)

The fourth run of the demand loop — the first multiclass one. Numbering
continues from the `flights` list; in observed-pain order:

14. ~~**The classification metric/plot surface is label-blind.**~~ —
    **resolved in P9**: an optional `labels: Mapping[int, str]` display
    mapping on `confusion_frame` / `per_class_metrics` /
    `plot_confusion_matrix`, exactly the candidate shape recorded — display
    names only, the metric math stays on the int codes, and `fit_baseline`'s
    deliberately numeric contract (item 6) is untouched. A mapping rather
    than a positional sequence because the frames' axes are the sorted union
    of *observed* codes: a class absent from a split would silently misalign
    positional names, while a mapping stays correct (and codes absent from
    the mapping keep their integer form, plain-rename semantics). With
    `labels=` the plot rotates its x-tick names 45° — five multi-word grades
    overlap horizontally — and without it every code path is byte-for-byte
    the old one, so titanic's readable 0/1 output is untouched. The project
    deleted `_named()` and `_relabel_confusion_axes()` for `labels=` at the
    four consumer-facing call sites; all persisted artifacts (CSVs *and*
    PNGs) verified byte-identical (sha256) to the pre-change run.
15. ~~**Validation asserts, nothing filters.**~~ — **struck in P9, not
    built**, as the item's own caveat anticipated. The observed pain: the 20
    physically impossible zero-dimension rows must be *removed*, the project
    hand-rolls a boolean mask (`drop_impossible_dimensions`) and re-states
    the same bound in `assert_in_range(min_value=0, inclusive="right")`, so
    the rule lives in two places that can drift. But the mask is three lines
    of well-documented pandas observed once — a `drop_out_of_range` would
    wrap it behind a new name to learn without removing meaningful code,
    below the helper bar (the item-13 precedent), and the drift risk, while
    real, has bitten nobody yet. Won't build until a second project
    hand-rolls the same mask, which will also show which shape recurs: a
    standalone row-dropper or a `filter=` mode on the range check. The
    project keeps its mask.
16. ~~**The project template scaffolds a shape no real project keeps.**~~ —
    **resolved in P9**, all three divergences the first real `ds new`
    dogfooding surfaced: the stub `run()` now carries the
    `settings: Settings | None = None` parameter every real pipeline needs
    (resolved via `settings or get_settings()`, threaded through `main()`),
    and the scaffolded test injects `Settings(data_dir=tmp_path / "data")` —
    a scaffold's first test run no longer writes into the shared data tree;
    the scaffolded README and module docstring say `ds run <slug>`, matching
    what `ds new` itself prints; and an empty `description` (the default)
    renders cleanly — conditional Jinja, so no dangling "`<Name> — `"
    docstring and no blank README paragraph. Pinned by the CLI tests and
    verified by scaffolding throwaway projects with and without a
    description: green out of the box, matching the real pipelines' shape.
17. ~~**`classification_metrics`' binary default forces a wrapper at every
    multiclass call site.**~~ — **resolved in P9 by documenting the idiom,
    not building a helper**: bind the average with
    `functools.partial(classification_metrics, average="macro")` and hand
    that to the `metrics_fn` hooks. The idiom is recorded where a multiclass
    consumer will actually meet the problem — the `average` parameter's own
    docstring (where the `"binary"` raise sends you) and the Guide's
    cross-validation section — and the project's wrapper `def` is now that
    one-liner (its macro-vs-weighted rationale kept as a comment), numbers
    identical. A library helper was not built: it would alias a stdlib
    one-liner behind a new name, and changing the `"binary"` default would
    silently change what precision/recall mean for every two-class consumer
    (titanic passes no `average` at all). Revisit only if a project needs an
    averaging shape `partial` cannot express.

Notes from the same run, for the record:

- **`plot_outliers` ranks by count, and count is not severity.** The plot
  put price/depth/carat first (thousands of honestly skewed values) while
  the physically impossible measurements — the actual data errors, a 58.9 mm
  `y` on a 2-carat stone — were near-invisible at 2–3 flagged values each;
  the clipping decision had to be made against `summarize()`'s max column
  instead. A magnitude-aware view would have shown it directly, but the
  count view is honest about what it claims and one consultation of an
  existing report is not much pain — a note, not an item, until it recurs.
- **Imputation stays unexercised at severity beyond titanic.** Diamonds has
  zero missing values, so this run adds nothing on that surface — recorded
  so the gap isn't mistaken for coverage.

Where the library did *not* fight: `fit_ordinal_categories(categories=...)` /
`apply_ordinal_encode` composed first-try — including the JSON round-trip of
the explicit worst-to-best orders through the persisted `Pipeline` (asserted
in the project's end-to-end test) and unseen-category behaviour never
triggering thanks to the vocabulary validation upstream; the stratified
five-class `cross_validate_kfold(make_pipeline=..., stratify=True)`
composition; `fit_baseline("majority")` on the int-coded target (item 6's
numeric-label scoping held exactly); `bin_column`'s quantile bins as an
EDA device; `drop_duplicate_rows`; `train_test_split_random(stratify=)` at
five classes; and the whole persistence story (`fit_pipeline` →
`save_params`/`load_params`, `save_model`/`load_model`).

## Friction backlog (from `projects/sms_spam`)

The fifth run of the demand loop — the first text one. Numbering continues
from the `diamonds` list; in observed-pain order:

18. ~~**The pipeline's step vocabulary cannot hold the fitted heart of a
    text pipeline.**~~ — **resolved in P11 by documenting the convention,
    not building** — the item's own warning ("don't design this from one
    consumer") ruled a vectorize step kind out. The pain stands as recorded:
    the TF-IDF vectorizer is the project's one genuinely fitted text
    transform and the closed `StepKind` vocabulary cannot hold it, so the
    fitted state splits across two artifacts (strict-JSON scoring pipeline,
    model joblib). But both build shapes stay dishonest with one consumer's
    evidence: every existing step maps named DataFrame columns to named
    DataFrame columns, while a vectorizer *manufactures* its column space —
    thousands of learned sparse columns — so forcing it through the dense
    frame contract would hide the real cost, and wrapping the sklearn
    object whole would smuggle a pickle into the strict-JSON `save_params`
    story. What sms_spam does is therefore now the *documented* convention
    rather than an accident: model-side transforms live in the estimator
    and persist via `save_model` — recorded as `ds.pipeline`'s fourth
    module-docstring design point and in the Guide's pipeline section, the
    item-17 precedent of resolving by documenting where the next consumer
    will actually look. Revisit only when a second text project shows
    whether the convention suffices or a first-class vectorize step kind
    earns a build.
    **Follow-on (P17): the convention suffices — reaffirmed, not built.**
    `projects/bbc_news`, the second text project, is again a TF-IDF-hearted
    pipeline and again puts the vectorizer inside the sklearn
    `ColumnTransformer` while the `ds` scoring `Pipeline` carries only the
    frame-shaped scale step — cleanly, with no workaround. That is exactly the
    "does the convention hold with a second consumer" test this item parked, and
    the answer is yes, so no vectorize step kind is built. It would still smuggle
    a pickle into the strict-JSON `save_params` story; the trigger to build one
    is now the *opposite* — a text project the model-side convention genuinely
    cannot serve.
19. ~~**`count_tokens`' graceful degradation is per-call and invisible to
    the caller.**~~ — **resolved in P11** (served first, as the recorded
    strongest candidate): `count_tokens` now resolves which counting path
    is live **once per process** per `model` — a memoized probe that caches
    success *and* failure — so with tiktoken installed but its vocabulary
    endpoint unreachable only the first call pays the failed download
    attempt (~0.4 s once, not × 5,171 messages ≈ 35 minutes), and a process
    never mixes counting paths mid-run even if connectivity changes. Of the
    two recorded candidate shapes, memoizing inside `count_tokens` was
    built and exposing the probe was not: with the stall gone, a consumer
    has no remaining reason to hold a counter callable, and none needs to
    *see* which path ran (the column is descriptive-only either way — the
    half-earned-verdict note below still governs the modeling path).
    Dogfood proof by deletion: sms_spam's hand-rolled
    `_resolve_token_counter` guard is gone, the pipeline calls
    `count_tokens` directly, no-extras artifacts byte-identical (sha256),
    and the `--extra all` run in the vocabulary-blocked sandbox — the live
    item-19 repro — now completes in seconds where it stalled ~35 minutes.
    Both paths are pinned by deterministic tests (a fake tiktoken module in
    `sys.modules`, no network), valid in the no-extras and `--extra all` CI
    jobs alike.
20. ~~**A silently-wrong boundary parse had to be caught out-of-band.**~~ —
    **struck in P11, not built**, as the item's own bar question
    anticipated: the check that caught the misparse is a one-line
    `if len(df) != expected: raise` against the published dataset size,
    observed once — an `assert_row_count` would wrap a single comparison
    behind a new name (the item-13/15 precedent, below the helper bar).
    The silent-parse *class* of failure is real and stays recorded; the
    trigger is a second project meeting a silently-wrong boundary read that
    only an expected-shape check catches, which will also show which shape
    recurs (a row-count assert vs a more general expected-shape check).
21. ~~**The features stage has no text helpers.**~~ — struck in P11 as
    observed-once, **served in P17** when the recorded trigger fired.
    `char_count`/`token_count` were two hand-rolled lines in `sms_spam`; when
    `projects/bbc_news` reached for the same length-feature family a second time,
    the trigger this item recorded ("a second text project hand-rolling the same
    columns decides both the bar and the shape") fired, and it decided the
    **frame helper**: `ds.features.text_features(df, column, *, features=None)`
    emits `<column>_char_count`, `_word_count` and `_avg_word_length` in one
    stateless call (the `add_datetime_features` shape), rather than a scatter of
    single-column counters. It deliberately carries only encoding-independent
    string features — `token_count` stays in `ds.modeling.nlp.count_tokens`,
    because its extras-dependent value is a different contract (that split is
    itself the item-18/26 "keep the reproducible part separate" lesson). `bbc_news`
    consumes it; `sms_spam` keeps its two inline lines (a finished project, not
    retrofitted).

Notes from the same run, for the record:

- **`count_tokens` half-earned its keep — the honest packaging verdict.**
  It finally has a real consumer, but only a *descriptive* one: its
  documented graceful degradation (BPE counts with tiktoken and its
  vocabulary, whitespace counts otherwise) means its values depend on the
  installed extras and network, which bars it from any feature a fitted
  model consumes — a pipeline that must reproduce across CI's no-extras and
  `--extra all` jobs cannot put an environment-dependent column in front of
  the estimator. So the first real text project scopes it to EDA artifacts
  and hands the modeling-side tokenization to the vectorizer. That is the
  P4 finding again at the function level: the degradation contract that
  keeps the module importable is exactly what keeps the function out of
  the modeling path (and, per item 19, the *shape* of that degradation —
  per-call, undetectable — is itself the sharpest friction this run hit).
  Recorded, not hidden; no action until a consumer wants the
  accurate-count path badly enough to declare a hard tiktoken dependency.
  **Upgrade (P17): the verdict was too absolute — "half-earned" becomes
  "earns a robust modeling consumer."** `bbc_news` puts `token_count` in front
  of the model, and it holds: the extras-dependence bars the count from a model
  that is *sensitive* to its exact value, but not from one that is *robust* to
  it — and a topic classifier with a TF-IDF heart, where `token_count` is one
  coarse length signal among thousands of terms, is robust (its predictions
  barely move whether the count is BPE or whitespace). So the count feeds the
  model there, with the tests asserting path-independent macro-F1 bounds instead
  of exact values. The reproducibility caveat is now *scoped* (it governs
  sensitive models), not blanket; the split between the reproducible length
  features (`ds.features.text_features`) and the extras-dependent count
  (`count_tokens`) is what lets a project choose per feature.
- **The `labels=` mapping's second consumer composed cleanly.** Binary
  `{0: "ham", 1: "spam"}` on the same three surfaces diamonds earned it
  for — no friction, mapping semantics held at two classes.
- **`bin_column`'s second consumer repeated the diamonds shape exactly**
  (quantile bins as an exploration device, never a model feature) — the
  helper earns its keep.
- **Items 13 and 15's second-project triggers did not fire.** No time axis
  is hand-assembled (no time dimension at all), and no out-of-range row
  mask is hand-rolled (the only row-dropping is `drop_duplicate_rows`).
  Both stay parked.
- **Imputation stays unexercised at severity beyond titanic.** The
  collection has zero missing values — recorded, as with diamonds, so the
  gap isn't mistaken for coverage.
- **The P9-fixed template held up in its first real use.** `ds new
  "SMS Spam"` scaffolded the shape this pipeline kept: the injectable
  `settings` parameter, the temp-`Settings` test pattern, and the
  `ds run sms_spam` instruction all survived into the finished project
  unchanged; only the placeholder stage comments and skeleton tests were
  replaced, which is what placeholders are for.

Where the library did *not* fight: `load_raw`'s kwargs forwarding took the
headerless quote-disabled read without a project-local loader; the
stratified `cross_validate_kfold(make_pipeline=..., stratify=True)`
composition at a 13% positive rate; `fit_baseline("majority")` on the
int-coded target (item 6's numeric-label scoping held again);
`drop_duplicate_rows`; `train_test_split_random(stratify=)`;
`ordinal_encode` as a stateless explicit-order label coder; and the whole
persistence story for what the vocabulary *can* express (`fit_pipeline` →
`save_params`/`load_params`, `save_model`/`load_model` for the sklearn
side).

## Friction backlog (from `projects/air_quality`)

The sixth run of the demand loop — the first against instrument-outage
missingness and a gapped hourly axis. Numbering continues from the
`sms_spam` list; in observed-pain order:

22. ~~**`cross_validate_by_time` cannot re-fit the transform chain per fold —
    item 9's parked trigger finally fired.**~~ — **served in P13**: the
    `make_pipeline` factory `cross_validate_kfold` already carried, added to
    `cross_validate_by_time` (the recorded strongest candidate) — the same
    `FitStep`-plan mechanism, re-fitted per fold on each fold's expanding
    rolling-origin window only. `air_quality` now hands the CV the raw
    training frame plus `make_pipeline=lambda frame: fit_pipeline(frame,
    plan)`, and its hand-rolled `_fold_fit_state` boundary reproduction (the
    `divmod` block-cutting copied from the library source, which existed
    *only* to measure the drift out-of-band because the function exposed
    per-fold sizes but not the windows) is deleted — the dogfood proof that
    the parameter's shape is right. The item-9 finding does **not** repeat
    here: on titanic the per-fold statistics changed but not one prediction
    flipped (logistic regression absorbed the affine shifts), whereas on this
    data the effect is real — the impute medians and scale centres vary enough
    (`nox_gt`'s median spans 115–147, a 28% swing) that the leak-free protocol
    measurably moves the per-fold CV numbers (every fold's MAE shifts; the CV
    mean goes 0.406 → 0.409). The held-out metrics and every persisted
    artifact stayed byte-identical (`cv_folds.csv` is the only output that
    changes, exactly as it should). Recorded honestly either way.
23. ~~**`fit_baseline`'s positional contract cannot align on a gapped time
    axis.**~~ — **resolved in P13 by documenting the completeness
    assumption**, not building a time-aware variant. `fit_baseline`'s
    `"naive_last"`/`"seasonal_naive"` docstring now spells out that the two
    naive strategies align *positionally* (`predict(n)` returns the next `n`
    values by position), which is the true same-time-ago reference only on a
    *gapless* continuation of the training axis; on a gapped axis (rows
    dropped for missingness) position `i − season_length` is no longer the
    same season back, so align by timestamp instead — the docstring gives the
    recipe, and names the completeness the flights note ("positional alignment
    is correct by construction") quietly assumed. A helper was not built: this
    is one consumer's evidence (the same bar every helper here clears), a
    time-aware baseline API from one data point would be a guess, and
    air_quality's `same_hour_yesterday_reference` is a four-line time-indexed
    lookup. Revisit if a second gapped-axis project hand-rolls the same
    alignment.
24. ~~**The time axis was assembled by hand — item 13's second-project
    trigger fired.**~~ — **served in P13**, with the shape resolved to a
    validation guard rather than the recorded parse-wrapper: `assert_unique(df,
    column)` in `ds.validation`. The recorded candidate bundled parse +
    uniqueness ("one helper taking the assembled string series + a required
    `format=` + the uniqueness check"); but the parse line differs per project
    (different formats, different concatenations), so only the uniqueness check
    was ever shared, and wrapping `pd.to_datetime` behind a new name is exactly
    what item 13 was struck for. `assert_unique` captures precisely the
    load-bearing repeated part — the guard raw `to_datetime` doesn't do — as a
    fluent guard matching the stage's family (returns the frame, raises
    `DataValidationError` listing the duplicates), leaves the one-line parse a
    one-liner (which the backlog said was fine), and generalizes to any
    unique-key check. Both flights and air_quality now call it after their
    (still project-local) concatenation-and-parse; both projects' persisted
    artifacts stayed equivalent.
25. ~~**A silently-wrong boundary parse had to be caught by an
    expected-shape check — item 20's second-project trigger fired.**~~ —
    **served in P13**: `assert_row_count(df, expected)` in `ds.validation`,
    the recorded candidate shape. Item 20 was struck in P11 as below the bar
    "observed once"; the trigger firing (air_quality's 114-trailing-NaN-row
    trap — structurally sms_spam's two-swallowed-rows save) is what that strike
    said would flip it, and the two data points agree on the shape (row-count
    assert against a published size, no general expected-shape check needed).
    What it adds over the one-line comparison is the stage-consistent
    `DataValidationError`, so a boundary check fails like every other guard —
    the same reason the whole `assert_*` family exists (`assert_no_nulls` is a
    one-liner too). air_quality's `trim_raw` now closes with it; artifacts
    unchanged.
26. ~~**Sentinel-coded missingness is invisible until hand-converted.**~~ —
    **resolved in P13 by documenting the gotcha**, not building
    `mask_sentinels`: a new Acquire-section paragraph in the Guide (the
    item-17/18 precedent — document where the next consumer meets the
    problem). The replace is one line observed once, below the aliasing-helper
    bar, but the *invisibility/ordering* interaction is the real lesson and
    documents better than it wraps: the sentinel is invisible to
    `missing_value_report`/`assert_in_range`/`summarize` until decoded, the
    read-time `na_values=` idiom silently misses the decimal-comma spelling
    (`-200,0`, 457 occurrences), and the robust post-parse numeric replace must
    run *before* any validation or EDA sees the frame — the ordering is
    load-bearing. air_quality keeps its one-line `mask_sentinels`. Revisit if a
    second sentinel-coded project repeats it.

Notes from the same run, for the record:

- **The imputation coverage gap is closed as *exercised*, not changed.**
  Recorded twice ("stays unexercised at severity beyond titanic"), the gap
  is now served: median fills fitted on the training window flow through
  the persisted pipeline over genuinely cell-level, independent-outage gaps
  (~5% of `nox_gt`/`no2_gt` within the labeled rows), and the pair
  composed without friction. The *severity* lesson was structural, though:
  most of this dataset's missingness was never imputation's to solve —
  90% missing means drop the column, all-channels-offline means drop the
  row, and a missing *target* means the row is the deployment condition,
  not training data. The report → triage → impute-the-remainder sequence
  is the real pattern, and `missing_value_report` carried it.
- **Item 15's trigger did not fire as recorded, but the row-mask family
  recurred.** No out-of-range mask was hand-rolled (the range checks all
  pass post-sentinel). But this is the third project to hand-roll
  row-dropping masks — diamonds' range mask, now an all-channels-null mask
  and a null-target mask — with a *different* shape each time, which is
  itself evidence: the recurring need is "drop rows failing a
  project-specific predicate", which `df.loc[mask]` already spells as
  clearly as any helper could. Stays parked, with the shape question now
  leaning "no helper".
- **Item 21's trigger did not fire** — no text columns, nothing
  hand-rolled from the text-features family. Stays parked.
- **Checksum-pinned fetch, a note not an item:** with the UCI archive
  unreachable from some networks, the fetch pins two byte-identical
  personal-repo mirrors and verifies sha256 before trusting either
  (including the cached copy — a partial earlier download must not poison
  later runs). Hand-rolled ~10 lines, first project to need it; if a
  second mirror-fetched project repeats the dance, that is the trigger for
  a fetch helper — recorded so the next session doesn't re-derive it.
- **The P9-fixed template held up again.** `ds new "Air Quality"`
  scaffolded the injectable-`settings` shape this pipeline kept; only the
  placeholder stage comments and skeleton tests were replaced.

Where the library did *not* fight: `assert_dtypes`' first real consumer is
the star — the decimal-comma misparse is exactly the silent failure a dtype
pin exists for, and it composed as one dict comprehension;
`load_raw`'s kwargs forwarding took `sep=";", decimal=","` without a
project-local loader (second consumer, the sms_spam path);
`missing_value_report`/`plot_missingness` carried the whole triage at real
severity; `fit_impute_values`/`apply_impute_missing` through
`fit_pipeline`'s three-step plan (impute → one-hot → scale, order
load-bearing and honored); `fit_one_hot_categories` on an *integer* hour
column, including the strict-JSON round-trip of int categories through the
persisted pipeline; `add_datetime_features(features=["hour", "is_weekend",
"elapsed_months"])` — the explicit-selection shape (item 11) at its third
consumer and the elapsed-months trend (item 12) at its second, as the
sensor-drift term; `train_test_split_by_time` (third consumer);
`cross_validate_by_time` itself, within its no-refit contract;
`fit_baseline("mean")` as the no-information floor;
`regression_metrics`/`compare_models`/`plot_model_comparison`/
`plot_residuals`; `plot_series` on an hourly window with a prediction
overlay; and the whole persistence story (`fit_pipeline` →
`save_params`/`load_params`, `save_model`/`load_model`, held-out window
scored from reloaded state only).

## Friction backlog (from `projects/adult_income`)

The seventh run of the demand loop — the first heavily-categorical,
high-cardinality one. It was picked precisely to stress the categorical cluster
by absence, and the honest result is a *short* backlog: that cluster existed and
composed first-try, so the project generated fewer new gaps than it validated
old surface (see "where the library did not fight" below). The two gaps it did
surface both have triggers that were *already recorded* by earlier projects and
now fire on their agreed second occurrence. Numbering continues from the
`air_quality` list; in observed-pain order:

27. ~~**Checksum-verified multi-mirror fetch is hand-rolled a second time — the
    fetch-helper trigger fires.**~~ — **served in P15**: `ds.io.fetch_dataset(name,
    urls, *, sha256, settings=None)`, the recorded candidate shape, returning the
    verified local path. `adult_income`'s `fetch_raw` was a ~25-line near-verbatim
    copy of `air_quality`'s (download into `settings.raw_dir`, try each mirror,
    verify the payload's sha256, re-verify not trust a cached copy so a partial
    earlier download can't poison later runs) — two projects carrying the same
    dance with only the URL tuple, filename and checksum varying, all *data* not
    code, above the helper bar exactly as `air_quality` predicted ("if a second
    mirror-fetched project repeats the dance, that is the trigger"). The two open
    questions resolved with evidence: the checksum is **required** (keyword-only) —
    the trigger fired on the *checksum-verified multi-mirror* shape, and the
    seaborn-mirror projects' plain un-pinned "download if absent" is a few inline
    lines below the aliasing bar, so they were **not** folded in (an optional
    checksum purely to absorb them would build surface with no demand, and pinning a
    live upstream repo is an unasked-for maintenance commitment); and the cache
    re-verify lives **inside** the helper, because it is part of what "verified
    fetch" means and a caller-side re-verify would be re-implemented at every call
    site. Stdlib only (`urllib`/`hashlib`) — no dependency added; a `ds.io`
    addition, so `tests/test_public_api.py` is untouched. The helper also hardens
    the dance: the destination resolves through `_resolve_within`, refusing a
    traversing `name` (the projects joined the raw dir directly). Dogfood proof by
    deletion: both `air_quality` and `adult_income` deleted their hand-rolled
    `fetch_raw` body — each keeps only its data plus a one-line `fetch_raw` binding
    it to `fetch_dataset` — and both end-to-end tests pass with held-out metrics and
    persisted artifacts equivalent (the verified bytes are identical). The seaborn
    projects' un-checksummed fetch stays recorded as the natural future consumer
    that would justify an optional-checksum widening only when one actually pulls it.
28. ~~**A string `"?"` missing-sentinel is decoded by hand — air_quality item
    26's second-sentinel trigger fires, in a new flavor.**~~ — **resolved in P15 as
    a documented one-liner, not built**, now that two differently-typed sentinels
    are on record (numeric −200, string `"?"`). Three categorical columns
    (`workclass`/`occupation`/`native_country`) tag an unknown with a literal `"?"`;
    `decode_sentinels` is the one-line `df[cols].replace("?", np.nan)` that makes
    the gaps visible to `missing_value_report` and fillable by the mode-impute step.
    The recurring need is "replace a per-project sentinel *value* in named columns
    with NaN", but the value (−200 vs `"?"`), its type, and the target columns
    differ every time, so the shared part is exactly the one-line `replace` — which
    spells it as clearly as any wrapper would, while the load-bearing *lesson* item
    26 documented (the sentinel is invisible to validation/EDA until decoded, and
    the decode must run before any of them sees the frame) is already the Guide's
    Acquire gotcha. A `mask_sentinels(df, columns, sentinel)` would alias the
    one-liner behind a new name (the item-13/15/20/26 precedent), so it stays
    documented; the project keeps its one-line `decode_sentinels`. A **third**
    differently-typed sentinel would force the thin helper.
29. ~~**`ds.eda` has no categorical↔target association view.**~~ — **struck /
    parked in P15, not built.** `top_correlations` is numeric-only, so on a dataset
    whose strongest predictors are categorical (marital status, occupation,
    education) the explore stage cannot rank them — the feature choices here leaned
    on domain knowledge and the confusion structure instead. But unlike the fetch
    and sentinel dances this gap was worked around by *thinking*, not by hand-rolled
    code, so there is nothing to promote and no consumer to document at, and the
    candidate shape is genuinely open (a Cramér's-V / mutual-information ranker vs a
    positive-rate-by-level group table vs a `bin_column`-style categorical
    profiler). Trigger: a second heavily-categorical project that actually
    hand-rolls the groupby, which also decides the shape. Not built speculatively.

    **Follow-on: served.** The goal-alignment pass that followed P15 promoted
    this from parked to built, because it is the one thin general-purpose stage
    (`eda`, three read-only helpers) and four of seven projects are
    heavily-categorical classification, so the signal has fired repeatedly even
    where each project reached for domain knowledge instead of code. The shape
    chosen among the three candidates is the **positive-rate-by-level group
    table**: `ds.eda.target_rate_by_category(df, column, target, *,
    min_count=1)` returns, per level of a categorical column, its `count`,
    `frac`, the mean `target` within the level (the "target rate" — a 0/1
    target's positive rate, or any numeric target's group mean) and the overall
    `baseline`, sorted by target rate descending — the categorical counterpart
    to `top_correlations` (which is numeric-only). It is *descriptive*, not a
    fitted feature (a target rate fed back as an input is textbook leakage; the
    docstring says to compute it on the training split when it informs a
    decision), and it pairs with `ds.viz.plot_target_rate` (per-level bars with a
    dashed baseline line) per the reuse-across-stages rule. The Cramér's-V /
    mutual-information *ranker* across columns was not built — it answers a
    different question (which column, not which level) and no project has
    hand-rolled it; it stays the open shape a future association-ranking friction
    would pull. `adult_income` adopts the group table in its Explore stage
    (>50K rate by `marital_status` and `occupation`) to prove it earns its place;
    its end-to-end test passes unchanged.

Notes from the same run, for the record:

- **The value-whitespace friction dissolved into a `load_raw` kwarg.** Every
  categorical value in the raw file is space-padded (`" Private"`), and the
  target labels too (`" <=50K"`). Rather than a hand-rolled `.str.strip()` pass
  (which `standardize_column_names` does for column *names* but nothing does for
  *values*), `skipinitialspace=True` forwarded through `load_raw` strips it at
  read time — the third consumer of that pandas-kwargs forwarding after
  `sms_spam` and `air_quality`. A `strip_string_values` helper was therefore
  *not* needed; recorded so a later heavily-categorical project that meets
  trailing (not leading) whitespace, which `skipinitialspace` doesn't touch,
  knows this is where that helper's trigger would sit.
- **Exact duplicate rows kept, deliberately (the titanic call).** 24 rows are
  exact duplicates, but with only these coarse survey attributes and no
  respondent identifier, identical vectors are expected distinct people —
  `drop_duplicate_rows` would delete real records. This is titanic's keep, the
  deliberate opposite of the diamonds/sms_spam drop (there the duplicates were
  split-leaking verbatim re-entries a model could memorize; a penalized linear
  model on census demographics cannot).
- **Item 15's row-mask trigger did not fire.** No predicate row-drop was
  hand-rolled at all (no range mask, no offline mask; duplicates kept, only the
  train/test split partitions rows). The row-mask family stays parked, still
  leaning "no helper" — a *same-shape* fourth mask would be needed to flip it,
  and this project added none.
- **Item 21's text-helper trigger did not fire** — no text columns; nothing
  from the text-features family was hand-rolled. Stays parked.
- **`count_tokens` stays half-earned.** No text pipeline here, so the
  accurate-count path is untouched; the verdict is unchanged — no action until a
  project needs it badly enough to declare a hard `tiktoken` dependency.

Where the library did *not* fight (the honest headline of this pick): the whole
categorical cluster it was chosen to stress composed first-try.
`fit_topk_categories`/`collapse_categories` took two high-cardinality columns at
once (`native_country` 41→10+other, `occupation` 14→10+other) — the second
consumer, and the first to lean on the collapse for genuine tail-sparsity
(Armed-Forces has nine rows) rather than nyc_taxis's cardinality alone —
including the strict-JSON round-trip of the kept-category tuples through the
persisted pipeline; `fit_one_hot_categories(drop_first=True)` produced the
full-rank wide matrix (48 indicator columns) a penalized logistic regression
needs, the first `drop_first` consumer; `flag_outliers` reported the capital
columns' IQR-fence counts without touching them (the first flag-not-clip
consumer, with `plot_outliers` on the same two columns); `fit_impute_values(
strategy="most_frequent")` filled the sentinel gaps (third consumer, at scale);
`check_schema(coerce=True)` pinned the six numeric dtypes; `assert_row_count`
and `assert_in_set` guarded the boundary; `train_test_split_random(stratify=)`
and `cross_validate_kfold(stratify=True, make_pipeline=...)` held the ~24/76
balance while re-fitting genuinely-varying per-fold state (an air_quality-style
real per-fold refit, not titanic's no-op); `fit_baseline("majority")` gave the
numeric-label floor (item 6's scoping held a fourth time); the `labels=` binary
mapping put the income names on the confusion/per-class axes (third consumer);
and the whole persistence story (`fit_pipeline` → `save_params`/`load_params`,
`save_model`/`load_model`, held-out split scored from reloaded state only).

## Friction backlog (from `projects/sunspots`)

The eighth run of the demand loop, and the **second forecasting** one — picked
after `flights` for a series a calendar-feature + naive approach handles badly
(a solar cycle aligned to nothing on the calendar), so the friction would pull
the forecasting surface deeper. Numbering continues from the `adult_income` list.
Because forecasting is a committed capability whose first project delegated its
model to raw scikit-learn, both items were **served in the same demand loop**
(the P3 precedent of building and dogfooding in one batch); in observed-pain
order:

30. ~~**The features stage has no autoregressive (lag) features.**~~ —
    **served in P16**: `ds.features.add_lagged_features(df, column, lags, *,
    dropna=True)`, the autoregressive counterpart to `add_datetime_features`.
    `flights` predicted from calendar position; this series has none worth
    using, so its only signal is its own recent history — every AR model needs
    lag columns, and the library had no way to make them. The helper adds
    `<column>_lag_<k>` (taken by row position — sort the time axis first),
    ascending-ordered regardless of request order, dropping the warm-up rows by
    default; it is stateless, so it precedes the split like the datetime
    features. Built on first-consumer strength rather than parked as a one-liner
    (the `plot_series` precedent, not the item-13/15 one): a single `.shift(k)`
    is a line, but a *whole model class* — autoregression — is impossible
    without the named multi-lag set, and the warm-up/ordering handling is the
    reusable part. `sunspots` is its first consumer.
31. ~~**No way to forecast past the edge of the data.**~~ — **served in P16**:
    `ds.modeling.timeseries.forecast_recursive(model, history, *, lags, steps)`.
    A lag-feature model does one-step-ahead scoring with a plain `model.predict`
    (each row reads the true recent values), but forecasting *further* than one
    step has no observed values to read — the later steps' lags are the model's
    own earlier predictions. `forecast_recursive` feeds each prediction back and
    slides the window, the multi-step forecast raw sklearn cannot do. Clearly
    above the aliasing bar — a fiddly, error-prone recursion (buffer management,
    feature-order alignment with `lags`, an optional `feature_names_in_` frame to
    avoid sklearn's name warning), not a wrapped one-liner. Scoped to pure
    autoregression by contract (an exogenous feature would need a future value it
    cannot supply), recorded in the docstring. `sunspots`'s held-out decade is
    its first consumer, and the honest finding rides along: recursive error
    compounds (r² 0.88 one-step → −0.35 recursive), yet still beats both
    calendar-naive references.

Notes from the same run, for the record:

- **A pure-AR forecaster needs no `ds.pipeline` scoring `Pipeline` — a scope
  finding, not a gap.** `flights` recorded that clean time-series data leaves the
  one-step fit plan with little to do; this project takes it to zero: lags are
  stateless, the series is complete (no impute), its swings are the signal (no
  clip), and OLS is scale-free (no scale), so nothing frame-shaped is fitted and
  only the model is persisted. Recorded so the missing `*_scoring.json` reads as
  intended, not as an omission. (Had this model wanted regularization, the scaler
  would live *inside* the sklearn estimator — the item-18 "model-side transforms
  live in the estimator" convention — keeping `forecast_recursive`'s
  raw-lags-in contract clean.)
- **A dedicated multi-step backtest was not built.** The one-step rolling-origin
  CV reused `cross_validate_by_time`, and the held-out multi-step forecast used
  `forecast_recursive` directly; no rolling *multi-step* backtest was
  hand-rolled, so nothing pulls a `ds.evaluation` backtest harness yet. Recorded
  as the trigger — a project that hand-rolls a rolling recursive-forecast
  evaluation — so the next session doesn't build it speculatively.
- **`fit_baseline`'s naive strategies are the *right* weak references here.**
  Their positional alignment (item 23's documented completeness assumption)
  holds — the held-out window continues the training axis with no gap — and both
  are honestly poor on a non-calendar cycle, which is the point of showing them.

Where the library did *not* fight: `fetch_dataset` took its third consumer (the
first with a genuinely live upstream repo, where the sha256 pin is clearly
right); `assert_unique` its third (the `YYYY-MM` parse); `cross_validate_by_time`
composed one-step rolling-origin folds on the lagged frame first-try;
`regression_metrics`/`compare_models`/`plot_residuals`/`plot_model_comparison`
and `plot_series` (the two-call history + forecast overlay, now with *two*
prediction series) all composed as on `flights`; and `save_model`/`load_model`
scored both forecasts from the reloaded model.

## Friction backlog (from `projects/store_sales`)

The tenth run of the demand loop, and the first on a **panel** (multi-entity)
dataset — picked by the grep-driven rule for the data *shape* no project had
(every prior project was one flat table or one univariate series). Numbering
continues from the `sunspots` list. The load-bearing item was **served in the
same demand loop** (a silent correctness bug, not a convenience gap); the rest
had clean inline workarounds on a shared-calendar panel and are recorded, not
built, in observed-pain order:

32. ~~**`add_lagged_features` bleeds history across entity boundaries.**~~ —
    **served in P18**: `ds.features.add_lagged_features(..., *, group=None)`. The
    helper lagged by row position over the *whole* frame, so on a panel — many
    series stacked in one frame — the first rows of each entity read the previous
    entity's tail as their history. That is wrong output, not a missing
    convenience, so it was fixed rather than parked as a one-liner: `group=` (one
    key or several) takes the lags *within* each entity via a grouped `shift`, and
    the reusable part is the guarantee that no value crosses a boundary plus
    per-group warm-up dropping. `store_sales` is its first consumer, with a test
    asserting store 2's first row carries no value bled from store 1.
33. **A panel wants a date-*cutoff* split, not `train_test_split_by_time`'s row
    fraction.** — *recorded, done inline.* `train_test_split_by_time` sorts one
    series and holds out the last fraction of *rows*; on an interleaved panel that
    straddles the boundary date across entities. A shared-calendar panel has a
    clean workaround worth two lines — a `date < cutoff` mask cuts every entity at
    the same instant — so the split was done inline (train 2013–2016, forecast
    2017). The trigger for a library `cutoff=`/`group=` split variant is a second
    panel project reaching for the same mask, or a panel whose entities *don't*
    share a calendar (where the inline mask no longer suffices).
34. **`assert_unique` guards a single column; a panel's key is composite.** —
    *recorded, done inline.* The panel's uniqueness key is `(store, item, date)`;
    `assert_unique` takes one column. A composite-key check is one
    `df.duplicated(subset=[...])` line, below the aliasing bar — done inline in
    `order_panel`. The trigger for widening `assert_unique` to accept a column
    list is a second project hand-rolling the same composite-key guard.
35. **No per-entity rolling-origin *backtest* — the parked harness now has one
    consumer's demand.** — *recorded, not built.* `fit_baseline`,
    `forecast_recursive` and `cross_validate_by_time` are all single-series; on a
    panel the naive references needed no fitting (each entity's `sales_lag_1` /
    `sales_lag_7` column *is* its naive-last / weekly-seasonal-naive one-step
    forecast, so evaluation read them directly), and the held-out evaluation was
    one-step-ahead. No grouped rolling-origin multi-step backtest was hand-rolled,
    so a `ds.evaluation` panel-backtest harness still has demand but no second
    consumer. This sharpens the sunspots item-31 note: the build trigger is a
    *second* panel (or multi-step) project that hand-rolls the grouped rolling
    evaluation — then the harness is built on first-consumer strength.

Notes from the same run, for the record:

- **A pooled panel model needs no impute/clip/scale step — the flights/sunspots
  scope finding, a third time.** The only fitted state is the entity + calendar
  one-hot vocabularies (`fit_pipeline`); the lags are stateless, the series are
  complete (no impute), and OLS is scale-free (no scale). Recorded so the
  single-step one-hot plan reads as intended, not as an omission.
- **Entity identity rides in as one-hot fixed effects.** Store and item enter as
  one-hot dummies (not raw integer codes, which OLS would misread as ordered), so
  a single pooled `LinearRegression` carries per-entity intercepts — the standard
  panel fixed-effects shape, built from the existing one-hot surface with no new
  helper.

Where the library did *not* fight: `fetch_dataset` took its fifth consumer;
`add_datetime_features`'s scoped `features=` subset selected exactly the calendar
terms a daily series uses; `fit_pipeline`/`one_hot_encode` fit the entity +
calendar vocabularies on train and applied to both; `split_features_target`,
`regression_metrics`/`compare_models`, `plot_residuals`/`plot_model_comparison`
and the two-call `plot_series` overlay all composed as on `flights`; and
`save_model`/`load_model` scored the held-out window from the reloaded model.

## Friction backlog (from `projects/bank_marketing`)

The eleventh run of the demand loop, and the first on an **imbalanced /
rare-event** target — picked by the grep-driven rule for the data *shape* no
project had (every prior classification project scored a roughly-balanced target
on hard labels). Numbering continues from the `store_sales` list. The
load-bearing item was **served in the same demand loop** (a whole missing
capability class, not a convenience gap); the rest had clean inline workarounds
and are recorded, not built, in observed-pain order:

36. ~~**No way to score a classifier's probabilities — accuracy is a trap at 11%
    prevalence.**~~ — **served in P19**: `ds.evaluation.probability_metrics(y_true,
    y_score)`. `classification_metrics` scores hard 0/1 predictions only, so it
    hides behind whatever threshold produced them; on a rare-event target a
    majority-class predictor scores 0.887 accuracy while finding no positives, so
    hard labels cannot tell an honest story. The new helper scores the predicted
    *probability* directly — ROC-AUC, average precision (PR-AUC, the
    imbalance-robust one whose no-skill floor is the positive rate, not 0.5) and
    Brier — the threshold-free capability the stage entirely lacked. Above the
    aliasing bar on two counts: a capability class, and a correctness guarantee
    (it raises on single-class `y_true`, where scikit-learn merely warns and
    returns `nan`). `bank_marketing` is its first consumer.
37. **No operating-point / threshold-selection helper.** — *recorded, done inline.*
    A rare-event model at the default 0.5 threshold predicts almost all "no"; the
    project handled this idiomatically with `class_weight="balanced"` on the
    estimator (a scikit-learn argument, one keyword — well below the aliasing
    bar), so the 0.5 threshold still yields a meaningful confusion matrix. A
    library `choose_threshold(y_true, y_score, *, criterion=...)` — sweeping the
    precision–recall curve to maximise F1 or hit a target precision/recall — is
    genuinely fiddly reusable logic (the `precision_recall_curve` off-by-one
    between thresholds and prec/rec is a classic bug), so it clears the bar the
    moment it has real demand. The trigger is a *second* imbalanced project that
    needs a **tuned** operating point rather than class reweighting (e.g. a fixed
    precision budget, or a cost-sensitive threshold) — then it is built on
    first-consumer strength beside `probability_metrics`.
38. **No ROC / precision–recall *curve* plot in `ds.viz`.** — *recorded, not
    built.* The probabilistic story was visualized with the existing
    `plot_model_comparison` bars on `average_precision` (model vs prevalence
    floor), which is enough to make the ranking gap legible. A dedicated
    `plot_roc_curve` / `plot_pr_curve` (the curve, with its no-skill baseline)
    would pair with `probability_metrics` the way `plot_confusion_matrix` pairs
    with `confusion_frame`, but a single point-metric bar covered this project.
    The trigger is a second probabilistic project wanting the operating-point
    *curve* (where the sweep, not the summary number, is the finding), or an
    explicit ask.

Notes from the same run, for the record:

- **`duration` dropped as leakage — the first leakage drop.** The call length is
  known only *after* the call whose outcome is the target (a 0-second call is a
  "no" by construction), so a model that keeps it scores spectacularly and
  predicts nothing usable. The UCI docs are explicit it must be excluded. This is
  a project-judgment boundary call, not a library gap (the earlier fnlwgt /
  education drops in `adult_income` were redundancy, not leakage) — recorded so
  the drop reads as intended.
- **The `pdays == 999` sentinel is the decode-a-sentinel pattern's third
  consumer.** Like air_quality's −200 and adult_income's "?", an in-band sentinel
  (here "never previously contacted", ~96% of rows) is folded to a clean binary
  flag inline — one `!=` comparison, below the aliasing bar. Reaffirms item 26:
  the pattern stays inline, no library sentinel-decoder is pulled.
- **`compare_models` composed with the new metric with no change.** Because
  `probability_metrics` shares the two-argument `(y_true, y_score)` shape of the
  other metric helpers, scoring the model against a probabilistic prevalence
  floor was the same `compare_models(..., metrics_fn=probability_metrics)` call —
  the `MetricsFunction` protocol paid off, no new comparison surface needed.

## Friction backlog (from `projects/mammography`)

The twelfth run of the demand loop, and the **second** on an imbalanced /
rare-event target — picked by the grep-driven rule to give `probability_metrics`
a second consumer and to fire the two items `bank_marketing` parked (37–38).
Numbering continues from the `bank_marketing` list. Unlike that project, this one
does *not* reweight the classes: it tunes the operating point, which is precisely
the shape items 37–38 named as their build trigger — so **both were served in the
same loop** (P20), on first-consumer strength. The remaining friction had clean
inline workarounds and is recorded, not built:

36. *(see the `bank_marketing` backlog — `probability_metrics`, served in P19; its
    second consumer is this project.)*
37. ~~**No operating-point / threshold-selection helper.**~~ — **served in P20**:
    `ds.evaluation.choose_threshold(y_true, y_score, *, criterion, target)`. A
    screening programme needs a *tuned* operating point (catch ≥80% of
    calcifications), which `class_weight` cannot express — so this project's
    demand fired the trigger `bank_marketing` recorded exactly. Above the aliasing
    bar as fiddly reusable logic with a correctness guarantee: it drops the
    trailing `(precision=1, recall=0)` point that `precision_recall_curve` returns
    with no matching threshold (the classic off-by-one) before searching, and
    raises when a `target_*` floor is unreachable rather than returning that
    phantom point. Tests pin the off-by-one alignment (a thresholded prediction
    reproduces the reported precision/recall), the unreachable-target raise, and
    the single-class guard.
38. ~~**No ROC / precision–recall *curve* plot in `ds.viz`.**~~ — **served in
    P20**: `ds.viz.plot_pr_curve` / `plot_roc_curve`. On a rare-event screening
    target the *sweep* is the finding (which recall you buy at which precision),
    not a single summary number, so the curve earned its place — the second
    probabilistic project wanting the operating-point curve, the trigger P19
    recorded. Each draws its no-skill baseline (the PR curve a dashed prevalence
    line, the ROC curve the chance diagonal) and returns the Axes, so the pipeline
    scatters the three tuned operating points onto the PR curve.
39. **~30% of rows are exact duplicates, kept not dropped.** — *recorded, done
    inline.* With only six coarse standardized attributes and no patient
    identifier, identical vectors are expected distinct screenings; dropping them
    would delete real records and distort the 2.3% prevalence (the titanic /
    bank_marketing precedent, a third consumer). The honest cost — a duplicated
    vector can land in both split halves — is a mild optimism shared with those
    projects, not a leakage bug fixable without deleting real rows. No library
    change: the "keep coarse-vector duplicates" judgment stays a per-project call.
40. **The threshold is tuned on the training scores, not a dedicated validation
    split.** — *recorded, done inline.* Choosing the operating point on the same
    data the model was fit on is a mild in-sample optimism; the clean fix is a
    held-out validation slice (or out-of-fold scores) to choose the threshold on.
    A one-line `train_test_split_random` carve is below the aliasing bar, and this
    project's held-out *test* metrics (scored at the train-chosen threshold) are
    still honest, so it was left inline. The trigger for a library
    `fit`/`choose`/`apply` threshold-calibration helper is a second project that
    needs the out-of-fold operating point (e.g. a cost-sensitive threshold tuned
    under CV), where the validation plumbing stops being one line.
41. **`choose_threshold` returns a point; applying it is still a caller
    list-comprehension.** — *recorded, done inline.* Turning the chosen threshold
    into hard labels is `[1 if s >= t else 0 for s in scores]` — one line, done
    inline in three places here. A library `apply_threshold(scores, t)` (or a
    `predict=` convenience on `choose_threshold`) is below the bar until a second
    consumer hand-rolls the same comprehension enough to want it named.

Notes from the same run, for the record:

- **A plain, un-reweighted model is the deliberate contrast with `bank_marketing`.**
  P19 handled imbalance with `class_weight="balanced"` (a scikit-learn argument);
  P20 leaves the loss alone and tunes the threshold instead, because the screening
  operating point ("catch ≥80%") is a *recall budget* reweighting can't target.
  Recorded so the un-weighted `LogisticRegression` reads as intended, not as an
  omission.
- **No categorical or missing-value surface fired — the shape is all-numeric and
  complete.** Six standardized float attributes, no NaN, no categoricals: the
  transform plan is a single `scale_features` step, and `target_rate_by_category`
  / one-hot / impute had nothing to consume. The friction is entirely on the
  probabilistic / threshold side, which is exactly the surface this loop set out
  to stress.
- **`probability_metrics`' single-class guard paid off on the tiny-positive
  class.** At 2.3% prevalence a careless split or a degenerate fold can hand a
  scorer one class; the P19 guard (raise, don't silently return `nan`) is what
  makes that fail loudly here rather than poison a mean.

Kept for the record — CLAUDE.md's engineering notes point here. Each was
re-checked in the 2026-07 evaluation; verdicts inline.

### The four thin stages fleshed out *(stands)*

Each stage every analysis touches carries its most-reached-for helpers, built
to the standard recipe: right stage module → Google-style docstring + full
type hints (`mypy --strict`) → mirroring test → export from `__all__`,
favouring the core deps (pandas, numpy, scikit-learn, matplotlib). When adding
more, keep pairing stage functions with `ds.viz` plots where it helps (as
`plot_outliers` visualizes `flag_outliers`).

### The worked example dogfoods the stages *(stands, superseded as proof)*

`projects/_example/pipeline.py` runs realistically dirty **synthetic** data
through the full lifecycle and `tests/test_example.py` asserts the split-safe
behavior. It remains the teaching reference; `projects/nyc_taxis` is now the
proof on data the library didn't design.

### Fit/apply (split-safe) transforms *(stands)*

The six statistic-learning transforms (`impute_missing`, `scale_features`,
`clip_outliers`/`flag_outliers`, `one_hot_encode`, `ordinal_encode`,
`collapse_categories`) each have a paired `fit_*`/`apply_*` form: `fit_*`
learns parameters from one frame and returns a small frozen dataclass,
`apply_*` applies them to any frame. The single-call forms remain as
fit-and-apply-on-the-same-frame conveniences and are implemented as exactly
that, so the two forms can't drift. Category vocabularies are fixed at fit
time (unseen categories → all-zero indicators / `-1` codes / the `"other"`
label).

### Persistable fit parameters *(revisited: scope was too narrow)*

The six `fit_*` dataclasses carry validated `to_dict`/`from_dict` round-trips
and `ds.io.save_params`/`load_params` persist them as strict JSON. Decisions
that stand: per-class methods rather than a generic `asdict` mechanism (honest
types under `mypy --strict`, per-class edge-case handling next to each
definition, shared plumbing in private `ds._serde`, `ds.io` typed against the
`FittedParams` protocol); strict JSON on disk (tagged non-finite floats, numpy
scalars unwrapped, tuples re-tupled, `from_dict` validates type tag + exact
field set). **Revisit resolved:** the cited goal — "score new rows in a later
run or another process" — was unmet without persisting the *model* too; P2
extended the story to the estimator via `ds.modeling.persistence` (JSON stays
the format for parameters, joblib is used only for the model).

### Composable fit/apply pipeline *(amended: the fit side gained an executor)*

`ds.pipeline.Pipeline` holds an ordered tuple of `PipelineStep`s (fitted
parameters + the `apply_*` kind they mean), applies them in order, and
persists through `save_params`/`load_params`. Decisions that stand: a
top-level `ds.pipeline` module (composes two stages; imports run strictly
pipeline → stages); a closed `StepParams` union + `StepKind` literal under
`mypy --strict`; steps tagged by *kind* because `OutlierBounds` serves two
apply forms; train-time-only parameters stay out (scoring rows have no
target). The per-pair API stays the primitive.

**Amendment (friction items 5 + 9, the promised design pass):** "the
pipeline is pure composition" was the fit-side half of the original
decision, and its stated trigger — a second project repeating the manual
fit → apply → fit dance — fired. Resolution: `Pipeline` *remains* pure
composition (construction, application and persistence are unchanged — a
pipeline still holds only fitted parameters), and the dance moved into
`fit_pipeline`, an executor over the same per-pair primitives: it runs an
ordered plan of `FitStep` entries (a step kind plus a fit callable) and
returns the assembled `Pipeline`. Two alternatives were weighed and
rejected: a fully declarative fit-spec (one dataclass per `fit_*` form,
making plans persistable data) would mirror every fit signature and drift
with them, and nothing demanded persisting *plans* — only fitted pipelines;
solving only the CV leak (a factory parameter alone) would have left item
5's dance in both projects. A `FitStep` carries a callable closing over the
varying keyword arguments (`columns=`, `strategy=`, `k=`), so the closed
unions stay closed and `mypy --strict` types the plan without a parallel
spec hierarchy. The same mechanism is what both cross-validators'
`make_pipeline` factory re-runs inside each fold (`cross_validate_kfold` on
shuffled folds, `cross_validate_by_time` on each expanding rolling-origin
window — the latter added in P13, item 22), so one plan serves the training
run and leak-free cross-validation on either path.

### API discoverability: import by stage *(stands)*

Stage helpers are imported from their stage (`from ds.eda import summarize`),
`Pipeline` from `ds.pipeline`; the top-level `ds` namespace re-exports only
stage-independent infrastructure (`Settings`, `get_settings`, `get_logger`,
`seed_everything`). The stage name is the teaching tool; a flat re-export
would force `import ds` to eagerly load matplotlib/scikit-learn and pile all
stages' names into one namespace. `tests/test_public_api.py` pins the exact
top-level surface (and that `import ds` stays cheap). `Pipeline` earns no
top-level re-export — a pipeline *composes* stage transforms, so flattening
the composer while its building blocks stay stage-scoped would be the one
inconsistent case.

### The `ds` CLI: `run` added, `check` rejected *(stands)*

`ds run <name>` cleared the bar as a *project-aware default*: it resolves
names against the real directories under `projects/` (literal or `ds new`
slug), lists the runnable projects on a miss, and never builds a path from
the name (same traversal discipline as `ds new`). `ds check` stays rejected:
it would either duplicate `make check`'s sequence (drift risk) or just call
`make` (adding nothing) — `make` is the canonical dev entry point. Don't
re-add it.

### Docs cookbook: cross-stage recipes *(stands)*

The highest-value cross-stage recipes are in `docs/guide.md` (validate at the
acquire boundary; screen redundant features before scaling; fit/evaluate/
diagnose with a real estimator). Add a recipe if and when a new combination
comes up in practice — pre-building a catalog of hypothetical ones was
considered and skipped.
