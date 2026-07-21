# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `projects/store_sales`: tenth **real-data** project (P18 — the first on a
  **panel**, i.e. multiple entities stacked in one frame), forecasting daily units
  sold across a store × item subset (50 entities) of the classic "Store Item
  Demand" dataset, 2013–2017. Chosen by the grep-driven demand rule for the data
  *shape* no prior project had (all were single flat tables or single univariate
  series), and its one library gap served in the same demand loop:
  - `ds.features.add_lagged_features` gains a keyword-only `group=` parameter
    (backlog item 32): lags taken *within* each entity via a grouped `shift`, so
    history never bleeds across an entity boundary. The ungrouped path lagged by
    row position over the whole frame, which on a panel silently read one entity's
    tail as the next entity's history — a correctness bug, so it was fixed rather
    than parked. `store_sales` is its first consumer, with a no-bleed test.

  Full lifecycle on `ds` + scikit-learn: checksum-verified fetch
  (`fetch_dataset`'s fifth consumer), boundary validation on the full 913k-row
  file, panel selection + within-entity ordering, calendar + grouped-lag features,
  a date-cutoff split (train 2013–2016, forecast 2017), a one-hot entity+calendar
  plan (`fit_pipeline`), a pooled `LinearRegression`, and the held-out window
  scored from the reloaded model. One-step held-out 2017: MAE 5.27 / r² 0.894,
  beating the weekly-seasonal-naive (`sales_lag_7`, MAE 6.70) and naive-last
  (`sales_lag_1`, MAE 7.99) references. The rest of the panel's single-series
  friction (a date-cutoff split, a composite-key uniqueness check, a per-entity
  rolling backtest) had clean inline workarounds and is recorded as backlog items
  33–35, not built.
- `tests/test_roadmap_size.py`: a size-budget gate that fails CI if `ROADMAP.md`
  grows past its line budget (currently 120), turning the "history goes to the
  archive, not the live roadmap" convention into an enforced invariant so the
  file can't silently re-bloat over future demand loops. Pairs with a written
  write-side rule in `ROADMAP.md`'s working agreement and `CLAUDE.md`'s roadmap
  section (append new backlogs and completed plan-of-record entries to
  `ROADMAP_ARCHIVE.md`, continuing the item numbering).
- `projects/bbc_news`: ninth **real-data** project (P17 — the second **text**
  one), a multiclass topic classifier (business/entertainment/politics/sport/tech)
  over the 2,225 BBC News articles, with the text-feature surface it pulled served
  in the same demand loop. Chosen after `sms_spam` to stress the text surface a
  second time and decide the triggers that project parked:
  - `ds.features.text_features(df, column, *, features=None)` (backlog item 21) —
    a stateless one-call expansion of a string column into `<column>_char_count`,
    `_word_count` and `_avg_word_length`, the text counterpart to
    `add_datetime_features`. `sms_spam` hand-rolled these length features; a
    second text project reaching for the same family is the trigger item 21
    recorded, and it decides the *frame helper* shape over single-column
    counters. Deliberately encoding-independent (pure string ops), which keeps
    the columns model-safe — unlike `count_tokens`.
  - `count_tokens` earns its first **modeling** consumer. `sms_spam` kept
    `token_count` descriptive-only; `bbc_news` puts it in front of the model as
    one coarse length signal beside thousands of TF-IDF terms, where the
    classifier is robust to which counting path (BPE/whitespace) runs. The
    "descriptive-only" verdict is scoped, not contradicted — it governs models
    *sensitive* to the exact count, and lifts where they are robust; the tests
    assert path-independent macro-F1 bounds.
  - Backlog item 18 (a `ds.pipeline` vectorization step kind) re-checked and
    **struck by reaffirmation**: the TF-IDF vectorizer again lives inside the
    sklearn `ColumnTransformer` while the `ds` scoring `Pipeline` carries the
    frame-shaped scale step, confirming the model-side-transform convention P11
    settled suffices with a second consumer. No step kind is built (it would
    smuggle a pickle into the strict-JSON `save_params` story).

  Full lifecycle on `ds` + scikit-learn: checksum-verified fetch
  (`fetch_dataset`'s fourth consumer), boundary validation, verbatim-duplicate
  drop (99 articles — the diamonds/sms_spam leak guard), the text features,
  ordinal-coded target, stratified split, a one-step scale plan (`fit_pipeline`),
  stratified 5-fold macro cross-validation with the plan re-fitted per fold,
  pipeline + model persisted and the held-out split scored from the reloaded
  model. Held-out macro-F1 0.964 / accuracy 0.965 vs a length-only model (0.329 —
  the honest headline that the topic signal is in the words) and the majority
  class (0.077); CV macro-F1 0.957 ± 0.009. A `ds.features` addition, so the
  top-level public surface (`tests/test_public_api.py`) is unchanged. `ROADMAP.md`
  records the P17 plan-of-record entry and the `bbc_news` friction serve.
- `projects/sunspots`: eighth **real-data** project (P16 — the second
  **forecasting** one), with the autoregressive surface it pulled served in the
  same demand loop. Forecasts the monthly Zurich/SILSO sunspot number, 1749–1983
  (2,820 months), chosen after `flights` for a series a calendar-feature + naive
  approach handles *badly*: the ~11-year solar cycle is aligned to nothing on the
  calendar, so month carries no signal and the model must predict from the
  series' own recent history. Two library gaps, both served here (because
  forecasting is a committed capability whose first project already delegated its
  model to raw scikit-learn):
  - `ds.features.add_lagged_features(df, column, lags, *, dropna=True)` — the
    autoregressive counterpart to `add_datetime_features`: adds
    `<column>_lag_<k>` columns taken by row position (sort the time axis first),
    ascending-ordered, dropping the warm-up rows by default. Stateless, so it
    precedes a split. Built on first-consumer strength — a whole model class
    needs it — not parked as a one-liner.
  - `ds.modeling.timeseries.forecast_recursive(model, history, *, lags, steps)` —
    forecast past the edge of the data by feeding each prediction back as later
    steps' lags, the multi-step forecast a single `model.predict` cannot produce.
    Pure-autoregression by contract; handles the `feature_names_in_` case so no
    sklearn feature-name warning is raised.

  Full lifecycle on `ds` + scikit-learn: checksum-verified fetch
  (`fetch_dataset`'s third consumer — a live upstream repo, so the sha256 pin is
  correct here), boundary validation, hand-assembled time axis (`assert_unique`
  again), an Explore step that shows the flat by-month profile, lag features, a
  chronological hold-out of the last decade, rolling-origin one-step CV
  (`cross_validate_by_time` again), the model persisted and both forecasts scored
  from the reloaded copy. Held-out (one-step-ahead) MAE 15.3 / r² 0.88 — strong;
  the recursive multi-step forecast over 120 months decays honestly (MAE 52.5 /
  r² −0.35, error compounds) yet still beats seasonal-naive (58.1) and naive-last
  (63.1). Scope finding recorded: a pure-AR forecaster needs no `ds.pipeline`
  scoring `Pipeline` (lags stateless, series complete, swings are signal, OLS
  scale-free), so only the model is persisted. A `ds.features` / `ds.modeling`
  addition, so the top-level public surface (`tests/test_public_api.py`) is
  unchanged. `ROADMAP.md` records the P16 plan-of-record entry and the `sunspots`
  friction backlog (items 30–31, both served).
- `ds.eda.target_rate_by_category(df, column, target, *, min_count=1)` and the
  paired `ds.viz.plot_target_rate` — the categorical counterpart to
  `top_correlations`, which is numeric-only. The helper returns, per level of a
  categorical column, its `count`, `frac`, the mean `target` within the level
  (the "target rate" — a 0/1 target's positive rate, or any numeric target's
  group mean) and the overall `baseline`, sorted by target rate descending, with
  missing values kept as their own level. It is *descriptive*, not a fitted
  feature (a target rate fed back as a model input is textbook leakage; the
  docstring says to compute it on the training split when it informs a
  decision). The plot draws per-level bars with a dashed baseline reference line,
  per the reuse-across-stages rule. This closes `ROADMAP.md` friction item 29
  (the `ds.eda` categorical↔target gap, parked in P15): `eda` was the one thin
  general-purpose stage and four of seven projects are heavily-categorical
  classification, so the signal had fired repeatedly. The
  positive-rate-by-level *group table* was chosen over the Cramér's-V /
  mutual-information *ranker* (a different question — which level, not which
  column — and no project has hand-rolled it). A `ds.eda` addition, so the
  top-level public surface (`tests/test_public_api.py`) is unchanged.
  `projects/adult_income` adopts it in its Explore stage (>50K rate by
  `marital_status` and `occupation`) to prove it earns its place; its end-to-end
  test passes unchanged.

### Changed
- Split `ROADMAP.md` into a lean live roadmap plus a history archive to control
  per-session token cost: the file had grown to ~1,440 lines / ~98 KB and
  `CLAUDE.md` told every session to read it before library work. `ROADMAP.md`
  now keeps only the live parts — the capability-per-stage table ("Where things
  stand"), the demand queue, and the working agreement (~75 lines). The
  completed plan of record (P1–P17), the goal evaluation, the per-project
  friction backlogs (items 1–31, still referenced by number from project code),
  and the settled-decision rationales moved verbatim to the new
  `ROADMAP_ARCHIVE.md`. Cross-references that pointed at the moved content
  (`CLAUDE.md`'s Roadmap section and its API/CLI/fetch rationale pointers,
  `README.md`'s planned-extras note, `pyproject.toml`'s extras comment) now
  point at the archive; `CLAUDE.md`'s "read it before library work" instruction
  became "read the small `ROADMAP.md`; grep the archive by item number when you
  need the why." No plan or decision was changed — only relocated. The archive
  lives at repo root (outside `docs/`) so `mkdocs build --strict` is unaffected.
- Bumped pre-commit hook pins so they stop drifting from the dev environment:
  `ruff-pre-commit` v0.6.9 → v0.15.21 (now matching the `ruff` version
  `uv.lock` resolves, with the `dev` group's `ruff>=` floor raised to the same
  version so a future `uv lock --upgrade` can't reopen the gap),
  `pre-commit-hooks` v4.6.0 → v6.0.0, and `nbstripout` 0.7.1 → 0.9.1. Absorbed
  the resulting reformat (`tests/test_public_api.py`); no new lint rule needed
  a fix or an ignore.

## [0.2.0] - 2026-07-18

### Added
- Weekly `deps-upgrade` GitHub Actions workflow
  (`.github/workflows/deps-upgrade.yml`, also `workflow_dispatch`): runs
  `uv lock --upgrade` and, when the lock moves, opens a single reviewable PR with
  the refreshed lock and the result of an inline `make check` against it. This
  restores automated Python-dependency *visibility* after dependabot was scoped to
  GitHub Actions only — in a lock-consistent way dependabot's pip stream could
  not (it can't regenerate `uv.lock`). Nothing auto-merges; a red inline check on
  breaking majors is expected and is the point.
- `ds.io.fetch_dataset(name, urls, *, sha256, settings=None)` (P15 — serving
  `adult_income` friction item 27): download a raw file into the project's
  raw-data directory, trying each mirror in order and verifying the payload's
  SHA-256 against a pinned digest before writing it, so a silently drifted mirror
  fails loudly rather than parsing strangely downstream. A cached copy is
  re-verified (not trusted) so a partial earlier download can't poison later runs.
  The checksum is **required** (keyword-only): the helper exists for the
  multiple-personal-mirror case where pinning the exact bytes is the whole point,
  and the seaborn-mirror projects' plain un-pinned "download if absent" stays
  inline (below the aliasing bar) rather than being folded in. Stdlib only
  (`urllib`/`hashlib`), so no dependency was added; a `ds.io` addition, so the
  top-level public surface (`tests/test_public_api.py`) is unchanged. `air_quality`
  and `adult_income` — the two projects that had hand-rolled the ~25-line
  multi-mirror/checksum/cache-reverify dance verbatim (the trigger `air_quality`
  recorded) — both deleted that body for a one-line binding to `fetch_dataset`,
  with their end-to-end tests passing unchanged (held-out metrics and persisted
  artifacts equivalent). ROADMAP records the P15 plan-of-record entry and strikes
  friction items 27 (served), 28 (the string `"?"` sentinel stays a documented
  one-liner now that two differently-typed sentinels are on record — a third
  would force a thin helper) and 29 (the categorical↔target EDA view struck and
  parked, no code workaround to promote), with "Next up" set to the eighth demand
  loop; CLAUDE.md gains an engineering note on the verified-fetch helper.
- `projects/adult_income`: seventh **real-data** project (P14 — the seventh run
  of the demand loop, and the first **heavily-categorical** one). Predicts
  whether a 1994 US census respondent earns over $50K from the 32,560-row
  training split of the UCI Adult / Census Income dataset (downloaded once into
  git-ignored `data/raw/` from a GitHub mirror, verified by pinned sha256
  because the UCI archive is not reachable from every network). Chosen — by the
  established rule of grepping which surfaces still had no real consumer —
  because the whole *categorical* cluster was thin or unused, and a
  high-cardinality tabular dataset stresses it by absence. First real consumers
  earned: `fit_one_hot_categories(drop_first=True)` (the dummy-trap guard the
  wide indicator matrix needs for a full-rank linear model) and `flag_outliers`
  on its *flag*-not-clip path (the ~92%-zero `capital_gain`/`capital_loss` are
  flagged for the record but kept — clipping would erase the large-gain signal
  that most cleanly separates the classes). Second consumers:
  `fit_topk_categories`/`collapse_categories` (the 41-country `native_country`
  and 14-trade `occupation` tails collapsed to `"other"` before one-hot) and
  the `labels=` display mapping (binary `{0: "<=50K", 1: ">50K"}`). Full
  lifecycle on `ds` + scikit-learn: checksum-verified fetch, boundary
  validation (`assert_row_count`, `check_schema` numeric-dtype pin), the `"?"`
  sentinels decoded to NaN and the target encoded 0/1, stratified split, a
  four-step fit plan (collapse tails / mode-impute the sentinels / wide one-hot
  with `drop_first` / scale) fitted on the training split and re-fitted inside
  every stratified CV fold, pipeline + model persisted and the held-out split
  scored from reloaded state. Held-out (>50K = positive): accuracy 0.861 /
  F1 0.679 / precision 0.763 / recall 0.611 vs an interpretable
  marital-status rule (0.713 / 0.590) and the majority class (0.759 / 0.000);
  CV F1 0.653 ± 0.007. `load_raw`'s `skipinitialspace=True` forwarding stripped
  the value whitespace without a project-local loader (third kwargs-forwarding
  consumer). Per the demand-first rule the project promotes nothing itself; its
  friction list is `ROADMAP.md` items 27–29 — led by the checksum-verified
  multi-mirror fetch, now hand-rolled a second time (air_quality's fetch-helper
  trigger fires), and the `"?"` sentinel decode (air_quality item 26's
  second-sentinel-project trigger fires, string flavor).
- The `air_quality` friction backlog served (P13 — `ROADMAP.md` items 22–26 in
  observed-pain order; three served, two resolved by documentation), each
  dogfooded by `projects/air_quality` (and `projects/flights` for the shared
  guard) in the same change, held-out metrics and persisted artifacts verified
  equivalent:
  - `cross_validate_by_time(make_pipeline=...)` (item 22, the headline —
    item 9's parked question): the rolling-origin twin of
    `cross_validate_kfold`'s factory, re-fitting the same `FitStep` plan per
    fold on each fold's expanding training window only. `air_quality` deleted
    its hand-rolled `_fold_fit_state` boundary reproduction (the dogfood
    proof). Unlike item 9's titanic finding, the effect is real on this data —
    the per-fold impute medians swing ~28%, so the leak-free protocol
    measurably moves the CV numbers (mean 0.406 → 0.409) while held-out metrics
    and every other persisted artifact stay byte-identical.
  - `ds.validation.assert_row_count(df, expected)` (item 25, item 20's
    second-project trigger) and `ds.validation.assert_unique(df, column)`
    (item 24, item 13's second-project trigger): two fluent guards. Each adds
    the stage-consistent `DataValidationError`; `assert_unique` also adds a
    correctness check raw `pd.to_datetime` doesn't do (a duplicated key a later
    sort would silently interleave). `flights` and `air_quality` both call
    `assert_unique` after their time-axis parse; `air_quality`'s `trim_raw`
    closes with `assert_row_count`.
- `fit_baseline`'s `"naive_last"`/`"seasonal_naive"` docstring now documents
  that the naive strategies align *positionally* and so assume a gapless axis;
  on a gapped axis, align by timestamp instead (item 23, resolved by
  documentation).
- Guide Acquire-section gotcha on sentinel-coded missingness: decode it with a
  post-parse numeric replace *before* validation/EDA, not a read-time
  `na_values=` that silently misses decimal-comma spellings (item 26, resolved
  by documentation).
- `projects/air_quality`: sixth **real-data** project (P12 — the sixth run of
  the demand loop, and the first against real instrument-outage missingness
  on a gapped hourly axis). Reconstructs the reference CO analyzer's reading
  (`co_gt`, mg/m³) from the rest of a road-level monitoring station on the
  UCI Air Quality dataset (9,357 hourly rows, an Italian city, March 2004 to
  April 2005; downloaded once into git-ignored `data/raw/` from two
  byte-identical GitHub mirrors, verified by pinned sha256 because the UCI
  archive is not reachable from every network). Chosen — by the established
  rule of grepping which surfaces still had no real consumer, weighted toward
  the open watch-list — for its real missingness (−200 sentinels: one column
  90.2% missing, the target 18.0%, the NOx/NO2 feature channels independently
  gapped), its silently-failing parse (semicolons + decimal commas + trailing
  junk), and a rolling-origin CV whose per-fold fitted state genuinely varies.
  First real consumers earned: `assert_dtypes` (the parse pin that makes the
  decimal-comma misparse loud) and the impute surface at real severity
  (`fit_impute_values`/`apply_impute_missing` through `fit_pipeline` over
  genuinely cell-level gaps). Full lifecycle on `ds` + scikit-learn:
  checksum-verified fetch, expected-row-count check, a three-way missingness
  triage (drop the 90% column, drop 366 device-offline hours, drop 1,647
  unlabeled hours, impute the partial remainder), hand-assembled hourly time
  axis, a three-step fit plan (median impute / 24-level hour one-hot /
  standardize), rolling-origin CV with a companion table measuring the
  per-fold fitted state the single up-front transform cannot re-fit, pipeline
  + model persisted and the held-out window scored from reloaded state.
  Held-out: MAE 0.305 / RMSE 0.458 / r² 0.876 vs the same-hour-yesterday
  reading (MAE 0.799, r² 0.223) and the training mean (MAE 1.068, r² −0.030);
  rolling-origin CV MAE 0.406 ± 0.111. Per the demand-first rule the project
  promotes nothing itself; its friction list is `ROADMAP.md` items 22–26, and
  item 22 finally fires item 9's parked `cross_validate_by_time(make_pipeline=...)`
  trigger.
- The `sms_spam` friction backlog served (P11 — `ROADMAP.md` items 18–21 in
  observed-pain order: one built, one resolved by documentation, two struck;
  the served item dogfooded by `projects/sms_spam` in the same change):
  - `ds.modeling.nlp.count_tokens` now resolves which counting path is live
    **once per process** per `model` (item 19): the encoding probe is
    memoized, caching success *and* failure, so with tiktoken installed but
    its vocabulary endpoint unreachable only the first call pays the failed
    download attempt (previously ~0.4 s on *every* call — a ~35-minute stall
    over the sms_spam project's 5,171 messages, hit live in CI-like
    sandboxes) and a process never mixes BPE and whitespace counts mid-run.
    The other candidate shape recorded for the item — exposing the probe —
    was deliberately not built: with the stall gone, no consumer needs to
    pick a counter or see which path ran. `projects/sms_spam` deleted its
    hand-rolled `_resolve_token_counter` guard and calls the library
    directly; no-extras artifacts verified byte-identical (sha256), held-out
    metrics unchanged (accuracy 0.968 / spam F1 0.864). Both paths are
    pinned by deterministic fake-`tiktoken` tests that need no network and
    hold in the no-extras and `--extra all` CI jobs alike.
  - Item 18 (no vectorization step kind in the `ds.pipeline` vocabulary) —
    **resolved by documenting the convention, not building from one
    consumer**, per the item's own warning: model-side transforms (a
    transform whose fitted state manufactures its column space, like a text
    vectorizer's learned sparse vocabulary) live inside the estimator and
    persist via `save_model`, while the ds `Pipeline` carries the
    frame-shaped steps around it. Recorded as `ds.pipeline`'s fourth
    module-docstring design point and a Guide paragraph; a second text
    project decides whether a first-class vectorize step earns a build.
  - Items 20 (an `assert_row_count`-style boundary check) and 21 (text
    feature helpers) — **struck, not built**: a one-line row-count guard and
    two well-documented count lines, each observed once, are below the
    helper bar (the item-13/15 precedent); the second-project triggers are
    recorded in `ROADMAP.md`.
- `projects/sms_spam`: fifth **real-data** project (P10 — the fifth run of
  the demand loop, and the first **text** one). Flags the 5,574 labelled
  messages of the SMS Spam Collection (~13% spam; single headerless TSV
  mirrored in the pycon-2016-tutorial repo, downloaded once into git-ignored
  `data/raw/`) as ham or spam, chosen — by the established rule of grepping
  which surfaces still had no real consumer — because
  `ds.modeling.nlp.count_tokens` and the `nlp`/tiktoken extra were the
  library's only entirely-unconsumed module and a text pipeline stresses the
  text gaps by absence. First real consumers earned: `count_tokens`
  (per-message `token_count`, deliberately descriptive-only — its graceful
  degradation makes the values extras-dependent, which bars it from the
  modeling path; the decision is recorded in the pipeline docstring) and the
  headerless-file path through `load_raw`'s pandas-kwargs forwarding
  (`header=None, names=..., quoting=csv.QUOTE_NONE` — the file's quotes are
  message text, and default quoting silently swallows two rows). Second
  consumers: the `labels=` display mapping (binary `{0: "ham", 1: "spam"}`)
  and `bin_column` (label mix per length quantile band). Full lifecycle on
  `ds` + scikit-learn: boundary validation, exact duplicates dropped as
  split-leaking verbatim repeats, hand-rolled length features, a one-step
  fit plan (standardize `char_count` — the only step the closed `StepKind`
  vocabulary can express here; the TF-IDF vectorizer lives inside the
  persisted sklearn model instead), stratified 5-fold CV with the plan
  re-fitted per fold, pipeline + model persisted and the held-out split
  scored from reloaded state. Held-out (spam = positive): accuracy 0.968 /
  F1 0.864 vs a hand-written keyword rule (0.916 / 0.659) and the majority
  class (0.873 / 0.000). Per the demand-first rule the project promotes
  nothing itself; its friction is recorded as items 18–21 of `ROADMAP.md`'s
  backlog (no vectorization step kind in the pipeline vocabulary,
  `count_tokens`' per-call graceful degradation stalling its very first
  consumer for ~35 minutes when tiktoken is installed but its vocabulary is
  unreachable, a silently-wrong boundary parse only an out-of-band row
  count caught, and no text helpers in the features stage).
- The `diamonds` friction backlog served (P9 — `ROADMAP.md` items 14–17, in
  observed-pain order; the served items dogfooded by `projects/diamonds` in
  the same change, with every persisted artifact — CSVs *and* PNGs —
  verified byte-identical to the pre-change run, held-out accuracy 0.655 /
  macro F1 0.551 untouched):
  - `labels=` display mapping (item 14) — an optional `Mapping[int, str]` on
    `ds.evaluation.confusion_frame` / `ds.evaluation.per_class_metrics` /
    `ds.viz.plot_confusion_matrix` that puts class *names* on the
    consumer-facing axes and tick labels; the metric math stays on the int
    codes, codes absent from the mapping keep their integer form, and
    `fit_baseline`'s numeric-label contract is untouched. With `labels=` the
    confusion plot rotates its x-tick names 45° so long names stay legible;
    without it, behaviour is unchanged. The project deleted its `_named()`
    and `_relabel_confusion_axes()` workarounds.
  - Item 15 (a row-filtering counterpart to `assert_in_range`) — **struck,
    not built**: a three-line mask observed once is below the helper bar;
    the second-project trigger is recorded in `ROADMAP.md`.
  - Template fixes (item 16) — the `ds new` scaffold now matches the shape
    every real pipeline keeps: the stub `run()` takes
    `settings: Settings | None = None`, the scaffolded test injects a
    temporary data directory instead of writing into the shared data tree,
    the scaffolded README and module docstring say `ds run <slug>` (what
    `ds new` itself prints), and an empty description renders cleanly with
    no dangling "—".
  - Item 17 (the multiclass metrics wrapper) — **resolved by documenting
    the idiom, not a helper**: bind
    `functools.partial(classification_metrics, average="macro")` for the
    two-argument `metrics_fn` hooks, recorded in the `average` parameter's
    docstring and the docs Guide; the project's wrapper `def` is now that
    one-liner.
- `projects/diamonds`: fourth **real-data** project (P8 — the fourth run of
  the demand loop, and the first **multiclass** one). Grades the cut of the
  53,940 classic ggplot2 diamonds (seaborn-data mirror; downloaded once into
  git-ignored `data/raw/`) into the five ordered classes Fair < Good <
  Very Good < Premium < Ideal, chosen — after grepping which helpers still
  had no real consumer — for quirks that pull several untouched surfaces at
  once. First real consumers earned: `fit_ordinal_categories` /
  `apply_ordinal_encode` with the explicit `categories=` domain ordering
  (color J→D, clarity I1→IF), `bin_column` (cut mix per carat quantile
  band), `plot_outliers` on non-synthetic data, and the multiclass metric
  surface (`confusion_frame` / `per_class_metrics` / `plot_confusion_matrix`
  at 5×5, `classification_metrics(average="macro")`). Full lifecycle on
  `ds` + scikit-learn: boundary validation with the three grade
  vocabularies, the physically impossible zero-dimension rows dropped, exact
  duplicates dropped as split-leaking re-entries, a three-step fit plan
  (clip the measurement-error dimension columns only, ordinal-encode,
  scale) fitted via `fit_pipeline`, stratified 5-fold CV with the plan
  re-fitted per fold, pipeline + model persisted and the held-out split
  scored from reloaded state. Held-out: accuracy 0.655 / macro F1 0.551 vs
  a proportions-only grading rule (0.546 / 0.357) and the majority class
  (0.400 / 0.114). Per the demand-first rule the project promotes nothing
  itself; its friction list is the new `ROADMAP.md` backlog (items 14–17:
  the label-blind metric surface, validation that asserts but cannot
  filter, template-stub divergences from what real projects keep, and the
  macro-average wrapper every multiclass call site rewrites).
- The `flights` friction backlog served (P7 — `ROADMAP.md` items 10–13, in
  observed-pain order, each dogfooded by `projects/flights` in the same
  change with equivalent held-out metrics):
  - `ds.viz.plot_series` (item 10) — the stage's first time-axis plot: a
    solid observed line plus optional dashed, named prediction overlays
    (the standard forecast-vs-actual visual), colours drawn from the Axes'
    cycle so repeated calls on one `ax` compose. One helper replaced both
    of the project's hand-rolled matplotlib figures: the raw series is a
    single call, the forecast view two composed calls (training tail, then
    the held-out window with the model and seasonal-naive overlays).
  - `ds.features.add_datetime_features` gains `features=` (item 11) — an
    explicit selection of which datetime features to emit, in a fixed
    documented order; the default (the full calendar set) is unchanged.
    Explicit selection was chosen over a resolution-aware default because
    inferring resolution from the frame is fitted state in disguise (a
    later scoring batch can be too small to infer the same answer from)
    and misfires silently. The project's hand-drop of the
    weekday-of-the-1st noise columns is gone — the scoped call never emits
    them.
  - `"elapsed_months"` (item 12) — an opt-in selectable feature of
    `add_datetime_features` (not a second helper: same source column, same
    expansion mechanism, and item 11's parameter already provides opt-in)
    emitting whole calendar months since a fixed epoch (January of year 0,
    `year * 12 + month - 1`) — the monotone trend counter a linear
    forecaster needs. The fixed epoch keeps scoring stateless: nothing is
    learned from the frame, and for a trend term only differences matter.
    Kept out of the default set (a modeling device, near-collinear with
    `_year`); replaces the project's hand-rolled `month_index` with
    equivalent metrics (the counter differs by a constant the intercept
    absorbs).
  - Item 13 (the hand-assembled time axis) — **struck, not built**: one
    `pd.to_datetime(..., format=)` call plus a project-specific uniqueness
    check, observed once, is below the helper bar. Won't build until a
    second project repeats the pain; rationale recorded in `ROADMAP.md`.
- Third real-data project, and the first **forecasting** one:
  `projects/flights` forecasts the 144 monthly international-airline-
  passenger totals, 1949–1960 (the classic Box & Jenkins series,
  seaborn-data mirror), chosen for a genuine time axis with strong yearly
  seasonality — the first data to stress the time-series surface.
  `train_test_split_by_time` gains its second consumer;
  `cross_validate_by_time` (rolling-origin folds) and `fit_baseline`'s
  `"naive_last"`/`"seasonal_naive"` strategies gain their first real ones.
  Full lifecycle on `ds` + scikit-learn: a hand-assembled time axis (year +
  month-name → one sorted datetime column, duplicates rejected), calendar
  features with the monthly-resolution noise dropped (`drop_constant_columns`
  catches the constant half), a hand-rolled `month_index` trend, a one-step
  fit plan (the month one-hot vocabulary) persisted as the scoring
  `Pipeline`, the model persisted and the held-out window scored from the
  reloaded copy, and a linear trend + month-effects model evaluated against
  both naive references on the strictly future 29-month window: MAE 34.3 vs
  seasonal-naive 64.8 and naive-last 81.4 (thousands of passengers), r² 0.63
  — honest about an additive model under multiplicative seasonality. Per the
  demand-first rule the project promotes nothing; its friction is recorded
  as items 10–13 of `ROADMAP.md`'s new backlog (P6 of the plan of record),
  led by the missing time-series plot in `ds.viz`. Item 9's parked
  `cross_validate_by_time(make_pipeline=...)` question stays parked —
  measured on this project, per-fold re-fitting would re-learn the identical
  month vocabulary on every fold.
- The fit-side design pass (`ROADMAP.md` friction items 5 + 9 — the deferred
  reopening of the settled pure-composition decision; the amended rationale
  is recorded there):
  - `ds.pipeline.fit_pipeline` + `FitStep` — execute an ordered fit plan
    (each entry a step kind plus a fit callable closing over that transform's
    keyword arguments) as the fit → apply → fit chain both real projects
    hand-strung, returning the assembled `Pipeline`. `Pipeline` itself is
    unchanged — still pure composition, persistence untouched. Both
    `projects/nyc_taxis` and `projects/titanic` replaced their five-pair
    dance with a plan; persisted scoring pipelines and held-out metrics came
    out byte-identical.
  - `ds.evaluation.cross_validate_kfold` gains `make_pipeline` — a
    per-fold pipeline factory (the `make_model` twin, typically
    `lambda frame: fit_pipeline(frame, plan)`) so the transform chain is
    re-fitted on each fold's training rows only, closing the leak where
    every fold's test rows influenced the imputation/scaling statistics.
    `titanic` now cross-validates its raw training split with the same plan
    it fits the scoring pipeline from. Honest finding, recorded in the
    ROADMAP strike: the fold statistics genuinely change but not one fold
    prediction flips on this frame/model, so the per-fold metrics are
    unchanged — the protocol is sound either way. `cross_validate_by_time`
    deliberately keeps no such parameter until a project pulls it.
- Classification-shaped Model/Evaluate helpers, serving friction items 6–8 of
  `ROADMAP.md`'s `titanic` backlog (demand-first, smallest win first; item 9
  stays queued for its own design pass):
  - `ds.modeling.fit_baseline` gains `strategy="majority"` (item 6) — predict
    the modal training label (nulls ignored, ties to the smallest label), the
    classification twin of `"mean"`. Scoped to numeric labels (the int-coded
    0/1 target that raised the item); string labels raise a clear error until
    a project demands them. `titanic`'s hand-rolled `y_train.mode()` reference
    is gone, with byte-identical held-out metrics.
  - `ds.modeling.tabular.train_test_split_random` (item 7) — the order-free
    twin of `train_test_split_by_time`: shuffled, optionally stratified on a
    named column whose class balance both halves preserve, seeded through
    numpy's global generator like the rest of the stage (so `seed_everything`
    reproduces the split). Replaces `titanic`'s raw scikit-learn
    `train_test_split(stratify=...)` call, byte-identically.
  - `ds.evaluation.cross_validate_kfold` gains `stratify=True` (item 8) —
    `StratifiedKFold` under the flag, composing with
    `metrics_fn=classification_metrics`; `titanic` passes it. Honest finding,
    recorded in the ROADMAP strike: it fixes the fold class-balance drift it
    controls (per-fold positive counts now within ±1, was a ~15-row spread)
    but the per-fold recall drift the backlog blamed on that imbalance did
    not shrink (~0.13 spread with and without, across 30 seeds) — that drift
    is sampling variance in *which* positives land in a fold.
- Second real-data project, and the first classification one:
  `projects/titanic` predicts passenger survival on the classic 891-row
  manifest (seaborn-data mirror), chosen for what it stresses — missing
  values at three severities (`age` ~20%, `deck` ~77% → a `deck_known`
  indicator, `embarked` two rows), the target respelled as a feature
  (`alive`, verified one-to-one before dropping), derived duplicate columns,
  a skewed `fare`, and 107 duplicate rows that are distinct passengers (kept,
  deliberately). Full lifecycle on `ds` + scikit-learn: stratified split, a
  five-step persisted scoring `Pipeline` (including two `impute_missing`
  steps with different strategies), the model persisted and the held-out
  split scored from the reloaded copy, and the previously untouched
  classification surface exercised end to end — `classification_metrics`,
  `confusion_frame`, `per_class_metrics`, `plot_confusion_matrix`, and the
  first real composition of `cross_validate_kfold` with
  `metrics_fn=classification_metrics`. Held-out accuracy 0.799 / F1 0.731
  vs the sex-only rule (0.777 / 0.692) and the majority class (0.615 / 0.0).
  Per the demand-first rule the project promotes nothing; the friction it
  surfaced is recorded as items 6–9 of `ROADMAP.md`'s backlog (P5 of the
  plan of record).
- High-cardinality categorical strategy in `ds.features` (friction item 4 of
  `ROADMAP.md`'s backlog — the last open item from the real-data `nyc_taxis`
  project): `fit_topk_categories`/`apply_collapse_categories` (plus the
  single-call `collapse_categories`) keep each column's `k` most frequent
  levels and collapse the rest — including values unseen at fit time — into
  an `"other"` label, so the existing one-hot/ordinal encoders handle the
  now-small vocabulary. The fitted `TopKCategories` round-trips through
  `to_dict`/`from_dict` like the other parameter dataclasses and joins
  `ds.pipeline` as the `"collapse_categories"` step kind. `nyc_taxis`
  dogfoods it on the ~200-level zone columns it originally had to drop:
  vs a boroughs-only variant, held-out MAE improves 2.62 → 2.26 (−14%) and
  r² 0.729 → 0.765.
- Model/Evaluate build-out (P3 of `ROADMAP.md`'s plan of record, re-ranked
  against the `nyc_taxis` friction backlog first):
  - `ds.features.add_datetime_features` now also emits ``<column>_hour``
    (friction item 2 — hour of day was the taxi data's strongest temporal
    signal and had to be hand-rolled; on date-only data the column is
    constantly zero, where `drop_constant_columns` removes it).
  - `ds.modeling.baseline`: `fit_baseline` (``mean`` / ``naive_last`` /
    ``seasonal_naive``) returns a frozen `Baseline` whose `predict(n)` feeds
    `ds.evaluation` directly — the reference point every first metric needs
    (friction item 3), deliberately not a scikit-learn estimator since a
    baseline needs no feature matrix.
  - `ds.evaluation.cross_validate_by_time` (rolling-origin folds — every
    test window strictly in its training data's future, the repeated-fold
    counterpart to `train_test_split_by_time`) and `cross_validate_kfold`
    (order-free data), both building a fresh model per fold via a
    `make_model` factory and scoring with a `metrics_fn` that defaults to
    `regression_metrics` (pass `classification_metrics` or a custom scorer
    to compose instead of forking the API).
  - `ds.evaluation.compare_models` (one row of metrics per named model) and
    its paired `ds.viz.plot_model_comparison`, per the stage↔viz
    convention.
  - `projects/nyc_taxis` dogfoods it all: library `pickup_hour`,
    `fit_baseline` and a persisted comparison frame/plot replace its
    hand-rolled versions with identical metrics.
- `ds.modeling.persistence`: `save_model`/`load_model` persist a fitted
  estimator with `joblib` (now a declared core dependency — it was already
  present transitively via scikit-learn), completing the fit-once/score-later
  story: fitted transform parameters travel as validated JSON
  (`save_params`/`load_params`) and the model as a joblib file, so a later
  run reloads both and scores with no refitting or in-memory carryover.
  `load_model` documents the unpickling trust boundary (only load files you
  or a trusted process wrote — the reason parameters deliberately stay in
  JSON). Both worked projects now run the reload loop end to end, closing the
  top item of the `nyc_taxis` friction backlog; the guide's Model section
  documents the pattern. Next up per `ROADMAP.md`: the Model/Evaluate
  build-out (P3).
- `projects/nyc_taxis`: the first project on **real** data — fare prediction
  over the 6,433-ride March-2019 NYC taxis sample (seaborn `taxis`, mirrored
  from the NYC TLC trip records; downloaded once into git-ignored
  `data/raw/`). It runs the full lifecycle on `ds` + scikit-learn alone, with
  the split-safe transforms persisted as one scoring `Pipeline` and the model
  evaluated against a naive train-mean baseline, and exists to drive the
  hybrid workspace's demand loop: the library friction it surfaced (no `hour`
  from `add_datetime_features`, no high-cardinality encoder, no model
  persistence, no baseline estimators) is recorded as the friction backlog in
  `ROADMAP.md`.
- `docs/guide.md` cookbook gained three cross-stage recipes for combinations
  the per-stage walkthrough didn't cover: validating right after
  `load_raw`/`load_table` with `check_schema` (acquire + validate), using
  `top_correlations` to screen out redundant features before
  `scale_features` (explore + feature), and fitting a scikit-learn estimator
  and closing the loop with `regression_metrics` and `plot_residuals` (model +
  evaluate + visualize). Closes the "Docs cookbook" item in `ROADMAP.md`.
- `ds run <name>`: run a project's `pipeline.py` by name. It resolves the name
  against the real directories under `projects/`, matching the literal name or
  the same slug `ds new` derives (`"Customer Churn"`, `customer_churn` and
  `customer-churn` all reach `projects/customer_churn/`; `_example` is reachable
  as `example`), then runs that project with the current interpreter and
  propagates its exit code. A miss lists the runnable projects. Because it
  selects an existing `projects/` entry rather than building a path from the
  name, a traversal attempt like `../evil` matches nothing — the same
  path-traversal discipline as `ds new`'s slug. The decision to add `ds run`
  (and to *reject* `ds check` as a redundant shell-out to `make check`) is
  recorded in `ROADMAP.md` and `CLAUDE.md`; `docs/guide.md` and `README.md`
  document the command.
- `tests/test_public_api.py` pins the curated top-level `ds` surface
  (`Settings`, `get_settings`, `get_logger`, `seed_everything`, `__version__`),
  asserts stage helpers and `ds.pipeline.Pipeline` are *not* re-exported, and
  checks that `import ds` stays cheap (loads no matplotlib/scikit-learn).
- `ds.pipeline`: a composable fit-once/apply-many `Pipeline` over the
  `fit_*`/`apply_*` pairs. A `Pipeline` holds an ordered tuple of
  `PipelineStep`s — each pairing fitted parameters with the `apply_*`
  transform it means via a `kind` tag (which is how one `OutlierBounds`
  serves both `"clip_outliers"` and `"flag_outliers"`; the flag form adds
  boolean `<column>_outlier` columns) — applies them all with one
  `apply(df)` call, and persists through the existing
  `ds.io.save_params`/`load_params` machinery by delegating to the per-class
  `to_dict`/`from_dict` round-trips. Step order and same-type duplicate
  steps survive the round-trip; `from_dict` fails with an error naming the
  offending step on unknown kinds or malformed/stale payloads. Train-time-only
  parameters (target-column fits) deliberately stay out of a pipeline —
  see the new "Compose the applies into one pipeline" cookbook section in
  `docs/guide.md`.
- Persistable fit parameters: every `fit_*` dataclass (`OutlierBounds`,
  `ImputeValues`, `ScaleParams`, `OneHotCategories`, `OrdinalCategories`) now
  has a validated `to_dict`/`from_dict` round-trip, and `ds.io` gained
  `save_params`/`load_params` (plus the `FittedParams` protocol) to persist
  them as strict JSON — so a pipeline can save its fitted state alongside the
  model and score new rows in a later run or another process. Numpy scalar
  fills are stored as plain numbers, non-finite bounds as tagged mappings
  (JSON has no `inf`/`nan` literal), and vocabulary tuples round-trip through
  lists; `from_dict` validates the payload so a stale, hand-edited or
  wrong-type file fails with a clear error. Shared encode/decode plumbing
  lives in the private `ds._serde` module.
- Split-safe `fit_*`/`apply_*` pairs for every statistic-learning transform,
  each returning/consuming a small frozen parameters dataclass so statistics
  can be fitted on the training split and reused on test data or new rows:
  `ds.preprocessing.fit_outlier_bounds` (`OutlierBounds`) with
  `apply_clip_outliers`/`apply_flag_outliers`, `fit_impute_values`
  (`ImputeValues`) with `apply_impute_missing`, and
  `ds.features.fit_scale_params` (`ScaleParams`) with `apply_scale_features`,
  `fit_one_hot_categories` (`OneHotCategories`) with `apply_one_hot_encode`,
  `fit_ordinal_categories` (`OrdinalCategories`) with `apply_ordinal_encode`.
  The fixed category vocabulary guarantees identical encoded columns on every
  frame (unseen categories → all-zero indicators / `-1` codes). The existing
  single-call forms are now thin fit-and-apply-on-the-same-frame wrappers with
  unchanged behavior. `docs/guide.md` gained a "Fit on train, apply to test"
  cookbook section.
- `ROADMAP.md` and an "Engineering notes" section in `CLAUDE.md` capturing the
  planned stage build-out and hard-won tooling gotchas.
- `ds.eda`: `missing_value_report` (columns with gaps, ranked) and
  `top_correlations` (most-correlated numeric pairs, for redundancy/leakage
  screening).
- `ds.evaluation`: `confusion_frame` (labeled confusion matrix) and
  `per_class_metrics` (per-class precision/recall/f1/support).
- `ds.viz` plotting helpers that return a matplotlib `Axes`: `plot_missingness`,
  `plot_confusion_matrix` and `plot_residuals` (the first two visualize
  `missing_value_report` and `confusion_frame`).
- `ds` command-line interface (`ds version`, `ds new <name>`); `ds new`
  scaffolds a project from the Copier template.
- CI now runs a `test-extras` job that adds the lightweight `tiktoken`
  dependency and re-runs the suite, exercising the accurate token-counting path
  instead of only its fallback (kept lean — it avoids the heavier `nlp`/`all`
  extras that no code exercises yet).
- Documentation `Guide` page: a per-stage cookbook and the new-project workflow.
- Project template now scaffolds a stage-by-stage `pipeline.py` skeleton, a
  per-project `notebooks/` folder, and a `tests/` folder with a starter test.
- `tests/test_example.py` runs the worked `_example` pipeline end to end so a
  regression in any lifecycle stage fails CI.
- `nbstripout` pre-commit hook to strip notebook outputs before they reach git.

### Changed
- CI now installs with `uv sync --locked` in every job, so a `uv.lock` that has
  drifted from `pyproject.toml` fails CI loudly instead of being silently
  re-resolved. Dependabot is scoped to **GitHub Actions only**
  (`.github/dependabot.yml`): it cannot regenerate a uv lockfile, so its Python
  ("pip") PRs would be un-mergeable against the `--locked` gate — Python deps are
  now bumped deliberately with `uv lock --upgrade` and the refreshed lock
  committed. CLAUDE.md records the workflow.
- `ROADMAP.md` restructured around a goal evaluation (2026-07): per-goal
  verdicts, a P1–P4 plan of record (next up: model persistence, then
  Model/Evaluate build-out), the `nyc_taxis` friction backlog, and the settled
  decisions kept with their rationale. A "demand first" rule joins the working
  agreement: new library work traces to real-project friction, not a
  brainstormed candidate list.
- The `test-extras` CI job now installs `--extra all` as declared instead of
  hand-injecting `tiktoken` — cheap by construction now that extras only carry
  consumed dependencies.
- **API discoverability settled: DS keeps the strict import-by-stage
  convention** — no flat top-level re-export of stage helpers, and
  `ds.pipeline.Pipeline` stays `from ds.pipeline import Pipeline`. The top-level
  `ds` namespace re-exports only stage-independent infrastructure, so `import
  ds` stays cheap and stages can't collide in one flat namespace. Documented in
  `docs/guide.md` ("Importing from DS"), `docs/index.md`, `README.md` and
  `CLAUDE.md`; recorded with its rationale in `ROADMAP.md`. No behavior change —
  every existing import keeps working.
- `projects/_example/pipeline.py`'s fresh-rows step now saves and reloads one
  `ds.pipeline.Pipeline` (impute region → encode region → scale calendar
  features) instead of three separate parameter files, keeping the
  train-time-only target-column bounds/fill persisted individually;
  `tests/test_example.py` mirrors the new flow.
- `projects/_example/pipeline.py` now saves all five fitted parameter objects
  next to its processed data with `save_params` and rebuilds them from disk
  with `load_params` to score fresh rows that did not exist at fit time
  (including a missing and an unseen region), closing the "new incoming rows"
  loop; `tests/test_example.py` asserts the reloaded state matches the fitted
  state and the fresh rows encode to the training feature columns.
  `docs/guide.md`'s split-safe cookbook section documents the pattern.
- `projects/_example/pipeline.py` now splits chronologically *before* any
  statistic-learning transform and runs clip → impute → one-hot → scale as
  fit-on-train/apply-to-both via the new `fit_*`/`apply_*` pairs, so the
  held-out window no longer leaks into the learned statistics;
  `tests/test_example.py` asserts the split-safe behavior.
- Package version is now single-sourced from `__version__` in
  `src/ds/__init__.py` via Hatch's dynamic version, instead of being duplicated
  in `pyproject.toml`.
- CLAUDE.md branching guidance no longer hard-codes a (stale) branch name.
- `projects/_example/pipeline.py` now generates realistically dirty synthetic
  data (missing values, outliers, a categorical column, duplicate rows) and
  runs it through `ds.io.load_raw`/`save_processed`, `ds.validation.check_schema`,
  `ds.preprocessing.coerce_dtypes`/`drop_duplicate_rows`/`clip_outliers`/
  `impute_missing`, and `ds.features.one_hot_encode`/`scale_features`, so the
  worked example actually exercises the stage functions fleshed out above
  instead of data too clean to need them. `docs/guide.md` and the project
  template's `pipeline.py.jinja` comment now point at `load_raw` instead of
  `load_table`/`settings.raw_dir`.

### Removed
- Unused dependency pins, so the install surface matches what code consumes:
  `polars` from the core dependencies, the `timeseries` extra entirely
  (`statsmodels`, `sktime` had zero importers), and
  `sentence-transformers`/`anthropic` from the `nlp` extra — which is now
  exactly `tiktoken`, the one extra dependency existing code exercises. A
  dependency is (re-)added in the same change as its first consumer; intended
  future extras live in `ROADMAP.md` until then.

### Fixed
- `ds.modeling.nlp.count_tokens` now falls back to its whitespace estimate when
  `tiktoken` is installed but its encoding cannot be loaded (e.g. the vocabulary
  download fails offline), instead of raising — honoring the module's
  "always callable" contract.
- Release workflow no longer attaches a stray `dist/.gitignore` asset; it now
  uploads only the built wheel and sdist.

## [0.1.0] - 2026-07-15

### Added
- Initial toolkit scaffold: `uv` packaging, `ruff` + `mypy --strict` + `pytest`
  tooling, and GitHub Actions CI.
- `ds` library organized by data-science process: `io`, `validation`,
  `preprocessing`, `eda`, `features`, `modeling` (tabular / timeseries / nlp),
  `evaluation`, `viz`, plus cross-cutting `config`, `logging`, `reproducibility`.
- Hybrid workspace: `projects/` (with a worked `_example` pipeline), a `copier`
  project template under `templates/`, and `notebooks/` + git-ignored `data/`.
- Documentation site (`mkdocs-material` + `mkdocstrings`) and contributor docs.
- Continuous-improvement config: pre-commit hooks, Dependabot, issue/PR templates.
- CI hardening: `pre-commit`-driven lint/type job, an enforced coverage gate
  (`--cov-fail-under`), a strict docs build that validates on PRs and deploys to
  GitHub Pages on `master`, and a release workflow (tag push or manual
  `workflow_dispatch`) that builds the package and publishes a GitHub Release.
