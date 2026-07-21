---
name: demand-loop
description: >-
  Run one demand loop for the DS toolkit — add a new real-data project under
  projects/ chosen to stress the thinnest library surface, let its friction
  regenerate the backlog, and grow src/ds/ only for gaps that friction proves.
  Use this whenever the task is "what's next / the next item", "run a demand
  loop", "add a new project", "extend the library", "pick a dataset", or any
  open-ended "improve the toolkit" request — the loop is THE process by which
  this repo grows, so reach for it even when the user doesn't name it. Also use
  it to answer "what should I work on?" against ROADMAP.md. Do NOT use it for a
  narrow, already-specified change (fix this bug, edit this docstring) — that is
  ordinary work, not a loop.
---

# Demand loop

This repo grows by a disciplined loop, not by a brainstormed feature list. Each
loop adds one **real-data project** under `projects/`, chosen so its data *shape*
stresses a library surface no existing project exercises well. Building it
surfaces genuine friction; you fix only the friction that is a real library gap,
and record the rest. **The regenerated friction backlog is as much the
deliverable as the project.**

Read `ROADMAP.md` (small, live) first — it holds the current state, the demand
queue, and the working agreement. The history (completed plan of record, the
per-project friction backlogs by item number, settled-decision rationales) lives
in `ROADMAP_ARCHIVE.md`; grep it by item number when you need a "why", don't read
it whole.

## The governing principles

Internalize these — every step below serves them.

- **Demand first.** New library work must trace to friction from a *real*
  project, never a candidate list. If nothing pulls a helper, don't build it.
- **Friction is the deliverable.** The point of the project is to *surface* where
  the library fights. Build the pipeline honestly on today's `ds` + scikit-learn
  and let the gaps appear; don't pre-emptively smooth them.
- **The aliasing bar.** Only add a library function when it is *above the bar*: a
  correctness guarantee, a whole capability class, or fiddly reusable logic — not
  a one-or-two-line wrapper a caller can write inline. A silent **correctness
  bug** (wrong output) is always above the bar; a mere convenience usually is not.
  When unsure, do it inline in the project and *record* the friction instead.
- **Fix only what bleeds (default scope).** Land the minimum library change the
  project genuinely forces this loop; defer the rest to the archive with a
  recorded revisit trigger. Confirm scope with the user if a loop tempts more.
- **Keep the roadmap split honest.** New history goes to `ROADMAP_ARCHIVE.md`
  (continuing item numbering); `ROADMAP.md` keeps only the live parts. A CI size
  gate (`tests/test_roadmap_size.py`) fails if `ROADMAP.md` grows past its budget.

## The loop, step by step

### 1. Orient

Read `ROADMAP.md`. If the demand queue names a committed next item, do that. If
it is **empty**, run a fresh loop: the steps below. Either way, confirm with the
user before a large build — the data shape and library scope are theirs to steer.

### 2. Select the data shape (grep-driven)

Find which surfaces are thinnest by *real consumers*, then pick a real dataset
whose shape stresses the thinnest cluster by its absence.

```bash
python .claude/skills/demand-loop/scripts/consumer_matrix.py
```

It prints, per public `ds` name, how many `projects/` consume it, and lists the
`<=1`-consumer surfaces. Read it with judgment the grep can't have (the script
reminds you inline): discount `apply_*` (consumed via `fit_pipeline`) and
`load_table`/`save_table` (via `load_raw`/`save_processed`) — they are not truly
unconsumed. A one-consumer *cluster* (e.g. an EDA→target pair, a forecasting
pair) is a candidate to stress a second time; an entirely *absent data shape*
(a shape no project has — panel/multi-entity, rare-event/probabilistic, grouped)
is the stronger pull. Cross-check against the "deprioritized until pulled" list
at the end of `ROADMAP.md`'s demand queue.

Then **recommend a specific dataset + data shape and confirm with the user**
(shape and library scope are a real fork — use `AskUserQuestion`). Respect
`ROADMAP.md`'s ordering rule: trace the choice to a thin surface, not a wishlist.

### 3. Source & pin the dataset

Real-data projects download through `ds.io.fetch_dataset` against a pinned
third-party mirror (see `sunspots`, `adult_income`, `air_quality`, `bbc_news`,
`store_sales` for the convention: a `raw.githubusercontent.com` mirror URL + a
sha256). **Verify before you trust** — never hardcode an unverified hash:

```bash
curl -sSIL "<url>" | grep -iE "HTTP/|content-length"   # reachable?
curl -sSL "<url>" -o /tmp/probe.csv && sha256sum /tmp/probe.csv && head -3 /tmp/probe.csv && wc -l /tmp/probe.csv
```

A single verified mirror matches most projects; a second byte-identical mirror
(same sha256) slots into the URL tuple. Pin `DATA_URLS`, `RAW_NAME`,
`RAW_SHA256`, and the published `EXPECTED_ROWS` as module constants. The
seaborn-mirror projects (`titanic`/`nyc_taxis`/`diamonds`) keep a plain inline
"download if absent" and are below the `fetch_dataset` bar — don't fold them in.

### 4. Scaffold

```bash
uv run ds new "<name>"        # -> projects/<slug>/ (pipeline.py, tests/, README, notebooks/)
```

The slug collapses non-`[a-z0-9]` runs to `_` (path-traversal safe). This wraps
copier over `templates/project/`.

### 5. Build the pipeline (let friction emerge)

Open the **closest existing project by data shape** and mirror its structure —
that is the fastest way to match house conventions and it is where the toolkit's
idioms live:

| Shape | Mirror |
|-------|--------|
| Tabular regression | `nyc_taxis`, `air_quality` |
| Binary / multiclass classification | `titanic`, `adult_income`, `diamonds` |
| Text | `sms_spam`, `bbc_news` |
| Forecasting (univariate) | `flights`, `sunspots` |
| Panel / multi-entity | `store_sales` |

Run the full lifecycle on `ds` + scikit-learn **only** (projects consume `ds`,
never re-implement it): fetch → validate at the boundary → explore (persist the
profile the modeling rests on) → features → split (chronological for time data)
→ fit the transform plan on train via `ds.pipeline.fit_pipeline` → model →
evaluate against honest baselines → visualize with `ds.viz`. Build it *naively*
first: where a single-series/single-table helper meets your new shape and does
the wrong thing, that is the friction — note it, don't paper over it.

Run it: `uv run ds run <slug>`. Confirm the metrics tell an honest, ordered story
(the model should beat its baselines; if it doesn't, say why — that is a finding,
not a failure).

### 6. Fix only what bleeds

Apply the aliasing bar to each friction point:

- **Above the bar (a correctness gap / capability class)** → add the minimal
  library surface under `src/ds/<stage>/`. Follow the contributor recipe:
  Google-style docstring, full type hints (`mypy --strict`), export from the
  module `__all__`, **add a mirroring test under `tests/`**, heavy deps go in the
  right extra and degrade gracefully. Prefer growing an existing function with a
  keyword-only parameter over a new name when the shape is a generalization (e.g.
  `add_lagged_features(..., group=...)` for panels). Add a test that pins the new
  behavior — for a correctness fix, a test that would fail on the old code.
- **Below the bar (a one-/two-line inline pattern)** → do it inline in the
  project and **record it as deferred friction** in the archive with the trigger
  that would justify building it later (usually "a second project reaches for the
  same pattern").

### 7. Project tests

Mirror the fetch-based test pattern (`sunspots`, `store_sales`): unit tests for
the pure helper functions (parsing, selection, and a guard for the correctness
fix), plus one end-to-end test that downloads via `fetch_raw` and **skips (not
fails) on network error**. Run them (the coverage gate is whole-package, so a
single project runs with `--no-cov`):

```bash
uv run pytest projects/<slug> --no-cov
```

### 8. Keep the docs honest (same change)

Update, in this loop's commit:

- **`ROADMAP.md`** (live only): the "Where things stand" stage row if a surface
  changed, the project roster line/count, and the demand queue (mark this loop
  done, refresh the deprioritized-until-pulled list). Stay under the size budget.
- **`ROADMAP_ARCHIVE.md`**: append a `## Friction backlog (from projects/<slug>)`
  section continuing the item numbering, and a completed plan-of-record `P<N>`
  entry. This is where the *history* goes — never back into `ROADMAP.md`.
- **`CLAUDE.md`**: the project list in the repo tree comment (and the item/P
  range if you closed items).
- **`README.md`**: the `projects/` listing line.
- **`CHANGELOG.md`**: an `[Unreleased] › Added` entry for the project and any new
  library surface.
- **`docs/guide.md`**: only if you added/changed a public library surface.

### 9. Verify & ship

```bash
make format        # ruff-format — CI's lint job runs it; `make lint` alone is not enough
make check         # lint + typecheck + test, exactly what CI runs
uv run pytest projects/<slug> --no-cov
uv run mkdocs build --strict   # only if you touched docs/
```

Then commit to the task's feature branch (never `master`) with a descriptive
message, and push with `git push -u origin <branch>`. Open a PR only if asked.

## Gotchas that cost a CI round-trip

These live in `CLAUDE.md`'s "Engineering notes" — the ones that bite in a loop:

- Run `make format` before committing, not just `make lint` (CI runs
  `ruff-format`; `ruff check` passing doesn't mean the formatter is satisfied).
- After editing any dependency, `uv lock` and commit the refreshed `uv.lock` in
  the same change, and bump the `ruff-pre-commit` rev if `ruff` moved — every CI
  job runs `uv sync --locked`.
- `mypy --strict` + pandas-stubs: `DataFrame.corr(method=...)` wants a `Literal`,
  `df.loc[a, b]` needs positional `.to_numpy()` before `float()`, and plot
  helpers annotate `-> Axes`.
- The public top-level surface is pinned by `tests/test_public_api.py`; stage
  functions import by stage, not from top-level `ds`.
