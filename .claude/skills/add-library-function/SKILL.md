---
name: add-library-function
description: >-
  Add one function (or generalize one) in the DS toolkit's src/ds/ library the
  way this codebase expects — correct lifecycle-stage module, Google docstring,
  full type hints for mypy --strict, __all__ export, a mirroring test, the
  fit_*/apply_* split-safe pattern for anything that learns from data, and heavy
  deps behind an extra. Use this whenever the task adds or grows reusable library
  surface: "add X to ds", "the projects keep hand-rolling Y — put it in the
  library", a friction item pulling a helper, extending a stage module, or a
  demand-loop's "fix what bleeds" step. Reach for it even when the user just
  says "add a helper" — this repo has firm conventions (mypy --strict, the
  fit/apply split, import-by-stage) that are easy to violate and cost a CI
  round-trip. Do NOT use it for project pipeline code under projects/ (that
  consumes the library, it doesn't extend it) or for non-library edits.
---

# Add a library function

`src/ds/` is a reusable library organized by **data-science process**, held to
`mypy --strict` and an 85% coverage gate, with a top-level API pinned by tests.
A new helper that ignores the conventions fails CI or drifts the public surface.
This is the recipe (the short version lives in `CONTRIBUTING.md`; the gotchas in
`CLAUDE.md`'s "Engineering notes").

## First: does it belong in the library at all?

The library grows from **real demand**, not speculation — the ordering rule in
`ROADMAP.md`. A function earns its place when a real project (under `projects/`)
has hand-rolled it, or a demand-loop friction item pulls it. If nothing consumes
it yet, don't add it.

And apply the **aliasing bar**: build a library function only when it is a
correctness guarantee, a whole capability class, or fiddly reusable logic — not a
one-/two-line wrapper a caller can write inline. When in doubt, leave it inline in
the project and record the friction. A silent **correctness bug** (wrong output)
is always above the bar.

## Placement — by lifecycle stage, not data type

Put the function in the module for its stage:

`io` → `validation` → `preprocessing` → `eda` → `features` → `modeling` →
`evaluation` → `viz`  (cross-cutting: `config`, `logging`, `reproducibility`).

**Prefer generalizing an existing function** with a keyword-only parameter over a
new name when the need is a generalization of an existing one (e.g. a `group=`
param that makes a transform panel-aware, a `features=` subset selector). A new
public name is a new thing to learn; a new keyword extends a known one. **Reuse
across stages** rather than duplicate — `ds.viz` plots call `ds.eda` /
`ds.evaluation` functions rather than re-deriving.

## The recipe for each new function

1. **Signature + docstring.** Google-style docstring with `Args`/`Returns`/
   `Raises`, and **full type hints** — the codebase is `mypy --strict`, so no
   bare `Any` escapes, annotate return types, and prefer `Sequence`/`Mapping`
   over concrete containers in signatures. Explain the *why* in the docstring
   (when to reach for it, how it differs from its neighbours), not just the what —
   the docstrings are a teaching surface here.
2. **Export it** from the module's `__all__`.
3. **Mirror a test** under `tests/` (mirroring `src/ds/`'s layout). Cover the
   happy path, the documented `Raises`, and — for a correctness fix — a case that
   would *fail on the old behavior*. Keep coverage meaningful (the gate is 85%
   over the whole `ds` package).
4. **Heavy dependency?** Put it in the right extra in `pyproject.toml` (an extra
   carries only deps code actually consumes) and **degrade gracefully when it is
   absent** — see `src/ds/modeling/nlp.py` for the import-guard pattern. Run
   `uv lock` and commit the refreshed `uv.lock` in the same change. The
   `test-extras` CI job installs `--extra all` and re-runs the suite.

## The fit_* / apply_* split-safe pattern (stateful transforms)

If the transform **learns anything from the data** — imputation values, scale
params, a category vocabulary, outlier bounds, top-k levels — it must **not** be a
single function that both learns and applies, because fitting on the test frame
leaks. Split it:

- `fit_<thing>(df, ...) -> <FrozenParams>` learns and returns a frozen dataclass
  of just the learned parameters (JSON-round-trippable — see `src/ds/_serde.py`).
- `apply_<thing>(df, params) -> df` applies previously-learned params, so train
  and test transform identically and an unseen category degrades predictably.
- Often keep a convenience `<thing>(df, ...)` that fits-and-applies in one call
  for exploratory use.

Existing pairs to mirror: `fit_impute_values`/`apply_impute_missing`,
`fit_scale_params`/`apply_scale_features`, `fit_one_hot_categories`/
`apply_one_hot_encode`, `fit_outlier_bounds`/`apply_flag_outliers`. Export the
`fit_*`, `apply_*`, the frozen dataclass, and the convenience name from `__all__`.

**Make it composable into the pipeline** if it is a fit/apply transform: register
the step in `src/ds/pipeline.py` — add the kind to the `StepKind` literal and map
it to its frozen-params dataclass in the type map there — so it can appear in a
`FitStep` plan run by `fit_pipeline`. Read the surrounding entries in
`pipeline.py` before adding one; match them.

## Public-API discipline (settled — don't fight it)

Stage functions are imported **by stage** (`from ds.eda import ...`); only the
stage-independent infra (`Settings`, `get_settings`, `get_logger`,
`seed_everything`) is re-exported from `src/ds/__init__.py`.
`tests/test_public_api.py` pins that exact top-level surface — **do not widen it**
to re-export your new stage function (that test will fail, by design). The package
version is single-sourced via Hatch's dynamic version in `src/ds/__init__.py`.

## mypy --strict + pandas-stubs gotchas

These bite specifically here:

- `DataFrame.corr(method=...)` wants a `Literal["pearson","kendall","spearman"]`,
  not `str`.
- `df.loc[a, b]` returns a broad scalar union that `float()` rejects — index
  positionally via `.to_numpy()` first.
- matplotlib is type-checked (`py.typed`): annotate plot helpers `-> Axes` and
  resolve the optional `ax` explicitly (create one if `None`).

## Verify

```bash
make format        # ruff-format — run before committing, not just `make lint`
make check         # lint + typecheck + test, exactly what CI runs
```

Then keep the docs honest in the **same change**: add the function to
`docs/guide.md`'s stage section and a `CHANGELOG.md` `[Unreleased] › Added`
entry. (If this add is part of a demand loop, the `demand-loop` skill's docs step
covers the roadmap side too.)
