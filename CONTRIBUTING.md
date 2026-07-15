# Contributing to DS

Thanks for improving the toolkit. This project is designed to stay clean and
current, so a little structure goes a long way.

## Setup

```bash
uv sync                 # environment + library + dev tools
uv run pre-commit install
```

## The one command you need

```bash
make check              # runs lint + typecheck + test, exactly like CI
```

Individual targets: `make lint`, `make format`, `make typecheck`, `make test`,
`make docs`. Run `make help` to list them.

## Adding to the library

The library is organized by **data-science process**, not by data type. When you
add a utility, put it in the module for its lifecycle stage:

`io` → `validation` → `preprocessing` → `eda` → `features` → `modeling` →
`evaluation` → `viz` (cross-cutting: `config`, `logging`, `reproducibility`).

For each new function:

1. Add it under `src/ds/<stage>/` with a Google-style docstring and full type
   hints (the codebase is `mypy --strict`).
2. Add a mirroring test under `tests/` and keep coverage meaningful.
3. Export it from the module's `__all__`.
4. If it needs a heavy dependency, put that dependency in the right extra
   (`nlp`, `timeseries`) and degrade gracefully when it is absent — see
   `ds/modeling/nlp.py` for the pattern.

## Starting a project

```bash
uv run copier copy templates/project projects/<name>
```

Projects consume `ds`; they don't re-implement it. If you copy-paste a helper
between projects, that's a signal it belongs back in `src/ds/`.

## Keeping docs honest

If you change structure, tooling, or commands, update `README.md` and `CLAUDE.md`
in the **same** change. Documentation must never drift ahead of reality.
