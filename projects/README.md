# Projects

Individual analyses and experiments live here — one directory per project. They
*consume* the reusable `ds` library rather than re-implementing utilities.

- Keep exploratory notebooks in the repo-level `notebooks/` directory or a
  `notebooks/` subfolder within a project.
- Read raw data from `data/raw/` and write outputs to `data/processed/`
  (both are git-ignored).
- Anything genuinely reusable belongs back in `src/ds/`, not copy-pasted between
  projects.

## Scaffolding a new project

```bash
uv run copier copy templates/project projects/my-analysis
```

This generates a `pipeline.py` skeleton (one section per lifecycle stage), a
`notebooks/` folder, and a `tests/` folder. Run the project's tests with
`uv run pytest projects/my-analysis --no-cov`.

## Example

[`_example/`](_example/) is a worked, runnable pipeline that touches every stage
of the lifecycle (load → validate → clean → feature → split → model → evaluate →
visualize):

```bash
uv run python projects/_example/pipeline.py
```
