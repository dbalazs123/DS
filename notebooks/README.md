# Notebooks

Exploratory and one-off notebooks. Keep them **out of** `src/ds/` — reusable code
belongs in the library, notebooks are for exploration and narrative.

Conventions:

- Import the toolkit (`import ds`) rather than redefining helpers.
- Call `ds.seed_everything()` and `ds.viz.set_theme()` at the top for
  reproducible, consistently-styled output.
- Clear outputs before committing (`.ipynb_checkpoints/` is git-ignored).
