# Diamonds

Grade the **cut of a diamond** from its measurements — the fourth project on
data that was *not* generated to fit the toolkit, and the first
**multiclass** one.

**Dataset:** the classic ggplot2 `diamonds` dataset — 53,940 round-cut
diamonds with price, the three graded quality scales (cut, color, clarity)
and six physical measurements, mirrored in the
[seaborn-data](https://github.com/mwaskom/seaborn-data) repository (the same
fetch pattern as the other three projects). It was chosen deliberately, by
first grepping which helpers still had no real consumer:

- **A five-class ordered target.** `cut` (Fair < Good < Very Good < Premium
  < Ideal) pushes the classification metric/plot surface past the binary
  case `titanic` covered: `confusion_frame`, `per_class_metrics` and
  `plot_confusion_matrix` at 5×5, and the first real use of
  `classification_metrics(average="macro")` — macro because the class mix is
  imbalanced (40% Ideal, 3% Fair) and a weighted average would hide exactly
  the minority classes the per-class view exists for.
- **Genuinely ordinal features.** `color` (J→D) and `clarity` (I1→IF) are
  ranked scales, giving `fit_ordinal_categories` / `apply_ordinal_encode`
  their first real consumer — and specifically the explicit `categories=`
  ordering, because the default (sorted unique values) would rank color
  backwards and interleave the clarity grades alphabetically.
- **Genuine outliers of two different kinds.** Physically impossible rows
  (20 stones with a zero dimension), gross measurement errors (a 58.9 mm
  width, a 31.8 mm depth) *and* honest right skew in carat/price — real
  material for `flag_outliers`/`plot_outliers` (previously only consumed by
  the synthetic `_example`) and for deciding where clipping is appropriate.
- **Exact duplicate rows** (145 after the impossible rows go) — re-entries,
  not distinct stones, so `drop_duplicate_rows` earns its keep guarding the
  split against leaked twins.

The pipeline downloads the CSV once into `data/raw/` (git-ignored) and runs
fetch → validate at the boundary (dtypes, the three grade vocabularies,
plausibility ranges) → drop the impossible rows and duplicates → explore
(correlation pairs, the cut mix per carat band via `bin_column`, the outlier
profile via `plot_outliers`) → encode the ordered target → stratified split
(`train_test_split_random`) → fit the three-step transform plan
(`ds.pipeline.fit_pipeline`: clip the dimension columns, ordinal-encode the
graded scales, standardize) → stratified 5-fold cross-validation with the
plan re-fitted per fold (`cross_validate_kfold(make_pipeline=...)`) → persist
the scoring `Pipeline` and the fitted model → score the held-out split from
the *reloaded* model → evaluate against the majority class and a
proportions-only rule → visualize.

This project exists to run the workspace's demand loop a fourth time.
Friction it surfaced in the library is recorded as the backlog in
[`ROADMAP.md`](../../ROADMAP.md); nothing is promoted in the same change
(demand first, one step per change).

## Modeling decisions worth knowing

- **Validation flags, a hand-rolled mask filters.** The zero-dimension rows
  have to be dropped before `assert_in_range` can pin strict positivity on
  the surviving rows — the validation stage only asserts, so the filter is
  project code (`drop_impossible_dimensions`).
- **Duplicates are dropped here, kept in titanic.** Same helper, opposite
  call: titanic's duplicate rows are distinct passengers, while ties across
  all ten measured columns (price to the dollar, dimensions to 0.01 mm) with
  no identifier are re-entries — and a duplicate pair straddling the split
  would leak test rows into training.
- **Clipping is scoped to where extremes are errors.** The fit plan clips
  only `x`/`y`/`z` (the measurement-error columns) — deliberately *not*
  `depth`/`table`, whose extreme values are exactly the poor proportions
  that make a cut Fair, nor `carat`/`price`, whose skew is real signal.
- **The target is int-coded through the ordinal encoder.** `encode_cut` maps
  Fair..Ideal to 0..4 with the explicit quality order — stateless (nothing
  learned from the frame), so safe pre-split, and it lands the labels in the
  integer form the metric surface and `fit_baseline("majority")` are typed
  for. The persisted artifacts and plots show the grade names via the
  `labels=` display mapping on `confusion_frame` / `per_class_metrics` /
  `plot_confusion_matrix` (which this project's friction earned — the metric
  math stays on the codes).
- **Two references.** The majority class (predict Ideal for everything:
  accuracy 0.400, macro F1 0.114) and a proportions rule reading only the
  raw `depth`/`table` bands from the standard round-brilliant grading charts
  (accuracy 0.546, macro F1 0.357) — cut grade is *defined* by proportions,
  so the model must beat the rule to justify its other seven columns.
  Held-out, the multinomial logistic regression reaches accuracy 0.655 /
  macro F1 0.551 (5-fold CV macro F1 0.549 ± 0.009 agrees).
- **The confusion structure is the honest headline.** Errors are almost
  entirely between *adjacent* grades — the model is strong on Fair, Premium
  and Ideal but `Good` collapses into `Very Good` (recall 0.12) — the
  pattern `per_class_metrics` and the 5×5 matrix exist to show and a single
  averaged number would bury.

## Run

```bash
uv run ds run diamonds
```

## Test

```bash
uv run pytest projects/diamonds --no-cov
```

(`--no-cov` skips the library's repo-wide coverage gate, which doesn't apply
to a single project's tests. The end-to-end test downloads the dataset on
first run and skips itself when the network is unavailable.)

## Layout

- `pipeline.py` — the analysis, stage by stage.
- `notebooks/` — exploratory notebooks for this project.
- `tests/` — unit tests for the hand-rolled row-filter/encoding/rule helpers
  and an end-to-end run.

Built on the shared [`ds`](../../src/ds) toolkit. Reads inputs from
`data/raw/`, writes outputs to `data/processed/`.
