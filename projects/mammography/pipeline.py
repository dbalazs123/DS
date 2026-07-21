"""Mammography — flag the rare calcification, then *tune the operating point*.

The twelfth project on *real* data, and the **second** deliberately imbalanced /
rare-event one after ``bank_marketing``: 11,183 rows of the Woods mammography
dataset, each a screened region described by six standardized image attributes,
with a binary severity label — ``1`` for a calcification (the thing a screening
programme exists to catch) and ``-1`` otherwise. Only **2.3%** of regions are
positive, rarer than ``bank_marketing``'s 11%, and that skew is the whole point.

The dataset was chosen by the established rule — grep which library surfaces
still have no real consumer — to give :func:`ds.evaluation.probability_metrics`
its *second* consumer (it had only ``bank_marketing``) and, in doing so, to fire
two friction items that first project parked. ``bank_marketing`` handled its
imbalance by *reweighting* the loss (``class_weight="balanced"``), which keeps
the default 0.5 threshold meaningful; this project deliberately does **not**. It
fits a plain logistic regression and then **tunes the decision threshold**,
because a screening programme has an *operating point* — "catch at least this
fraction of the calcifications" — that reweighting can't express. That is the
shape ``bank_marketing`` said would justify building the tools it left parked:

- :func:`ds.evaluation.choose_threshold` — sweep the precision–recall curve for
  the threshold that maximises F1, or the cheapest one that meets a recall
  budget (the screening ask). At the naive 0.5 cut the model misses most of the
  calcifications (recall ~0.37); tuning to an 80% recall budget lifts recall to
  ~0.92 at a read-off cost in precision — the trade reweighting can't target.
- :func:`ds.viz.plot_pr_curve` / :func:`ds.viz.plot_roc_curve` — the
  operating-point *curve*, where the whole sweep is the finding, not a single
  summary number. The chosen operating points are scattered onto the PR curve.

Two data facts shape the pipeline and are load-bearing:

- The severity label ships quoted (``'-1'`` / ``'1'``); it is stripped and
  encoded to a 0/1 ``severity`` column with the calcification as the positive 1.
- ~30% of rows are exact duplicates. With only six coarse standardized
  attributes, identical vectors are *expected distinct* screenings (the
  titanic / bank_marketing precedent — no client identifier, so deduplicating
  would delete real records and distort the 2.3% prevalence), so the rows are
  kept. The honest caveat that follows — an identical vector can land in both
  split halves — is noted where the split happens.

The pipeline runs the full lifecycle on ``ds`` + scikit-learn alone: fetch →
validate the boundary and pin the numeric dtypes → strip/encode the target →
explore (the six attributes' correlation with the label; attr5 leads) →
stratified split → a one-step scale plan fitted on train and re-fit inside every
CV fold → fit a plain logistic regression → **choose the threshold on the
training scores** (never the test set) → score the held-out split
probabilistically (ROC-AUC / PR-AUC vs a prevalence floor) *and* at each tuned
operating point (vs a majority-class floor) → visualize the curves.

Run it with::

    uv run ds run mammography

The raw CSV is downloaded once into ``<data_dir>/raw/`` (git-ignored) and reused
on later runs. The fetch pins a GitHub mirror and verifies its sha256 before
trusting it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ds import Settings, get_logger, get_settings, seed_everything
from ds.eda import missing_value_report, summarize, top_correlations
from ds.evaluation import (
    choose_threshold,
    classification_metrics,
    compare_models,
    confusion_frame,
    cross_validate_kfold,
    per_class_metrics,
    probability_metrics,
)
from ds.features import fit_scale_params
from ds.io import fetch_dataset, load_raw, save_params, save_processed
from ds.modeling.baseline import fit_baseline
from ds.modeling.persistence import load_model, save_model
from ds.modeling.tabular import split_features_target, train_test_split_random
from ds.pipeline import FitStep, fit_pipeline
from ds.preprocessing import standardize_column_names
from ds.validation import (
    assert_in_set,
    assert_no_nulls,
    assert_row_count,
    check_schema,
    require_columns,
)
from ds.viz import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pr_curve,
    plot_roc_curve,
    set_theme,
)

logger = get_logger(__name__)

# The dataset's canonical hosts (UCI / OpenML) are not reachable from every
# network, so the fetch uses a GitHub mirror of the Woods mammography CSV — six
# standardized attributes + a quoted severity label, no header row — and
# ds.io.fetch_dataset verifies the checksum below before trusting the download.
# (A second byte-identical mirror would slot straight into this tuple.)
DATA_URLS = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/mammography.csv",)
RAW_NAME = "mammography.csv"
RAW_SHA256 = "7d3dea3f075f30bbdbb8980e7725684059b4fe3d0f850fadd0f635f6993d8730"

# The published size of the Woods mammography dataset: 11,183 regions.
EXPECTED_ROWS = 11183

# The file ships without a header; these are the six image-derived attributes
# (already standardized in the source) followed by the severity label.
_FEATURES = ["attr1", "attr2", "attr3", "attr4", "attr5", "attr6"]
_TARGET = "severity"
_COLUMNS = [*_FEATURES, _TARGET]

# The raw severity values are single-quoted; "'1'" is the calcification — the
# rare positive a screen exists to catch — and maps to 1, everything else to 0.
_RAW_LABELS = {"-1", "1"}
_POSITIVE_LABEL = "1"

# The screening budget: the operating point must catch at least this fraction of
# the calcifications, as cheaply in precision as choose_threshold can manage.
_RECALL_BUDGET = 0.80

# Display names for the confusion / per-class artifacts; the math stays on codes.
_LABELS = {0: "benign", 1: "calcification"}


def fetch_raw(settings: Settings) -> Path:
    """Download the mammography CSV into ``settings.raw_dir`` and verify it.

    A thin binding of this project's dataset (mirror, filename, pinned digest)
    to :func:`ds.io.fetch_dataset`, which does the download, checksum
    verification and cache re-verify (ROADMAP_ARCHIVE.md item 27's shared dance).

    Args:
        settings: Resolves the raw-data directory.

    Returns:
        Path to the verified local copy of the dataset.

    Raises:
        ValueError: If no mirror serves a file matching the pinned checksum.
        urllib.error.URLError: If every mirror is unreachable.
    """
    return fetch_dataset(RAW_NAME, DATA_URLS, sha256=RAW_SHA256, settings=settings)


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Strip the quotes from ``severity`` and encode the calcification as 1.

    The raw label is single-quoted (``'-1'`` / ``'1'``); the quotes are removed,
    the two values validated (an unexpected label would otherwise be silently
    coded 0), and ``'1'`` — the rare calcification the screen exists to find —
    mapped to the positive 1.

    Args:
        df: Frame with a string ``severity`` column of ``'-1'`` / ``'1'``.

    Returns:
        A new frame with ``severity`` as an ``int64`` 0/1 column.

    Raises:
        DataValidationError: If ``severity`` carries a value outside the two
            expected labels once unquoted.
    """
    out = df.copy()
    out[_TARGET] = out[_TARGET].astype(str).str.strip("'")
    assert_in_set(out, _TARGET, _RAW_LABELS)
    out[_TARGET] = (out[_TARGET] == _POSITIVE_LABEL).astype("int64")
    return out


def run(output_dir: Path, settings: Settings | None = None) -> dict[str, float]:
    """Run the mammography calcification-screening pipeline end to end.

    Args:
        output_dir: Directory to write artifacts (figures, tables) into.
        settings: Paths to read/write data under; resolved from the
            environment when omitted. Tests pass a temporary one so a run
            never touches the shared data tree.

    Returns:
        Held-out metrics at the recall-budget operating point, plus the
        threshold-free ``roc_auc`` / ``average_precision`` / ``brier``, the
        ``f1_``- and ``default_``-prefixed metrics from the other two operating
        points, ``majority_``-prefixed hard-label floors and the chosen
        thresholds under ``threshold_``.
    """
    seed_everything()
    settings = settings or get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire — one idempotent, checksum-verified download into the
    # git-ignored data tree. The file has no header row, so load_raw forwards
    # header/names to the reader; no project-local loader is needed.
    fetch_raw(settings)
    df = load_raw(RAW_NAME, settings=settings, header=None, names=_COLUMNS)

    # 2. Validate the boundary: the published row count (a silently-wrong parse
    # would land elsewhere), the columns we depend on, and — the parse pin — the
    # six attributes actually numeric. check_schema coerces them so a stray
    # stringified column fails here, loudly, rather than as a mangled scale
    # downstream.
    df = standardize_column_names(df)
    assert_row_count(df, EXPECTED_ROWS)
    require_columns(df, _COLUMNS)
    df = check_schema(df, dict.fromkeys(_FEATURES, "float64"), coerce=True)

    # 3. Clean — strip the quotes from severity and encode the calcification to
    # 1. No drop_duplicate_rows: ~30% of rows are exact duplicates, but with
    # only six coarse standardized attributes and no patient identifier,
    # identical vectors are expected distinct screenings, so deduplicating would
    # delete real records and distort the 2.3% prevalence (the titanic /
    # bank_marketing precedent).
    df = encode_target(df)

    # 4. Explore — persist the profile the modeling rests on: the summary, the
    # missing-value report (this dataset has none) and the numeric correlations
    # with the target. attr5 leads, attr4 next; the remaining four are weak. All
    # features are numeric, so there is no categorical (target_rate_by_category)
    # read to take here.
    set_theme("notebook")
    summarize(df).to_csv(output_dir / "summary.csv")
    missing_value_report(df).to_csv(output_dir / "missing.csv")
    top_correlations(df, n=10).to_csv(output_dir / "top_correlations.csv", index=False)

    # 5. Split before anything fit-based. No time axis, so a shuffled split
    # (seeded by seed_everything), stratified on the target to hold the ~2.3%
    # positive rate in both halves. Honest caveat: because identical vectors are
    # kept (step 3), a duplicated region can appear in both halves — a mild
    # optimism shared with the other coarse-vector projects, not a leakage bug
    # we can fix without deleting real rows.
    train, test = train_test_split_random(df, test_size=0.2, stratify=_TARGET)

    # 6. Fit the one-step transform plan on the training split only, re-fit
    # inside every CV fold via the same object: standardize the six attributes
    # for the penalized logistic regression. (They arrive already standardized
    # globally; re-centring on the train split keeps the fit honest to its own
    # window.) No one-hot — there are no categoricals; no impute — no missing.
    plan = [FitStep("scale_features", lambda frame: fit_scale_params(frame, columns=_FEATURES))]
    scoring = fit_pipeline(train, plan)

    # A plain logistic regression: unlike bank_marketing this does NOT reweight
    # the classes. The imbalance is handled downstream by tuning the threshold,
    # which is the whole point of the project — a screening programme has an
    # operating point that class_weight cannot express.
    def make_model() -> LogisticRegression:
        return LogisticRegression(max_iter=1000)

    # 7. Cross-validate on the *raw* training split, the scale re-fit inside each
    # fold and stratified to hold the class balance. Scored on hard labels at the
    # default 0.5 threshold, this shows the operating-point problem: accuracy
    # sits at the prevalence (~0.98) while recall is only ~0.39 — the model
    # misses most calcifications at 0.5, the motivation for tuning the threshold.
    cv_scores = cross_validate_kfold(
        train,
        target=_TARGET,
        make_model=make_model,
        make_pipeline=lambda frame: fit_pipeline(frame, plan),
        stratify=True,
        metrics_fn=classification_metrics,
    )
    cv_scores.to_csv(output_dir / "cv_folds.csv")
    logger.info(
        "5-fold CV @0.5 — recall %.3f, accuracy %.3f (misses most calcifications at 0.5)",
        float(cv_scores["recall"].mean()),
        float(cv_scores["accuracy"].mean()),
    )

    # 8. Apply the fitted plan to both windows and persist everything a later
    # scoring run needs: the processed frame, the strict-JSON scoring pipeline,
    # and the model.
    train = scoring.apply(train)
    test = scoring.apply(test)
    assert_no_nulls(train)
    assert_no_nulls(test)
    save_processed(pd.concat([train, test]), "mammography_features.parquet", settings=settings)
    params_dir = settings.processed_dir / "params"
    save_params(scoring, params_dir / "mammography_scoring.json")

    x_train, y_train = split_features_target(train, _TARGET)
    x_test, y_test = split_features_target(test, _TARGET)
    save_model(make_model().fit(x_train, y_train), params_dir / "mammography_model.joblib")
    model = load_model(params_dir / "mammography_model.joblib")
    y_test_raw = list(y_test)
    # Positive-class probabilities on both splits: thresholds are chosen on the
    # TRAIN scores (never the test set), then applied to the held-out TEST.
    train_scores = [float(p) for p in model.predict_proba(x_train)[:, 1]]
    test_scores = [float(p) for p in model.predict_proba(x_test)[:, 1]]
    y_train_raw = list(y_train)

    # 9. Choose operating points on the training scores. Three thresholds:
    #   - default 0.5 (the trap, for contrast);
    #   - the F1-optimal cut (choose_threshold's balanced default);
    #   - the cheapest cut meeting the screening recall budget (catch >=80% of
    #     calcifications) — the operating point reweighting cannot target.
    f1_point = choose_threshold(y_train_raw, train_scores, criterion="f1")
    recall_point = choose_threshold(
        y_train_raw, train_scores, criterion="target_recall", target=_RECALL_BUDGET
    )
    operating_points = {
        "default_0.5": 0.5,
        "f1_optimal": f1_point["threshold"],
        "recall_budget": recall_point["threshold"],
    }
    logger.info(
        "Chosen thresholds — f1_optimal=%.3f, recall_budget(>=%.0f%%)=%.3f",
        f1_point["threshold"],
        _RECALL_BUDGET * 100,
        recall_point["threshold"],
    )

    # 10. Evaluate on the held-out split. Two views:
    #
    #   (a) Probabilistic (threshold-free) — the honest headline. roc_auc /
    #       average_precision score how well the model *ranks* calcifications,
    #       against a prevalence floor (a constant = the training positive rate,
    #       which ranks at chance: ROC-AUC 0.5, AP = prevalence). This is
    #       probability_metrics' second consumer.
    #
    #   (b) Operating points — hard labels at each tuned threshold, against a
    #       majority-class floor (all-benign: high accuracy, zero recall — the
    #       trap made concrete).
    prevalence = fit_baseline(y_train, strategy="mean")
    prevalence_scores = [float(v) for v in prevalence.predict(len(y_test_raw))]
    probabilistic = compare_models(
        y_test_raw,
        {"logistic_regression": test_scores, "prevalence_rate": prevalence_scores},
        metrics_fn=probability_metrics,
    )
    probabilistic.to_csv(output_dir / "probabilistic_comparison.csv")

    hard_preds = {
        name: [1 if s >= threshold else 0 for s in test_scores]
        for name, threshold in operating_points.items()
    }
    majority = fit_baseline(y_train, strategy="majority")
    hard_preds["majority_class"] = [int(v) for v in majority.predict(len(y_test_raw))]
    operating = compare_models(y_test_raw, hard_preds, metrics_fn=classification_metrics)
    operating.to_csv(output_dir / "operating_points.csv")

    budget_preds = hard_preds["recall_budget"]
    confusion_frame(y_test_raw, budget_preds, labels=_LABELS).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    per_class_metrics(y_test_raw, budget_preds, labels=_LABELS).to_csv(
        output_dir / "per_class_metrics.csv"
    )

    # The returned headline: the recall-budget operating point (what a screen
    # ships), the threshold-free ranking, the other two operating points and the
    # naive floor — every number the run stands on, flat for the CLI to print.
    metrics = classification_metrics(y_test_raw, budget_preds)
    metrics.update(probability_metrics(y_test_raw, test_scores))
    metrics.update(
        {
            f"f1_{name}": value
            for name, value in classification_metrics(y_test_raw, hard_preds["f1_optimal"]).items()
        }
    )
    metrics.update(
        {
            f"default_{name}": value
            for name, value in classification_metrics(y_test_raw, hard_preds["default_0.5"]).items()
        }
    )
    metrics.update(
        {
            f"majority_{name}": value
            for name, value in classification_metrics(
                y_test_raw, hard_preds["majority_class"]
            ).items()
        }
    )
    metrics["threshold_f1_optimal"] = f1_point["threshold"]
    metrics["threshold_recall_budget"] = recall_point["threshold"]
    logger.info("Held-out metrics vs references: %s", metrics)

    # 11. Visualize — the two operating-point curves (the finding), the chosen
    # points scattered onto the PR curve, the confusion matrix at the recall
    # budget, and the probabilistic ranking bars (average precision).
    fig_pr, ax_pr = plt.subplots()
    plot_pr_curve(y_test_raw, test_scores, label="logistic regression", ax=ax_pr)
    for name in operating_points:
        point = classification_metrics(y_test_raw, hard_preds[name])
        ax_pr.scatter(point["recall"], point["precision"], zorder=5, label=name)
    ax_pr.set_title("Precision–recall curve with tuned operating points")
    ax_pr.legend()
    fig_pr.savefig(output_dir / "pr_curve.png", bbox_inches="tight")
    plt.close(fig_pr)

    fig_roc, ax_roc = plt.subplots()
    plot_roc_curve(y_test_raw, test_scores, label="logistic regression", ax=ax_roc)
    ax_roc.set_title("ROC curve — calcification ranking")
    fig_roc.savefig(output_dir / "roc_curve.png", bbox_inches="tight")
    plt.close(fig_roc)

    fig_cm, ax_cm = plt.subplots()
    plot_confusion_matrix(y_test_raw, budget_preds, labels=_LABELS, ax=ax_cm)
    ax_cm.set_title(f"Confusion matrix — recall budget (>={_RECALL_BUDGET:.0%})")
    fig_cm.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig_cm)

    fig_ap, ax_ap = plt.subplots()
    plot_model_comparison(probabilistic, metric="average_precision", ax=ax_ap)
    ax_ap.set_title("Ranking quality (average precision) vs prevalence floor")
    fig_ap.savefig(output_dir / "probabilistic_comparison.png", bbox_inches="tight")
    plt.close(fig_ap)

    return metrics


def main() -> None:
    settings = get_settings()
    metrics = run(settings.processed_dir / "mammography", settings=settings)
    print("Pipeline finished. Held-out metrics (recall-budget operating point + references):")
    for name, value in metrics.items():
        print(f"  {name:>28}: {value:,.3f}")


if __name__ == "__main__":
    main()
