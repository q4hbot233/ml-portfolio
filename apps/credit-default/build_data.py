#!/usr/bin/env python3
"""Export the small derived slice of the credit-default analysis that the public app ships.

Reads the private repo's prepared table and its frozen model/threshold spec, rebuilds the
*already selected* pipeline, scores the test split, and writes three files:

    data/test_predictions.csv   one row per test client: calibrated score, outcome, and the
                                four audited attributes. Derived from UCI dataset 350
                                (CC BY 4.0), which permits redistribution with attribution.
    data/permutation_nulls.csv  the 5,000 null draws per (attribute, parity statistic) at
                                the frozen threshold, seed 907 -- the same run the report
                                summarises.
    data/reference.json         frozen constants, published headline numbers, the
                                frozen-threshold group table, and the permutation summaries.

Nothing is selected here. The model spec, the calibration method and the threshold are all
read back from reports/results/, where they were frozen on validation. This script only
re-evaluates them, and asserts that what it reproduces matches what the report says.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Path to the private analysis repository this slice is derived from.
# Set CREDIT_DEFAULT_REPO, or place that repo alongside this checkout.
REPO = Path(
    os.environ.get(
        "CREDIT_DEFAULT_REPO",
        Path(__file__).resolve().parents[3] / "credit-default-risk-interpretability-fairness",
    )
)
OUT = Path(__file__).resolve().parent / "data"
sys.path.insert(0, str(REPO))

from src import config, costs, credit_data as cd, fairness as F, pipelines as P  # noqa: E402

KEYS = ("demographic_parity_difference", "fpr_difference", "tpr_difference")
ATTRS = ("sex", "age_band", "education", "marriage")


def close(a, b, tol=1e-9, what=""):
    if not (abs(float(a) - float(b)) <= tol):
        raise AssertionError(f"{what}: reproduced {a!r} != reported {b!r}")
    print(f"  ok  {what:<52} {float(a):.10g}")


def main() -> int:
    analysis = json.loads((REPO / "reports/results/analysis.json").read_text())
    selection = json.loads((REPO / "reports/results/02_model_selection.json").read_text())
    thr_rep = json.loads((REPO / "reports/results/03_calibration_threshold.json").read_text())
    audit_path = REPO / "reports/audit/selection_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("_environment_matches_requirements") is False:
        missing = [k for k, v in audit["_environment"].items() if v is None]
        # Only refuse if something the audit actually computes with is off-pin. lime, shap
        # and fairlearn are absent from that script's import graph entirely.
        load_bearing = {"numpy", "pandas", "scipy", "scikit-learn", "lightgbm"}
        if load_bearing.intersection(missing):
            raise SystemExit(f"selection audit ran without {sorted(load_bearing & set(missing))}")
    swap = json.loads((REPO / "reports/audit/swap_comparison.json").read_text())
    swap.pop("_environment", None)
    selection_audit = {"design": audit["design"], "summary": audit["summary"],
                       "selected_family": audit["selected_family"],
                       "selection_reason": audit["selection_reason"],
                       "n_outer_folds": len(audit["folds"]) // len(audit["summary"])}

    spec = selection["selected_spec"]
    calibration = thr_rep["calibration"]["method"]
    threshold = float(thr_rep["frozen_operating_point"]["threshold"])
    cost_ratio = float(config.HEADLINE_COST_RATIO)
    print(f"spec        {spec['family']} {spec['params']} + {calibration}")
    print(f"threshold   {threshold:.17g}   cost ratio r = {cost_ratio:g}\n")

    df = cd.load()
    frames = cd.split_frames(df)
    X_tr, y_tr = cd.xy(frames["train"])
    X_te, y_te = cd.xy(frames["test"])

    model = P.build_scoring_model(spec, calibration).fit(X_tr, y_tr)
    proba = P.predict_positive_proba(model, X_te)
    pred = costs.apply_threshold(proba, threshold)
    y = np.asarray(y_te).astype(int)

    # ---- gate 1: the confusion matrix the report publishes -----------------------------
    print("reproducing the published test-split numbers")
    conf = analysis["test"]["confusion"]
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    for name, got, want in (("tn", tn, conf["tn"]), ("fp", fp, conf["fp"]),
                            ("fn", fn, conf["fn"]), ("tp", tp, conf["tp"])):
        close(got, want, 0, f"confusion {name}")

    from sklearn.metrics import average_precision_score, roc_auc_score
    close(average_precision_score(y, proba),
          analysis["test"]["metrics"]["average_precision"]["point"], 1e-12, "average precision")
    close(roc_auc_score(y, proba),
          analysis["test"]["metrics"]["roc_auc"]["point"], 1e-12, "ROC-AUC")
    close(costs.expected_cost(y, pred, cost_ratio),
          analysis["test"]["expected_cost_units_of_C_FP"], 0, "expected cost (units of C_FP)")
    close(costs.cost_per_client(y, pred, cost_ratio),
          analysis["test"]["cost_per_client_units_of_C_FP"], 1e-9, "cost per client")

    # ---- gate 2: the per-group table ----------------------------------------------------
    print("\nreproducing the frozen-threshold group table")
    reported = {(r["attribute"], r["group"]): r for r in analysis["fairness"]["group_metrics"]}
    group_rows = []
    for attr in ATTRS:
        table = F.group_table(y, pred, frames["test"][attr].to_numpy())
        for grp, row in table.iterrows():
            rep = reported[(attr, grp)]
            for m in ("n", "base_rate", "selection_rate", "tpr", "fpr", "fnr"):
                if abs(float(row[m]) - float(rep[m])) > 1e-12:
                    raise AssertionError(f"{attr}/{grp}/{m}: {row[m]} != {rep[m]}")
            group_rows.append({"attribute": attr, "group": str(grp),
                               **{m: float(row[m]) for m in
                                  ("n", "base_rate", "selection_rate", "tpr", "fpr", "fnr")},
                               "below_min_n": bool(rep["below_min_n"])})
        print(f"  ok  {attr:<12} {len(table)} groups match on n, base rate, "
              f"selection rate, TPR, FPR, FNR")

    # ---- gate 3: the permutation nulls --------------------------------------------------
    print("\nreproducing the permutation nulls (5,000 shuffles, seed 907)")
    rep_perm = analysis["fairness"]["permutation_test"]
    draws, summary = {}, {}
    for attr in ATTRS:
        out = F.permutation_parity_pvalues(
            y, pred, frames["test"][attr].to_numpy(), keys=KEYS,
            n_perm=5000, seed=config.BOOTSTRAP_SEED, return_draws=True,
        )
        summary[attr] = {}
        for key in KEYS:
            got, want = out[key], rep_perm[attr][key]
            for field in ("observed", "p_value", "null_median", "null_q95"):
                close(got[field], want[field], 1e-12, f"{attr}/{key}/{field}")
            draws[f"{attr}__{key}"] = np.asarray(out[key]["draws"], dtype=float)
            summary[attr][key] = {k: float(got[k]) for k in
                                  ("observed", "p_value", "null_median", "null_q95")}
            summary[attr][key]["n_perm"] = 5000

    # ---- write ---------------------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)

    rows = pd.DataFrame({
        "score": np.round(proba, 9),
        "defaulted": y,
        "sex": frames["test"]["sex"].to_numpy(),
        "age_band": frames["test"]["age_band"].to_numpy(),
        "education": frames["test"]["education"].to_numpy(),
        "marriage": frames["test"]["marriage"].to_numpy(),
    })
    rows.to_csv(OUT / "test_predictions.csv", index=False)

    # The null draws themselves are no longer shipped: the page reports the p-values from
    # reference.json rather than drawing the distributions, and the draws were 520 KB of a
    # 816 KB payload. The summary below is what survives of that run.

    reference = {
        "_source": "credit-default-risk-interpretability-fairness (private repo)",
        "_derived_from": "UCI Default of Credit Card Clients, dataset 350, CC BY 4.0",
        "_note": ("Derived artefacts only: a calibrated score per test client, the outcome, "
                  "the four audited attributes, and the permutation nulls. No raw predictors "
                  "are shipped."),
        "dataset": analysis["dataset"],
        "splits": analysis["splits"],
        "model": {"family": spec["family"], "params": spec["params"],
                  "calibration": calibration, "n_features": len(spec["features"]),
                  "cv_scoring": spec["cv_scoring"], "cv_best_score": spec["cv_best_score"],
                  "lgbm_cv": analysis["model_selection"]["lgbm_cv"],
                  "logistic_cv": analysis["model_selection"]["logistic_cv"],
                  "lgbm_minus_logistic_ap": analysis["model_selection"]["lgbm_minus_logistic_ap"]},
        "frozen": {"threshold": threshold, "cost_ratio_ASSUMED": cost_ratio,
                   "bayes_threshold": analysis["decision"]["bayes_threshold"],
                   "flat_region": analysis["decision"]["flat_region"],
                   "min_audit_cell": config.MIN_AUDIT_CELL,
                   "primary_attribute": config.PRIMARY_PROTECTED},
        "published_test": {"average_precision": analysis["test"]["metrics"]["average_precision"],
                           "roc_auc": analysis["test"]["metrics"]["roc_auc"],
                           "brier": analysis["test"]["metrics"]["brier"],
                           "ece": analysis["test"]["ece"],
                           "confusion": conf,
                           "expected_cost_units_of_C_FP":
                               analysis["test"]["expected_cost_units_of_C_FP"],
                           "cost_per_client_units_of_C_FP":
                               analysis["test"]["cost_per_client_units_of_C_FP"],
                           "share_of_cost_from_false_negatives":
                               analysis["test"]["share_of_cost_from_false_negatives"]},
        "validation_cost_ratio_sweep": analysis["decision"]["cost_ratio_sweep"],
        "validation_naive_half_vs_chosen": analysis["decision"]["naive_half_vs_chosen"],
        "group_metrics_at_frozen_threshold": group_rows,
        "parity_at_frozen_threshold": analysis["fairness"]["parity_by_attribute"],
        "permutation_test_at_frozen_threshold": summary,
        "mitigation": analysis["fairness"]["mitigation"],
        # Standalone audit, not a notebook result: read straight out of reports/audit/ and
        # carried through verbatim. It changed no shipped number, which is the point of it.
        "selection_audit": selection_audit,
        # What the swap was actually worth on the test split, and where the rest of the
        # nested-CV estimate went. Both models at their frozen cut, and both at their own
        # cheapest cut on test -- the difference between those two columns is the part the
        # nested comparison could not see, because that design gives each fold its own cut.
        "swap": swap,
        "attribution": ("Yeh, I-C. (2009). Default of Credit Card Clients [Dataset]. "
                        "UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H. "
                        "Licensed CC BY 4.0."),
    }
    (OUT / "reference.json").write_text(json.dumps(reference, indent=1) + "\n")

    print("\nwrote")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:<26} {p.stat().st_size/1024:8.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
