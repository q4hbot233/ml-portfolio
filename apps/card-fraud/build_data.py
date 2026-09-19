#!/usr/bin/env python3
"""Export the small derived slice of the fraud analysis that the public app ships.

Reads the private repo's prepared table and its frozen selection, refits the *already
selected* model, scores the test split, and writes two files:

    data/test_scores.csv    one row per held-out transaction: the calibrated score and the
                            outcome. No features -- V1..V28 are the issuer's data, however
                            anonymised, and the app does not need them to argue about a queue.
    data/reference.json     frozen constants and the published numbers, carried through
                            verbatim so the page cannot drift from the analysis.

Nothing is selected here. The model and the selection rule are read back from
reports/results/analysis.json, where they were frozen on validation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("FRAUD_REPO",
            Path(__file__).resolve().parents[3] / "card-fraud-detection-imbalance"))
OUT = Path(__file__).resolve().parent / "data"
sys.path.insert(0, str(REPO))

from src import budget, config, fraud_data as fd, pipelines as P, resampling as R  # noqa: E402


def close(a, b, tol, what):
    if not abs(float(a) - float(b)) <= tol:
        raise AssertionError(f"{what}: reproduced {a!r} != published {b!r}")
    print(f"  ok  {what:<44} {float(a):.6g}")


def main() -> int:
    analysis = json.loads((REPO / "reports/results/analysis.json").read_text())
    family, treatment = analysis["selection"]["selected"]
    print(f"selected: {family} + {treatment}\n")

    df = fd.load()
    frames = fd.split_frames(df)
    X_tr, y_tr = fd.xy(frames["train"])
    X_te, y_te = fd.xy(frames["test"])
    model = R.fit_with_treatment(P.MODEL_FAMILIES[family](), X_tr, y_tr, treatment)
    score = P.predict_positive_proba(model, X_te)
    y = np.asarray(y_te).astype(int)

    print("reproducing the published test numbers")
    from sklearn.metrics import average_precision_score, roc_auc_score
    close(average_precision_score(y, score),
          analysis["test"]["metrics"]["average_precision"]["point"], 1e-10, "average precision")
    close(roc_auc_score(y, score), analysis["test"]["metrics"]["roc_auc"]["point"], 1e-10, "ROC-AUC")
    for row in analysis["test"]["queue"]:
        got = budget.queue_metrics(y, score, int(row["k"]))
        close(got["caught"], row["caught"], 0, f"caught at k={int(row['k'])}")

    pd.DataFrame({"score": np.round(score, 8), "fraud": y}).to_csv(
        OUT / "test_scores.csv", index=False)

    reference = {
        "_source": "card-fraud-detection-imbalance (private repo)",
        "_derived_from": analysis["dataset"]["source"],
        "_note": ("Derived artefacts only: one score and one outcome per held-out "
                  "transaction. No features are shipped."),
        "citation": analysis["dataset"]["citation"],
        "dataset": analysis["dataset"],
        "split": analysis["split"],
        "prevalence_drift": analysis["prevalence_drift"],
        "split_comparison": analysis["split_comparison"],
        "treatments": analysis["treatments"],
        "treatment_shapes": analysis["treatment_shapes"],
        "paired_vs_untouched": analysis["paired_vs_untouched"],
        "best_tree_minus_untouched_logistic_queue":
            analysis["best_tree_minus_untouched_logistic_queue"],
        "smote_tomek_rows_removed": analysis["smote_tomek_rows_removed"],
        "calibration": analysis["calibration"],
        "noise_floor": analysis["noise_floor"],
        "selection": analysis["selection"],
        "published_test": analysis["test"],
        "budgets": list(config.REVIEW_BUDGETS),
        "headline_budget": config.HEADLINE_BUDGET,
    }
    (OUT / "reference.json").write_text(json.dumps(reference, indent=1) + "\n")
    print("\nwrote")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:<24} {p.stat().st_size/1024:8.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
