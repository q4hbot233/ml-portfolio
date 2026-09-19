#!/usr/bin/env python3
"""Export the derived slice of the HMDA analysis that the public app ships.

Writes two files:

    data/applications.csv   one row per application that reached an outcome: the outcome,
                            the model's predicted denial probability where one exists, and
                            the four audited attributes. No loan amounts, no incomes, no
                            geography -- the page argues about bounds and does not need them.
    data/reference.json     the published numbers, carried through verbatim.

HMDA is already public loan-level disclosure, but the smallest slice that supports the
argument is still the right one to ship.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("HMDA_REPO",
            Path(__file__).resolve().parents[3] / "hmda-mortgage-denial-audit"))
OUT = Path(__file__).resolve().parent / "data"
sys.path.insert(0, str(REPO))

from src import bounds as B, hmda_data as hd, pipelines as P  # noqa: E402


def close(a, b, tol, what):
    if not abs(float(a) - float(b)) <= tol:
        raise AssertionError(f"{what}: reproduced {a!r} != published {b!r}")
    print(f"  ok  {what:<46} {float(a):.6g}")


def main() -> int:
    analysis = json.loads((REPO / "reports/results/analysis.json").read_text())

    df = hd.load()
    modelled = hd.modelled(df)
    frames = hd.split_frames(modelled)
    X_tr, y_tr = hd.xy(frames["train"])
    model = P.make_lgbm().fit(X_tr, y_tr)

    print("reproducing the published test numbers")
    X_te, y_te = hd.xy(frames["test"])
    s_te = P.predict_positive_proba(model, X_te)
    from sklearn.metrics import average_precision_score, roc_auc_score
    close(average_precision_score(y_te, s_te), analysis["test"]["average_precision"], 1e-9,
          "test average precision")
    close(roc_auc_score(y_te, s_te), analysis["test"]["roc_auc"], 1e-9, "test ROC-AUC")
    test = frames["test"].copy()
    test["score"] = s_te
    st = B.stratified_rate_difference(test[hd.TARGET].to_numpy(), test["race"].to_numpy(),
                                      test["score"].to_numpy(), reference="white", focal="black")
    close(st["stratified_difference"], analysis["test"]["stratified_difference"], 1e-9,
          "test stratified difference")

    # Per-row data is shipped only for what the page actually recomputes: moving a decision
    # threshold and watching the group rates move. That needs a score, an outcome and a
    # group -- three columns, not the file. Everything else on the page (the stated-reason
    # table, the strata, the E-values) is a published aggregate carried through as-is.
    #
    # Scores are rounded to 4 decimals: the threshold slider cannot resolve more than that,
    # and full precision would triple the page's weight for nothing.
    scored = modelled.copy()
    scored["score"] = np.round(P.predict_positive_proba(model, scored[hd.MODEL_FEATURES]), 4)
    race_codes = {g: i for i, g in enumerate(sorted(scored["race"].unique()))}
    compact = pd.DataFrame({
        "s": scored["score"],
        "d": scored[hd.TARGET].astype("int8"),
        "r": scored["race"].map(race_codes).astype("int8"),
    })
    compact.to_csv(OUT / "scored.csv", index=False)

    # The withdrawal sweep needs counts only -- four numbers per group.
    reached = df.loc[df["outcome"].isin(["originated", "denied", "withdrawn", "incomplete"])]
    outcome_counts = (reached.groupby(["race", "outcome"], observed=True).size()
                      .unstack(fill_value=0))

    reference = {
        "_source": "hmda-mortgage-denial-audit (private repo)",
        "_derived_from": analysis["source"],
        "_note": ("Derived artefacts only: outcome, predicted denial probability and the "
                  "four audited attributes. No loan amounts, incomes or geography."),
        "slice": analysis["slice"],
        "n_published": analysis["n_published"],
        "n_modelled": analysis["n_modelled"],
        "dropped_by_the_convention": analysis["dropped_by_the_convention"],
        "manski": analysis["manski"],
        "stated_reasons": analysis["stated_reasons"],
        "models": analysis["models"],
        "valid_strata": analysis["valid_strata"],
        "valid_strata_rows": analysis["valid_strata_rows"],
        "e_values": analysis["e_values"],
        "parity_permutation": analysis["parity_permutation"],
        "published_test": analysis["test"],
        "excluded_on_purpose": hd.EXCLUDED_ON_PURPOSE,
        "min_audit_cell": 300,
        "race_codes": race_codes,
        "outcome_counts_by_race": json.loads(outcome_counts.to_json(orient="index")),
    }
    (OUT / "reference.json").write_text(json.dumps(reference, indent=1) + "\n")
    print("\nwrote")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:<24} {p.stat().st_size/1024:8.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
