"""The label is a decision, and the variable that explains it was left out -- public demo.

A standard supervised project on HMDA: what a row is, what was engineered, two families
scored against a floor, and a rule you can move a threshold on. The finding it is built
around is not a score -- it is that the lenders themselves name the variable the regulation
does not collect, on a third of the denials, and more often for the group with the larger
gap.

Nothing here is causal, nothing is a legal finding, and no lender is named.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / "data"

INK = "#1b1b1f"; MUTED = "#6b6b76"; GRID = "#dcdce2"
BLUE = "#2f5d9e"; RED = "#b3402f"; GREY = "#9aa0aa"; SAND = "#c2882f"


def use_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 9.5, "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED, "text.color": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.alpha": 0.7, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
    })


def load_reference() -> dict:
    return json.loads((DATA_DIR / "reference.json").read_text())


def load_scored() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "scored.csv")


# =====================================================================================
# THE BOUND THAT THE CONVENTION HIDES
# =====================================================================================



def group_rates(df: pd.DataFrame, threshold: float, codes: dict) -> pd.DataFrame:
    """Denial rate, selection rate and error rates per group at one decision threshold."""
    inv = {v: k for k, v in codes.items()}
    out = []
    pred = (df["s"].to_numpy() >= threshold).astype(int)
    y = df["d"].to_numpy()
    r = df["r"].to_numpy()
    for code, name in sorted(inv.items()):
        m = r == code
        if m.sum() == 0:
            continue
        yy, pp = y[m], pred[m]
        pos, neg = yy.sum(), (1 - yy).sum()
        out.append({"group": name, "n": int(m.sum()), "base_rate": float(yy.mean()),
                    "selection_rate": float(pp.mean()),
                    "fpr": float(pp[yy == 0].mean()) if neg else np.nan,
                    "fnr": float(1 - pp[yy == 1].mean()) if pos else np.nan})
    return pd.DataFrame(out).set_index("group")


# =====================================================================================
# FIGURES
# =====================================================================================

def figure_eda(eda: dict):
    """Denial rate by the three levers that move it most, with cell sizes attached — a rate
    over 800 applications and a rate over 90,000 are not the same claim."""
    cols = [("dti_band", "debt-to-income band"), ("loan_purpose", "loan purpose"),
            ("lien_status", "lien status")]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    for ax, (col, title) in zip(axes, cols):
        rows = eda["breakdowns"].get(col, [])
        rows = sorted(rows, key=lambda r: r["denial_rate"])
        y = np.arange(len(rows))
        ax.barh(y, [r["denial_rate"] * 100 for r in rows], 0.66, color=BLUE)
        ax.set_yticks(y, [r["label"] for r in rows], fontsize=7.5)
        for i, r in enumerate(rows):
            ax.text(r["denial_rate"] * 100 + 1.5, i, f"n={r['n']:,}", va="center",
                    fontsize=6.5, color=MUTED)
        ax.axvline(eda["outcome"]["denial_rate"] * 100, color=GREY, ls="--", lw=1)
        ax.set_xlim(0, 118)
        ax.set_title(title, fontsize=9.5, color=MUTED, loc="left")
        ax.set_xlabel("denial rate (%)", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    return fig


def figure_missing(eda: dict):
    """Missingness is a property of this file, not an accident to be imputed away quietly."""
    rows = eda["missingness"]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.1))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r["share"] * 100 for r in rows], 0.6, color=SAND)
    ax.set_yticks(y, [r["feature"] for r in rows], fontsize=8.5)
    for i, r in zip(y, rows):
        ax.text(r["share"] * 100 + 0.2, i, f"{r['n_missing']:,}", va="center", fontsize=7.5,
                color=MUTED)
    ax.set_xlabel("share of training applications with the field missing (%)", fontsize=8.5,
                  color=MUTED)
    fig.tight_layout()
    return fig


def figure_models(models: list, denial_rate: float):
    """What each family scores, against the floor that matters."""
    m = pd.DataFrame(models)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    y = np.arange(len(m))[::-1]
    ax.barh(y, m["average_precision"], 0.6, color=[GREY if n == "majority" else BLUE
                                                   for n in m["model"]])
    ax.axvline(denial_rate, color=RED, ls="--", lw=1.2)
    ax.text(denial_rate, len(m) - 0.35, f"  prevalence floor {denial_rate:.3f}",
            fontsize=8, color=RED)
    ax.set_yticks(y, m["model"], fontsize=9)
    ax.set_xlabel("average precision", fontsize=9, color=MUTED)
    ax.set_xlim(0, 1)
    for i, v in zip(y, m["average_precision"]):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    return fig






# =====================================================================================
# PAGE
# =====================================================================================

def main() -> None:
    st.set_page_config(page_title="The variable that was left out",
                       page_icon="🏚️", layout="centered")
    use_style()
    cache = st.cache_data(show_spinner=False)
    ref = cache(load_reference)()
    scored = cache(load_scored)()

    slice_ = ref["slice"]
    st.title("The label is a decision, and the variable that explains it was left out")
    st.markdown(
        f"**{ref['n_modelled']:,} mortgage applications in {slice_['state']} in "
        f"{slice_['year']}**, 29.4% of them denied. Every ingredient for a standard fairness "
        f"project is here, and the standard project gets written thousands of times: fit a "
        f"model, find that Black applicants are denied more often, publish the gap.\n\n"
        f"This page is about why that number is harder to interpret than it looks. HMDA is a "
        f"**disclosure** file, so the label is a decision a human made, not an event that "
        f"happened — a model that predicts denial well has learned the lender's policy, "
        f"including whatever is wrong with it.\n\n"
        f"Then the hole that no better model closes: the single variable the lenders cite "
        f"most often for denying somebody is one Regulation C does not collect."
    )
    st.warning(
        "Nothing on this page is causal, nothing is a legal finding, and no lender is named "
        "or examined. It describes rates in one state in one year.",
        icon=":material/gavel:")

    st.divider()
    st.subheader("1 · The data, and what a row of it is")
    eda = ref["eda"]
    oc, sl = eda["outcome"], eda["slice"]
    st.markdown(
        f"Every mortgage application a covered lender reported in **{sl['state']}, "
        f"{sl['year']}** under Regulation C — {ref['n_published']:,} published rows, of which "
        f"**{ref['n_modelled']:,} reached an approve-or-deny decision** and are modelled. "
        f"Of the {oc['n']:,} in the training split, **{oc['n_denied']:,} were denied — "
        f"{oc['denial_rate']:.2%}**.\n\n"
        f"Everything in this section is computed on the training split alone. Describing a "
        f"dataset with its held-out rows folded in is a leak that never shows up as a score; "
        f"it shows up as a modelling choice made knowing the answer.\n\n"
        f"**The outcome is not an event, it is a decision somebody made.** That is the whole "
        f"premise: a model that predicts it well has learned an institution's policy, "
        f"including whatever is wrong with the policy."
    )
    st.pyplot(figure_eda(eda), use_container_width=True)
    dti = {r["label"]: r for r in eda["breakdowns"]["dti_band"]}
    st.markdown(
        f"**Debt-to-income does most of the work.** Above 60% the denial rate is "
        f"**{dti['>60%']['denial_rate']:.1%}** on {dti['>60%']['n']:,} applications; in the "
        f"36–49 band it is {dti['36-49 (exact)']['denial_rate']:.1%}. Two details worth "
        f"noticing rather than smoothing over: applications with DTI **not reported** are "
        f"denied at {dti['not reported']['denial_rate']:.1%}, so the missingness is "
        f"informative and gets its own level; and the **<20%** band is denied at "
        f"{dti['<20%']['denial_rate']:.1%} — higher than the bands above it, which is not "
        f"what a story about affordability alone would predict."
    )

    st.divider()
    st.subheader("2 · Features: what was built, cleaned, and deliberately left out")
    ft = eda["features"]
    st.markdown(
        f"**{ft['total']} features** — {len(ft['numeric'])} numeric, "
        f"{len(ft['categorical'])} categorical. Two of them needed work before they were "
        f"usable, and both decisions are the kind that quietly change a result:\n\n"
        f"- **`dti_band`.** Regulation C requires an *exact integer* only between 36 and 49; "
        f"outside that window lenders report a band. The column is therefore a mixture of "
        f"point values and intervals. Treating it all as numeric invents precision that does "
        f"not exist outside 36–49; treating it all as categorical throws away the precision "
        f"inside it. It is split into a band with `36-49 (exact)` as its own level.\n"
        f"- **`loan_to_value_ratio`.** The raw column contains values up to 33,480,000. "
        f"Those are **set to missing, not clipped** — clipping a typo to 200 turns it into a "
        f"confident data point sitting at the edge of the distribution, where it reads as a "
        f"high-risk applicant rather than as a mistake."
    )
    st.pyplot(figure_missing(eda), use_container_width=True)
    st.markdown(
        f"Missingness is left visible rather than imputed away: `loan_to_value_ratio` is "
        f"absent on {eda['missingness'][0]['share']:.1%} of training rows, partly because of "
        f"the cleaning above."
    )

    st.markdown("**Ten columns in the file are excluded on purpose.** Not omitted — excluded, "
                "with the reason recorded next to each one, and a test that fails if any of "
                "them reaches a `fit` call.")
    exc = pd.DataFrame([{"column": k, "why it is not a feature": v}
                        for k, v in ft["excluded_on_purpose"].items()])
    st.dataframe(exc, use_container_width=True, hide_index=True)
    st.markdown(
        "Three different reasons are mixed in that table and they are worth separating. "
        "`denial_reason-1` is **leakage** — it is recorded after the decision it would be "
        "predicting. `interest_rate` and its siblings are **structurally missing**: they are "
        "priced after approval, so they are blank on every denied row, and a model given "
        "them would learn that blank means denied. `tract_minority_population_percent` is "
        "neither — it is available and predictive, and it is excluded because using "
        "neighbourhood racial composition to decide lending is redlining by construction."
    )

    pre = {r["label"]: r for r in eda["breakdowns"]["preapproval"]}
    req = pre.get("Preapproval requested")
    if req is not None and req["n_denied"] == 0:
        st.warning(
            f"**An eleventh column probably belongs in that table, and this page found it "
            f"while being rebuilt.** `preapproval` = *requested* has a denial rate of "
            f"**{req['denial_rate']:.1%} across {req['n']:,} training applications — zero "
            f"denials**. Not because asking for a preapproval gets you a mortgage, but "
            f"because a denied preapproval is recorded under a *different* action code "
            f"(`preapproval_denied`, {ref['outcome_counts']['preapproval_denied']} rows in "
            f"the file) which the modelling convention drops. Inside the modelled population "
            f"the feature is an artefact of how the outcome is coded.\n\n"
            f"It affects {req['n'] / oc['n']:.1%} of training rows, so it is small — but it "
            f"is the same failure the rest of this page is about, found in the feature set "
            f"rather than in the label. Removing it means re-running the analysis; it is "
            f"documented here rather than quietly left in.",
            icon=":material/warning:")

    st.divider()
    st.subheader("3 · Training, and what a model has to beat")
    models = ref["models"]
    st.pyplot(figure_models(models, oc["denial_rate"]), use_container_width=True)
    byname = {m["model"]: m for m in models}
    st.markdown(
        f"Three fits on the {ft['total']} features above. The floor is not zero — average "
        f"precision for a model that ranks at random is the base rate, "
        f"**{oc['denial_rate']:.3f}** — and predicting \"approved\" for everyone is right "
        f"{oc['majority_accuracy']:.1%} of the time while being useless.\n\n"
        f"Logistic regression reaches **{byname['logistic']['average_precision']:.3f}**, "
        f"LightGBM **{byname['lgbm']['average_precision']:.3f}**. The tree wins, and the "
        f"margin is not the interesting part: **a model that predicts this label well is a "
        f"model that has learned a lending policy.** Accuracy here is not evidence that the "
        f"decisions were right, only that they were consistent — which is exactly why the "
        f"rest of this page is about what the file cannot tell you rather than about the "
        f"score."
    )

    st.divider()
    st.subheader("4 · The variable the regulation does not collect")
    sr = ref["stated_reasons"]
    st.markdown(
        f"HMDA records the lender's own stated primary reason for each denial. It is the "
        f"file's account of itself, and it is worth reading before fitting anything.\n\n"
        f"**Credit history is the stated primary reason for "
        f"{sr['credit_history_share_of_all_denials']:.1%} of all denials.**"
    )
    reasons = pd.DataFrame(sr["by_race"])
    # The frame arrives from JSON with the group as an ordinary column; selecting "columns
    # whose max exceeds 4%" over a string column is a TypeError, not a filter.
    index_col = next((c for c in ("race", "index") if c in reasons.columns), None)
    if index_col:
        reasons = reasons.set_index(index_col)
    reasons = reasons.select_dtypes("number")
    keep = [c for c in reasons.columns if reasons[c].max() >= 0.04]
    st.dataframe(reasons[keep].style.format("{:.1%}"), use_container_width=True)
    st.markdown(
        "Credit history is cited for **35.9% of denials of Black applicants against 30.3% of "
        "white ones**.\n\n"
        "**Regulation C does not collect credit score.** So the lenders are telling us, in "
        "the file itself, that the decisions turned substantially on a variable the file "
        "withholds — and that they turned on it differentially. That is not a reason to "
        "stop. It is a reason to stop pretending a residual gap has one interpretation."
    )

    st.divider()
    st.subheader("5 · The rule, at a threshold you choose")
    st.markdown(
        "Everything above is about the lenders' decisions. This is about a rule built from "
        "them: set a cut on the model's score and watch which groups it selects. Computed "
        f"live from {len(scored):,} held-out and training applications."
    )
    thr = st.slider("Flag the application at or above this predicted denial probability",
                    0.05, 0.95, 0.50, 0.01)
    rates = group_rates(scored, thr, ref["race_codes"])
    shown = rates.loc[rates["n"] >= ref["min_audit_cell"]]
    st.dataframe(shown.style.format({"n": "{:,.0f}", "base_rate": "{:.1%}",
                                     "selection_rate": "{:.1%}", "fpr": "{:.1%}",
                                     "fnr": "{:.1%}"}), use_container_width=True)
    perm = ref["parity_permutation"]
    st.caption(
        f"At the published cut of 0.50 the selection-rate gap is "
        f"{perm['selection_rate_difference']['observed']:.4f} "
        f"(p = {perm['selection_rate_difference']['p_value']:.4f}) and the false-alarm gap "
        f"{perm['fpr_difference']['observed']:.4f} "
        f"(p = {perm['fpr_difference']['p_value']:.4f}), both larger than relabelling "
        f"produces. The missed-denial gap is "
        f"{perm['fnr_difference']['observed']:.4f} "
        f"(p = {perm['fnr_difference']['p_value']:.3f}) — inside what noise makes. Those "
        f"p-values come from shuffling the group labels 2,000 times with every outcome and "
        f"decision held fixed; a bootstrap interval cannot test a max−min statistic, because "
        f"it is non-negative by construction."
    )

    st.divider()
    with st.expander("What this is, and what it is not"):
        st.markdown(f"""
**The features.** The underwriting information HMDA records: loan amount, income, property
value, loan-to-value, debt-to-income band, loan type and purpose, occupancy, lien status and
a few more. It excludes race, sex, ethnicity and age — and it also excludes **census-tract
racial composition**, which is available in the file and is the variable that would make a
model redline by construction. A guard runs before every fit.

**Also excluded:** interest rate and loan costs, which are set after approval and are blank
on every denied row, and the stated denial reason, which is recorded after the decision it
would be predicting.

**A detail that turned out to matter.** Regulation C requires an exact debt-to-income integer
only between 36 and 49; outside that window lenders report a band. The column's *resolution
depends on its value* — precise exactly where the underwriting cutoffs are. It is modelled as
a band with `36-49 (exact)` as its own level, because converting bands to midpoints would
assert precision the regulation forbade.

**Georgia 2023 only.** Denial rates, group composition and lender mix all differ by state and
year, and every number here would move. `race` is a derived HMDA field built from
self-reported categories with a large *not provided* group — and *not provided* is itself an
outcome of the application process rather than missing data.

*{ref['_derived_from']}*
""")


if __name__ == "__main__":
    main()
