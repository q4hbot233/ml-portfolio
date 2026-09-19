"""The label is a decision, and the variable that explains it was left out -- public demo.

The page exists to make two bounds tangible. The first is a slider: the standard analysis
drops every application that was withdrawn or closed incomplete, and the slider asks what
those applications would have become. The second is an E-value: how strong the variable
HMDA does not collect would have to be.

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

def denial_rate_under_assumption(counts: dict, share_denied: float) -> float:
    """Denial rate if ``share_denied`` of the never-decided applications had been denied.

    ``share_denied = 0`` and ``= 1`` are the Manski worst cases. Everything between is an
    assumption, which is the point: the standard analysis makes one implicitly by dropping
    those applications, and never says which.
    """
    denied = counts.get("denied", 0)
    originated = counts.get("originated", 0)
    undecided = counts.get("withdrawn", 0) + counts.get("incomplete", 0)
    total = denied + originated + undecided
    if total == 0:
        return float("nan")
    return (denied + share_denied * undecided) / total


def e_value(rr: float) -> float:
    """VanderWeele & Ding (2017). The minimum strength an unmeasured confounder would need
    with BOTH the grouping and the outcome to explain an association away entirely."""
    rr = float(rr)
    if rr < 1:
        rr = 1.0 / rr
    if rr <= 1:
        return 1.0
    return rr + np.sqrt(rr * (rr - 1.0))


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

def figure_sweep(counts_by_race: dict, share: float, groups: list):
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    xs = np.linspace(0, 1, 101)
    for g, colour in zip(groups, (BLUE, RED, SAND, GREY)):
        ys = [denial_rate_under_assumption(counts_by_race[g], x) for x in xs]
        ax.plot(xs, ys, lw=2.0, color=colour, label=g)
        ax.plot([share], [denial_rate_under_assumption(counts_by_race[g], share)],
                "o", ms=6, color=colour)
    ax.axvline(share, color=INK, lw=1.2)
    ax.set(xlabel="share of never-decided applications assumed to have been denials",
           ylabel="denial rate", xlim=(0, 1), ylim=(0, None))
    ax.legend(fontsize=8.5, labelcolor=MUTED, loc="upper left")
    ax.set_title("Every point on this chart is an assumption. The convention picks one silently.",
                 loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    return fig


def figure_gap_sweep(counts_by_race: dict, share: float, focal: str, reference: str):
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    xs = np.linspace(0, 1, 101)
    ys = [denial_rate_under_assumption(counts_by_race[focal], x)
          - denial_rate_under_assumption(counts_by_race[reference], x) for x in xs]
    ax.plot(xs, ys, lw=2.2, color=SAND)
    here = (denial_rate_under_assumption(counts_by_race[focal], share)
            - denial_rate_under_assumption(counts_by_race[reference], share))
    ax.plot([share], [here], "o", ms=7, color=INK)
    ax.axhline(0, color=RED, lw=1.2, ls=(0, (4, 3)))
    ax.annotate(f"{here:+.3f}", xy=(share, here), xytext=(8, 6),
                textcoords="offset points", fontsize=10, color=INK, fontweight="semibold")
    ax.set(xlabel="share of never-decided applications assumed to have been denials",
           ylabel=f"{focal} minus {reference}", xlim=(0, 1))
    ax.set_title("The gap, as a function of an assumption nobody states", loc="left",
                 color=INK, fontsize=10)
    fig.tight_layout()
    return fig


def figure_strata(rows: list):
    s = pd.DataFrame(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.3))
    x = s["mean_predicted"]
    ax1.plot(x, s["rate_focal"], marker="o", ms=4, lw=1.8, color=RED, label="black")
    ax1.plot(x, s["rate_reference"], marker="o", ms=4, lw=1.8, color=BLUE, label="white")
    ax1.plot([0, 1], [0, 1], ls=(0, (3, 3)), lw=1, color=GRID)
    ax1.set(xlabel="mean predicted denial probability", ylabel="observed denial rate",
            xlim=(0, 1), ylim=(0, 1))
    ax1.legend(fontsize=8.5, labelcolor=MUTED)
    ax1.set_title("Both groups, against what the policy expected", loc="left",
                  color=INK, fontsize=10)
    ax2.bar(range(len(s)), s["difference"], color=SAND)
    ax2.axhline(0, color=GRID, lw=1)
    ax2.set_xticks(range(len(s)), [f"{v:.2f}" for v in x], fontsize=7.5, rotation=45)
    ax2.set(xlabel="stratum, by predicted risk", ylabel="black minus white")
    ax2.set_title("The residual vanishes at both ends", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    return fig


def figure_evalue(raw_rr: float, res_rr: float):
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    vals = [e_value(raw_rr), e_value(res_rr)]
    ax.barh([0, 1], vals, color=[GREY, BLUE], height=0.5)
    for i, (v, rr) in enumerate(zip(vals, [raw_rr, res_rr])):
        ax.text(v + 0.06, i, f"E-value {v:.2f}   (risk ratio {rr:.2f})",
                va="center", fontsize=9.5, color=INK)
    ax.axvline(1.0, color=RED, lw=1.2, ls=(0, (4, 3)))
    ax.set_yticks([0, 1], ["raw gap", "after conditioning on\nwhat HMDA records"])
    ax.set_xlim(0, max(vals) * 1.8)
    ax.set(xlabel="strength an unmeasured variable would need with BOTH race and denial")
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
        f"Then two holes, neither fixable with a better model, both answerable with a bound."
    )
    st.warning(
        "Nothing on this page is causal, nothing is a legal finding, and no lender is named "
        "or examined. It describes rates in one state in one year.",
        icon=":material/gavel:")

    # ---------------------------------------------------------------- hole 1
    st.divider()
    st.subheader("1 · The applications that never reached a decision")
    drop = ref["dropped_by_the_convention"]
    st.markdown(
        f"An application can end in eight ways. The analysis everybody writes uses two: "
        f"originated and denied. It drops **withdrawn** and **file closed for "
        f"incompleteness** — {drop['withdrawn']:,} + {drop['incomplete']:,} = "
        f"**{drop['share_of_decided_applications']:.1%} of everything that reached an "
        f"outcome**.\n\n"
        f"Those are not missing at random. A loan officer who tells an applicant informally "
        f"that the file will not fly produces a *withdrawal*, not a denial. And the rate at "
        f"which applications land there differs by group."
    )
    counts = ref["outcome_counts_by_race"]
    groups = [g for g in ("white", "black", "asian", "not_provided") if g in counts]
    shares = pd.DataFrame([{
        "race": g,
        "n": sum(counts[g].values()),
        "originated": counts[g].get("originated", 0) / sum(counts[g].values()),
        "denied": counts[g].get("denied", 0) / sum(counts[g].values()),
        "never decided": (counts[g].get("withdrawn", 0) + counts[g].get("incomplete", 0))
                         / sum(counts[g].values()),
    } for g in groups]).set_index("race")
    st.dataframe(shares.style.format({"n": "{:,.0f}", "originated": "{:.1%}",
                                      "denied": "{:.1%}", "never decided": "{:.1%}"}),
                 use_container_width=True)

    st.markdown("##### So what would they have become?")
    st.markdown(
        "Nobody knows. The file contains nothing that would say. **But the answer can be "
        "swept**, and that is more honest than dropping them and not mentioning it — which "
        "is what dropping them does."
    )
    share = st.slider(
        "Assume this share of never-decided applications would have been DENIED",
        min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f",
        help="0 and 1 are the Manski worst cases: nothing is assumed beyond arithmetic. "
             "Everything in between is an assumption you are making on purpose.")

    st.pyplot(figure_sweep(counts, share, groups), use_container_width=True)
    st.pyplot(figure_gap_sweep(counts, share, "black", "white"), use_container_width=True)

    gap_here = (denial_rate_under_assumption(counts["black"], share)
                - denial_rate_under_assumption(counts["white"], share))
    observed = ref["manski"]["black_minus_white"]["observed_gap_among_decided"]
    lo = ref["manski"]["black_minus_white"]["gap_lower"]
    hi = ref["manski"]["black_minus_white"]["gap_upper"]
    a, b, c = st.columns(3)
    a.metric("Gap under your assumption", f"{gap_here:+.3f}")
    b.metric("Gap the convention reports", f"{observed:+.3f}",
             "computed among decided applications only", delta_color="off")
    c.metric("Worst-case bounds", f"[{lo:+.2f}, {hi:+.2f}]",
             "sign not determined", delta_color="off")
    st.info(
        f"**The bounds are deliberately extreme** — they assume every dropped application in "
        f"one group would have gone one way and every one in the other the opposite way. "
        f"Nobody believes that and the truth is nowhere near either end. What the width shows "
        f"is that the convention is doing work nobody states: \"the denial gap is "
        f"{observed:.3f}\" really means \"{observed:.3f} among applications that survived a "
        f"filter I did not model, which the two groups pass at rates differing by 4.6 "
        f"points.\"",
        icon=":material/straighten:")

    # ---------------------------------------------------------------- hole 2
    st.divider()
    st.subheader("2 · The variable the regulation does not collect")
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
    st.subheader("3 · Where the residual lives, and how big the hole is")
    models = pd.DataFrame(ref["models"])
    best = models.loc[models["average_precision"].idxmax()]
    st.markdown(
        f"A model of the *decision* reaches average precision **{best['average_precision']:.3f}** "
        f"against a floor of 0.294. That is a statement about how legible the policy is, not "
        f"about how good the model is — underwriting follows rules, and rules are "
        f"predictable.\n\n"
        f"Stratifying on its predicted denial probability compares applications the lenders' "
        f"own revealed policy treats as alike."
    )
    st.pyplot(figure_strata(ref["valid_strata_rows"]), use_container_width=True)
    vs = ref["valid_strata"]
    st.markdown(
        f"Conditioning on everything HMDA records removes about "
        f"{1 - vs['share_of_raw_gap_remaining']:.0%} of the raw gap and leaves "
        f"**{vs['stratified_difference']:.3f}**. And the residual is near zero in the lowest "
        f"stratum and near zero in the highest, largest in the middle — in the top stratum "
        f"both groups are denied about 99.5% of the time and there is no room for a gap. "
        f"**The disparity lives where the decision was genuinely marginal**, which is the "
        f"only place a decision can express a preference."
    )

    ev = ref["e_values"]
    st.pyplot(figure_evalue(ev["raw_risk_ratio"], ev["residual_risk_ratio"]),
              use_container_width=True)
    st.info(
        f"**An unmeasured variable would need a risk ratio of about "
        f"{ev['residual']:.1f} with both race and denial**, on top of every recorded "
        f"underwriting input, to account for the residual entirely.\n\n"
        f"Is credit score plausibly that strong? Almost certainly yes, and I am not going to "
        f"pretend otherwise to make the finding louder. So the honest conclusion of the "
        f"most-used fairness dataset in US lending is **this data cannot settle it** — and "
        f"the useful contribution is saying exactly how large the hole is, and noting that "
        f"the lenders themselves named the variable in it.",
        icon=":material/flag:")

    # ---------------------------------------------------------------- live
    st.divider()
    st.subheader("4 · The rule, at a threshold you choose")
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

**Method sources.** Worst-case bounds under unknown selection: Manski (1990). E-value:
VanderWeele & Ding (2017), *Annals of Internal Medicine* 167(4).

*{ref['_derived_from']}*
""")


if __name__ == "__main__":
    main()
