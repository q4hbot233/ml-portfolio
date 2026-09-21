"""What 492 frauds can and cannot tell you -- the public demo.

Everything on the page is computed live from the model's 42,721 held-out scores, which are
shipped with it. The point of the page is the queue: a threshold is the wrong object for
fraud detection, because a team fixes a COUNT of alerts it can work and the threshold is
whatever the last one scored.
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

INK = "#E9ECF1"; MUTED = "#99A0AC"; GRID = "#232830"
BLUE = "#6B9BD8"; RED = "#E2644E"; GREY = "#7D8594"; GREEN = "#4DB38A"
BG = "#08090B"


def use_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 9.5, "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED, "text.color": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.alpha": 0.7, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.titlecolor": INK, "legend.labelcolor": MUTED,
    })


def load_reference() -> dict:
    return json.loads((DATA_DIR / "reference.json").read_text())


def load_scores() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "test_scores.csv")


# =====================================================================================
# WHAT REBALANCING IS WORTH, PAIRED
# =====================================================================================

def paired_deltas(search_rows: list) -> pd.DataFrame:
    """Each treatment against the same hyperparameters left untreated.

    Best treated minus best untreated compares a maximum over 380 configurations with a
    maximum over 76, and the bigger pool tends to win even when the treatment does nothing.
    The grid runs inside every treatment, so each treated configuration has an untreated
    twin with identical settings, and the difference between the two is what the treatment
    did at those settings.
    """
    idx = {(r["family"], json.dumps(r["params"], sort_keys=True), r["treatment"]):
           r["average_precision"] for r in search_rows}
    cells = {(f, p) for (f, p, _) in idx}
    rows = []
    for family, params in cells:
        base = idx.get((family, params, "none"))
        if base is None:
            continue
        for (f, p, treatment), value in idx.items():
            if treatment != "none" and f == family and p == params:
                rows.append({"treatment": treatment, "family": family,
                             "untreated": base, "treated": value,
                             "delta": value - base})
    return pd.DataFrame(rows)


def paired_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    return (deltas.groupby("treatment")
            .agg(n=("delta", "size"), median=("delta", "median"),
                 p10=("delta", lambda s: s.quantile(0.10)),
                 p90=("delta", lambda s: s.quantile(0.90)),
                 wins=("delta", lambda s: (s > 0).sum()))
            .sort_values("median", ascending=False)
            .reset_index())


# =====================================================================================
# THE QUEUE
# =====================================================================================

def top_k(score: np.ndarray, k: int, seed: int = 5150) -> np.ndarray:
    """Indices of the k highest scores, ties broken at random.

    Ties are not cosmetic here. The table is sorted by time, so an argsort tie-break means
    "whichever happened first", which is not a rule anyone implements and can move
    precision@20 by a whole transaction.
    """
    s = np.asarray(score, dtype=float)
    k = int(min(k, s.size))
    jitter = np.random.default_rng(seed).random(s.size)
    return np.lexsort((jitter, -s))[:k]


def queue_metrics(y: np.ndarray, score: np.ndarray, k: int) -> dict:
    y = np.asarray(y).astype(int)
    idx = top_k(score, k)
    caught = int(y[idx].sum())
    total = int(y.sum())
    k_eff = int(idx.size)
    return {"k": k_eff, "caught": caught, "missed": total - caught,
            "precision": caught / k_eff if k_eff else np.nan,
            "recall": caught / total if total else np.nan,
            "alerts_per_fraud": k_eff / caught if caught else np.inf,
            "lift": (caught / k_eff) / y.mean() if k_eff and y.mean() else np.nan}


def queue_curve(y: np.ndarray, score: np.ndarray, ks: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([queue_metrics(y, score, int(k)) for k in ks])


def bootstrap_precision(y: np.ndarray, score: np.ndarray, k: int, n_boot: int = 400,
                        seed: int = 5150) -> tuple[float, float, float]:
    """Stratified bootstrap: positives and negatives resampled separately.

    With 52 frauds an ordinary resample draws a different number each time, and the interval
    it returns mixes "how good is the model" with "how many frauds were in this draw".
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pos, pos.size, True), rng.choice(neg, neg.size, True)])
        draws.append(queue_metrics(y[i], s[i], k)["precision"])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return queue_metrics(y, s, k)["precision"], float(lo), float(hi)


# =====================================================================================
# FIGURES
# =====================================================================================

def figure_queue(curve: pd.DataFrame, k: int, prevalence: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.3))
    ax1.plot(curve["k"], curve["precision"], lw=2.0, color=BLUE)
    ax1.axhline(prevalence, color=RED, ls=(0, (4, 3)), lw=1.1)
    ax1.annotate(f"reviewing at random: {prevalence:.4f}", xy=(curve['k'].iloc[2], prevalence),
                 xytext=(0, 7), textcoords="offset points", fontsize=8, color=RED)
    ax1.set(xlabel="review budget k", ylabel="precision@k: of what we look at, how much is fraud")
    ax2.plot(curve["k"], curve["recall"], lw=2.0, color=GREEN)
    ax2.set(xlabel="review budget k", ylabel="recall@k: of the fraud, how much we reach")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.axvline(k, color=INK, lw=1.3)
        ax.set_ylim(0, 1)
    ax1.set_title("What the team's day is made of", loc="left", color=INK, fontsize=10)
    ax2.set_title("What the loss report is made of", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    return fig


def ordinal(n: int) -> str:
    """22nd, not 22th."""
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }".replace(" ", "")


def figure_eda(eda: dict):
    """Three things about the data before any model: how rare fraud is, where it sits in
    amount, and whether the hour of day carries anything."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.2, 3.0))
    bal = eda["balance"]

    ax1.bar([0], [bal["n_legit"]], 0.6, color=GREY, label="legitimate")
    ax1.bar([1], [bal["n_fraud"]], 0.6, color=RED, label="fraud")
    ax1.set_yscale("log")
    ax1.set_xticks([0, 1], ["legit", "fraud"])
    ax1.set_ylabel("training rows (log scale)", fontsize=8.5, color=MUTED)
    ax1.set_title(f"1 in {bal['one_in']:,} rows is fraud", fontsize=9.5, color=MUTED,
                  loc="left")
    for x, v in ((0, bal["n_legit"]), (1, bal["n_fraud"])):
        ax1.text(x, v * 1.35, f"{v:,}", ha="center", fontsize=8.5, color=MUTED)

    bins = eda["amount"]["bins"]
    y = np.arange(len(bins))
    ax2.barh(y, [b["fraud_rate"] * 100 for b in bins], 0.65, color=RED)
    ax2.set_yticks(y, [f"{int(b['lo'])}–{int(b['hi'])}" for b in bins], fontsize=7)
    ax2.set_xlabel("fraud rate in that band (%)", fontsize=8.5, color=MUTED)
    ax2.set_title("by transaction amount", fontsize=9.5, color=MUTED, loc="left")
    ax2.invert_yaxis()

    hrs = eda["hours"]
    ax3.plot([h["hour"] for h in hrs], [h["fraud_rate"] * 100 for h in hrs],
             color=BLUE, lw=1.8)
    ax3.axhline(bal["prevalence"] * 100, color=GREY, ls="--", lw=1)
    ax3.set_xlabel("hours since the file starts, mod 24", fontsize=8.5, color=MUTED)
    ax3.set_title("by hour of cycle", fontsize=9.5, color=MUTED, loc="left")
    fig.tight_layout()
    return fig


def figure_separability(eda: dict, top: int = 14):
    """One-feature ROC-AUC, folded so a component that separates downwards is not reported
    as useless. The two engineered columns are marked."""
    rows = eda["separability"][:top]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    y = np.arange(len(rows))[::-1]
    colours = [RED if r["derived"] else BLUE for r in rows]
    ax.barh(y, [r["strength"] for r in rows], 0.68, color=colours)
    ax.set_yticks(y, [r["feature"] + ("  (engineered)" if r["derived"] else "")
                      for r in rows], fontsize=8.5)
    ax.set_xlabel("separating power of that column alone  |2·AUC − 1|", fontsize=9,
                  color=MUTED)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


def figure_split(split_comparison: list):
    """The same model scored under a shuffled split and a chronological one.

    The percentage goes on the shuffled bar and reads as inflation. On the chronological bar
    as a negative number it would say the chronological split "performs worse", which gets
    it backwards: the model is identical in both and only the evaluation differs.
    """
    rows = {r["index"]: r for r in split_comparison}
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    metrics = ["average_precision", "roc_auc", "precision_at_100"]
    labels = ["average precision", "ROC-AUC", "precision@100"]
    x = np.arange(len(metrics))
    a = [rows["shuffled"][m] for m in metrics]
    b = [rows["chronological"][m] for m in metrics]
    ax.bar(x - 0.19, a, 0.36, color=GREY, hatch="///", edgecolor=BG, linewidth=0,
           label="shuffled, inflated by leakage")
    ax.bar(x + 0.19, b, 0.36, color=BLUE, label="chronological, what deployment would give")
    for i, (u, v) in enumerate(zip(a, b)):
        ax.text(i - 0.19, u + 0.04, f"+{(u - v) / v:.0%}", ha="center", fontsize=10,
                color=RED, fontweight="semibold")
        ax.text(i + 0.19, v + 0.04, f"{v:.2f}", ha="center", fontsize=9.5, color=BLUE)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("higher is not better here", fontsize=9, color=MUTED)
    ax.set_title("the red figure is how much shuffling overstates the blue one",
                 fontsize=9.5, color=MUTED, loc="left", pad=8)
    # Below the axes: at upper right it sat on top of the ROC-AUC bars, which reach 0.98.
    ax.legend(fontsize=8.5, labelcolor=MUTED, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    return fig


def figure_paired(deltas: pd.DataFrame):
    """One dot per (hyperparameters, treatment) pair: what the treatment did there.

    Drawn against zero rather than against an average, because the claim being
    made is "this helped at these settings", and the reader should be able to see
    how many of the dots are on the wrong side of the line. Few of them are.
    """
    order = (deltas.groupby("treatment")["delta"].median()
             .sort_values(ascending=False).index.tolist())
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    rng = np.random.default_rng(5150)
    for i, treatment in enumerate(order):
        v = deltas.loc[deltas["treatment"] == treatment, "delta"].to_numpy()
        ax.scatter(v, np.full(v.size, i) + rng.uniform(-0.17, 0.17, v.size),
                   s=13, alpha=0.45, color=BLUE, linewidths=0)
        ax.plot([np.median(v)], [i], marker="|", ms=22, mew=2.2, color=RED)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(range(len(order)), order, fontsize=9)
    ax.set_xlabel("change in average precision against the same model left untreated",
                  fontsize=9, color=MUTED)
    ax.set_title("Each treatment against its own twin, not against the best of the others",
                 loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    return fig


def figure_search(rows: list, by_treatment: list, budget: int):
    """456 configurations, one dot each, grouped by what was done to the class balance.

    The scatter is the point. A single bar per treatment would say rebalancing lifts the
    tree a long way; the spread says the setting of the model matters more than the
    treatment does, and that an untreated bar drawn at one arbitrary configuration can be
    put almost anywhere on this axis.
    """
    df = pd.DataFrame(rows)
    order = [r["treatment"] for r in by_treatment][::-1]
    pos = {t: i for i, t in enumerate(order)}
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    rng = np.random.default_rng(11)
    for fam, colour in (("lgbm", BLUE), ("logistic", GREEN)):
        blk = df[df["family"] == fam]
        y = blk["treatment"].map(pos) + rng.uniform(-0.17, 0.17, len(blk))
        ax.scatter(blk["average_precision"], y, s=11, color=colour, alpha=0.5,
                   linewidths=0, label=fam)
    for r in by_treatment:
        ax.plot([r["average_precision"]], [pos[r["treatment"]]], marker="|", ms=17,
                mew=2.2, color=INK)
        ax.text(r["average_precision"] + 0.012, pos[r["treatment"]] + 0.30,
                f"best {r['average_precision']:.3f}", fontsize=8, color=INK)
    ax.set_yticks(range(len(order)), order, fontsize=9)
    ax.set_xlabel("average precision on validation, one dot per configuration", fontsize=9,
                  color=MUTED)
    ax.set_xlim(0, 1)
    # upper left: the top row is the best-scoring treatment, so its dots are all far right
    ax.legend(fontsize=8.5, labelcolor=MUTED, loc="upper left")
    fig.tight_layout()
    return fig


# =====================================================================================
# PAGE
# =====================================================================================

def main() -> None:
    st.set_page_config(page_title="What 492 frauds can and cannot tell you",
                       page_icon="🔎", layout="centered")
    use_style()
    cache = st.cache_data(show_spinner=False)
    ref = cache(load_reference)()
    scores = cache(load_scores)()
    y, score = scores["fraud"].to_numpy(), scores["score"].to_numpy()

    ds, sp, sel, test = ref["dataset"], ref["split"], ref["selection"], ref["test"]
    k0 = ref["headline_budget"]
    w = sel["selected"]
    untreated, by_t = ref["untreated_best"], ref["by_treatment"]
    best_t = by_t[0]
    deltas = cache(paired_deltas)(ref["search_rows"])
    pair_tbl = paired_summary(deltas)
    pair_head = pair_tbl["median"].max()
    untreated_aps = [r["average_precision"] for r in ref["search_rows"]
                     if r["treatment"] == "none"]
    treated_aps = [r["average_precision"] for r in ref["search_rows"]
                   if r["treatment"] != "none"]
    collapsed_un = sum(1 for v in untreated_aps if v < 0.10)
    collapsed_tr = sum(1 for v in treated_aps if v < 0.10)

    st.title("What 492 frauds can and cannot tell you")
    st.markdown(
        f"**{ds['rows']:,} card transactions over 48 hours, {ds['frauds']} of them fraud, "
        f"{ds['prevalence']:.3%}.** The advice everyone repeats at this prevalence is to "
        f"rebalance: weight the classes, drop negatives, or synthesise positives with "
        f"SMOTE.\n\n"
        f"So this fits **{ref['search']['n_configurations']} configurations**: two model "
        f"families across a hyperparameter grid, each crossed with six treatments. Every one "
        f"is scored on a validation split, one is picked by a rule written down first, and "
        f"the test split is opened once.\n\n"
        f"What comes out is that **the advice is right and the usual way of checking it is "
        f"not**. Against a tuned untreated model, rebalancing buys "
        f"**{best_t['average_precision'] - untreated['average_precision']:+.3f}** average "
        f"precision, almost nothing. Against the *same* hyperparameters left untreated it "
        f"buys **{pair_head:+.2f}**, and it never loses. Those are the same 456 fits read "
        f"two ways, and the gap between them is the finding: rebalancing does not raise the "
        f"ceiling, it stops the floor falling out. Section 3 is about which of those two "
        f"numbers a write-up should quote."
    )
    st.caption(f"Live from the selected model's {len(scores):,} held-out scores, shipped "
               f"with the page. The full analysis is in a private repository.")

    # ------------------------------------------------------------------ 1
    st.divider()
    st.subheader("1 · The data, and how it is split")
    eda = ref["eda"]
    st.markdown(
        f"Splits are **chronological**: the first {sp['train']['rows']:,} transactions "
        f"train, the next {sp['valid']['rows']:,} select, the last {sp['test']['rows']:,} "
        f"are opened once at the end. A compromised card produces several frauds minutes "
        f"apart, so a shuffled split puts some of them on each side of the line and scores "
        f"the model partly on cards it has already seen. Section 5 measures what that is "
        f"worth.\n\n"
        f"Prevalence falls across the three: **{sp['train']['prevalence']:.3%} → "
        f"{sp['valid']['prevalence']:.3%} → {sp['test']['prevalence']:.3%}**, on "
        f"{sp['train']['frauds']}, {sp['valid']['frauds']} and {sp['test']['frauds']} "
        f"frauds. Every interval on this page is wide for that reason and says so."
    )
    st.pyplot(figure_eda(eda), use_container_width=True)
    st.markdown(
        f"Accuracy is unusable before anything is fitted: predicting \"not fraud\" every "
        f"time is right **{1 - sp['train']['prevalence']:.2%}** of the time. Everything "
        f"below is **average precision**, the area under the precision–recall curve, whose "
        f"floor for random ranking is the base rate, {sp['valid']['prevalence']:.4f}, not "
        f"0.5. Amount is not the giveaway people expect: the fraud rate is highest in the "
        f"smallest band, which is card testing."
    )

    # ------------------------------------------------------------------ 2
    st.divider()
    st.subheader("2 · Features, and what they were worth")
    st.markdown(
        "28 of the 31 columns are **V1..V28**, principal components published instead of "
        "the raw fields because the raw fields are a European issuer's records. You cannot "
        "engineer on top of a component you cannot interpret, so the work was two derived "
        "columns and one deliberate exclusion: `hour_of_cycle` (hours since the file "
        "starts, mod 24; the file has no wall clock), `log_amount`, and **`Time` left "
        "out**, because a model given it fits where the fraud bursts happen to sit in these "
        "particular 48 hours."
    )
    st.pyplot(figure_separability(eda), use_container_width=True)
    ranks = {r["feature"]: i + 1 for i, r in enumerate(eda["separability"])}
    st.markdown(
        f"**And they were worth very little.** Ranked by how well each column separates the "
        f"classes alone, `hour_of_cycle` comes {ordinal(ranks['hour_of_cycle'])} of "
        f"{len(eda['separability'])} and `log_amount` {ordinal(ranks['log_amount'])}. The "
        f"top of the list is components somebody else's PCA already found. Reporting the "
        f"engineering without this chart would imply it did work it did not do."
    )

    # ------------------------------------------------------------------ 3
    st.divider()
    st.subheader("3 · Selection: every model, every treatment, scored on validation")
    g = ref["search"]
    st.markdown(
        f"Each of **{g['n_configurations']}** configurations is fitted on the training "
        f"split and scored on validation: {len(g['lgbm_grid']['num_leaves']) * len(g['lgbm_grid']['learning_rate']) * len(g['lgbm_grid']['n_estimators']) * len(g['lgbm_grid']['min_child_samples'])} "
        f"LightGBM settings and {len(g['logistic_grid']['C'])} logistic ones, each crossed "
        f"with all {len(g['treatments'])} treatments. Hyperparameters are searched **inside "
        f"every treatment**, because the best shape of a tree depends on what was done to "
        f"the rows underneath it."
    )
    st.pyplot(figure_search(ref["search_rows"], by_t, k0), use_container_width=True)
    st.markdown(
        f"**Read the spread before the ranking.** The best untreated LightGBM reaches "
        f"**{untreated['average_precision']:.3f}**; the best rebalanced one reaches "
        f"**{best_t['average_precision']:.3f}**. That is "
        f"**{best_t['average_precision'] - untreated['average_precision']:+.3f}** for every "
        f"resampler and weighting scheme in the list, applied to a model that was already "
        f"tuned. Meanwhile the untreated dots alone run from "
        f"{min(r['average_precision'] for r in ref['search_rows'] if r['treatment'] == 'none'):.2f} "
        f"to {untreated['average_precision']:.2f}. **The choice of hyperparameters moves "
        f"this model further than the choice of treatment does.** A write-up that fits one "
        f"untreated configuration, rebalances it, and reports the difference is mostly "
        f"reporting where its baseline happened to land."
    )
    st.markdown(
        f"**So pair them.** The grid is run inside every treatment, so every treated "
        f"configuration has an untreated twin with identical settings, and the difference "
        f"between the two is what the treatment did *there*: no maximum, no comparison "
        f"across arms of different sizes."
    )
    st.pyplot(figure_paired(deltas), use_container_width=True)
    ptbl = pair_tbl.copy()
    ptbl["wins"] = ptbl["wins"].astype(str) + " of " + ptbl["n"].astype(str)
    st.dataframe(
        ptbl[["treatment", "median", "p10", "p90", "wins"]].rename(columns={
            "median": "median change in AP", "p10": "10th pct", "p90": "90th pct",
            "wins": "configurations improved"}),
        use_container_width=True, hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="%+.3f")
                       for c in ("median change in AP", "10th pct", "90th pct")})
    tomek = next(s for s in ref["treatment_shapes"] if s["treatment"] == "smote_tomek")
    st.caption(
        f"`smote` and `smote_tomek` have identical rows in the table above. That is correct: "
        f"on these training rows they produce the same training set. Tomek links are pairs "
        f"of opposite-class points that are each other's nearest neighbour, and the raw "
        f"training data has {tomek['tomek_rows_before_oversampling']} rows in them. Once SMOTE "
        f"has added {tomek['rows_added']:,} synthetic frauds, every point near the boundary "
        f"has a synthetic fraud as its nearest neighbour, the pairs are gone, and the "
        f"cleaning step removes {tomek['rows_removed']} rows. It is one of the six treatments "
        f"in name only on a problem this imbalanced.")
    st.warning(
        f"**The two readings disagree by an order of magnitude, and both are correct.** "
        f"Best-against-best says "
        f"{best_t['average_precision'] - untreated['average_precision']:+.3f}. Paired says "
        f"{pair_head:+.3f}. The reason is in the left tail of the scatter above: "
        f"**{collapsed_un} of {len(untreated_aps)} untreated configurations land below 0.10 "
        f"average precision** (at this prevalence an unweighted tree with a large "
        f"`min_child_samples` has no reason to split on 384 positives and returns something "
        f"close to a constant), while **{collapsed_tr} of {len(treated_aps)} treated ones "
        f"do**.\n\n"
        f"So what rebalancing is worth depends on how much tuning surrounds it: "
        f"{best_t['average_precision'] - untreated['average_precision']:+.3f} average "
        f"precision if you also search the grid exhaustively, roughly {pair_head:.1f} if "
        f"you do not. It protects against a hyperparameter choice that would collapse the "
        f"model and does little for one that was already tuned. A write-up that reports "
        f"only one of the two numbers is mostly telling you how hard its author searched.",
        icon=":material/compare_arrows:")
    tbl = pd.DataFrame([{"treatment": r["treatment"], "best family": r["family"],
                         "average precision": r["average_precision"],
                         f"precision@{k0}": r[f"precision_at_{k0}"]} for r in by_t])
    st.dataframe(tbl.style.format({"average precision": "{:.4f}",
                                   f"precision@{k0}": "{:.3f}"}),
                 use_container_width=True, hide_index=True)
    st.info(
        f"**The rule, fixed before the validation scores were read:**\n\n> *{sel['rule']}*\n\n"
        f"**{sel['n_tied_at_top']}** configurations tie at the top on validation "
        f"precision@{k0} = **{sel['best_validation_precision_at_budget']:.3f}**, so it goes "
        f"to the tie-break. The selected model is **{w['family']} + {w['treatment']}**, "
        f"`{', '.join(f'{k}={v}' for k, v in w['params'].items())}`. The two tied "
        f"configurations are {' and '.join(sel['tied_treatments'])} at the same settings, "
        f"which is one model trained twice on the same rows, so the tie-break had nothing "
        f"to decide.\n\n"
        f"**And the rule is deciding on very little.** Validation holds "
        f"{sp['valid']['frauds']} frauds, so precision@{k0} moves in steps of "
        f"{1/k0:.2f}, one transaction. "
        f"**{sum(1 for r in ref['search_rows'] if abs(r[f'precision_at_{k0}'] - sel['best_validation_precision_at_budget']) < 0.011) - sel['n_tied_at_top']}** "
        f"more configurations sit exactly one transaction behind the winner. Section 5 "
        f"measures what shuffling the split is worth to three decimal places; this is the "
        f"same kind of optimism in the other direction, and the honest version is that the "
        f"selected model was picked out of a crowd it did not clearly beat.",
        icon=":material/rule:")

    # ------------------------------------------------------------------ 4
    st.divider()
    st.subheader("4 · The test split, opened once")
    m = test["average_precision"]
    pk = test[f"precision_at_{k0}"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Average precision", f"{m['point']:.3f}",
              f"95% [{m['ci_lo']:.3f}, {m['ci_hi']:.3f}]", delta_color="off")
    c2.metric("ROC-AUC", f"{test['roc_auc']['point']:.3f}",
              f"95% [{test['roc_auc']['ci_lo']:.3f}, {test['roc_auc']['ci_hi']:.3f}]",
              delta_color="off")
    c3.metric(f"Precision@{k0}", f"{pk['point']:.3f}",
              f"95% [{pk['ci_lo']:.3f}, {pk['ci_hi']:.3f}]", delta_color="off")
    res = test["resolution"]
    st.markdown(
        f"**Now the object the team actually fixes.** Nobody sets a probability; a fraud "
        f"desk sets how many alerts it can work in a shift, and the threshold is whatever "
        f"the last one scored. Move the budget and watch both numbers, computed live from "
        f"the {len(scores):,} held-out scores."
    )
    k = st.slider("How many transactions can the team review?", 10, 2000, k0, 10)
    ks = np.unique(np.round(np.logspace(1, np.log10(2000), 44)).astype(int))
    curve = queue_curve(y, score, ks)
    st.pyplot(figure_queue(curve, k, float(y.mean())), use_container_width=True)
    q = queue_metrics(y, score, k)
    p, lo, hi = bootstrap_precision(y, score, k)
    a, b, c = st.columns(3)
    a.metric(f"Precision@{k}", f"{p:.3f}", f"95% [{lo:.3f}, {hi:.3f}]", delta_color="off")
    b.metric("Frauds caught", f"{q['caught']} of {int(y.sum())}",
             f"{q['recall']:.0%} of the fraud", delta_color="off")
    c.metric("Alerts per fraud found", f"{q['alerts_per_fraud']:.1f}",
             f"{q['lift']:.0f}× better than random", delta_color="off")
    st.warning(
        f"**What the interval is made of.** The test split holds **{res['n_positives']} "
        f"frauds**, so at a budget of {res['k']} one transaction moving in or out of the "
        f"queue is **{res['one_transaction_is_pct']:.0f} percentage point**, and no model "
        f"can exceed precision **{res['max_possible_precision']:.2f}** there because there "
        f"are not {res['k']} frauds to find. At k=20 this model's precision is "
        f"**{queue_metrics(y, score, 20)['precision']:.2f}**: the top twenty are all "
        f"fraud. For a fraud team that is a more useful number than the average precision.",
        icon=":material/straighten:")

    # ------------------------------------------------------------------ 5
    st.divider()
    st.subheader("5 · The split decided more than the model did")
    sc = ref["split_comparison"]
    st.pyplot(figure_split([{"index": kk, **vv} for kk, vv in sc.items()
                            if isinstance(vv, dict)]), use_container_width=True)
    st.markdown(
        f"Same configuration, same rows, one difference: how the held-out set was chosen. "
        f"Shuffling puts several frauds from one compromised card on both sides of the "
        f"line, so the model is scored partly on cards it has already seen, a situation no "
        f"deployed detector is ever in. It reports average precision "
        f"**{sc['shuffled']['average_precision']:.3f}** against the honest "
        f"**{sc['chronological']['average_precision']:.3f}**, an overstatement of "
        f"**{sc['overstatement']:.0%}**."
    )
    st.info(
        f"**So what would I tell a fraud team?** Use the tuned model, expect about "
        f"**{pk['point']:.0%} of a {k0}-alert queue to be fraud** and the top twenty to be "
        f"nearly all of it, and do not believe any evaluation that shuffled. Rebalance: "
        f"it costs nothing and it is the difference between a model and a constant at "
        f"{sp['train']['prevalence']:.3%}, but do not expect it to be worth much once the "
        f"model is tuned, and do not let anyone quote you a number for it that was not "
        f"paired. Of the three decisions on this page, splitting the data honestly is worth "
        f"the most, tuning is next, and the treatment is the one that matters least at the "
        f"top and most at the bottom.",
        icon=":material/flag:")

    st.divider()
    with st.expander("Scope and data limits"):
        st.markdown(
            f"- **A prototype**, not a deployed system. One dataset, one machine, no "
            f"monitoring, no drift handling.\n"
            f"- **{ds['frauds']} frauds total and {sp['test']['frauds']} in the test "
            f"split.** Every interval here is wide and the fourth decimal of any of these "
            f"numbers is noise.\n"
            f"- **The features are somebody else's PCA.** Nothing here can say which real "
            f"behaviour drives a score, so none of it is an explanation of fraud.\n"
            f"- **Two days of one issuer's traffic in 2013.** Prevalence, amounts and "
            f"patterns all differ elsewhere, and prevalence already drifts "
            f"{sp['train']['prevalence']:.3%} → {sp['test']['prevalence']:.3%} inside this "
            f"file.\n"
            f"- The shipped page carries one score and one outcome per held-out "
            f"transaction and **no features at all**.\n\n"
            f"*{ref['_derived_from']} — {ref['citation']}*")


if __name__ == "__main__":
    main()
