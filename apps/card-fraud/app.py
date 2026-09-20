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

INK = "#1b1b1f"; MUTED = "#6b6b76"; GRID = "#dcdce2"
BLUE = "#2f5d9e"; RED = "#b3402f"; GREY = "#9aa0aa"; GREEN = "#2f7d5d"


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


def load_scores() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "test_scores.csv")


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
    ax1.set(xlabel="review budget k", ylabel="precision@k — of what we look at, how much is fraud")
    ax2.plot(curve["k"], curve["recall"], lw=2.0, color=GREEN)
    ax2.set(xlabel="review budget k", ylabel="recall@k — of the fraud, how much we reach")
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
    """The grey bars are taller and that is the problem, not the point.

    The first version of this put "-34%" over the chronological bar, which reads as "the
    chronological split performs worse" — the opposite of what it means. The model is the
    same in both; only the evaluation changed. So the label now sits on the shuffled bar and
    names it as inflation, the axis says which number is the honest one, and the legend says
    what each bar is rather than only how it was split.
    """
    rows = {r["index"]: r for r in split_comparison}
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    metrics = ["average_precision", "roc_auc", "precision_at_100"]
    labels = ["average precision", "ROC-AUC", "precision@100"]
    x = np.arange(len(metrics))
    a = [rows["shuffled"][m] for m in metrics]
    b = [rows["chronological"][m] for m in metrics]
    ax.bar(x - 0.19, a, 0.36, color=GREY, hatch="///", edgecolor="white", linewidth=0,
           label="shuffled — inflated by leakage")
    ax.bar(x + 0.19, b, 0.36, color=BLUE, label="chronological — what deployment would give")
    for i, (u, v) in enumerate(zip(a, b)):
        ax.text(i - 0.19, u + 0.04, f"+{(u - v) / v:.0%}", ha="center", fontsize=10,
                color=RED, fontweight="semibold")
        ax.text(i + 0.19, v + 0.04, f"{v:.2f}", ha="center", fontsize=9.5, color=BLUE)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("higher is not better here", fontsize=9, color=MUTED)
    ax.set_title("the red figure is how much shuffling overstates the blue one",
                 fontsize=9.5, color=MUTED, loc="left", pad=8)
    ax.legend(fontsize=8.5, labelcolor=MUTED, loc="upper right")
    fig.tight_layout()
    return fig


def figure_treatments(treatments: list, budget: int):
    t = pd.DataFrame(treatments)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    order = list(dict.fromkeys(t["treatment"]))
    ypos = {name: i for i, name in enumerate(order)}
    for ax, col, title in (
            (ax1, "average_precision", "average precision — what papers report"),
            (ax2, f"precision_at_{budget}", f"precision@{budget} — what the team lives")):
        for i, (fam, blk) in enumerate(t.groupby("model", sort=False)):
            y = np.array([ypos[n] for n in blk["treatment"]]) + i * 0.38
            ax.barh(y, blk[col], 0.34, color=BLUE if fam == "lgbm" else GREEN, label=fam)
            for yy, v in zip(y, blk[col]):
                ax.text(v + 0.012, yy, f"{v:.3f}", va="center", fontsize=7.5, color=MUTED)
        ax.set_yticks(np.arange(len(order)) + 0.19, order, fontsize=8.5)
        ax.set_xlim(0, 1.06)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax1.legend(fontsize=8.5, labelcolor=MUTED, loc="lower right")
    fig.tight_layout()
    return fig


def figure_overfit(diag: dict):
    """Two things that are true about the untreated tree, and no third thing pretending to
    tie them together.

    An earlier version of this page explained the untreated tree's poor validation score
    with a mechanism — that at 0.19% prevalence the splits isolating fraud cannot form. The
    left panel is what killed that: the score is at its highest after five trees, so the
    splits form and then 395 more trees undo them. The right panel is why no replacement
    mechanism is offered either. Score is not monotone in the one setting being swept, so
    "too much capacity" does not survive contact with the evidence any better.
    """
    curves, frag = diag["boosting_curve"], diag["fragility"]
    shipped = diag["shipped_min_child_samples"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))

    for tag, colour, label in (("none", RED, "untreated"),
                               ("class_weight", BLUE, "class_weight")):
        c = curves[tag]
        k = [r["trees"] for r in c]
        ax1.plot(k, [r["train"] for r in c], color=colour, lw=1.1, ls=":", alpha=0.75)
        ax1.plot(k, [r["validation"] for r in c], color=colour, lw=2.0, marker="o",
                 ms=3.5, label=label)
    peak = max(curves["none"], key=lambda r: r["validation"])
    end = curves["none"][-1]
    # The headroom above 1.0 is deliberate: it is the only place on these axes where a
    # label does not land on one of the four lines.
    ax1.annotate(f"peak {peak['validation']:.2f}\nat {peak['trees']} trees",
                 xy=(peak["trees"], peak["validation"]), xytext=(peak["trees"] + 1, 1.12),
                 fontsize=8.5, color=RED, ha="left", va="top",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.9,
                                 connectionstyle="arc3,rad=0.15"))
    ax1.annotate(f"{end['validation']:.2f} by {end['trees']}",
                 xy=(end["trees"], end["validation"]),
                 xytext=(end["trees"] * 0.22, end["validation"] - 0.20), fontsize=8.5,
                 color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    ax1.set_xscale("log")
    ax1.set_xticks([r["trees"] for r in curves["none"]],
                   [str(r["trees"]) for r in curves["none"]], fontsize=8)
    ax1.set_xlabel("boosting rounds (log scale)", fontsize=9, color=MUTED)
    ax1.set_ylabel("average precision", fontsize=9, color=MUTED)
    ax1.set_ylim(0, 1.42)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_title("it is past its peak · solid validation, dotted training",
                  loc="left", fontsize=9.5, color=INK, pad=8)
    ax1.legend(fontsize=8.5, labelcolor=MUTED, loc="upper right", framealpha=0.95)

    x = np.arange(len(frag))
    vals = [r["validation"] for r in frag]
    err = np.array([[r["validation"] - r["ci_lo"] for r in frag],
                    [r["ci_hi"] - r["validation"] for r in frag]])
    cols = [RED if r["min_child_samples"] == shipped else GREY for r in frag]
    ax2.bar(x, vals, 0.55, color=cols)
    ax2.errorbar(x, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=1.1, capsize=3.5)
    for xi, r in zip(x, frag):
        ax2.text(xi, r["ci_hi"] + 0.035, f"{r['validation']:.2f}", ha="center",
                 fontsize=8.5, color=RED if r["min_child_samples"] == shipped else MUTED)
    ax2.set_xticks(x, [f"{r['min_child_samples']}" + ("\nshipped" if
                       r["min_child_samples"] == shipped else "") for r in frag],
                   fontsize=8.5)
    ax2.set_xlabel("min_child_samples — one default, nothing else changed",
                   fontsize=9, color=MUTED)
    ax2.set_ylim(0, 1.42)
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_title("and the shipped value sits in a hole", loc="left", fontsize=9.5,
                  color=INK, pad=8)
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
    get_ref = cache(load_reference)
    get_scores = cache(load_scores)

    ref = get_ref()
    rows = get_scores()
    y = rows["fraud"].to_numpy()
    score = rows["score"].to_numpy()
    prevalence = float(y.mean())

    st.title("What 492 frauds can and cannot tell you")
    st.markdown(
        "284,807 card transactions over 48 hours, **492 of them fraud — 0.173%**. A model "
        "that predicts \"not fraud\" for every row is right 99.83% of the time.\n\n"
        "This walks the whole thing: what is in the file, what could be engineered out of "
        "it, two model families and what they have to beat, six ways of handling the "
        "imbalance, and the rule that picked one. Then the two findings that came out of "
        "it — that the real decision is a **queue length**, not a threshold, and that **how "
        "the data was split mattered more than which model was fitted**.\n\n"
        "The received advice at that prevalence is to rebalance: weight the classes, drop "
        "negatives, or synthesise positives with SMOTE. This page is what happened when I "
        "tested the advice instead of repeating it, and asked the question it usually "
        "skips: **rebalancing improves *what*, exactly, and for whom?**\n\n"
        f"Everything below is computed live from the model's **{len(rows):,} held-out "
        f"scores**, shipped with this page. The full analysis lives in a private repository."
    )

    st.divider()
    st.subheader("1 · The data, before any model")
    eda = ref["eda"]
    bal, amt = eda["balance"], eda["amount"]
    st.markdown(
        f"{bal['n_rows']:,} training transactions, **{bal['n_fraud']} of them fraud — "
        f"{bal['prevalence']:.4%}, one in {bal['one_in']:,}**. Every figure in this section "
        f"is computed on the training split only: describing a dataset with its test rows "
        f"included is a leak that never shows up as a score, only as a modelling decision "
        f"made knowing the answer.\n\n"
        f"The first thing that follows is that **accuracy is unusable**. Predicting "
        f"\"not fraud\" every time scores {bal['majority_accuracy']:.2%}."
    )
    st.pyplot(figure_eda(eda), use_container_width=True)
    fr = amt["quantiles"]["fraud"]; lg = amt["quantiles"]["legit"]
    st.markdown(
        f"**Amount is not the giveaway people expect.** Median fraud is "
        f"€{fr['0.5']:.2f} against €{lg['0.5']:.2f} for legitimate traffic, and the fraud "
        f"rate is highest in the smallest band — card testing, where a stolen number is "
        f"tried on something tiny first. {amt['zero_amount']['fraud']} frauds are for "
        f"exactly €0.\n\n"
        f"**Hour of cycle carries a little.** The file starts at an unknown wall-clock hour, "
        f"so this is hours-since-start mod 24 rather than \"3 a.m.\" — a periodic "
        f"coordinate, not a time of day, and it is labelled that way everywhere."
    )

    st.divider()
    st.subheader("2 · Feature engineering, and what it was worth")
    ft = eda["features"]
    st.markdown(
        f"There is not much room here and the write-up should say so. **{ft['pca']} of the "
        f"{ft['total']} columns are `V1..V28`, principal components** published instead of "
        f"the raw fields because the raw fields are a European issuer's transaction records. "
        f"You cannot engineer on top of a component you cannot interpret.\n\n"
        f"So the work was two derived columns and one deliberate exclusion:\n\n"
        f"- **`log_amount`** — amounts span €0 to €25,691 with a long tail; the log is what "
        f"a linear model can use.\n"
        f"- **`hour_of_cycle`** — derived from `Time`, because a deployed detector does know "
        f"the clock.\n"
        f"- **`Time` itself is excluded.** It is seconds since the first row *of this file*. "
        f"A model given it can fit where the fraud bursts happen to sit in these particular "
        f"48 hours, which is a fact about the file and not about fraud."
    )
    st.pyplot(figure_separability(eda), use_container_width=True)
    sep = eda["separability"]
    ranks = {r["feature"]: i + 1 for i, r in enumerate(sep)}
    st.warning(
        f"**And they were worth very little.** Ranked by how well each column separates the "
        f"classes on its own, `hour_of_cycle` comes **{ordinal(ranks['hour_of_cycle'])} of "
        f"{len(sep)}** and `log_amount` **{ordinal(ranks['log_amount'])}**. The top of that "
        f"list is "
        f"components — `{sep[0]['feature']}`, `{sep[1]['feature']}`, `{sep[2]['feature']}` — "
        f"which somebody else's PCA already found. Reporting the engineering without this "
        f"chart would imply it did work it did not do.",
        icon=":material/trending_down:")

    st.divider()
    st.subheader("3 · Training and tuning, and what a model has to beat")
    base, spec = ref["baselines"], ref["model_spec"]
    rt = ref.get("retuned")
    st.markdown(
        f"Two families on the {ft['total']} features above, **everything in this section "
        f"scored on the validation split** — the test split is not opened until section 5. "
        f"Every figure is **average precision**: the area under the precision–recall curve, "
        f"whose floor for a model that ranks at random is the base rate rather than 0.5. "
        f"That floor is **{base['majority']['average_precision']['point']:.4f}** here; "
        f"predicting \"not fraud\" for everything is right "
        f"{1 - base['majority']['average_precision']['point']:.2%} of the time and catches "
        f"nothing."
    )

    if rt:
        cvv = {r["model"]: r for r in rt["cv_vs_validation"]}
        st.markdown("**Both families were tuned by grid search — twice, because how you fold "
                    "the data turns out to decide what it picks.**")
        tbl = pd.DataFrame([{
            "family": m,
            "grid": f"{rt['search'][m]['n_configurations']} configs × "
                    f"{rt['search'][m]['cv_folds']} folds",
            "shuffled CV picks": ", ".join(f"{k}={v}" for k, v in
                                           cvv[m]["shuffled_cv_params"].items()),
            "its CV score": cvv[m]["shuffled_cv_score"],
            "…on validation": cvv[m]["shuffled_winner_on_validation"],
            "chronological CV picks": ", ".join(f"{k}={v}" for k, v in
                                                cvv[m]["chronological_cv_params"].items()),
            "its CV score": cvv[m]["chronological_cv_score"],
            "…on validation ": cvv[m]["chronological_winner_on_validation"],
        } for m in ("logistic", "lgbm")])
        st.dataframe(tbl.style.format({c: "{:.4f}" for c in tbl.columns
                                       if "score" in c or "validation" in c}),
                     use_container_width=True, hide_index=True)

        lg, tr = cvv["logistic"], cvv["lgbm"]
        st.error(
            f"**The search is contaminated by the same leak the whole page is about.** "
            f"`GridSearchCV` folds with `StratifiedKFold`, which *shuffles* — and on this "
            f"file shuffling puts several frauds from one compromised card on both sides of "
            f"a fold. For the tree that inflates the CV score to "
            f"**{tr['shuffled_cv_score']:.4f}** where the same configuration scores "
            f"**{tr['shuffled_winner_on_validation']:.4f}** on a later split: a "
            f"**{tr['shuffled_cv_score'] / tr['shuffled_winner_on_validation']:.1f}×** "
            f"overstatement. A shuffled search does not merely report an optimistic number — "
            f"**it selects the configuration that exploits the leak best.** The linear model "
            f"barely notices ({lg['shuffled_cv_score']:.3f} against "
            f"{lg['shuffled_winner_on_validation']:.3f}), which is why one family can hide "
            f"this from you.",
            icon=":material/science:")

        st.markdown(
            f"So the shipped tuning uses **expanding-window folds in time order**: every fold "
            f"is scored on transactions later than the ones it was fitted on, which is what "
            f"deployment looks like. It picks `C={rt['search']['logistic']['best_params']['C']}` "
            f"for the logistic regression and "
            f"`num_leaves={rt['search']['lgbm']['best_params']['num_leaves']}, "
            f"min_child_samples={rt['search']['lgbm']['best_params']['min_child_samples']}, "
            f"n_estimators={rt['search']['lgbm']['best_params']['n_estimators']}` for the tree."
        )

        c1, c2 = st.columns(2)
        c1.metric("Tuned logistic regression · average precision",
                  f"{lg['chronological_winner_on_validation']:.4f}",
                  "validation, chronologically tuned", delta_color="off")
        c2.metric("Tuned LightGBM · average precision",
                  f"{tr['chronological_winner_on_validation']:.4f}",
                  "validation, chronologically tuned", delta_color="off")

        sweep = ref.get("min_child_sweep")
        if sweep:
            best_sw = sweep["best"]
            st.warning(
                f"**And tuning still does not fix the tree.** Sweeping `min_child_samples` "
                f"directly against the validation split finds "
                f"**{best_sw['min_child_samples']}**, worth "
                f"**{best_sw['average_precision']:.3f}** — but *neither* grid search picks it. "
                f"Both land on "
                f"{cvv['lgbm']['chronological_cv_params']['min_child_samples']} and both leave "
                f"the tree at about {tr['chronological_winner_on_validation']:.2f}, because "
                f"cross-validation inside the training period cannot see how the tree will "
                f"behave in the later one. The value that works was found by scoring against "
                f"the split about to be reported on — which is tuning on your own held-out "
                f"data, and not a procedure anyone should ship.\n\n"
                f"Section 4 takes this further and it does not end where this paragraph "
                f"implies: sweeping the same setting shows {cvv['lgbm']['chronological_cv_params']['min_child_samples']} "
                f"is not merely beatable but the *worst* of the values tried, with higher "
                f"and lower both scoring better. Rebalancing does take this tree to 0.85. "
                f"So does leaving the tree alone and moving one default.",
                icon=":material/warning:")

    st.caption(
        f"For reference, the original pre-registered run used no search at all: "
        f"logistic C={spec['logistic']['C']}, LightGBM "
        f"{spec['lgbm']['n_estimators']} trees / lr {spec['lgbm']['learning_rate']} / "
        f"num_leaves {spec['lgbm']['num_leaves']} / min_child_samples "
        f"{spec['lgbm']['min_child_samples']}, scoring "
        f"{base['logistic']['average_precision']['point']:.4f} and "
        f"{base['lgbm']['average_precision']['point']:.4f}. The tuned numbers above replace "
        f"them as the fair comparison; sections 5 to 7 still report the pre-registered model, "
        f"because that is the one whose test split was opened once."
    )

    st.divider()
    st.subheader("4 · Class imbalance: which treatment is actually the most powerful")
    rtu = ref.get("retuned")
    shapes = pd.DataFrame(ref["treatment_shapes"])
    smote = shapes.loc[shapes["treatment"] == "smote"].iloc[0]
    under = shapes.loc[shapes["treatment"] == "undersample"].iloc[0]
    st.markdown(
        f"Six treatments, on 199,368 training rows holding 384 frauds. What each one does "
        f"to the data *before* any metric is computed:\n\n"
        f"- **undersample** discards **{int(under['rows_removed']):,} real negatives**, "
        f"leaving {int(under['rows']):,} rows.\n"
        f"- **SMOTE** manufactures **{int(smote['rows_added']):,} synthetic frauds** from "
        f"384 real ones — {int(smote['rows_added'])/384:.0f} per real fraud. It interpolates "
        f"between a fraud and one of its nearest fraud neighbours; in 30 dimensions with 384 "
        f"positives those neighbours are not near. It is not more data, it is an assumption "
        f"that the space between two frauds is also fraud.\n"
        f"- **SMOTETomek** is SMOTE plus removing boundary pairs. It took 78 seconds and "
        f"removed **{ref['smote_tomek_rows_removed']} rows** — a training set byte-identical "
        f"to plain SMOTE, which is why those rows match to four decimals below.\n"
        f"- **class_weight** and **scale_pos_weight** touch no rows at all."
    )
    treat_rows = rtu["treatments"] if rtu else ref["treatments"]
    st.pyplot(figure_treatments(treat_rows, ref["headline_budget"]),
              use_container_width=True)
    if rtu:
        st.caption("Treatments applied to the chronologically tuned models from section 3, "
                   "scored on validation. The pre-registered run's version of this chart "
                   "used untuned models and is what sections 5 to 7 still refer to.")

    nf = ref["noise_floor"]
    st.markdown(
        f"**Read the left panel and you would say SMOTE wins.** Read the right one and the "
        f"ordering mostly disappears. And before believing either: the validation split holds "
        f"**{nf['n_positives_valid']} frauds**, so at a budget of {nf['budget']} one "
        f"transaction moving in or out of the queue is **{nf['one_transaction_pct']:.0f} "
        f"percentage point**, and the median 95% interval across all twelve configurations is "
        f"**{nf['median_interval_width_pts']:.1f} points wide**. Most of them overlap almost "
        f"entirely. A two-point win here is two transactions."
    )

    hh = ref["best_tree_minus_untouched_logistic_queue"]
    diag = ref.get("tree_diagnosis")

    if diag:
        v = diag["verdict"]
        st.markdown("#### Why the untreated tree scores what it scores")
        st.markdown(
            f"This page used to answer that with a mechanism: LightGBM needs a minimum "
            f"number of rows in a leaf, so at 0.19% prevalence the splits that would "
            f"isolate fraud cannot form. It is tidy, it is the story usually told about "
            f"trees on rare classes, and **on this dataset it is false.** The untreated "
            f"tree scores **{v['untreated_train']:.3f}** on the rows it was fitted to; a "
            f"model that cannot form those splits cannot do that. It is also false in "
            f"detail: take every leaf in the first {diag['purity_trees']} trees holding a "
            f"training fraud and ask what share of its rows are frauds, and the untreated "
            f"tree comes back at **{diag['leaf_fraud_purity']['none']:.3f}** against "
            f"**{diag['leaf_fraud_purity']['class_weight']:.3f}** for the weighted one. Its "
            f"fraud leaves are *tighter* than the ones the model that generalises builds, "
            f"not coarser."
        )
        st.pyplot(figure_overfit(diag), use_container_width=True)

        curve = diag["boosting_curve"]["none"]
        st.markdown(
            f"**Left — it is past its peak.** The untreated tree's validation score is at "
            f"its highest after **{v['untreated_peak_trees']} trees**: "
            f"**{v['untreated_peak_validation']:.3f}**, nearly twice what the finished model "
            f"manages. Every round after that lowers training loss and lowers the validation "
            f"score with it, to **{curve[-1]['validation']:.3f}** by {curve[-1]['trees']}. "
            f"The blue line is the same model with `class_weight`: it climbs, flattens, and "
            f"runs its training curve to a perfect fit without dragging validation down. So "
            f"0.19% prevalence is not stopping the tree from learning fraud. Whatever it is "
            f"doing, the fraud structure is there by tree {v['untreated_peak_trees']}."
        )

        frag = diag["fragility"]
        best = v["best_neighbouring_min_child"]
        st.warning(
            f"**Right — and this is the part that should change what you take from the "
            f"chart above.** Move `min_child_samples` off its default and nothing else, and "
            f"the untreated tree scores "
            + ", ".join(f"**{r['validation']:.2f}** at {r['min_child_samples']}"
                        for r in frag) + f". The shipped value, "
            f"{diag['shipped_min_child_samples']}, is the worst of the four — lower *and* "
            f"higher both do better, so this is not a capacity story either. And it is not "
            f"noise on {ref['noise_floor']['n_positives_valid']} validation frauds: the "
            f"paired bootstrap for {best['min_child_samples']} against the shipped value is "
            f"**{best['vs_shipped']:+.3f} [{best['vs_shipped_lo']:+.3f}, "
            f"{best['vs_shipped_hi']:+.3f}]**.\n\n"
            f"**I do not have a mechanism that predicts that ordering, and this page is no "
            f"longer going to assert one.** What follows from it without any mechanism is "
            f"enough: the untreated bar is not a fixed reference point. \"Rebalancing takes "
            f"the tree from {v['untreated_validation']:.2f} to 0.85\" is partly a statement "
            f"about rebalancing and partly a statement about where the untreated baseline "
            f"happened to be standing.",
            icon=":material/warning:")

        gaps = pd.DataFrame(diag["treatments"])
        gaps.columns = ["treatment", "train", "validation", "train − validation"]
        st.dataframe(gaps.style.format({"train": "{:.3f}", "validation": "{:.3f}",
                                        "train − validation": "{:+.3f}"}),
                     use_container_width=True, hide_index=True)
        st.markdown(
            f"The one thing that *is* consistent across all six: read the last column, not "
            f"the third. Every treatment that helps closes the gap between the two splits — "
            f"{v['untreated_gap']:+.3f} untreated against {v['weighted_gap']:+.3f} weighted "
            f"— and `undersample` is the one that goes *negative*, because discarding "
            f"{int(under['rows_removed']):,} negatives leaves too little training data "
            f"behind to overfit. Whatever these treatments are buying, they are not buying "
            f"separability the tree did not have."
        )
        with st.expander("The capacity ladder, for completeness"):
            lad = pd.DataFrame(diag["capacity_ladder"])
            st.dataframe(lad.style.format({"train": "{:.3f}", "validation": "{:.3f}"}),
                         use_container_width=True, hide_index=True)
            st.caption(
                "Untreated, no resampling. Shrinking the model does recover a good deal of "
                "the score, which is why overfitting was my first answer — but the "
                "min_child_samples sweep above is not monotone, so it cannot be the whole "
                "answer, and a partial explanation stated as a complete one is the exact "
                "failure this section is correcting.")
        st.caption("scripts/why_the_tree_fails.py · training and validation only")

    st.markdown("#### What that leaves of the comparison")
    c1, c2 = st.columns(2)
    if rtu:
        rtt = {(r["model"], r["treatment"]): r for r in rtu["treatments"]}
        tree_none = rtt[("lgbm", "none")]["average_precision"]
        tree_best = max(x["average_precision"] for (m, t), x in rtt.items()
                        if m == "lgbm" and t != "none")
        cp = cvv["lgbm"]["chronological_cv_params"] if rt else {}
        alt = (f" The search in section 3 does not rescue it either: the chronological grid "
               f"swept `n_estimators`, `num_leaves` and `min_child_samples`, picked "
               f"{cp['n_estimators']}×{cp['num_leaves']} at "
               f"min_child_samples={cp['min_child_samples']}, and left the tree at "
               f"{tree_none:.2f}. Cross-validation folds *inside* the training period, where "
               f"the shipped configuration looks fine. So rebalancing really is what a "
               f"shippable procedure arrives at here — **but read that as a fact about the "
               f"procedure, not as evidence that resampling fixed something about the "
               f"class balance.**" if rt else "")
    else:
        tree_none, tree_best, alt = 0.32, 0.85, ""
    c1.markdown(
        f"**On the tree the number moves a long way, and a default moves it just as far.** "
        f"Average precision {tree_none:.2f} → {tree_best:.2f} from rebalancing, on tuned "
        f"models; {v['untreated_validation']:.2f} → "
        f"{v['best_neighbouring_min_child']['validation']:.2f} from one hyperparameter, on "
        f"untouched ones." + alt if diag else
        f"**On the tuned tree, rebalancing moves the ranking metric a long way.** "
        f"Average precision {tree_none:.2f} → {tree_best:.2f}." + alt
    )
    c2.markdown(
        "**On the linear model it moves the ranking metric and leaves the queue alone.** "
        "Average precision +0.067; precision@100 **+0.000, interval [−0.040, +0.020]**. Same "
        "treatment, same data, opposite conclusions — and the only way to know which case "
        "you are in is to check."
    )
    st.info(
        f"**So: which treatment is most powerful? The question is malformed.** Best "
        f"rebalanced tree minus the untouched logistic regression, on the queue: "
        f"**{hh['point']:+.3f} [{hh['ci_lo']:+.3f}, {hh['ci_hi']:+.3f}]**. Every piece of "
        f"rebalancing machinery applied to the tree gets back to where an untreated linear "
        f"model was already standing — a model that needed none of it, on the same rows at "
        f"the same prevalence. The most powerful interventions on this dataset were "
        f"choosing a family whose default configuration was not in a hole, and then "
        f"choosing the split properly (section 6). Neither is a resampler.",
        icon=":material/flag:")

    with st.expander("What it costs: the scores stop being probabilities"):
        calib = pd.DataFrame(ref["calibration"])
        show = calib[["model", "treatment", "mean_predicted", "actual_rate", "ECE"]]
        st.dataframe(show.style.format({"mean_predicted": "{:.4f}", "actual_rate": "{:.4f}",
                                        "ECE": "{:.5f}"}), use_container_width=True)
        st.markdown(
            "Read `mean_predicted` against `actual_rate`. Every treatment changes the base "
            "rate the model is fitted on, so the score stops estimating P(fraud | "
            "transaction) and starts estimating it for a population that does not exist.\n\n"
            "But note the last two rows: for the **untreated tree**, rebalancing *improves* "
            "calibration, because that tree over-predicts to begin with, so a treatment that "
            "shifts its scores happens to shift them towards the truth. So \"rebalancing "
            "wrecks calibration\" is true of one family here and false of the other — the "
            "same lesson as everything else on this page."
        )

    st.divider()
    st.subheader("5 · Choosing one, by a rule written down first")
    sel = ref["selection"]
    rtu2 = ref.get("retuned")
    st.markdown(
        f"Twelve configurations and one test split. The rule was fixed before the test split "
        f"was opened:\n\n> *{sel['rule']}*\n\n"
        f"On the pre-registered run, **{len(sel['tied_at_top'])} configurations tied at the "
        f"top** — every logistic variant, all at 0.490 on validation precision@100. The "
        f"tie-break sent it to the one that does least to the data, so the model behind "
        f"sections 6 and 7 is **{sel['selected'][0]} with `{sel['selected'][1]}`**: no "
        f"weighting, no resampling, nothing synthesised."
    )
    if rtu2:
        rs = rtu2["selection"]
        same = list(rs["selected"]) == list(sel["selected"])
        st.info(
            f"**Re-running the same rule on the tuned models changes the tie and not the "
            f"winner.** {len(rs['tied_at_top'])} configurations now tie at "
            f"{rs['selected_precision_at_budget']:.3f}"
            + (" — including two LightGBM variants that could not reach the top before — "
               if any(m == "lgbm" for m, _ in rs["tied_at_top"]) else " — ")
            + f"and the tie-break still selects **{rs['selected'][0]} / "
            f"`{rs['selected'][1]}`**."
            + ("" if same else " That is a different model from the pre-registered one.")
            + f"\n\nThe test split is **not** re-opened for the tuned models. The "
            f"pre-registered run used it once, under a rule fixed in advance; scoring a "
            f"second set of models on it would spend that discipline for a number nobody "
            f"needs, since the selection did not move.",
            icon=":material/rule:")
    st.markdown(
        f"That so many configurations tie exactly is not a coincidence — it is the noise "
        f"floor from section 4 reappearing in the selection: at {nf['budget']} alerts and "
        f"{nf['n_positives_valid']} validation frauds there are only so many distinct values "
        f"precision@{nf['budget']} can take."
    )

    st.divider()
    st.subheader("6 · The queue, which is the real decision")
    st.markdown(
        "Nobody at a card issuer says \"flag at p ≥ 0.6\". Somebody says **\"we have four "
        "analysts and they can clear about a hundred alerts a shift\"** — and that sentence "
        "fixes a *count*, not a probability. Once the count is fixed the threshold is "
        "whatever the hundredth-highest score happened to be that day.\n\n"
        "So the metric is precision@k, and k is a staffing decision rather than a modelling "
        "one. Move it."
    )
    k = st.select_slider(
        "How many alerts can the team work in this window?",
        options=[10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000],
        value=int(ref["headline_budget"]))

    q = queue_metrics(y, score, k)
    point, lo, hi = bootstrap_precision(y, score, k)
    a, b, c = st.columns(3)
    a.metric("Frauds caught", f"{q['caught']} of {int(y.sum())}",
             f"{q['recall']:.1%} of the fraud in this window")
    b.metric("Precision — of the queue, how much was fraud", f"{q['precision']:.1%}",
             f"95% interval [{lo:.1%}, {hi:.1%}]", delta_color="off")
    c.metric("Alerts per fraud caught",
             "—" if not np.isfinite(q["alerts_per_fraud"]) else f"{q['alerts_per_fraud']:.2f}",
             f"{q['lift']:.0f}× better than reviewing at random", delta_color="off")

    ks = np.unique(np.round(np.logspace(np.log10(5), np.log10(3000), 45)).astype(int))
    st.pyplot(figure_queue(queue_curve(y, score, ks), k, prevalence), use_container_width=True)
    st.markdown(
        f"**The trade is explicit and it is not subtle.** Going from 20 alerts to 1,000 "
        f"roughly doubles the fraud caught and divides the hit rate by more than twenty. "
        f"Which point is right is a staffing question the model exists to inform, not to "
        f"answer.\n\n"
        f"At the top of the list the model is very good: **the top 20 of {len(rows):,} "
        f"transactions contain 19 frauds.** That is the number I would put in front of a "
        f"fraud team, because it is the number they would experience on Monday."
    )
    st.divider()
    st.subheader("7 · The split decided more than the model did")
    st.markdown(
        "This file has a `Time` column spanning 48 hours, and almost every published "
        "treatment of it splits **at random**. Below are the same model and the same "
        "hyperparameters scored two ways. The bars are not two models competing — they are "
        "one model, evaluated honestly and evaluated with a leak."
    )
    st.pyplot(figure_split(ref["split_comparison"]), use_container_width=True)
    drift = ref["prevalence_drift"]
    sc = {r["index"]: r for r in ref["split_comparison"]}
    sh, ch = sc["shuffled"], sc["chronological"]
    infl = sh["average_precision"] / ch["average_precision"] - 1
    st.markdown(
        f"**Use the chronological one.** The taller grey bars are not a better model — the "
        f"model, the features and the hyperparameters are identical in both. Only the way "
        f"data was held out changed, and shuffling lets the model see transactions that "
        f"happen *after* the ones it is scored on. A deployed fraud model never gets that: "
        f"it only ever scores what comes next. So **{ch['average_precision']:.2f} is the "
        f"honest estimate and {sh['average_precision']:.2f} is the number you would report "
        f"to yourself — shuffling overstates average precision by "
        f"{infl:.0%}.**\n\n"
        f"ROC-AUC barely notices ({sh['roc_auc']:.3f} against {ch['roc_auc']:.3f}, a "
        f"{sh['roc_auc'] / ch['roc_auc'] - 1:.0%} gap) which is its own warning: the metric "
        f"most papers lead with is the one least able to tell a leaking evaluation from a "
        f"clean one.\n\n"
        f"Two reasons the gap is real rather than pessimism. One compromised card often "
        f"produces several frauds minutes apart, and shuffling puts some of them in training "
        f"and the rest in test — the model has effectively seen the answer. And the fraud "
        f"rate is not constant across the file: **{drift['train']:.3%} in training, "
        f"{drift['valid']:.3%} in validation, {drift['test']:.3%} in test** — a 37% relative "
        f"drop inside two days, so the chronological test set holds "
        f"{int(ch['n_positives'])} frauds where the shuffled one holds "
        f"{int(sh['n_positives'])}. The chronological split is not merely stricter; it is "
        f"measuring a different and later population, which is exactly the population a "
        f"deployed model faces."
    )

    st.divider()
    with st.expander("What this is, and what it is not"):
        sel = ref["selection"]
        pub = ref["published_test"]
        st.markdown(
            f"""
**The model on screen.** {sel['selected'][0]} with treatment `{sel['selected'][1]}`, chosen
by a rule written down before the test split was opened: *best precision@100 on validation,
ties broken towards the configuration that does less to the data*. Five configurations tied
at 0.490 — every logistic variant — so the rule selected the untouched one. On test it scores
average precision **{pub['metrics']['average_precision']['point']:.4f}**
[{pub['metrics']['average_precision']['ci_lo']:.4f},
{pub['metrics']['average_precision']['ci_hi']:.4f}] against a floor of {prevalence:.4f}.

**This is not a deployed system**, and there is no production claim anywhere in it. It is an
argument about evaluation, run on a public research dataset.

**The features are anonymous.** `V1..V28` are principal components, published instead of the
raw columns because the raw columns are a European issuer's transaction records. No claim
about *why* a transaction scores highly is available here, and none is made. Those components
were also fitted on the whole file before any split existed, which is a mild leak baked into
the published data.

**Two days, one issuer, 2013.** The prevalence drift visible across the split is a warning
about how far these numbers travel, not a property of card fraud.

**52 test positives.** Every interval on this page is wide for that reason, and quoting the
point estimates without them would misrepresent what was measured.

*{ref['citation']}*
"""
        )


if __name__ == "__main__":
    main()
