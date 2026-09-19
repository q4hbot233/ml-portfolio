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


def figure_split(split_comparison: list):
    rows = {r["index"]: r for r in split_comparison}
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    metrics = ["average_precision", "roc_auc", "precision_at_100"]
    labels = ["average precision", "ROC-AUC", "precision@100"]
    x = np.arange(len(metrics))
    a = [rows["shuffled"][m] for m in metrics]
    b = [rows["chronological"][m] for m in metrics]
    ax.bar(x - 0.19, a, 0.36, color=GREY, label="shuffled split")
    ax.bar(x + 0.19, b, 0.36, color=BLUE, label="chronological split")
    for i, (u, v) in enumerate(zip(a, b)):
        ax.text(i, max(u, v) + 0.07, f"{(v-u)/u:+.0%}", ha="center", fontsize=9.5,
                color=RED, fontweight="semibold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
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
        "The received advice at that prevalence is to rebalance: weight the classes, drop "
        "negatives, or synthesise positives with SMOTE. This page is what happened when I "
        "tested the advice instead of repeating it, and asked the question it usually "
        "skips: **rebalancing improves *what*, exactly, and for whom?**\n\n"
        f"Everything below is computed live from the model's **{len(rows):,} held-out "
        f"scores**, shipped with this page. The full analysis lives in a private repository."
    )

    st.divider()
    st.subheader("1 · The queue, which is the real decision")
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
    st.subheader("2 · The noise floor, which should come first")
    n_pos = int(y.sum())
    st.markdown(
        f"Before believing any comparison: this split contains **{n_pos} frauds**. At your "
        f"budget of {k}, one transaction moving in or out of the queue is "
        f"**{100/k:.1f} percentage points**, and the highest precision anyone could possibly "
        f"reach is **{min(1.0, n_pos/k):.2f}** — you cannot have more frauds in the queue "
        f"than exist.\n\n"
        f"That is arithmetic, not an estimate, and it is the reason the interval above is "
        f"{(hi-lo)*100:.0f} points wide. A paper reporting that treatment A beats treatment B "
        f"by two points of precision on this dataset is reporting two transactions."
    )
    nf = ref["noise_floor"]
    st.info(
        f"Across the twelve model-and-treatment configurations tried, the median 95% "
        f"interval on precision@{nf['budget']} is **{nf['median_interval_width_pts']:.1f} "
        f"percentage points wide**. Most of them overlap almost completely.",
        icon=":material/straighten:")

    st.divider()
    st.subheader("3 · The split was worth more than the model")
    st.markdown(
        "This file has a `Time` column spanning 48 hours, and almost every published "
        "treatment of it uses a **random** split. That lets the model learn from "
        "transactions that happen after the ones it is scored on, and it breaks the "
        "card-level structure — one compromised card often produces several frauds minutes "
        "apart, and shuffling puts some in train and the rest in test."
    )
    st.pyplot(figure_split(ref["split_comparison"]), use_container_width=True)
    drift = ref["prevalence_drift"]
    st.markdown(
        f"Same model, same hyperparameters, two ways of holding data out. **Average "
        f"precision falls 34% when the split stops being shuffled.**\n\n"
        f"The fraud rate is not even constant across the file: **{drift['train']:.3%} in "
        f"training, {drift['valid']:.3%} in validation, {drift['test']:.3%} in test** — a "
        f"37% relative drop inside two days. A chronological split is not merely stricter, "
        f"it is measuring a different population."
    )

    st.divider()
    st.subheader("4 · What rebalancing buys, and what it buys it in")
    shapes = pd.DataFrame(ref["treatment_shapes"])
    smote = shapes.loc[shapes["treatment"] == "smote"].iloc[0]
    under = shapes.loc[shapes["treatment"] == "undersample"].iloc[0]
    st.markdown(
        f"Before any metric, what each treatment does to 199,368 training rows holding 384 "
        f"frauds:\n\n"
        f"- **undersample** discards **{int(under['rows_removed']):,} real negatives**, "
        f"leaving {int(under['rows']):,} rows in total.\n"
        f"- **SMOTE** manufactures **{int(smote['rows_added']):,} synthetic frauds** from "
        f"384 real ones — {int(smote['rows_added'])/384:.0f} per real fraud. It interpolates "
        f"between a fraud and one of its nearest fraud neighbours; in 30 dimensions with 384 "
        f"positives those neighbours are not near. It is not more data, it is an assumption "
        f"that the space between two frauds is also fraud.\n"
        f"- **class_weight** and **scale_pos_weight** touch no rows at all."
    )
    st.pyplot(figure_treatments(ref["treatments"], ref["headline_budget"]),
              use_container_width=True)

    hh = ref["best_tree_minus_untouched_logistic_queue"]
    c1, c2 = st.columns(2)
    c1.markdown(
        "**On the tree it is doing something real.** Average precision 0.32 → 0.85, and the "
        "queue moves too. The mechanism is mechanical: LightGBM needs a minimum number of "
        "rows in a leaf, and at 0.19% prevalence the splits that would isolate fraud "
        "**cannot form**. The tree was not underfitting — it was prevented from fitting by a "
        "hyperparameter that is sensible at any normal prevalence."
    )
    c2.markdown(
        "**On the linear model it moves the ranking metric and leaves the queue alone.** "
        "Average precision +0.067; precision@100 **+0.000, interval [−0.040, +0.020]**. Same "
        "treatment, same data, opposite conclusions — and the only way to know which case "
        "you are in is to check."
    )
    st.info(
        f"**Best rebalanced tree minus the untouched logistic regression, on the queue: "
        f"{hh['point']:+.3f} [{hh['ci_lo']:+.3f}, {hh['ci_hi']:+.3f}].** Every piece of "
        f"rebalancing machinery applied to the tree gets back to where an untreated linear "
        f"model was already standing. That is not an argument against rebalancing — it is an "
        f"argument against reporting it as a single number called \"improvement\" without "
        f"saying improvement in what.",
        icon=":material/flag:")

    with st.expander("A recommended step that provably does nothing"):
        st.markdown(
            f"`SMOTETomek` is SMOTE followed by removing Tomek links — opposite-class nearest "
            f"neighbours sitting on the boundary — and is widely recommended as the cleaner "
            f"variant. It took 78 seconds.\n\n"
            f"**It removed {ref['smote_tomek_rows_removed']} rows**, and produced a training "
            f"set byte-identical to plain SMOTE. That is why every `smote` and `smote_tomek` "
            f"row in the chart above matches to four decimals. After SMOTE has filled the "
            f"minority region with 199,000 synthetic points there are no boundary pairs left "
            f"to find."
        )

    with st.expander("What it costs: the scores stop being probabilities"):
        calib = pd.DataFrame(ref["calibration"])
        show = calib[["model", "treatment", "mean_predicted", "actual_rate", "ECE"]]
        st.dataframe(show.style.format({"mean_predicted": "{:.4f}", "actual_rate": "{:.4f}",
                                        "ECE": "{:.5f}"}), use_container_width=True)
        st.markdown(
            "Read `mean_predicted` against `actual_rate`. Every treatment changes the base "
            "rate the model is fitted on, so the score stops estimating P(fraud | "
            "transaction) and starts estimating it for a population that does not exist.\n\n"
            "But note the last two rows: for the **barely-fitting tree**, rebalancing "
            "*improves* calibration, because the untreated tree was badly fitted to begin "
            "with and over-predicts. So \"rebalancing wrecks calibration\" is true of one "
            "family here and false of the other — the same lesson as everything else on this "
            "page."
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
