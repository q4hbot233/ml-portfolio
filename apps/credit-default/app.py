"""Who pays for the model's mistakes? -- an interactive cost-sensitive decision explorer.

Runs in the browser under stlite. Everything heavy was pre-computed in the private repo and
exported to ``data/``; what happens here is arithmetic on 6,000 shipped test-set scores.

The file is deliberately split in two. Everything above the ``STREAMLIT UI`` banner is plain
Python -- functions that take arguments and return arrays, frames and dicts, with no
streamlit call anywhere inside them -- so the compute layer imports and runs under ordinary
Python. Below the banner there is no arithmetic: widgets in, those functions called, figures
out.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"

#: The four attributes the audit slices on. ``sex`` was the pre-specified primary; the other
#: three are exploratory and are labelled as such wherever they appear.
ATTRIBUTES = ("sex", "age_band", "education", "marriage")

#: The parity statistics with a shipped permutation null.

PARITY_LABEL = {
    "demographic_parity_difference": "selection-rate gap",
    "fpr_difference": "false-alarm-rate gap",
    "tpr_difference": "catch-rate gap",
}

ATTRIBUTE_LABEL = {
    "sex": "sex", "age_band": "age band",
    "education": "education", "marriage": "marital status",
}

GROUP_METRICS = ("n", "base_rate", "selection_rate", "tpr", "fpr", "precision", "fnr")

#: Groups smaller than this are shown with their n but kept out of every parity headline,
#: exactly as in the source analysis.
MIN_AUDIT_CELL = 100

#: Permutation seed, carried over from the source repo so the shipped nulls can be checked.
PERMUTATION_SEED = 907

#: Slider resolution for the threshold control.
THRESHOLD_STEP = 0.0005
THRESHOLD_MIN, THRESHOLD_MAX = 0.0050, 0.8000

INK = "#1b1b1f"
MUTED = "#6b6b76"
GRID = "#dcdce2"
BLUE = "#2f5d9e"
ORANGE = "#d1662a"
RED = "#b3402f"
GREY = "#8d8d98"
TEAL = "#2d7d74"


# =====================================================================================
# LOADING
# =====================================================================================

def load_reference(data_dir: Path | str = DATA_DIR) -> dict:
    """Frozen constants and the published headline numbers, as exported from the repo."""
    return json.loads((Path(data_dir) / "reference.json").read_text())


def load_rows(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """One row per test client: calibrated score, outcome, and the four audited attributes."""
    return pd.read_csv(Path(data_dir) / "test_predictions.csv")


# =====================================================================================
# THE DECISION
# =====================================================================================

def flag(score, threshold: float) -> np.ndarray:
    """Flag a client when the score is **at or above** the threshold.

    ``>=`` is the convention used throughout the source analysis, including in the sweep
    that chose the deployed cut, so a threshold read off the curve reproduces exactly the
    decision the curve evaluated.
    """
    return (np.asarray(score, dtype=float) >= float(threshold)).astype(int)


def confusion(y_true, y_pred) -> dict:
    """The 2x2 table and every rate read off it. ``precision`` is NaN when nobody is flagged."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    n = tp + fp + fn + tn
    pos, neg, sel = tp + fn, fp + tn, tp + fp
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n,
        "base_rate": pos / n if n else np.nan,
        "accuracy": (tp + tn) / n if n else np.nan,
        "selection_rate": sel / n if n else np.nan,
        "recall": tp / pos if pos else np.nan,
        "precision": tp / sel if sel else np.nan,
        "fpr": fp / neg if neg else np.nan,
        "fnr": fn / pos if pos else np.nan,
        "tnr": tn / neg if neg else np.nan,
    }


def expected_cost(conf: dict, cost_ratio: float) -> float:
    """Total cost in units of one false alarm: ``r * FN + FP``.

    No currency appears anywhere in this app. The dataset carries no exposure, no
    loss-given-default and no recovery rate, so a money figure would be invented.
    """
    r = float(cost_ratio)
    if r <= 0:
        raise ValueError("cost_ratio must be positive")
    return r * conf["fn"] + conf["fp"]


def bayes_threshold(cost_ratio: float) -> float:
    """``1 / (1 + r)`` -- the optimal cut for a perfectly calibrated score."""
    r = float(cost_ratio)
    if r <= 0:
        raise ValueError("cost_ratio must be positive")
    return 1.0 / (1.0 + r)


def cut_points(y_true, score) -> pd.DataFrame:
    """Confusion counts at every distinct cut of the score, in one O(n log n) pass.

    Row ``k`` describes the rule ``score >= threshold_k``; the first row is "flag nobody",
    at a threshold just above the largest score. Cutting inside a run of tied scores is not
    a rule anybody can implement, so only the last index of each run becomes a cut.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {s.shape}")
    n, n_pos = len(y), int(y.sum())
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    tp_cum, fp_cum = np.cumsum(y_sorted), np.cumsum(1 - y_sorted)
    last = np.flatnonzero(np.diff(s_sorted)) if n > 1 else np.array([], dtype=int)
    last = np.append(last, n - 1)
    tp = np.concatenate([[0], tp_cum[last]]).astype(int)
    fp = np.concatenate([[0], fp_cum[last]]).astype(int)
    return pd.DataFrame({
        "threshold": np.concatenate([[np.nextafter(s_sorted[0], np.inf)], s_sorted[last]]),
        "tp": tp, "fp": fp, "fn": n_pos - tp, "tn": (n - n_pos) - fp,
    })


def cost_curve(y_true, score, cost_ratio: float) -> pd.DataFrame:
    """Expected cost at every distinct cut, ascending in threshold, with the rates."""
    table = cut_points(y_true, score)
    n = table[["tp", "fp", "fn", "tn"]].sum(axis=1)
    pos, sel = table["tp"] + table["fn"], table["tp"] + table["fp"]
    table["n"] = n
    table["selection_rate"] = sel / n
    table["recall"] = np.where(pos > 0, table["tp"] / pos, np.nan)
    table["precision"] = np.where(sel > 0, table["tp"] / sel, np.nan)
    table["expected_cost"] = float(cost_ratio) * table["fn"] + table["fp"]
    table["cost_per_client"] = table["expected_cost"] / n
    return table.sort_values("threshold", ignore_index=True)


def cost_at(curve: pd.DataFrame, threshold: float) -> float:
    """Expected cost of the rule ``score >= threshold``, read off the curve exactly.

    The curve is a step function, so this is a lookup and not an interpolation: the rule
    ``score >= t`` flags exactly the clients that the first cut point at or above ``t``
    flags. Interpolating would draw a number that no threshold actually produces.
    """
    thresholds = curve["threshold"].to_numpy()
    idx = int(np.searchsorted(thresholds, float(threshold), side="left"))
    idx = min(idx, len(thresholds) - 1)
    return float(curve["expected_cost"].to_numpy()[idx])


def optimal_threshold(y_true, score, cost_ratio: float) -> float:
    """The cut minimising ``r * FN + FP`` on the rows given.

    Ties break towards the **higher** threshold, i.e. towards flagging fewer people:
    wherever two rules cost the same, the one that intervenes in fewer lives wins. The
    tie-break is stated rather than left to whatever order argsort happened to produce.
    """
    table = cost_curve(y_true, score, cost_ratio)
    best = table["expected_cost"].min()
    return float(table.loc[table["expected_cost"] <= best, "threshold"].max())


def flat_region(curve: pd.DataFrame, tolerance: float = 0.01) -> tuple[float, float, int]:
    """The band of cuts within ``tolerance`` of the cheapest one: ``(lo, hi, n_cuts)``.

    The minimum of this curve is a basin, not a point. Reporting only the argmin invites a
    reader to believe the third decimal place matters, which it does not.
    """
    best = float(curve["expected_cost"].min())
    inside = curve.loc[curve["expected_cost"] <= best * (1.0 + tolerance), "threshold"]
    return float(inside.min()), float(inside.max()), int(inside.size)


def snap_to_grid(value: float, step: float = THRESHOLD_STEP,
                 lo: float = THRESHOLD_MIN, hi: float = THRESHOLD_MAX) -> float:
    """Nearest value on the threshold slider's grid, so the control and the rule agree."""
    return float(np.clip(round(float(value) / step) * step, lo, hi))


# =====================================================================================
# WHO ABSORBS THE ERRORS
# =====================================================================================

def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise ``num/den`` with 0/0 -> NaN. An empty cell has no rate; 0.0 would be a lie."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(num.shape, np.nan, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return out


def group_codes(groups) -> tuple[np.ndarray, list]:
    """Integer codes plus the sorted level list, so group order never depends on row order."""
    s = pd.Series(np.asarray(groups)).astype("object")
    levels = sorted(pd.unique(s.dropna()), key=str)
    lookup = {lvl: i for i, lvl in enumerate(levels)}
    codes = s.map(lookup).to_numpy()
    if pd.isna(codes).any():
        raise ValueError("protected attribute contains missing values; audit cannot proceed")
    return codes.astype(int), levels


def group_counts(y_true, y_pred, codes: np.ndarray, n_levels: int) -> dict:
    """Per-group confusion counts in one vectorised pass. Runs on every permutation draw."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    c = np.asarray(codes).astype(int)
    return {
        "tp": np.bincount(c[(y == 1) & (p == 1)], minlength=n_levels),
        "fp": np.bincount(c[(y == 0) & (p == 1)], minlength=n_levels),
        "fn": np.bincount(c[(y == 1) & (p == 0)], minlength=n_levels),
        "tn": np.bincount(c[(y == 0) & (p == 0)], minlength=n_levels),
    }


def rates_from_counts(counts) -> dict:
    """Per-group confusion counts -> the per-group rates reported everywhere."""
    tp, fp, fn, tn = (np.asarray(counts[k], dtype=float) for k in ("tp", "fp", "fn", "tn"))
    n = tp + fp + fn + tn
    return {
        "n": n.astype(int),
        "base_rate": _ratio(tp + fn, n),
        "selection_rate": _ratio(tp + fp, n),
        "tpr": _ratio(tp, tp + fn),
        "fpr": _ratio(fp, fp + tn),
        "precision": _ratio(tp, tp + fp),
        "fnr": _ratio(fn, tp + fn),
    }


def group_table(y_true, y_pred, groups) -> pd.DataFrame:
    """Per-group rates at whatever decision ``y_pred`` encodes, one row per group."""
    codes, levels = group_codes(groups)
    rates = rates_from_counts(group_counts(y_true, y_pred, codes, len(levels)))
    return pd.DataFrame({k: rates[k] for k in GROUP_METRICS},
                        index=pd.Index(levels, name="group"))


def parity_gaps_from_rates(rates) -> dict:
    """Parity scalars from per-group rate arrays -- the arithmetic, spelled out.

    Every measure here is ``max - min`` across groups, which for two groups is an absolute
    difference. That shape is the reason a bootstrap interval cannot test any of them: the
    statistic is non-negative by construction, so its interval can approach zero but never
    straddle it. What tests them is a permutation null -- shuffle the group labels with
    everything else held fixed -- computed in the analysis repository and shipped here
    as p-values rather than recomputed on the page.
    """
    def gap(name: str) -> float:
        v = np.asarray(rates[name], dtype=float)
        v = v[np.isfinite(v)]
        return float(v.max() - v.min()) if v.size else float("nan")

    tpr_d, fpr_d = gap("tpr"), gap("fpr")
    return {
        "demographic_parity_difference": gap("selection_rate"),
        "fpr_difference": fpr_d,
        "tpr_difference": tpr_d,
        "equalized_odds_difference": float(np.nanmax([tpr_d, fpr_d])),
        "base_rate_difference": gap("base_rate"),
    }


def parity_gaps(table: pd.DataFrame) -> dict:
    """:func:`parity_gaps_from_rates` for a group table."""
    return parity_gaps_from_rates({c: table[c].to_numpy() for c in table.columns})


def audit_tables(rows: pd.DataFrame, threshold: float,
                 attributes=ATTRIBUTES, min_n: int = MIN_AUDIT_CELL) -> dict:
    """``{attribute: group table}`` at one threshold, with a ``below_min_n`` flag per group."""
    pred = flag(rows["score"].to_numpy(), threshold)
    y = rows["defaulted"].to_numpy()
    out = {}
    for attr in attributes:
        table = group_table(y, pred, rows[attr].to_numpy())
        table["below_min_n"] = table["n"] < min_n
        out[attr] = table
    return out


def headline_gaps(tables: dict, min_n: int = MIN_AUDIT_CELL) -> pd.DataFrame:
    """One row per attribute: the parity gaps, over the groups big enough to report."""
    rows = []
    for attr, table in tables.items():
        kept = table.loc[table["n"] >= min_n]
        gaps = parity_gaps(kept[list(GROUP_METRICS)])
        rows.append({"attribute": attr, "n_groups_used": int(len(kept)),
                     "n_groups_dropped": int(len(table) - len(kept)), **gaps})
    return pd.DataFrame(rows)



# =====================================================================================
# THE SAME COMPUTATIONS, ADDRESSED BY THE UI -- still plain Python, still testable
# =====================================================================================

def compute_curve(cost_ratio: float, data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    rows = load_rows(data_dir)
    return cost_curve(rows["defaulted"].to_numpy(), rows["score"].to_numpy(), cost_ratio)


def compute_optimal(cost_ratio: float, data_dir: Path | str = DATA_DIR) -> float:
    rows = load_rows(data_dir)
    return optimal_threshold(rows["defaulted"].to_numpy(), rows["score"].to_numpy(), cost_ratio)


def compute_decision(threshold: float, cost_ratio: float,
                     data_dir: Path | str = DATA_DIR) -> dict:
    """Everything the headline row shows, for one (threshold, r)."""
    rows = load_rows(data_dir)
    conf = confusion(rows["defaulted"].to_numpy(), flag(rows["score"].to_numpy(), threshold))
    cost = expected_cost(conf, cost_ratio)
    return {**conf, "cost_ratio": float(cost_ratio), "threshold": float(threshold),
            "expected_cost": cost, "cost_per_client": cost / conf["n"],
            "share_of_cost_from_false_negatives":
                (float(cost_ratio) * conf["fn"] / cost) if cost > 0 else float("nan")}


def compute_audit(threshold: float, data_dir: Path | str = DATA_DIR) -> dict:
    return audit_tables(load_rows(data_dir), threshold)


def compute_gap_curve(attribute: str, thresholds: tuple, data_dir: Path | str = DATA_DIR):
    """Cache-friendly wrapper: ``thresholds`` is a tuple so the key is hashable."""
    return gap_curve(load_rows(data_dir), attribute, np.asarray(thresholds, dtype=float))


# =====================================================================================
# FIGURES
# =====================================================================================

def _dress(ax, title=None, xlabel=None, ylabel=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED, labelpad=7)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED, labelpad=7)
    if title:
        ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=11, fontweight="semibold")
    return ax


def figure_cost_curve(curve: pd.DataFrame, threshold: float, cost_ratio: float,
                      frozen_threshold: float):
    """Expected cost against the decision threshold, with the current cut and 0.5 marked."""
    best = float(curve["expected_cost"].min())
    t_best = float(curve.loc[curve["expected_cost"] <= best, "threshold"].max())
    here, half = cost_at(curve, threshold), cost_at(curve, 0.5)
    lo, hi, n_cuts = flat_region(curve)

    # Frame the part of the curve worth looking at: always far enough right to show 0.5,
    # never so far that the wall beyond it flattens the basin into the axis.
    ceiling = max(half, here, best * 2.0) * 1.05
    reach = curve.loc[curve["expected_cost"] <= ceiling, "threshold"]
    xmax = float(np.clip(reach.max() if len(reach) else 0.6, 0.58, 0.85))
    view = curve.loc[curve["threshold"] <= xmax]
    span = max(float(view["expected_cost"].max()) - best, 1.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axvspan(lo, min(hi, xmax), color=BLUE, alpha=0.08, lw=0,
               label=f"within 1% of the cheapest cut — {n_cuts} distinct cuts qualify")
    ax.plot(view["threshold"], view["expected_cost"], color=BLUE, lw=2.1, zorder=3)

    ax.axvline(frozen_threshold, color=TEAL, lw=1.1, ls=(0, (5, 3)), alpha=0.9, zorder=2)
    ax.annotate(f"cut frozen on validation: {frozen_threshold:.4f}",
                xy=(frozen_threshold, best), xytext=(frozen_threshold + 0.013, best - span * 0.12),
                fontsize=8.5, color=TEAL, ha="left", va="center")

    ax.plot([0.5], [half], "o", ms=8, mfc="white", mec=GREY, mew=2, zorder=5)
    ax.annotate(f"0.5 — the cut you get for free\n{half:,.0f} units, {half / max(best, 1):.1f}x the minimum",
                xy=(0.5, half), xytext=(0.5 - 0.022, half + span * 0.06),
                fontsize=8.5, color=MUTED, ha="right", va="bottom")

    ax.plot([threshold], [here], "o", ms=10, mfc=ORANGE, mec="white", mew=1.8, zorder=6)
    ax.annotate(f"you are here: {threshold:.4f}\n{here:,.0f} units",
                xy=(threshold, here), xytext=(threshold + 0.02, here + span * 0.10),
                fontsize=9, color=ORANGE, ha="left", va="bottom", fontweight="semibold")

    _dress(
        ax,
        title=(f"At r = {cost_ratio:g}, the cheapest cut is {t_best:.3f} — "
               f"and 0.5 costs {half / max(best, 1):.1f}x as much"),
        xlabel="decision threshold — flag when the calibrated probability of default is at least this",
        ylabel="expected cost on 6,000 test clients\n(units of one false alarm)",
    )
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, best + span * 1.32)
    ax.set_xticks(np.arange(0, xmax + 0.001, 0.1))
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=MUTED)
    fig.tight_layout()
    return fig


def figure_group_errors(tables: dict, threshold: float, min_n: int = MIN_AUDIT_CELL):
    """False-negative and false-positive rates by group, for all four attributes."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), dpi=110)
    fig.patch.set_facecolor("white")

    for ax, attr in zip(axes.ravel(), ATTRIBUTES):
        table = tables[attr].sort_index(ascending=False)
        ax.set_facecolor("white")
        pos = np.arange(len(table))
        ax.barh(pos + 0.19, table["fnr"] * 100, height=0.34, color=RED, alpha=0.9,
                label="missed defaults — % of that group's defaulters")
        ax.barh(pos - 0.19, table["fpr"] * 100, height=0.34, color=BLUE, alpha=0.9,
                label="false flags — % of that group's non-defaulters")
        labels = [f"{g}\nn = {int(r.n):,}" + ("  (too small)" if r.below_min_n else "")
                  for g, r in table.iterrows()]
        ax.set_yticks(pos)
        ax.set_yticklabels(labels, fontsize=8.5, color=INK)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
        ax.tick_params(axis="y", colors=INK, length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, len(table) - 0.4)
        ax.xaxis.set_major_formatter(PercentFormatter())
        role = "primary" if attr == "sex" else "exploratory"
        ax.set_title(f"{ATTRIBUTE_LABEL[attr]}  ({role})", fontsize=10, color=INK,
                     loc="left", pad=8, fontweight="semibold")

    gaps = headline_gaps(tables, min_n).set_index("attribute")
    worst = gaps["fpr_difference"].idxmax()
    fig.suptitle(
        f"At threshold {threshold:.4f}, the false-alarm rate spreads "
        f"{gaps.loc[worst, 'fpr_difference'] * 100:.1f} points across "
        f"{ATTRIBUTE_LABEL[worst]} and {gaps.loc['sex', 'fpr_difference'] * 100:.1f} across sex",
        fontsize=11.5, color=INK, x=0.011, ha="left", y=0.995, fontweight="semibold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", ncol=1, frameon=False,
               fontsize=8.5, labelcolor=MUTED, bbox_to_anchor=(0.011, 0.0))
    fig.tight_layout(rect=(0, 0.075, 1, 0.955))
    return fig


def gap_curve(rows: pd.DataFrame, attribute: str, thresholds: np.ndarray,
              min_n: int = MIN_AUDIT_CELL) -> pd.DataFrame:
    """Each parity gap for one attribute, at every threshold in ``thresholds``.

    A single gap at a single cut is a fact about one arbitrary operating point. Sweeping
    the cut shows which gaps are a property of the *rule* and which are a property of
    *where the rule happens to sit* -- and those are different claims about fairness.
    """
    y = rows["defaulted"].to_numpy()
    groups = rows[attribute].to_numpy()
    score = rows["score"].to_numpy()
    out = []
    for t in thresholds:
        table = group_table(y, flag(score, t), groups)
        kept = table.loc[table["n"] >= min_n]
        out.append({"threshold": float(t), **parity_gaps(kept[list(GROUP_METRICS)]),
                    "selection_rate": float((score >= t).mean())})
    return pd.DataFrame(out)


GAP_LABEL = {
    "demographic_parity_difference": "selection-rate gap",
    "fpr_difference": "false-alarm-rate gap",
    "tpr_difference": "catch-rate gap",
}
GAP_COLOUR = {"demographic_parity_difference": BLUE,
              "fpr_difference": RED,
              "tpr_difference": "#7A8796"}


def figure_gap_vs_threshold(curve: pd.DataFrame, attribute: str, threshold: float,
                            frozen: float):
    """The three gaps as a function of where the line is drawn."""
    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=110)
    for key, label in GAP_LABEL.items():
        ax.plot(curve["threshold"], curve[key] * 100, lw=2.0, color=GAP_COLOUR[key],
                label=label, zorder=3)
    ax.axvline(frozen, color=GRID, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.axvline(threshold, color=INK, lw=1.4, zorder=4)
    ax.annotate(f"you are here\n{threshold:.4f}", xy=(threshold, ax.get_ylim()[1]),
                xytext=(6, -4), textcoords="offset points", ha="left", va="top",
                fontsize=8.5, color=INK, fontweight="semibold")
    ax.set_xlim(float(curve["threshold"].min()), float(curve["threshold"].max()))
    ax.set_ylim(bottom=0)
    _dress(ax, title=f"How the {ATTRIBUTE_LABEL[attribute]} gaps move as the line moves",
           xlabel="decision threshold", ylabel="gap between groups, percentage points")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=MUTED)
    fig.tight_layout()
    return fig


# =====================================================================================
# STREAMLIT UI -- widgets in, the functions above called, figures out. No arithmetic here.
# =====================================================================================

def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Who pays for the model's mistakes?",
                       page_icon="⚖️", layout="centered")
    cache = st.cache_data(show_spinner=False)
    get_reference = cache(load_reference)
    get_curve = cache(compute_curve)
    get_optimal = cache(compute_optimal)
    get_decision = cache(compute_decision)
    get_audit = cache(compute_audit)
    get_gap_curve = cache(compute_gap_curve)

    ref = get_reference()
    frozen_t = float(ref["frozen"]["threshold"])
    published = ref["published_test"]

    st.title("Who pays for the model's mistakes?")
    st.markdown(
        "**Most projects on this dataset finish where this one starts.** They fit a few "
        "models, report an AUC in the high 0.70s, draw a feature-importance chart and stop. "
        "But a fitted model is not a decision — it is a column of probabilities. Somebody "
        "still has to draw a line and say *these clients get flagged and those do not*, and "
        "the moment that line exists it hands a bill to somebody.\n\n"
        "So this page is about the part that comes after the model: **how you pick that "
        "line, why you pick it there, and what the assumption behind it turns out to "
        "control.** The answer to the last one was not what I expected — the assumption does "
        "not only move the line, it reaches back and changes which model you should have "
        "trained.\n\n"
        "Everything here is computed live from the model's 6,000 held-out test scores, "
        "shipped with this page. **The full analysis — model selection, calibration, "
        "interpretability, the mitigation attempts — lives in a private repository.**"
    )

    st.divider()
    st.subheader("1 · Draw the line")
    st.markdown(
        "**There is no threshold in the data.** A classifier hands you a probability per "
        "client; turning that into an action needs one number the data does not contain — "
        "how much worse a missed default is than a false alarm. Call it `r`. Fix `r` and the "
        "threshold follows: the rule minimising `r × FN + FP` is the cheapest line you can "
        "draw. Refuse to fix it and you have not avoided the assumption, you have made it "
        "silently — cutting at 0.5, which is what `predict()` hands you, *is* the choice "
        "`r = 1`.\n\n"
        "Three candidate lines, and they are different objects:\n\n"
        "- **0.5** — free, and optimal only if a missed default and a false alarm cost the "
        "same.\n"
        "- **1/(1+r)** — Bayes-optimal for a *perfectly calibrated* score. Needs no data "
        "beyond `r`, and is wrong exactly to the extent the score is miscalibrated. This is "
        "why calibration was checked before any cost rule was applied.\n"
        "- **the empirical minimum** — sweep every distinct cut on validation and take the "
        "cheapest. Uses the data, and pays for it in sampling noise.\n\n"
        "This project uses the third, chosen on validation and frozen before the test split "
        "was opened. Turn on *snap* below and the three land within about a percent of each "
        "other in cost — which is the finding, and the reason the third decimal place is not "
        "worth arguing about."
    )

    c1, c2 = st.columns([3, 2])
    with c1:
        cost_ratio = int(st.slider(
            "How much more does a missed default cost than a false alarm?   r = C_FN / C_FP",
            min_value=1, max_value=30, value=int(ref["frozen"]["cost_ratio_ASSUMED"]), step=1,
            help="An assumption, not a measurement. This dataset carries no exposure, no "
                 "loss-given-default and no recovery rate, so every cost here is in units "
                 "of one false alarm and no currency figure appears anywhere."))
    with c2:
        st.write("")
        snap = st.toggle("Snap to the cheapest cut", value=False,
                         help="Jump the threshold to the cost-minimising cut for this r, "
                              "computed on the rows on screen.")

    t_opt = get_optimal(cost_ratio)
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = snap_to_grid(float(ref["frozen"]["threshold"]))
    if snap:
        st.session_state["threshold"] = snap_to_grid(t_opt)
    threshold = float(st.slider(
        "Decision threshold — flag the client at or above this probability",
        min_value=THRESHOLD_MIN, max_value=THRESHOLD_MAX, step=THRESHOLD_STEP,
        format="%.4f", key="threshold", disabled=snap))
    if snap:
        st.caption(f"Snapped to **{threshold:.4f}**. The exact cost-minimising cut on this "
                   f"split at r = {cost_ratio} is {t_opt:.5f}, and a perfectly calibrated "
                   f"score would want 1/(1+r) = {bayes_threshold(cost_ratio):.4f} — the "
                   f"three agree to about a percent of cost, because the minimum is a basin.")

    d = get_decision(threshold, cost_ratio)
    base_cost = float(published["expected_cost_units_of_C_FP"])

    a, b, c = st.columns(3)
    a.metric("Expected cost", f"{d['expected_cost']:,.0f} units",
             f"{d['expected_cost'] - base_cost:+,.0f} vs the deployed rule",
             delta_color="inverse")
    b.metric("Cost per client", f"{d['cost_per_client']:.3f} units")
    c.metric("Share of the book flagged", f"{d['selection_rate']:.1%}")
    e, f, g = st.columns(3)
    e.metric("Accuracy", f"{d['accuracy']:.1%}",
             f"{d['accuracy'] - (1 - d['base_rate']):+.1%} vs flagging nobody")
    f.metric("Recall — defaulters caught", f"{d['recall']:.1%}")
    g.metric("Precision — flags that were right",
             "—" if np.isnan(d["precision"]) else f"{d['precision']:.1%}")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**The 2×2**, on 6,000 held-out clients")
        st.table(pd.DataFrame(
            {"not flagged": [f"{d['tn']:,}", f"{d['fn']:,}"],
             "flagged": [f"{d['fp']:,}", f"{d['tp']:,}"]},
            index=pd.Index([f"did not default ({d['fp'] + d['tn']:,})",
                            f"defaulted ({d['tp'] + d['fn']:,})"], name="")))
    with right:
        share = d["share_of_cost_from_false_negatives"]
        st.markdown("**Who the bill goes to**")
        st.markdown(
            f"Of {d['expected_cost']:,.0f} cost units, **{share:.0%}** is carried by the "
            f"**{d['fn']:,} missed defaults** and **{1 - share:.0%}** by the "
            f"**{d['fp']:,} people who would have paid** but got flagged anyway. Which of "
            f"those is *the harm* depends on whether a flag is a declined card or a "
            f"supportive phone call — a product decision, not a modelling one."
        )

    st.pyplot(figure_cost_curve(get_curve(cost_ratio), threshold, cost_ratio, frozen_t),
              use_container_width=True)
    st.caption(
        f"The minimum is a basin, not a point. At the assumed r = 10, "
        f"{ref['frozen']['flat_region']['n_cuts']} distinct cuts on validation sat within 1% "
        f"of the cheapest one, while 0.5 — the threshold `predict()` hands you for free — "
        f"sits far up the right-hand wall. The third decimal place does not matter; the "
        f"choice between 0.5 and 0.11 matters enormously. The deployed cut "
        f"{frozen_t:.4f} was chosen on the validation split and only ever evaluated here."
    )

    st.divider()
    st.subheader("2 · Who absorbs the errors")
    st.markdown(
        "The same rule, sliced four ways. These four attributes were carried through the "
        "whole project for this panel alone and never reached a `fit` call — the model has "
        "never seen any of them. Drag the threshold above and watch the bars move."
    )
    tables = get_audit(threshold)
    st.pyplot(figure_group_errors(tables, threshold), use_container_width=True)
    st.markdown(
        "**Read the two bars separately: they have different denominators.** The red bar is "
        "a share of that group's *defaulters* — the ones it let through. The blue bar is a "
        "share of that group's *non-defaulters* — the ones it flagged anyway. A single "
        "error rate would average these two into a number that hides the thing worth seeing, "
        "which is that a group can be treated worse in **two opposite directions at once**."
    )

    sex_tab = tables["sex"].loc[["female", "male"]] if "male" in tables["sex"].index else None
    if sex_tab is not None:
        f, m = sex_tab.loc["female"], sex_tab.loc["male"]
        st.markdown(
            f"At this cut men absorb more false alarms ({m['fpr']:.1%} of male non-defaulters "
            f"against {f['fpr']:.1%} of female ones) and women absorb more missed defaults "
            f"({f['fnr']:.1%} against {m['fnr']:.1%}). **Which of those is the harm is not a "
            f"question the data can answer.** If a flag is a declined card, the men are the "
            f"ones being hurt. If a flag is a phone call before the account goes bad, the "
            f"women are the ones not getting it. Same numbers, opposite conclusion, and the "
            f"choice belongs to whoever decides what a flag *does*."
        )

    gaps = headline_gaps(tables)
    gaps = gaps.set_index("attribute")
    worst = gaps["demographic_parity_difference"].idxmax()
    st.markdown(
        f"**The protected attribute everyone audits is not where the spread is.** At this "
        f"cut the selection-rate gap across `{worst}` is "
        f"{gaps.loc[worst, 'demographic_parity_difference']:.1%} against "
        f"{gaps.loc['sex', 'demographic_parity_difference']:.1%} across `sex` — and the model "
        f"was given neither. Excluding an attribute from the feature matrix removes it as an "
        f"*input*, not as a *pattern*; the correlated proxies are all still there. That is why "
        f"this panel measures outcomes rather than inspecting inputs."
    )

    st.markdown("##### One gap at one cut is a fact about that cut")
    st.markdown(
        "Everything above describes a single operating point. Sweeping the threshold "
        "separates gaps that are a property of the **rule** from gaps that are a property of "
        "**where the rule happens to sit** — which are different claims, and only the first "
        "survives someone changing the cost assumption."
    )
    gap_attr = st.selectbox(
        "Attribute to sweep", ATTRIBUTES, index=ATTRIBUTES.index("sex"),
        format_func=lambda a: f"{ATTRIBUTE_LABEL[a]}"
        f"{'  —  pre-specified primary' if a == ref['frozen']['primary_attribute'] else '  —  exploratory'}",
        key="gap_attr")
    grid = np.linspace(max(THRESHOLD_MIN, 0.01), min(THRESHOLD_MAX, 0.60), 70)
    st.pyplot(figure_gap_vs_threshold(get_gap_curve(gap_attr, tuple(np.round(grid, 5))),
                                      gap_attr, threshold, frozen_t),
              use_container_width=True)
    st.caption(
        "Dashed line: the cut this project deployed. Solid line: where you have put it."
    )
    st.markdown(
        "**Every gap vanishes at both ends, and that is arithmetic rather than fairness.** "
        "Flag almost everyone and no group can differ from another; flag almost nobody and "
        "the same. The gaps live in the middle, and what separates them is *where* in the "
        "middle.\n\n"
        "Across the whole high-flagging half of this range the **catch-rate gap sits near "
        "0.9 points** while the other two climb to 6 and 7 — at a cut this low almost every "
        "defaulter in every group is flagged, so the catch rate has no room left to differ. "
        "That is the reading most audits of this model would stop at. **Push the line the "
        "other way, to where only 15% of the book is flagged, and the catch-rate gap becomes "
        "the largest of the three** (4.5 points against 3.3 and 1.8).\n\n"
        "So \"the catch-rate gap is negligible\" is not a property of this rule. It is a "
        "property of this rule *at this threshold*, and the threshold came from a cost ratio "
        "I made up. The jaggedness on the right is real rather than rendering: past that "
        "point so few clients are flagged that one person moves a group rate."
    )

    with st.expander("The numbers behind those bars"):
        frame = pd.concat({ATTRIBUTE_LABEL[a]: tables[a] for a in ATTRIBUTES},
                          names=["attribute"])
        show = frame[["n", "base_rate", "selection_rate", "tpr", "fpr", "fnr"]].copy()
        show.columns = ["n", "actually defaulted", "flagged", "defaulters caught",
                        "false-alarm rate", "missed-default rate"]
        st.dataframe(show.style.format({"n": "{:,.0f}", "actually defaulted": "{:.1%}",
                                        "flagged": "{:.1%}", "defaulters caught": "{:.1%}",
                                        "false-alarm rate": "{:.1%}",
                                        "missed-default rate": "{:.1%}"}),
                     use_container_width=True)
        st.caption(
            "Base rates differ across every attribute audited — look at the second column — "
            "so demographic parity and equalised odds cannot both hold: a calibrated score "
            "applied to groups that default at different rates must flag them at different "
            "rates. Both are shown and no winner is declared. Groups under "
            f"{MIN_AUDIT_CELL} rows are printed with their n but kept out of every gap."
        )

        perm = ref.get("permutation_test_at_frozen_threshold") or {}
        if perm:
            st.markdown("**Which of these gaps is bigger than chance produces?**")
            rows_p = []
            for attr in ATTRIBUTES:
                for key, label in GAP_LABEL.items():
                    e = perm.get(attr, {}).get(key)
                    if not e:
                        continue
                    rows_p.append({
                        "attribute": ATTRIBUTE_LABEL[attr], "gap": label,
                        "observed": f"{e['observed'] * 100:.2f} pts",
                        "p": f"{e['p_value']:.3f}",
                        "verdict": "larger than chance" if e["p_value"] < 0.05
                                   else "inside what noise makes",
                    })
            st.dataframe(pd.DataFrame(rows_p).set_index(["attribute", "gap"]),
                         use_container_width=True)
            st.caption(
                "A bootstrap interval cannot answer this. Every gap here is a `max − min` "
                "statistic, so it is non-negative by construction and its interval can "
                "approach zero but never straddle it — \"excludes zero\" describes how "
                "precisely the gap is estimated, not whether there is one. The exact null is "
                "cheap instead: hold the model, the cut and every client's outcome fixed and "
                "shuffle the group labels 5,000 times. These p-values are from the deployed "
                "cut, not from wherever your slider is."
            )

    st.divider()
    st.subheader("3 · Why this model and not the simpler one")
    st.markdown(
        "Two families were fitted, tuned and calibrated the same way: an L2 logistic "
        "regression and LightGBM. One of them has to ship. **The interesting part of this "
        "project is not which one won — it is that the obvious way to decide was the wrong "
        "way, and it took a specific piece of machinery to see that.**"
    )

    st.markdown("##### The default answer, and why it is not obviously right")
    ms = ref["model"]
    d_ap = ms["lgbm_minus_logistic_ap"]
    st.markdown(
        f"The reflex is to pick whichever scores better on the cross-validation metric. Here "
        f"that metric is **average precision**, and by it the two are a tie: LightGBM wins the "
        f"cross-validation ({ms['lgbm_cv']:.4f} against {ms['logistic_cv']:.4f}) and then "
        f"produces a paired difference on validation of **{d_ap['point']:+.5f}, 95% interval "
        f"[{d_ap['ci_lo']:+.4f}, {d_ap['ci_hi']:+.4f}]** — an interval straddling zero. On that "
        f"reading you keep the simpler model, and the first version of this project did.\n\n"
        f"Average precision integrates precision over the **whole** recall axis. It gives the "
        f"stretch at recall 0.1 — where precision is high and this rule never operates — the "
        f"same standing as recall {published['confusion']['tpr']:.2f}, which is where the rule "
        f"actually lives. **Choosing with a number that averages over everywhere, in order to "
        f"act in one place, is a decision rather than a default**, and it is not one I had "
        f"made deliberately."
    )

    st.markdown("##### The obstacle: one validation split cannot answer the better question")
    st.markdown(
        "The better question is which family is cheaper at the operating point — expected "
        "cost at r = 10, the quantity the decision actually pays. Asked on the 6,000-row "
        "validation split, the paired bootstrap on that difference runs from about "
        "**−318 to +81**: LightGBM cheaper in 88% of resamples and still not separated from "
        "zero. The effect is real and smaller than one split of this size can resolve, which "
        "is exactly why the reassuring paragraph above sounded so safe."
    )

    sa = ref.get("selection_audit")
    sw = ref.get("swap")
    if sa:
        st.markdown("##### The design, and what each piece of it is for")
        st.markdown(
            f"Nested cross-validation over the **{sa['design']['n_rows']:,} pooled training "
            f"and validation rows** ({sa['design']['outer']}). Four choices, each one closing "
            f"a specific way the comparison could have flattered one side:\n\n"
            "- **The outer loop holds out a fold and never touches it.** The number reported "
            "for a fold is scored on rows nothing in that fold's pipeline has seen.\n"
            "- **The inner loop tunes each family separately, on outer-training rows only.** "
            "Neither family gets a hyperparameter chosen with a peek at what judges it, and "
            "neither gets a longer look than the other.\n"
            "- **Both arms are built exactly as the deployed model is** — same calibration "
            "wrapper, same folds, same settings. Give one family a five-fold calibration and "
            "the other a bare three-fold and you are comparing calibration states, not model "
            "families. That was a real bug in the first version of this comparison.\n"
            "- **Each fold is charged its own cheapest cut**, not a threshold fixed elsewhere. "
            "That makes the statistic a property of the *ranking*, so a family is not "
            "penalised for a cut that happens to suit the other one.\n\n"
            "And the inner loop runs **twice** — once selecting hyperparameters on average "
            "precision, once on expected cost — which turns \"would a cost-aware rule have "
            "caught this on its own?\" into a measurement instead of an opinion."
        )
        rows = []
        for key, label in (("average_precision", "average precision"),
                           # r is the audit's own frozen 10, NOT the slider above: that run is
                           # fixed, and relabelling it with whatever r is on screen would claim
                           # a result it never had.
                           ("expected_cost",
                            f"expected cost at r = {sa['design']['cost_ratio_ASSUMED']:g}")):
            v = sa["summary"][key]
            d = v["lgbm_minus_logistic"]
            rows.append({
                "inner selection metric": label,
                "logistic": f"{v['logistic_mean_cost']:,.0f}",
                "LightGBM": f"{v['lgbm_mean_cost']:,.0f}",
                "difference": f"{d['point']:+,.1f}",
                "95% interval": f"[{d['ci_lo']:+,.1f}, {d['ci_hi']:+,.1f}]",
                "LightGBM cheaper on": f"{d['cheaper_on_n_folds']} of {d['n_folds']} folds",
            })
        st.table(pd.DataFrame(rows).set_index("inner selection metric"))
        st.caption("Costs in units of one false alarm, per fold of 4,800 held-out clients.")

        st.markdown("##### The rule that was applied to that table")
        st.info(
            "**Switch away from the simpler incumbent only if the paired interval on expected "
            "cost is clear of zero.** Winning on average is not enough. This is the *same* "
            "rule that kept the logistic regression in the first version — only the statistic "
            "it is applied to changed. Writing it down before looking is the difference "
            "between a rule and a rationalisation.",
            icon=":material/rule:",
        )
        c1, c2 = st.columns(2)
        c1.markdown(
            "**LightGBM clears it, on every fold, under both criteria.** The tie on average "
            "precision was a real tie — about a question this project was never going to act "
            "on."
        )
        c2.markdown(
            "**But changing the selection metric is not what found it.** Tuning on cost rather "
            "than average precision moves the answer by less than its own interval. "
            "**The gain is in the model family, not the metric** — \"optimise the business "
            "number directly\" would not have got here on its own."
        )

    rs = ref.get("selection_vs_cost_ratio")
    if rs:
        st.markdown("##### The part I did not see coming: `r` chooses the model too")
        st.markdown(
            "Everything above treats `r = 10` as fixed. But `r` is the input to the "
            "selection criterion, not just to the threshold — so it decides *which part of "
            "the curve is being graded*. If the two families are not equally good "
            "everywhere on that curve, then changing the assumption should change the "
            "winner. Running the whole nested comparison again at each `r` says whether it "
            "does."
        )
        sweep = pd.DataFrame(rs["ratios"])
        show = pd.DataFrame({
            "r": sweep["cost_ratio"].map(lambda v: f"{v:g}"),
            "logistic": sweep["logistic_mean_cost"].map(lambda v: f"{v:,.0f}"),
            "LightGBM": sweep["lgbm_mean_cost"].map(lambda v: f"{v:,.0f}"),
            "difference": sweep["difference"].map(lambda v: f"{v:+,.1f}"),
            "95% interval": [f"[{lo:+,.1f}, {hi:+,.1f}]"
                             for lo, hi in zip(sweep["ci_lo"], sweep["ci_hi"])],
            "the rule picks": sweep["selected"],
        }).set_index("r")
        # Mark the row nearest the slider so the assumption at the top of the page and the
        # model choice at the bottom are visibly the same number.
        nearest = (sweep["cost_ratio"] - cost_ratio).abs().idxmin()
        mark = f"{sweep.loc[nearest, 'cost_ratio']:g}"
        st.table(show.style.apply(
            lambda row: ["background-color: rgba(47,93,158,0.10)"] * len(row)
                        if row.name == mark else [""] * len(row), axis=1))
        st.caption(
            f"Highlighted: the row nearest the `r` you set at the top of this page "
            f"(r = {cost_ratio:g}). Costs per fold of 4,800 held-out clients; the same "
            f"nested design and the same pre-registered rule at every `r`. The test split "
            f"is not read here."
        )
        picks = sweep.set_index("cost_ratio")["selected"]
        c1, c2 = st.columns(2)
        c1.markdown(
            f"**At r = 1 the rule keeps the logistic regression.** The interval is "
            f"[{sweep.iloc[0]['ci_lo']:+.1f}, {sweep.iloc[0]['ci_hi']:+.1f}] — it crosses "
            f"zero, LightGBM is ahead on only "
            f"{int(sweep.iloc[0]['lgbm_cheaper_on_n_folds'])} of "
            f"{int(sweep.iloc[0]['n_folds'])} folds, and the rule refuses to switch. "
            f"**Everywhere from r = 2 up it switches.** One assumption, made by me and not "
            f"by the data, decides which model ships."
        )
        best = sweep.loc[sweep['difference'].idxmin()]
        c2.markdown(
            f"**And the advantage has a shape.** It peaks at r = {best['cost_ratio']:g} "
            f"({best['difference']:+,.0f}) and falls away on both sides — at r = 1 you flag "
            f"almost nobody and at r = 50 almost everybody, and a rule that intervenes "
            f"everywhere or nowhere cannot express a better model. **Which model you use "
            f"matters most exactly where the decision is hardest**, and not at all where it "
            f"is already made."
        )
        st.markdown(
            "This is the thing I would put first if I had to keep one sentence from the "
            "project: **the number I could not measure did not just set the operating "
            "point, it selected the model.** Everything downstream of it — the threshold, "
            "the confusion matrix, who absorbs the errors in section 2 — inherits an "
            "assumption that never appears in the data, and a reader who believes "
            "`r = 1` is entitled to a different model, not just a different cut."
        )

    if sw:
        a, b = sw["arms"]["superseded"], sw["arms"]["current"]
        st.markdown("##### What the decision was worth when it met the held-out data")
        st.markdown(
            f"That is the rationale, and it is only half the story. The rule changed, the "
            f"model changed with it, the test split was opened a second time — and the swap "
            f"was worth **{abs(sw['difference_at_frozen_cuts']):,.0f} units out of "
            f"{a['cost_at_frozen_cut']:,.0f}**, against the roughly 180 the nested comparison "
            f"implied for 6,000 rows."
        )
        st.table(pd.DataFrame([
            {"model": a["label"], "at the frozen cut": f"{a['cost_at_frozen_cut']:,.0f}",
             "at its own best cut on test": f"{a['cost_at_own_best_cut_on_test']:,.0f}",
             "threshold transfer loss": f"{a['threshold_transfer_loss']:+,.0f}"},
            {"model": b["label"], "at the frozen cut": f"{b['cost_at_frozen_cut']:,.0f}",
             "at its own best cut on test": f"{b['cost_at_own_best_cut_on_test']:,.0f}",
             "threshold transfer loss": f"{b['threshold_transfer_loss']:+,.0f}"},
            {"model": "difference",
             "at the frozen cut": f"{sw['difference_at_frozen_cuts']:+,.0f}",
             "at its own best cut on test": f"{sw['difference_at_own_best_cuts']:+,.0f}",
             "threshold transfer loss": ""},
        ]).set_index("model"))
        c1, c2 = st.columns(2)
        c1.markdown(
            f"**One of my four design choices came back.** Charging every fold its own "
            f"cheapest cut made the statistic a property of the ranking, which is what I "
            f"wanted — and it made the design blind to the fact that LightGBM transfers a "
            f"*frozen* threshold worse ({b['threshold_transfer_loss']:+,.0f} against "
            f"{a['threshold_transfer_loss']:+,.0f}). Its cost curve is less flat near the "
            f"minimum. That is {sw['explained_by_threshold_transfer']:+,.0f} units of the gap."
        )
        c2.markdown(
            f"**The rest is noise.** Expected cost on these 6,000 clients has a bootstrap "
            f"standard deviation of **{sw['bootstrap_sd_of_expected_cost']:,.0f} units**, so "
            f"{sw['difference_at_own_best_cuts']:+,.0f} and −180 are not distinguishable from "
            f"each other, and neither is distinguishable from zero."
        )
        st.info(
            "**So was the decision wrong?** The reasoning holds and the payoff did not arrive. "
            "Ten nested folds over 24,000 rows can see a difference that one 6,000-row split "
            "cannot confirm; both of those are true and the second governs what I am allowed "
            "to claim. What it cost is concrete: \"held out, opened once\" became \"opened "
            "twice under a rule that changed in between\". The unambiguous gain was somewhere "
            "I was not looking — re-running the calibration rule for the new model **rejected "
            "isotonic** and adopted sigmoid, more than halving expected calibration error on "
            "test, 0.0150 → 0.0062.",
            icon=":material/flag:",
        )

    st.divider()
    with st.expander("What this is, and what it is not"):
        ms = ref["model"]
        diff = ms["lgbm_minus_logistic_ap"]
        st.markdown(
            f"""
**The rule on screen.** LightGBM on 19 predictors, sigmoid-calibrated, selected on **expected
cost at r = {ref['frozen']['cost_ratio_ASSUMED']:g} under nested cross-validation** — accuracy
was ruled out before anything was fitted, because flagging nobody is right 77.9% of the time,
and average precision was ruled out later, for the reason in section 4. On average precision
the two families are a tie: LightGBM wins the cross-validation
({ms['lgbm_cv']:.4f} against {ms['logistic_cv']:.4f}) and then produces a paired difference on
validation of **{diff['point']:+.5f}, 95% interval [{diff['ci_lo']:+.4f}, {diff['ci_hi']:+.4f}]**.

On the test split it scores average precision **{published['average_precision']['point']:.4f}**
[{published['average_precision']['ci_lo']:.4f}, {published['average_precision']['ci_hi']:.4f}]
against a floor of {ref['dataset']['prevalence']:.4f}, ROC-AUC
**{published['roc_auc']['point']:.4f}**, and expected calibration error
**{published['ece']:.4f}** — calibration checked first, because a cost rule applied to an
uncalibrated score is arithmetic on the wrong quantity. The calibration method was chosen by a
rule written down before either model existed: adopt only if **both** Brier and ECE improve on
validation. For this model that rule **rejected isotonic** and adopted sigmoid.

**The test split has been opened twice** — once for a logistic regression selected on average
precision, and again for this model after the selection rule changed. Nothing on the test split
informed that change, but "opened once" is a stronger claim than this project can now make.

**This is behavioural scoring, not underwriting.** Every client already holds a card and six
months of history, so it says nothing about who should have been given one. Everyone here was
approved — declined applicants are absent and their counterfactual outcomes are
unobservable, so any bias in the original granting decision is invisible. Every group result
is scoped to *among clients who held a card*.

**One month, one country, twenty years ago** — October 2005, Taiwan, in the aftermath of a
domestic card-debt crisis. The base rate is not a general fact about consumer credit.

**`sex` is a binary 2005 administrative code**, not gender identity, with no non-binary
category. `education` and `marriage` contain undocumented codes pooled into an
`other_unknown` bucket too small to report. Nothing here is causal and nothing here is a
legal finding. The three non-`sex` attributes are exploratory: four attributes times three
statistics is twelve chances for something to look large, so a p = 0.035 found among twelve
is not a discovery.

**r = 10 was mine, not the data's.** That is why it is a slider.

**Data.** {ref['attribution']} Only derived outputs travel with this page: one calibrated
score per test client, the outcome, the four audited attributes, and the permutation draws.
"""
        )


if __name__ == "__main__":
    main()
