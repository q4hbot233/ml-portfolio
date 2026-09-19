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
PARITY_KEYS = ("demographic_parity_difference", "fpr_difference", "tpr_difference")

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


def load_nulls(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """The 5,000 permutation draws per ``attribute__statistic``, at the frozen threshold."""
    return pd.read_csv(Path(data_dir) / "permutation_nulls.csv")


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
    straddle it. :func:`permutation_null` is what tests them.
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
# IS THE GAP REAL?
# =====================================================================================

def permutation_null(y_true, y_pred, groups, keys=PARITY_KEYS, n_perm: int = 5000,
                     seed: int = PERMUTATION_SEED, min_n: int = MIN_AUDIT_CELL) -> dict:
    """Shuffle the group labels; keep the model, the cut and every outcome fixed.

    Under the null that group membership is unrelated to how this rule treats a client,
    every relabelling is as likely as the observed one, so the share of shuffles reaching
    the observed gap is a one-sided p-value. Group sizes are preserved by construction,
    which is the right conditioning -- they are a property of the sample, not an estimate.

    Cells below ``min_n`` are dropped once, before shuffling, so the observed statistic and
    the null are computed over the same groups.

    The p-value is the add-one estimator ``(1 + #{null >= observed}) / (1 + n_perm)``: the
    observed labelling is itself an arrangement under the null, so a finite run of shuffles
    cannot honestly report zero.

    Returns ``{key: {"observed", "p_value", "null_median", "null_q95", "n_perm", "draws"}}``.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    g = np.asarray(groups).astype(object)

    full = group_table(y, p, g)
    keep = np.isin(g, list(full.index[full["n"] >= min_n]))
    y, p, g = y[keep], p[keep], g[keep]

    observed = parity_gaps(group_table(y, p, g))
    codes, levels = group_codes(g)
    k = len(levels)
    index = {name: np.flatnonzero(mask) for name, mask in
             {"tp": (y == 1) & (p == 1), "fp": (y == 0) & (p == 1),
              "fn": (y == 1) & (p == 0), "tn": (y == 0) & (p == 0)}.items()}

    rng = np.random.default_rng(seed)
    draws = {key: np.empty(int(n_perm), dtype=float) for key in keys}
    for i in range(int(n_perm)):
        permuted = rng.permutation(codes)
        counts = {name: np.bincount(permuted[idx], minlength=k) for name, idx in index.items()}
        gaps = parity_gaps_from_rates(rates_from_counts(counts))
        for key in keys:
            draws[key][i] = gaps[key]

    out = {}
    for key in keys:
        d = draws[key]
        d = d[np.isfinite(d)]
        out[key] = {
            "observed": float(observed[key]),
            "p_value": (float((1.0 + np.sum(d >= observed[key])) / (1.0 + d.size))
                        if d.size else float("nan")),
            "null_median": float(np.median(d)) if d.size else float("nan"),
            "null_q95": float(np.quantile(d, 0.95)) if d.size else float("nan"),
            "n_perm": int(n_perm),
            "draws": d,
        }
    return out


def shipped_null(nulls: pd.DataFrame, reference: dict, attribute: str, keys=PARITY_KEYS) -> dict:
    """The published 5,000-shuffle null for one attribute, read back from ``data/``.

    Same shape as :func:`permutation_null`, so the two are interchangeable at the call site.
    """
    published = reference["permutation_test_at_frozen_threshold"][attribute]
    return {key: {**{k: float(published[key][k]) for k in
                     ("observed", "p_value", "null_median", "null_q95")},
                  "n_perm": int(published[key]["n_perm"]),
                  "draws": nulls[f"{attribute}__{key}"].to_numpy(dtype=float)}
            for key in keys}


def null_for_threshold(rows: pd.DataFrame, nulls: pd.DataFrame, reference: dict,
                       attribute: str, threshold: float,
                       n_perm: int = 2000) -> tuple[dict, bool]:
    """The null for the decision currently on screen, and whether it is the published one.

    The null depends on the decision, not on the number that produced it. Whenever the
    threshold on screen flags exactly the clients the frozen cut flagged, the shipped
    5,000-shuffle result *is* the right answer and is returned unchanged; only a genuinely
    different decision triggers a fresh run.
    """
    score = rows["score"].to_numpy()
    pred = flag(score, threshold)
    frozen = flag(score, float(reference["frozen"]["threshold"]))
    if np.array_equal(pred, frozen):
        return shipped_null(nulls, reference, attribute), True
    return permutation_null(rows["defaulted"].to_numpy(), pred,
                            rows[attribute].to_numpy(), n_perm=n_perm), False


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


def compute_null(attribute: str, threshold: float, n_perm: int = 2000,
                 data_dir: Path | str = DATA_DIR) -> tuple[dict, bool]:
    return null_for_threshold(load_rows(data_dir), load_nulls(data_dir),
                              load_reference(data_dir), attribute, threshold, n_perm)


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


def figure_permutation_nulls(null: dict, attribute: str):
    """The null each gap is tested against, with the observed gap drawn on it."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), dpi=110)
    fig.patch.set_facecolor("white")
    n_sig = sum(null[k]["p_value"] < 0.05 for k in PARITY_KEYS)

    for ax, key in zip(axes, PARITY_KEYS):
        row = null[key]
        draws, obs = row["draws"] * 100, row["observed"] * 100
        ax.set_facecolor("white")
        top = max(float(np.quantile(draws, 0.999)), obs) * 1.14
        bins = np.linspace(0, top, 46)
        ax.hist(draws, bins=bins, color=GREY, alpha=0.5, lw=0)
        tail = draws[draws >= obs]
        if tail.size:
            ax.hist(tail, bins=bins, color=RED, alpha=0.7, lw=0)
        ax.axvline(obs, color=RED, lw=2.0, zorder=5)
        inside = row["p_value"] >= 0.05
        ax.annotate(f"observed {obs:.2f} pts\np = {row['p_value']:.3f}",
                    xy=(obs, ax.get_ylim()[1] * 0.98),
                    xytext=(-5 if inside else 5, 0), textcoords="offset points",
                    ha="right" if inside else "left", va="top",
                    fontsize=8.5, color=RED, fontweight="semibold")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
        ax.set_yticks([])
        ax.set_xlim(0, top)
        ax.set_title(PARITY_LABEL[key], fontsize=9.5, color=INK, loc="left", pad=7,
                     fontweight="semibold")
        ax.set_xlabel("gap between groups, percentage points", fontsize=8, color=MUTED,
                      labelpad=5)

    verdict = {0: "none of the three gaps is", 1: "one of the three gaps is",
               2: "two of the three gaps are", 3: "all three gaps are"}[n_sig]
    fig.suptitle(
        f"Shuffling {ATTRIBUTE_LABEL[attribute]} {null[PARITY_KEYS[0]]['n_perm']:,} times: "
        f"{verdict} bigger than chance produces",
        fontsize=11.5, color=INK, x=0.011, ha="left", y=0.99, fontweight="semibold")
    fig.text(0.011, 0.875,
             "grey — gaps a random relabelling produces at these group sizes;   "
             "red — the shuffles that reach the observed gap",
             fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.835))
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
    get_null = cache(compute_null)

    ref = get_reference()
    frozen_t = float(ref["frozen"]["threshold"])
    published = ref["published_test"]

    st.title("Who pays for the model's mistakes?")
    st.markdown(
        "A credit-default classifier on 30,000 Taiwanese card accounts, pushed past the ROC "
        "curve to the decision it implies. Every project on this dataset ends at an AUC in "
        "the high 0.70s — but a probability is not a decision. Somebody has to draw a line "
        "and say *these clients get flagged and those do not*, and the moment that line "
        "exists it hands a bill to somebody. Move the line below and watch the bill move: "
        "what the cut costs, who absorbs the errors, and whether the gaps between groups "
        "are bigger than random relabelling would produce.\n\n"
        "Everything here is computed live from the model's 6,000 held-out test scores, "
        "shipped with this page. **The full analysis — model selection, calibration, "
        "interpretability, the mitigation attempts — lives in a private repository.**"
    )

    st.divider()
    st.subheader("1 · Draw the line")

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
        "whole project for this panel alone and never reached a `fit` call. Drag the "
        "threshold above and watch the bars move."
    )
    tables = get_audit(threshold)
    st.pyplot(figure_group_errors(tables, threshold), use_container_width=True)
    st.caption(
        "At the deployed cut the two sexes are wrong in opposite directions: men are "
        "flagged more often and absorb more false flags, while women absorb more missed "
        "defaults. Notice too that the spread across age band and education is roughly "
        "four to five times the spread across sex."
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
            "Base rates differ across every attribute audited, so demographic parity and "
            "equalised odds cannot both hold. Both are shown and no winner is declared. "
            "Groups under 100 rows are printed with their n but kept out of every gap."
        )

    st.divider()
    st.subheader("3 · Is the gap real?")
    st.markdown(
        "My first reading of these gaps was the obvious one: the bootstrap intervals "
        "excluded zero, so there were four real disparities. That reading is wrong. Every "
        "measure here is a `max − min` statistic, so its bootstrap distribution is "
        "non-negative by construction — an interval can approach zero but never straddle "
        "it. *Excludes zero* describes how precisely a gap is estimated, not whether there "
        "is one. The null I actually care about is easy to sample exactly: hold the model, "
        "the cut and every client's outcome fixed, and shuffle the group labels."
    )
    attribute = st.selectbox(
        "Attribute", ATTRIBUTES, index=0,
        format_func=lambda a: ATTRIBUTE_LABEL[a] +
        ("  —  pre-specified primary" if a == "sex" else "  —  exploratory"))
    n_perm = 2000
    with st.spinner("Shuffling group labels…"):
        null, is_published = get_null(attribute, threshold, n_perm)
    st.pyplot(figure_permutation_nulls(null, attribute), use_container_width=True)
    if is_published:
        st.caption(
            "This is the published run: 5,000 shuffles, seed 907, at the frozen cut. The "
            "sex catch-rate gap is 0.0197 and randomly relabelling 6,000 clients exceeds it "
            "about 29% of the time — squarely inside what noise generates at these group "
            "sizes — while the selection-rate and false-alarm gaps for the same attribute "
            "sit far outside their nulls (p = 0.0006 and p = 0.005). Three of the four sex "
            "gaps are real in this sample, one is not, and nothing in the bootstrap "
            "intervals distinguishes them."
        )
    else:
        st.caption(
            f"Your threshold flags a different set of clients than the frozen cut, so this "
            f"null was recomputed for the decision on screen: {n_perm:,} shuffles, seed 907. "
            f"The published run used 5,000 at the frozen cut — set the threshold back to "
            f"{snap_to_grid(frozen_t):.4f} to see it."
        )
    st.caption(
        "The pattern repeats across all four attributes: selection-rate and false-alarm "
        "gaps real and large, catch-rate gaps mostly not. That is mechanically sensible — "
        "at a cut this low nearly every defaulter is flagged in every group, so the catch "
        "rate has almost no room to differ while the selection rate has all the room in "
        "the world."
    )

    st.divider()
    st.subheader("4 · What changing the model actually bought")
    st.markdown(
        "The rule on screen is the **second** model this project put on the test split, and the "
        "story of why is the most useful thing here. The first version chose between logistic "
        "regression and LightGBM on **average precision**, got an interval straddling zero, "
        "called it a tie and kept the simpler model. Average precision integrates precision over "
        "the whole recall axis \u2014 and this rule operates at recall 0.92, one end of it."
    )
    sa = ref.get("selection_audit")
    sw = ref.get("swap")
    if sa:
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
        st.caption(
            f"Nested cross-validation on {sa['design']['n_rows']:,} pooled training and "
            f"validation clients, {sa['design']['outer']}. The inner loop tunes **each family "
            f"separately on outer-training rows only**, and both arms are built exactly as the "
            f"deployed model is, so this compares families and not calibration states. A single "
            f"6,000-row validation split cannot see this difference at all \u2014 its paired "
            f"bootstrap on cost runs from about \u2212318 to +81. Ten folds over 24,000 rows can."
        )
        st.markdown(
            "So the selection rule changed, the model changed with it, and the test split was "
            "opened a second time."
        )

    if sw:
        a, b = sw["arms"]["superseded"], sw["arms"]["current"]
        st.markdown(f"#### And then it was worth {abs(sw['difference_at_frozen_cuts']):,.0f} units "
                    f"out of {a['cost_at_frozen_cut']:,.0f}")
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
        st.caption(
            f"The nested comparison implied roughly 180 units on 6,000 rows. The frozen cuts "
            f"delivered {abs(sw['difference_at_frozen_cuts']):,.0f} \u2014 "
            f"{abs(sw['difference_at_frozen_cuts']) / a['cost_at_frozen_cut']:.2%}."
        )
        c1, c2 = st.columns(2)
        c1.markdown(
            f"**LightGBM's threshold travels worse.** Its cost curve is less flat near the "
            f"minimum, so a cut chosen on a different 6,000 rows lands further from optimal "
            f"({b['threshold_transfer_loss']:+,.0f} against {a['threshold_transfer_loss']:+,.0f}). "
            f"That is {sw['explained_by_threshold_transfer']:+,.0f} units of the gap, and the "
            f"nested design is blind to it \u2014 it gives every fold its own cheapest cut."
        )
        c2.markdown(
            f"**The rest is noise.** Expected cost on these 6,000 clients has a bootstrap "
            f"standard deviation of **{sw['bootstrap_sd_of_expected_cost']:,.0f} units**. So "
            f"{sw['difference_at_own_best_cuts']:+,.0f} and \u2212180 are not distinguishable from "
            f"each other, and neither is distinguishable from zero."
        )
        st.info(
            "**What this cost.** \"Held out, opened once\" became \"opened twice under a rule "
            "that changed in between\", in exchange for 0.18%. Nothing on the test split informed "
            "the change \u2014 the nested comparison never reads those rows \u2014 but that is a defence, "
            "not a justification. The one unambiguous gain is elsewhere: re-running the "
            "calibration rule for the new model **rejected isotonic** (it improves Brier and "
            "worsens ECE) and adopted sigmoid, which more than halves expected calibration error "
            "on test, 0.0150 \u2192 0.0062.",
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
