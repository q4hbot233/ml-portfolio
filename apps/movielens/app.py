"""Who does the recommender fail? - interactive version of the MovieLens write-up.

Runs in the browser under stlite (Streamlit on Pyodide), so nothing here fits a model.
Every number comes out of the aggregate tables in ``data/``, which were exported from the
executed analysis; this file only re-slices, re-aggregates and plots them.

Layout rule for this module: all loading and computation lives in plain functions that
take arguments and return frames or figures, with no Streamlit call inside them. The
Streamlit section at the bottom reads the widgets, calls those functions and renders.
That way the compute layer can be exercised from ordinary Python.

Two evaluation bases are shipped and never mixed inside a panel. ``*_test`` files are
fitted on train+validation and scored on the test set - the published comparison.
``*_val`` files are fitted on train and scored on validation - the analysis the model
choices were made on, and where the per-user and cold-start results come from.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
#: Warm = models that recommend from the popular end of the catalogue, cool = models that
#: recommend from the long tail, grey = the non-personalised floors. The grouping is a
#: finding - see the agreement matrix - not a convention.
COLOR = {
    "SVD": "#B2182B",
    "SVD+pop(alpha=3)": "#67001F",
    "UserItemBias": "#E08214",
    "Popularity": "#8C510A",
    "UserKNN": "#2166AC",
    "ItemKNN": "#4393C3",
    "NMF": "#762A83",
    "ItemMean": "#9E9E9E",
    "GlobalMean": "#C9C9C9",
}
#: A separate ramp for user segments, so people are never coloured like models.
PEOPLE = ["#D6E3E8", "#9CBCC9", "#5E8CA0", "#2C5468"]
BLOC_ORDER = ["SVD", "UserItemBias", "Popularity", "NMF", "ItemKNN", "UserKNN"]
PICKABLE = ["SVD", "SVD+pop(alpha=3)", "UserItemBias", "Popularity",
            "NMF", "ItemKNN", "UserKNN", "ItemMean", "GlobalMean"]
ENGINES = ["SVD", "NMF", "ItemKNN", "UserKNN"]
ACTIVITY_LABELS = ["Q1 lightest", "Q2", "Q3", "Q4 heaviest"]
TASTE_LABELS = ["Q1 most eccentric", "Q2", "Q3", "Q4 most mainstream"]
COLDSTART_LEVELS = ["0", "1", "2", "5", "10", "20", "50", "100", "all"]

TABLES = (
    "topn_test", "rating_test", "longtail_test", "rec_items_test",
    "activity_error_val", "activity_topn_val", "taste_error_val",
    "coldstart_curve_val", "coldstart_new_users", "agreement_val",
    "pop_prior_activity_val", "split_summary", "user_summary_val",
    "example_top10_user1", "tuning_popularity_alpha",
)

PLOT_STYLE = {
    "figure.dpi": 110, "figure.facecolor": "white",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E4E4E4", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "legend.frameon": False,
    "xtick.labelsize": 8.8, "ytick.labelsize": 8.8, "figure.titlesize": 11.5,
}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def data_dir(base: str | Path | None = None) -> Path:
    """Where the aggregate CSVs live, with a fallback for the browser bundle."""
    if base is not None:
        return Path(base)
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    for candidate in (here / "data", Path("data"), Path("/data")):
        if candidate.is_dir():
            return candidate
    return here / "data"


def load_tables(base: str | Path | None = None, names=TABLES) -> dict[str, pd.DataFrame]:
    """Read every shipped aggregate into a dict of frames. No model, no raw ratings."""
    root = data_dir(base)
    return {name: pd.read_csv(root / f"{name}.csv") for name in names}


# --------------------------------------------------------------------------- #
# compute - metrics at a chosen list length
# --------------------------------------------------------------------------- #
def metrics_at_n(topn: pd.DataFrame, rating: pd.DataFrame, n: int,
                 models=None) -> pd.DataFrame:
    """One row per model: its ranking metrics at top-N alongside its rating error.

    Those two columns are the whole argument. ``rmse`` is what the coursework reported;
    ``precision`` is what the system actually does. Models that only produce an ordering
    (Popularity ranks, it does not estimate stars) keep an NaN in the rating columns.
    """
    if n not in set(topn["n"]):
        raise ValueError(f"no exported metrics at n={n}")
    frame = topn[topn["n"] == n].merge(rating[["model", "rmse", "mae"]],
                                       on="model", how="left")
    if models is not None:
        frame = frame[frame["model"].isin(models)]
        order = {m: i for i, m in enumerate(models)}
        frame = frame.sort_values("model", key=lambda s: s.map(order))
    return frame.reset_index(drop=True)


def headline_gap(topn: pd.DataFrame, rating: pd.DataFrame, n: int,
                 a: str = "SVD", b: str = "UserKNN") -> dict:
    """The one-line case against RMSE: two models a rounding error apart on one metric
    and orders of magnitude apart on the other."""
    frame = metrics_at_n(topn, rating, n).set_index("model")
    rmse_a, rmse_b = float(frame.loc[a, "rmse"]), float(frame.loc[b, "rmse"])
    prec_a, prec_b = float(frame.loc[a, "precision"]), float(frame.loc[b, "precision"])
    return {
        "model_a": a, "model_b": b, "n": n,
        "rmse_a": rmse_a, "rmse_b": rmse_b,
        "rmse_gap": abs(rmse_a - rmse_b),
        "rmse_gap_pct": 100 * abs(rmse_a - rmse_b) / min(rmse_a, rmse_b),
        "precision_a": prec_a, "precision_b": prec_b,
        "precision_ratio": (max(prec_a, prec_b) / min(prec_a, prec_b)
                            if min(prec_a, prec_b) > 0 else float("inf")),
    }


def engine_rmse_spread(frame: pd.DataFrame, engines=ENGINES) -> float:
    """Best engine minus worst engine on RMSE - the gap the coursework argued over."""
    r = frame[frame["model"].isin(engines)]["rmse"].dropna()
    return float(r.max() - r.min())


def metric_rankings(frame: pd.DataFrame, models=None) -> pd.DataFrame:
    """Rank every model by RMSE and by precision, to show the two orders disagree."""
    f = frame if models is None else frame[frame["model"].isin(models)]
    f = f.dropna(subset=["rmse"]).copy()
    f["rank_rmse"] = f["rmse"].rank(method="min")
    f["rank_precision"] = f["precision"].rank(method="min", ascending=False)
    f["rank_change"] = f["rank_precision"] - f["rank_rmse"]
    return f[["model", "rmse", "precision", "rank_rmse",
              "rank_precision", "rank_change"]].sort_values("rank_rmse").reset_index(drop=True)


def display_models(highlight: str, base=BLOC_ORDER) -> list[str]:
    """The six models the figures compare, plus whatever the reader has selected."""
    return list(base) if highlight in base else [*base, highlight]


# --------------------------------------------------------------------------- #
# compute - people
# --------------------------------------------------------------------------- #
def rating_proxy(model: str, available: set[str]) -> str | None:
    """Whose rating errors describe this model.

    ``Popularity`` never estimates a star rating, so it has none. The popularity prior
    re-ranks SVD without touching a single prediction, so its rating errors *are* SVD's -
    which is itself one of the findings.
    """
    if model in available:
        return model
    if model.startswith("SVD+pop") and "SVD" in available:
        return "SVD"
    return None


def bucket_errors(activity_error: pd.DataFrame, model: str,
                  labels=ACTIVITY_LABELS) -> pd.DataFrame:
    """Per-user RMSE distribution inside each user bucket, for one model."""
    frame = activity_error[activity_error["model"] == model].copy()
    if frame.empty:
        raise ValueError(f"no per-user error exported for {model}")
    order = {b: i for i, b in enumerate(labels)}
    return frame.sort_values("bucket", key=lambda s: s.map(order)).reset_index(drop=True)


def bucket_ranking(activity_topn: pd.DataFrame, model: str, n: int,
                   labels=ACTIVITY_LABELS) -> pd.DataFrame:
    """Hit rate and precision at top-N for one model, bucket by bucket."""
    frame = activity_topn[(activity_topn["model"] == model) &
                          (activity_topn["n"] == n)].copy()
    if frame.empty:
        raise ValueError(f"no bucket ranking exported for {model} at n={n}")
    order = {b: i for i, b in enumerate(labels)}
    return frame.sort_values("bucket", key=lambda s: s.map(order)).reset_index(drop=True)


def bucket_summary(activity_error: pd.DataFrame, activity_topn: pd.DataFrame,
                   model: str, bucket: str, n: int) -> dict:
    """Everything the page says about one (model, bucket, N) cell.

    The error half is NaN for a model that emits no star estimate; the ranking half
    always exists, because ranking is the thing every model here does.
    """
    rank = bucket_ranking(activity_topn, model, n)
    rrow = rank[rank["bucket"] == bucket]
    if rrow.empty:
        raise ValueError(f"unknown bucket {bucket!r}")
    rrow = rrow.iloc[0]
    out = {
        "model": model, "bucket": bucket, "n": n,
        "users": int(rrow["users"]),
        "hit_rate": float(rrow["hit_rate"]), "precision": float(rrow["precision"]),
        "mean_relevant": float(rrow["mean_relevant"]),
        "share_with_nothing": 1.0 - float(rrow["hit_rate"]),
        "mean_rmse": float("nan"), "sd_rmse": float("nan"),
        "iqr": float("nan"), "mean_history": float("nan"),
        "error_model": None,
    }
    proxy = rating_proxy(model, set(activity_error["model"]))
    if proxy is not None:
        row = bucket_errors(activity_error, proxy)
        row = row[row["bucket"] == bucket].iloc[0]
        out.update({
            "mean_rmse": float(row["mean_rmse"]), "sd_rmse": float(row["sd_rmse"]),
            "iqr": float(row["p75"] - row["p25"]),
            "mean_history": float(row["mean_history"]),
            "error_model": proxy,
        })
    return out


def spread_summary(activity_error: pd.DataFrame, model: str) -> dict:
    """How the per-user error moves across activity quartiles: level against spread."""
    err = bucket_errors(activity_error, model)
    return {
        "mean_lightest": float(err["mean_rmse"].iloc[0]),
        "mean_heaviest": float(err["mean_rmse"].iloc[-1]),
        "sd_lightest": float(err["sd_rmse"].iloc[0]),
        "sd_heaviest": float(err["sd_rmse"].iloc[-1]),
        "widest_iqr": float((err["p75"] - err["p25"]).max()),
    }


# --------------------------------------------------------------------------- #
# compute - catalogue
# --------------------------------------------------------------------------- #
def popularity_profile(rec_items: pd.DataFrame, n: int, models=None) -> pd.DataFrame:
    """Where each model shops: the popularity of what it actually hands out.

    ``pop_pct`` is the film's percentile on the training-support curve, 1.0 being the
    most-rated film of the fitting period. A model averaging 0.15 is not being
    adventurous on purpose - it is taking an argmax over candidates it has no evidence
    about, and thin evidence is what produces an extreme estimate.
    """
    f = rec_items[rec_items["slot"] <= n]
    if models is not None:
        f = f[f["model"].isin(models)]
    rows = []
    for name, g in f.groupby("model"):
        w = g["n_users"].to_numpy(float)
        rows.append({
            "model": name,
            "mean_pop_pct": float(np.average(g["pop_pct"], weights=w)),
            "mean_pop_rank": float(np.average(g["pop_rank"], weights=w)),
            "median_train_ratings": _weighted_median(
                g["train_ratings"].to_numpy(float), w),
            "mean_train_ratings": float(np.average(g["train_ratings"], weights=w)),
            "distinct_films": int(g["pop_rank"].nunique()),
            "recommendations": int(w.sum()),
        })
    return (pd.DataFrame(rows).sort_values("mean_pop_pct", ascending=False)
            .reset_index(drop=True))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, cum[-1] / 2.0)])


def coverage_at_n(topn: pd.DataFrame, n: int, models=None) -> pd.DataFrame:
    """Share of the catalogue that reaches at least one person in a top-N list."""
    f = topn[topn["n"] == n][["model", "coverage", "novelty", "mean_train_ratings"]]
    if models is not None:
        f = f[f["model"].isin(models)]
    return f.sort_values("coverage", ascending=False).reset_index(drop=True)


def tail_stats(longtail: pd.DataFrame) -> dict:
    """What the catalogue looks like before any model touches it."""
    s = longtail["train_ratings"].to_numpy()
    return {
        "films": int(len(s)),
        "zero_support": int((s == 0).sum()),
        "under_five": int((s < 5).sum()),
        "median": float(np.median(s)),
        "max": int(s.max()),
        "top_decile_share": float(np.sort(s)[::-1][:max(1, len(s) // 10)].sum() / s.sum()),
    }


# --------------------------------------------------------------------------- #
# compute - cold start and the fix
# --------------------------------------------------------------------------- #
def coldstart_frame(curve: pd.DataFrame, column: str = "rmse",
                    levels=COLDSTART_LEVELS) -> pd.DataFrame:
    """Truncation level x model table for one metric, ordered along the sweep."""
    f = curve.dropna(subset=[column]).copy()
    f["level"] = f["level"].astype(str)
    wide = f.pivot_table(index="level", columns="model", values=column)
    return wide.loc[[lvl for lvl in levels if lvl in wide.index]]


def coldstart_dip(curve: pd.DataFrame, model: str = "SVD") -> dict:
    """The result I did not see coming: a few ratings are worse than none at all."""
    wide = coldstart_frame(curve, "rmse")[model]
    cold = float(wide.loc["0"])
    early = wide.loc[[lvl for lvl in ("1", "2", "5", "10", "20") if lvl in wide.index]]
    recovered = [lvl for lvl in wide.index[1:] if float(wide.loc[lvl]) < cold]
    return {
        "model": model, "rmse_at_zero": cold,
        "worst_level": str(early.idxmax()), "rmse_at_worst": float(early.max()),
        "penalty": float(early.max()) - cold,
        "recovers_at": recovered[0] if recovered else "never",
        "rmse_with_full_history": float(wide.loc["all"]),
    }


def prior_effect(pop_prior: pd.DataFrame, labels=ACTIVITY_LABELS) -> pd.DataFrame:
    order = {b: i for i, b in enumerate(labels)}
    return pop_prior.sort_values("bucket", key=lambda s: s.map(order)).reset_index(drop=True)


def agreement_matrix(agreement: pd.DataFrame, names=BLOC_ORDER) -> pd.DataFrame:
    m = agreement.pivot(index="model_a", columns="model_b", values="mean_shared")
    return m.reindex(index=names, columns=names)


def agreement_blocs(agreement: pd.DataFrame, warm=BLOC_ORDER[:3],
                    cool=BLOC_ORDER[3:]) -> dict:
    """Two models with the same RMSE need not recommend the same films - by how much."""
    m = agreement_matrix(agreement, list(warm) + list(cool))
    within = [m.loc[a, b] for grp in (warm, cool) for a in grp for b in grp if a != b]
    between = [m.loc[a, b] for a in warm for b in cool]
    return {
        "within_mean": float(np.mean(within)), "between_mean": float(np.mean(between)),
        "zero_pairs": int(sum(1 for v in between if v == 0)),
        "cross_pairs": int(len(between)),
        "svd_userknn": float(m.loc["SVD", "UserKNN"]),
        "n_users": int(agreement["n_users"].max()),
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _fmt_pct(x, _pos):
    return f"{100 * x:.0f}%"


def fig_two_metrics(frame: pd.DataFrame, n: int, gap: dict, highlight: str) -> plt.Figure:
    """Scatter of the two metrics against each other, and the two orderings they imply."""
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.4), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.3, 1]})
        rated = frame.dropna(subset=["rmse"])

        ax = axes[0]
        floor = max(float(frame["precision"].replace(0, np.nan).min()) / 3, 1e-4)
        for _, r in rated.iterrows():
            name = r["model"]
            y = max(float(r["precision"]), floor)
            ax.scatter(r["rmse"], y, s=min(30 + 0.9 * r["mean_train_ratings"], 620),
                       color=COLOR.get(name, "#888888"), zorder=3,
                       edgecolor="#222222" if name == highlight else "white",
                       linewidth=2.0 if name == highlight else 1.2)
            ax.annotate(name, (r["rmse"], y), textcoords="offset points",
                        xytext=(9, -3), fontsize=8.2, color=COLOR.get(name, "#888888"),
                        fontweight="bold" if name == highlight else "semibold")
        pop = frame[frame["model"] == "Popularity"]
        if not pop.empty:
            level = max(float(pop["precision"].iloc[0]), floor)
            ax.axhline(level, color=COLOR["Popularity"], ls="--", lw=1.3, zorder=1)
            ax.annotate("Popularity baseline (ranks only, no x position)",
                        xy=(1.0, level), xycoords=("axes fraction", "data"),
                        xytext=(-4, 4), textcoords="offset points", ha="right",
                        fontsize=7.6, color=COLOR["Popularity"])
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.set_xlabel("test RMSE, stars   (lower is better)")
        ax.set_ylabel(f"test precision@{n}, log scale   (higher is better)")
        ax.set_title("marker area = mean training ratings\nbehind a recommended film",
                     fontsize=9.2)

        ax = axes[1]
        order = metric_rankings(frame)
        for _, r in order.iterrows():
            name = r["model"]
            ax.plot([0, 1], [r["rank_rmse"], r["rank_precision"]],
                    color=COLOR.get(name, "#888888"),
                    lw=3.0 if name == highlight else 1.7,
                    alpha=1.0 if name == highlight else 0.7, zorder=3,
                    marker="o", markersize=6, markeredgecolor="white")
            ax.annotate(name, (0, r["rank_rmse"]), textcoords="offset points",
                        xytext=(-9, 0), ha="right", va="center", fontsize=8.0,
                        color=COLOR.get(name, "#888888"))
        ax.set_xlim(-0.9, 1.08)
        ax.set_xticks([0, 1], ["by RMSE", f"by precision@{n}"])
        ax.set_ylim(len(order) + 0.5, 0.5)
        ax.set_yticks(range(1, len(order) + 1))
        ax.set_ylabel("rank  (1 = best)")
        ax.grid(axis="x", visible=False)
        ax.set_title("the lines cross: a choice made on the\nleft says little about the right",
                     fontsize=9.2)

        ratio = gap["precision_ratio"]
        ratio_txt = "never-scores" if not np.isfinite(ratio) else f"{ratio:.0f}x"
        fig.suptitle(f"{gap['model_a']} and {gap['model_b']}: "
                     f"{gap['rmse_gap_pct']:.1f}% apart on RMSE, "
                     f"{ratio_txt} apart on precision@{n}", fontweight="bold")
    return fig


def fig_error_by_bucket(err: pd.DataFrame | None, rank: pd.DataFrame, model: str,
                        bucket: str, n: int, engine_spread: float) -> plt.Figure:
    """Per-user error and per-user usefulness, sliced by how much history a person has."""
    with plt.rc_context(PLOT_STYLE):
        if err is None:
            fig, ax_hit = plt.subplots(figsize=(7.6, 3.8), constrained_layout=True)
            axes_err = None
        else:
            fig, (axes_err, ax_hit) = plt.subplots(1, 2, figsize=(7.6, 4.1),
                                                   constrained_layout=True)

        if axes_err is not None:
            ax = axes_err
            stats = [{"med": r["median"], "q1": r["p25"], "q3": r["p75"],
                      "whislo": r["p10"], "whishi": r["p90"], "fliers": [],
                      "label": r["bucket"].replace(" ", "\n", 1)}
                     for _, r in err.iterrows()]
            bp = ax.bxp(stats, showfliers=False, widths=0.62, patch_artist=True)
            for patch, face, label in zip(bp["boxes"], PEOPLE, err["bucket"]):
                patch.set_facecolor(face)
                patch.set_edgecolor("#1F3A45" if label == bucket else "#9CBCC9")
                patch.set_linewidth(2.2 if label == bucket else 1.0)
            for part in ("medians", "whiskers", "caps"):
                for line in bp[part]:
                    line.set_color("#1F3A45")
            ax.plot(range(1, len(err) + 1), err["mean_rmse"], "o", color="#B2182B",
                    markersize=5, zorder=5, label="mean per-user RMSE")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_ylabel("per-user RMSE, stars   (box = p25-p75, whiskers p10-p90)")
            ax.set_xlabel("user activity: training-history length, quartiles")
            ax.set_title(f"{model}: the spread inside one quartile dwarfs the "
                         f"{engine_spread:.3f} stars\nbetween all four engines",
                         fontsize=9.2)

        ax = ax_hit
        colors = ["#1F3A45" if b == bucket else c for b, c in zip(rank["bucket"], PEOPLE)]
        bars = ax.bar(range(len(rank)), rank["hit_rate"], color=colors, width=0.62)
        for rect, v in zip(bars, rank["hit_rate"]):
            ax.annotate(f"{100 * v:.0f}%", (rect.get_x() + rect.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8.6)
        ax.set_xticks(range(len(rank)), [b.replace(" ", "\n", 1) for b in rank["bucket"]])
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_pct))
        ax.set_ylim(0, max(0.15, float(rank["hit_rate"].max()) * 1.3))
        ax.set_ylabel(f"users with >=1 useful film in their top {n}")
        ax.set_xlabel("user activity: training-history length, quartiles")
        lo, hi = float(rank["hit_rate"].iloc[0]), float(rank["hit_rate"].iloc[-1])
        ax.set_title(f"{model}: {100 * lo:.0f}% of the lightest quartile get anything,\n"
                     f"against {100 * hi:.0f}% of the heaviest", fontsize=9.2)
    return fig


def fig_long_tail(longtail: pd.DataFrame, rec_items: pd.DataFrame,
                  profile: pd.DataFrame, n: int, highlight: str,
                  models=None) -> plt.Figure:
    """The catalogue, and the slice of it each model actually hands out."""
    models = models or display_models(highlight)
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.6), constrained_layout=True,
                                 sharex=True, gridspec_kw={"height_ratios": [1.25, 1.75]})
        size = len(longtail)

        ax = axes[0]
        counts = longtail["train_ratings"].to_numpy()
        ax.fill_between(longtail["pop_rank"], 0.5, np.maximum(counts, 0.5),
                        color="#C6C6C6", lw=0)
        ax.set_yscale("log")
        ax.set_ylim(0.5, counts.max() * 2.0)
        ax.set_ylabel("ratings per film\n(log scale)")
        first_zero = int((counts > 0).sum()) + 1
        ax.axvline(first_zero, color="#666666", ls=":", lw=1.1)
        ax.annotate(f"{int((counts == 0).sum())} films with no\ntraining ratings at all",
                    xy=(first_zero, 25), xytext=(-10, 0), textcoords="offset points",
                    ha="right", fontsize=8, color="#555555")
        ax.set_title(f"Half the catalogue has under {np.median(counts):.0f} ratings behind "
                     f"it - and that half is where an unguarded argmax goes", fontsize=9.6)

        ax = axes[1]
        prof = profile.set_index("model")
        for row, name in enumerate(models):
            g = rec_items[(rec_items["model"] == name) & (rec_items["slot"] <= n)]
            if g.empty:
                continue
            w = g["n_users"].to_numpy(float)
            ax.scatter(g["pop_rank"], np.full(len(g), row), s=15,
                       color=COLOR.get(name, "#888888"),
                       alpha=0.25 + 0.55 * (w / w.max()), linewidths=0, zorder=3)
            ax.plot([prof.loc[name, "mean_pop_rank"]], [row], marker="|", markersize=20,
                    markeredgewidth=2.6, color="#111111", zorder=5)
            ax.annotate(f"mean percentile {prof.loc[name, 'mean_pop_pct']:.2f}   "
                        f"median film has {prof.loc[name, 'median_train_ratings']:.0f} ratings",
                        xy=(size, row), xytext=(-2, 7), textcoords="offset points",
                        ha="right", fontsize=7.6, color="#444444")
        ax.set_yticks(range(len(models)), models)
        for tick, name in zip(ax.get_yticklabels(), models):
            tick.set_color(COLOR.get(name, "#888888"))
            if name == highlight:
                tick.set_fontweight("bold")
        ax.set_ylim(len(models) - 0.45, -0.65)
        ax.set_xlim(0, size + 25)
        ax.set_xlabel("popularity rank of the film   (1 = most-rated film of the fitting "
                      "period)   |   black tick = mean")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"every film any user is shown in a top-{n} list", fontsize=9.6)
    return fig


def fig_coverage(cov: pd.DataFrame, n: int, highlight: str,
                 catalogue_size: int) -> plt.Figure:
    """How much of the catalogue ever reaches anybody."""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 3.8), constrained_layout=True)
        f = cov.iloc[::-1]
        bars = ax.barh(range(len(f)), f["coverage"],
                       color=[COLOR.get(m, "#888888") for m in f["model"]], height=0.66)
        for rect, m, v in zip(bars, f["model"], f["coverage"]):
            rect.set_edgecolor("#111111" if m == highlight else "white")
            rect.set_linewidth(1.8 if m == highlight else 0.6)
            ax.annotate(f"{100 * v:.1f}%   ({int(round(v * catalogue_size))} films)",
                        (v, rect.get_y() + rect.get_height() / 2),
                        textcoords="offset points", xytext=(5, 0), va="center",
                        fontsize=8.2, color="#333333")
        ax.set_yticks(range(len(f)), f["model"])
        ax.set_xlim(0, float(f["coverage"].max()) * 1.5)
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_pct))
        ax.set_xlabel(f"share of the {catalogue_size:,}-film catalogue reaching at least "
                      f"one person in a top-{n} list")
        ax.grid(axis="y", visible=False)
        best, worst = f.iloc[-1], f.iloc[0]
        ax.set_title(f"At top-{n} the widest model reaches "
                     f"{100 * best['coverage']:.0f}% of the catalogue, "
                     f"the narrowest {100 * worst['coverage']:.0f}%", fontsize=9.8)
    return fig


def fig_coldstart(curve: pd.DataFrame, dip: dict) -> plt.Figure:
    """Error and usefulness as a simulated new user rates their first films."""
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0), constrained_layout=True)
        rmse = coldstart_frame(curve, "rmse")
        hit = coldstart_frame(curve, "hit_rate@10")

        ax = axes[0]
        x = np.arange(len(rmse.index))
        for name in ("SVD", "UserKNN", "UserItemBias", "ItemMean"):
            if name not in rmse.columns:
                continue
            ax.plot(x, rmse[name].to_numpy(), "-o", lw=2.0, markersize=4.6,
                    color=COLOR.get(name, "#888888"), markeredgecolor="white",
                    label=name + (" (control)" if name == "ItemMean" else ""))
        worst_x = list(rmse.index).index(dip["worst_level"])
        ax.annotate(f"{dip['model']} with {dip['worst_level']} ratings is worse\n"
                    f"than {dip['model']} with none",
                    xy=(worst_x, dip["rmse_at_worst"]), xytext=(10, 40),
                    textcoords="offset points", fontsize=8.2, color=COLOR["SVD"],
                    arrowprops=dict(arrowstyle="->", color=COLOR["SVD"], lw=1.3))
        ax.set_xticks(x, rmse.index)
        ax.set_xlabel("ratings the model may see for these users\n(ordinal axis, not linear)")
        ax.set_ylabel("RMSE on their held-out ratings, stars")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("a handful of ratings is worse\nthan no ratings at all", fontsize=9.4)

        ax = axes[1]
        x = np.arange(len(hit.index))
        for name in ("SVD", "UserItemBias", "Popularity"):
            if name not in hit.columns:
                continue
            ax.plot(x, hit[name].to_numpy(), "-o", lw=2.0, markersize=4.6,
                    color=COLOR.get(name, "#888888"), markeredgecolor="white", label=name)
        ax.set_xticks(x, hit.index)
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_pct))
        ax.set_xlabel("ratings the model may see for these users\n(ordinal axis, not linear)")
        ax.set_ylabel("users with >=1 useful film in their top 10")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("switch to the personalised model at the\nend of that window, not the start",
                     fontsize=9.4)
    return fig


def fig_prior_effect(effect: pd.DataFrame, alpha_sweep: pd.DataFrame) -> plt.Figure:
    """The popularity prior lifts everybody, moves nobody, and is paid for in catalogue."""
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), constrained_layout=True)

        ax = axes[0]
        y = np.arange(len(effect))
        for i, r in effect.iterrows():
            ax.annotate("", xy=(r["hit_after"], i), xytext=(r["hit_before"], i),
                        arrowprops=dict(arrowstyle="-|>", color="#B9C6CC", lw=3))
        ax.scatter(effect["hit_before"], y, s=56, color="#9CBCC9", zorder=3,
                   label="SVD", edgecolor="white")
        ax.scatter(effect["hit_after"], y, s=56, color=COLOR["SVD"], zorder=3,
                   label="SVD + popularity prior", edgecolor="white")
        ax.set_yticks(y, [b.replace(" ", "\n", 1) for b in effect["bucket"]])
        ax.set_ylim(len(effect) - 0.5, -0.5)
        ax.set_xlim(0, float(effect["hit_after"].max()) * 1.25)
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_pct))
        ax.set_xlabel("users with >=1 useful film in their top 10")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("everybody moves up,\nnobody changes places", fontsize=9.4)

        ax = axes[1]
        s = alpha_sweep.sort_values("alpha")
        ax.plot(s["coverage"], s["ndcg@10"], "-o", color=COLOR["SVD"], lw=2.0,
                markersize=5, markeredgecolor="white")
        for _, r in s.iterrows():
            ax.annotate(f"{r['alpha']:g}", (r["coverage"], r["ndcg@10"]),
                        textcoords="offset points", xytext=(6, -10), fontsize=7.6,
                        color="#555555")
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_pct))
        ax.set_xlabel("catalogue coverage   (point labels = prior strength alpha)")
        ax.set_ylabel("validation NDCG@10")
        ax.set_title("the bill: the relevance is bought\nwith half the catalogue",
                     fontsize=9.4)
    return fig


def fig_agreement(matrix: pd.DataFrame, blocs: dict) -> plt.Figure:
    """Mean films shared between every pair of top-10 lists."""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
        ax.grid(False)
        shown = matrix.to_numpy(dtype=float).copy()
        np.fill_diagonal(shown, np.nan)
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#F2F2F2")
        im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=10)
        names = list(matrix.index)
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    ax.text(j, i, "-", ha="center", va="center", color="#AAAAAA")
                else:
                    v = shown[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.6,
                            color="white" if v > 5.5 else "#222222")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
            for tick, name in zip(ticks, names):
                tick.set_color(COLOR.get(name, "#888888"))
        ax.axhline(2.5, color="#333333", lw=2)
        ax.axvline(2.5, color="#333333", lw=2)
        i, j = names.index("SVD"), names.index("UserKNN")
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#B2182B", lw=2.4, zorder=5))
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("mean films shared out of 10, per user")
        ax.set_title(f"Two blocs: {blocs['within_mean']:.2f} films shared inside one, "
                     f"{blocs['between_mean']:.2f} across", fontsize=9.8)
    return fig


# --------------------------------------------------------------------------- #
# streamlit layer - widgets, calls, rendering; no computation of its own
# --------------------------------------------------------------------------- #
def main() -> None:  # pragma: no cover - exercised by the browser, not by the tests
    import streamlit as st

    st.set_page_config(page_title="Who does the recommender fail?", page_icon="🎬",
                       layout="centered")

    @st.cache_data(show_spinner=False)
    def _tables():
        return load_tables()

    t = _tables()
    catalogue_size = int(t["longtail_test"]["pop_rank"].max())
    split = t["split_summary"].set_index("part")

    st.title("Who does the recommender fail?")
    st.markdown(
        "The assignment was: build four recommenders on MovieLens-100K, compare them, "
        "declare a winner. Mine ended where thousands of MovieLens notebooks end - four "
        "RMSEs within a couple of hundredths of each other and a winner picked on the "
        "third decimal place. I came back to it with a different question: **when this "
        "thing is wrong, who is it wrong about?** RMSE cannot answer that, and it turns "
        "out it answers very little else either.\n\n"
        "*The full analysis lives in a private repository. This page ships only the "
        "aggregate tables it produced - no ratings, no user records.*")

    with st.sidebar:
        st.header("Controls")
        model = st.selectbox("Model", PICKABLE, index=0)
        bucket = st.selectbox("User activity bucket", ACTIVITY_LABELS, index=0,
                              help="Quartiles of training-history length: how many films "
                                   "the person had already rated when the model was fitted.")
        n = st.slider("Top-N list length", min_value=1, max_value=20, value=10,
                      help="Every ranking metric on this page is recomputed at this "
                           "list length.")
        st.caption(
            f"**The split.** {int(split.loc['train', 'rows']):,} training ratings, "
            f"{int(split.loc['validation', 'rows']):,} validation, "
            f"{int(split.loc['test', 'rows']):,} test; 943 users, "
            f"{catalogue_size:,} films. Per user and chronological - the last 20% of each "
            "person's history is held out - so no model trains on a user's future in "
            "order to predict their past.")

    shown = display_models(model)
    frame = metrics_at_n(t["topn_test"], t["rating_test"], n, models=shown)
    gap = headline_gap(t["topn_test"], t["rating_test"], n)
    spread = engine_rmse_spread(metrics_at_n(t["topn_test"], t["rating_test"], n))
    row = frame[frame["model"] == model].iloc[0]

    st.subheader("1. Two metrics, two different answers")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{model} - RMSE",
              "n/a" if pd.isna(row["rmse"]) else f"{row['rmse']:.4f}",
              help="Average star error over held-out ratings. The only number the "
                   "coursework reported.")
    c2.metric(f"precision@{n}", f"{row['precision']:.4f}",
              help=f"Share of the {n} recommended films the user went on to rate 4 or 5.")
    c3.metric("ratings behind a recommended film",
              f"{row['mean_train_ratings']:.0f}",
              help="Mean training support of the films this model puts in a list. Single "
                   "digits mean it is recommending films almost nobody rated.")
    st.pyplot(fig_two_metrics(frame, n, gap, model))
    ratio = gap["precision_ratio"]
    st.markdown(
        f"On the test set **{gap['model_a']} and {gap['model_b']} are "
        f"{gap['rmse_gap_pct']:.1f}% apart on RMSE** ({gap['rmse_a']:.4f} against "
        f"{gap['rmse_b']:.4f}) and "
        + (f"**{ratio:.0f}x apart on precision@{n}** "
           f"({gap['precision_a']:.4f} against {gap['precision_b']:.4f})."
           if np.isfinite(ratio) else
           f"**incomparable on precision@{n}**: {gap['model_b']} puts nothing useful in "
           f"{n} slot{'s' if n > 1 else ''} for anybody.")
        + f" The entire spread between the four engines on RMSE is {spread:.3f} stars. "
        "A model chosen on the rating metric tells you almost nothing about the list a "
        "user would actually see.")

    st.subheader("2. So who does it fail?")
    proxy = rating_proxy(model, set(t["activity_error_val"]["model"]))
    err = bucket_errors(t["activity_error_val"], proxy) if proxy else None
    rank = bucket_ranking(t["activity_topn_val"], model, n)
    summary = bucket_summary(t["activity_error_val"], t["activity_topn_val"],
                             model, bucket, n)
    st.pyplot(fig_error_by_bucket(err, rank, model, bucket, n, spread))
    lines = [
        f"**{bucket}** is {summary['users']} of the people this model was scored on. "
        f"{100 * summary['share_with_nothing']:.0f}% of them get nothing they later rated "
        f"4 or 5 anywhere in their top {n}, against "
        f"{100 * (1 - float(rank['hit_rate'].iloc[-1])):.0f}% of the heaviest quartile."]
    if proxy is not None:
        sp = spread_summary(t["activity_error_val"], proxy)
        lines.append(
            f" {model if proxy == model else proxy} is wrong about them by "
            f"{summary['mean_rmse']:.3f} stars on average, with a standard deviation of "
            f"{summary['sd_rmse']:.2f} *across individuals*. Notice what activity does and "
            f"does not change: the quartile means barely move "
            f"({sp['mean_lightest']:.2f} to {sp['mean_heaviest']:.2f}) while the spread "
            f"halves ({sp['sd_lightest']:.2f} to {sp['sd_heaviest']:.2f}). A light user is "
            "not reliably served worse - they are served *unpredictably*, which for a "
            "product is the worse of the two problems.")
    else:
        lines.append(f" {model} produces an ordering, not a star estimate, so it has no "
                     "rating error to slice.")
    if model.startswith("SVD+pop"):
        lines.append(" The popularity prior re-ranks SVD without changing a single "
                     "rating prediction, so the error panel is SVD's by construction.")
    st.markdown("".join(lines))
    st.caption("The comparison across quartiles is not perfectly clean: a light user has "
               "fewer held-out ratings and so fewer chances for one of ten guesses to "
               "count as a hit. Hit rate and per-user RMSE carry this result; the raw "
               "precision ratio is not an effect size.")

    with st.expander("The other slice: taste, not activity"):
        taste_model = proxy or "SVD"
        taste = bucket_errors(t["taste_error_val"], taste_model, labels=TASTE_LABELS)
        st.dataframe(
            taste[["bucket", "users", "mean_rmse", "sd_rmse", "median"]]
            .rename(columns={"bucket": f"taste quartile ({taste_model})",
                             "mean_rmse": "mean per-user RMSE", "sd_rmse": "sd",
                             "median": "median per-user RMSE"})
            .style.format({"mean per-user RMSE": "{:.3f}", "sd": "{:.3f}",
                           "median per-user RMSE": "{:.3f}"}),
            hide_index=True)
        st.caption(
            "Taste is the correlation between a user's ratings and what everybody else "
            "gave the same films, leave-one-out so a user cannot correlate with their own "
            "contribution to the item mean. Activity moves the *spread* of the error; "
            "taste moves its *level*. Short history, unusual taste - exactly the people a "
            "recommender is for.")

    st.subheader("3. Where in the catalogue does each model shop?")
    profile = popularity_profile(t["rec_items_test"], n)
    st.pyplot(fig_long_tail(t["longtail_test"], t["rec_items_test"], profile, n, model))
    tail = tail_stats(t["longtail_test"])
    prof_row = profile.set_index("model").loc[model]
    st.markdown(
        f"{tail['zero_support']} of the {tail['films']:,} films have no training ratings "
        f"at all, {tail['under_five']} have fewer than five, and the median film has "
        f"{tail['median']:.0f}. **{model}** recommends at popularity percentile "
        f"{prof_row['mean_pop_pct']:.2f} on average, its median recommended film carrying "
        f"{prof_row['median_train_ratings']:.0f} training ratings, and over all users it "
        f"ever names {prof_row['distinct_films']} distinct films. The two KNNs sit at the "
        "bottom of that curve, and it is not taste: an argmax over fifteen hundred "
        "candidates systematically selects the *noisiest* estimates. A film with two "
        "ratings, both 5, gets a confident-looking score; a film with three hundred is "
        "pinned near its true mean and can never win. RMSE never sees this, because RMSE "
        "is only scored on pairs the user actually rated - and nobody rates the films with "
        "two ratings. The pathology lives entirely in the part of the catalogue the rating "
        "metric never visits.")

    st.subheader("4. What that costs the catalogue")
    st.pyplot(fig_coverage(coverage_at_n(t["topn_test"], n, models=PICKABLE), n, model,
                           catalogue_size))
    st.caption("Coverage is the share of films that reach at least one person. It is a "
               "descriptor, not a target: a model can buy relevance by collapsing onto the "
               "same forty famous films, which is exactly what the fix in panel 6 does.")

    st.subheader("5. Cold start is a curve, not a switch")
    dip = coldstart_dip(t["coldstart_curve_val"])
    st.pyplot(fig_coldstart(t["coldstart_curve_val"], dip))
    st.markdown(
        f"150 users had their history truncated to their first 0, 1, 2, 5 ... ratings and "
        f"every model was refitted at each level. **{dip['model']} with "
        f"{dip['worst_level']} ratings is worse than {dip['model']} with none** "
        f"({dip['rmse_at_worst']:.3f} against {dip['rmse_at_zero']:.3f}), and it does not "
        f"get back below its no-history score until {dip['recovers_at']} ratings. With no "
        "history the model falls back to the global and item terms; give it five and it "
        "fits a user factor vector to them, and a factor vector estimated from five "
        "observations is mostly noise. `ItemMean` is the control - no user representation "
        "at all - and its line is flat, which is what makes that reading defensible.")
    with st.expander("The all-or-nothing version: 100 brand-new users"):
        cold = t["coldstart_new_users"]
        st.dataframe(
            cold[["model", "rmse", "precision@10", "hit_rate@10"]]
            .style.format({"rmse": "{:.4f}", "precision@10": "{:.4f}",
                           "hit_rate@10": "{:.4f}"}, na_rep="-"),
            hide_index=True)
        st.caption("UserKNN, ItemKNN and NMF cannot represent an unseen user, so they fall "
                   "back to the global mean and score identically to it - which is the "
                   "whole problem. SVD does better only because its user bias defaults to "
                   "zero rather than its prediction.")

    st.subheader("6. A fix for the ranking, and what the fix costs")
    effect = prior_effect(t["pop_prior_activity_val"])
    st.pyplot(fig_prior_effect(effect, t["tuning_popularity_alpha"]))
    st.markdown(
        "Adding `alpha * log10(1 + training ratings)` to the ranking score - alpha chosen "
        "on validation - changes no rating prediction at all, which is one more sign that "
        "RMSE was never watching this part of the system. It lifts the lightest quartile "
        f"from {effect['hit_before'].iloc[0]:.2f} to {effect['hit_after'].iloc[0]:.2f} and "
        f"the heaviest from {effect['hit_before'].iloc[-1]:.2f} to "
        f"{effect['hit_after'].iloc[-1]:.2f}; "
        f"{100 * effect['users_hurt'].mean():.0f}% of users are made worse off on average. "
        "Everybody gains and nobody changes places - it fixes the symptom, ranking by "
        "predicted score alone, and does nothing about who the system serves badly. Part "
        "of what it is being rewarded for, offline, is guessing which films the user had "
        "already heard of.")

    with st.expander("Do two models with the same RMSE recommend the same films?"):
        blocs = agreement_blocs(t["agreement_val"])
        st.pyplot(fig_agreement(agreement_matrix(t["agreement_val"]), blocs))
        st.markdown(
            f"Over {blocs['n_users']} users and ten slots each: **"
            f"{blocs['svd_userknn']:.2f} films shared between SVD and the user-based "
            f"KNN**, the pair that sits two hundredths of a star apart on RMSE. "
            f"{blocs['zero_pairs']} of the {blocs['cross_pairs']} cross-bloc pairs share "
            "nothing whatsoever. Whatever \"SVD beat ItemKNN by 0.01 RMSE\" means, it does "
            "not mean the two systems would show a person similar things.")

    with st.expander("One user's actual top ten"):
        st.dataframe(t["example_top10_user1"], hide_index=True)
        st.caption("`train_ratings` is the evidence behind each film. The coursework's own "
                   "blended model fills its list with films carrying one or two ratings "
                   "each; the same fitted model re-ranked with the prior does not. No "
                   "rating prediction differs between those two lists.")

    st.divider()
    st.caption(
        "**Basis.** Panels 1, 3 and 4 are test-set numbers - fitted on train+validation, "
        "scored once on 20,381 held-out ratings, 908 users with a held-out 4-or-5 to rank "
        "against. Panels 2, 5 and 6 are the validation-set analysis the model choices were "
        "made on. One split, one seed: treat any RMSE gap under about 0.01 as noise. "
        "Everything is offline - nobody ever saw these lists, and the only evidence a "
        "recommendation was good is that the user happened to rate that film highly later, "
        "so every ranking number here is a lower bound. One dataset from one era; nothing "
        "here generalises to recommender algorithms at large.\n\n"
        "**Data.** MovieLens-100K, GroupLens Research, University of Minnesota. Used under "
        "research-use terms that forbid redistribution, so this page ships aggregates "
        "only. Please cite F. M. Harper and J. A. Konstan (2015), *The MovieLens Datasets: "
        "History and Context*, ACM TiiS 5(4), article 19.")


if __name__ == "__main__":
    main()
