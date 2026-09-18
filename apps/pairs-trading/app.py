"""A strategy that dies at 6 basis points.

Interactive cost-sensitivity explorer for an out-of-sample pairs-trading study:
40 US-listed ETFs, 780 pairs per window, 37 non-overlapping walk-forward windows,
4,662 out-of-sample days.

STRUCTURE. Everything above the `# ---- streamlit ----` marker is plain Python:
loaders and compute functions that take arguments and return dataframes, arrays
or dicts, with no streamlit call anywhere inside them. The plotting helpers
likewise take data and return a matplotlib Figure. The streamlit section only
reads widgets, calls those functions and renders what comes back. That is so the
compute layer can be exercised in ordinary Python, which is how the numbers on
this page were checked.

Every number displayed comes from a shipped artifact derived from the study's
executed outputs. No price data is shipped -- only my own derived return and
spread series.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
try:  # pragma: no cover - backend selection differs in the browser runtime
    matplotlib.use("Agg")
except Exception:
    pass
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# Constants. These mirror the study's config and are not free parameters here.
# --------------------------------------------------------------------------
BPS = 1e-4
TRADING_DAYS_PER_YEAR = 252
DEFAULT_COST_BPS = 5.0
DEFAULT_BORROW_BPS = 50.0

C = {
    "ink": "#1b1b1b",
    "muted": "#8a8a8a",
    "faint": "#d9d9d9",
    "gross": "#6f6f6f",
    "net": "#b3452b",
    "random": "#c9a227",
    "bench": "#2f6f9f",
    "stop": "#c0392b",
    "converge": "#1f8a70",
    "entry": "#2f6f9f",
    "grid": "#b9b9b9",
}

PLOT_STYLE = {
    "figure.dpi": 110,
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#585858",
    "axes.labelcolor": C["ink"],
    "text.color": C["ink"],
    "xtick.color": "#585858",
    "ytick.color": "#585858",
    "axes.grid": True,
    "grid.color": C["grid"],
    "grid.alpha": 0.32,
    "grid.linewidth": 0.6,
    "axes.titlelocation": "left",
    "axes.titlepad": 8.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
}


# ==========================================================================
# LOADERS
# ==========================================================================
def data_dir() -> Path:
    """Locate the shipped data directory in either a local or browser runtime."""
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent / "data")
    except NameError:  # pragma: no cover - stlite may not define __file__
        pass
    candidates += [Path("data"), Path("/data"), Path.cwd() / "data"]
    for c in candidates:
        if (c / "daily_oos.csv").exists():
            return c
    return candidates[0]


def load_daily(d: Path) -> pd.DataFrame:
    """Daily out-of-sample series, cost-agnostic.

    Columns: date, gross (daily return on committed capital, before any cost),
    turnover (notional traded that day, in committed-capital units),
    short_notional (notional short that day), n_active (pairs in position).
    """
    df = pd.read_csv(d / "daily_oos.csv", parse_dates=["date"])
    return df.set_index("date").sort_index()


def load_grid(d: Path) -> pd.DataFrame:
    return pd.read_csv(d / "sensitivity_grid.csv")


def load_per_fold(d: Path) -> pd.DataFrame:
    return pd.read_csv(d / "per_fold.csv", parse_dates=["trading_start", "trading_end"])


def load_selected_pairs(d: Path) -> pd.DataFrame:
    return pd.read_csv(d / "selected_pairs.csv")


def load_exemplar(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "exemplar_pair.csv", parse_dates=["date"])
    df["event"] = df["event"].fillna("")
    return df


def load_facts(d: Path) -> dict:
    return json.loads((d / "study_facts.json").read_text())


# ==========================================================================
# COMPUTE
# ==========================================================================
def transaction_cost(daily: pd.DataFrame, cost_bps_per_side: float) -> pd.Series:
    """Spread/commission/slippage charged on the day's notional change."""
    return daily["turnover"] * float(cost_bps_per_side) * BPS


def borrow_cost(daily: pd.DataFrame, borrow_bps_per_year: float) -> pd.Series:
    """Daily accrual on whatever leg is actually short."""
    return daily["short_notional"] * float(borrow_bps_per_year) * BPS / TRADING_DAYS_PER_YEAR


def net_returns(daily: pd.DataFrame, cost_bps_per_side: float,
                borrow_bps_per_year: float) -> pd.Series:
    """Net daily return. The cost model is linear, so one backtest prices at any level.

    net_t = gross_t - turnover_t * c * 1e-4 - short_notional_t * b * 1e-4 / 252
    """
    net = (daily["gross"]
           - transaction_cost(daily, cost_bps_per_side)
           - borrow_cost(daily, borrow_bps_per_year))
    net.name = "net"
    return net


def equity_curve(returns: pd.Series) -> pd.Series:
    """Growth of one unit of committed capital."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    curve = equity_curve(returns)
    return float((curve / curve.cummax() - 1.0).min())


def newey_west_t(x: np.ndarray | pd.Series, lags: int | None = None) -> tuple[float, float]:
    """HAC t-statistic on the mean daily return, Bartlett kernel.

    Daily strategy returns are autocorrelated while a position is held, so an
    OLS t-stat is untrustworthy. Lag truncation is the usual 4*(T/100)^(2/9).
    Reproduces the study's statsmodels HAC figure to ~1e-9.
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 30:
        return float("nan"), float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mean = v.mean()
    e = v - mean
    s = float(e @ e) / n
    for k in range(1, lags + 1):
        s += 2.0 * (1.0 - k / (lags + 1.0)) * float(e[k:] @ e[:-k]) / n
    if s <= 0:
        return float("nan"), float("nan")
    t = mean / math.sqrt(s / n)
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return float(t), float(p)


def sharpe_standard_error(sharpe_ann: float, n_periods: int,
                          periods_per_year: float = TRADING_DAYS_PER_YEAR) -> float:
    """Lo (2002) i.i.d. standard error of an ANNUALISED Sharpe.

    SE(SR_ann) = sqrt(q) * sqrt((1 + SR_period^2 / 2) / T). The (1 + SR^2/2)
    term takes the PER-PERIOD Sharpe; putting the annualised one in there
    inflates the interval.
    """
    if not np.isfinite(sharpe_ann) or n_periods <= 0:
        return float("nan")
    per_period = float(sharpe_ann) / math.sqrt(periods_per_year)
    return float(math.sqrt((1.0 + 0.5 * per_period ** 2) / n_periods)
                 * math.sqrt(periods_per_year))


def performance(returns: pd.Series, turnover: pd.Series | None = None) -> dict:
    """Full performance record for one daily series of returns on committed capital.

    The book is dollar-neutral and self-financing to first order, so the series
    is already an excess return and no risk-free rate is subtracted again.
    """
    r = returns.dropna().astype(float)
    n = len(r)
    sd = float(r.std(ddof=1))
    sharpe = float(r.mean()) / sd * math.sqrt(TRADING_DAYS_PER_YEAR) if sd > 0 else float("nan")
    se = sharpe_standard_error(sharpe, n)
    t, p = newey_west_t(r)
    mdd = max_drawdown(r)
    ann_ret = float((1.0 + r).prod() ** (TRADING_DAYS_PER_YEAR / n) - 1.0) if n else float("nan")
    return {
        "n_days": n,
        "ann_return": ann_ret,
        "ann_vol": sd * math.sqrt(TRADING_DAYS_PER_YEAR),
        "sharpe": sharpe,
        "sharpe_se": se,
        "sharpe_ci95": (sharpe - 1.96 * se, sharpe + 1.96 * se),
        "nw_tstat": t,
        "nw_pvalue": p,
        "max_drawdown": mdd,
        "cumulative": float((1.0 + r).prod() - 1.0),
        "hit_rate_daily": float((r > 0).mean()),
        "ann_turnover": (float(turnover.sum() / n * TRADING_DAYS_PER_YEAR)
                         if turnover is not None and n else float("nan")),
    }


def breakeven_cost_bps(daily: pd.DataFrame, borrow_bps_per_year: float) -> float:
    """Cost per side at which the mean daily return -- and so the Sharpe -- is zero.

    Costs are linear in turnover, so this is solved exactly, not interpolated:
        mean(gross) - mean(turnover)*c*1e-4 - mean(short)*b*1e-4/252 = 0
    """
    gm = float(daily["gross"].mean())
    tm = float(daily["turnover"].mean())
    sm = float(daily["short_notional"].mean())
    if tm <= 0:
        return float("nan")
    return float((gm - sm * float(borrow_bps_per_year) * BPS / TRADING_DAYS_PER_YEAR)
                 / (tm * BPS))


def sharpe_vs_cost(daily: pd.DataFrame, borrow_bps_per_year: float,
                   cost_grid: np.ndarray) -> pd.DataFrame:
    """Net annualised Sharpe as a function of cost per side, on one grid."""
    rows = []
    for c in np.asarray(cost_grid, dtype=float):
        r = net_returns(daily, float(c), borrow_bps_per_year)
        sd = float(r.std(ddof=1))
        rows.append({
            "cost_bps_per_side": float(c),
            "sharpe": float(r.mean()) / sd * math.sqrt(TRADING_DAYS_PER_YEAR) if sd > 0 else np.nan,
            "ann_return": float((1.0 + r).prod() ** (TRADING_DAYS_PER_YEAR / len(r)) - 1.0),
        })
    return pd.DataFrame(rows)


def cost_decomposition(daily: pd.DataFrame, cost_bps_per_side: float,
                       borrow_bps_per_year: float) -> dict:
    """What the sample paid, in units of committed capital, over the whole period."""
    tc = float(transaction_cost(daily, cost_bps_per_side).sum())
    bc = float(borrow_cost(daily, borrow_bps_per_year).sum())
    return {
        "transaction_cost_total": tc,
        "borrow_cost_total": bc,
        "total": tc + bc,
        "ann_turnover": float(daily["turnover"].sum() / len(daily) * TRADING_DAYS_PER_YEAR),
    }


def per_year_table(daily: pd.DataFrame, cost_bps_per_side: float,
                   borrow_bps_per_year: float) -> pd.DataFrame:
    """Calendar-year gross and net returns, recomputed at the chosen cost level."""
    net = net_returns(daily, cost_bps_per_side, borrow_bps_per_year)
    frame = pd.DataFrame({"gross": daily["gross"].values, "net": net.values},
                         index=daily.index)
    by_year = frame.groupby(frame.index.year)
    out = pd.DataFrame({
        "n_days": by_year.size(),
        "gross_return": by_year["gross"].apply(lambda s: float((1.0 + s).prod() - 1.0)),
        "net_return": by_year["net"].apply(lambda s: float((1.0 + s).prod() - 1.0)),
    })
    out.index.name = "year"
    return out.reset_index()


def grid_configurations(grid: pd.DataFrame) -> pd.DataFrame:
    """The distinct (formation, entry_z, max_pairs) cells in the declared grid."""
    cols = ["formation_days", "entry_z", "max_pairs"]
    return grid[cols].drop_duplicates().sort_values(cols).reset_index(drop=True)


def grid_slice(grid: pd.DataFrame, formation_days: int, entry_z: float,
               max_pairs: int) -> pd.DataFrame:
    """One configuration's net Sharpe across the six declared cost levels."""
    m = ((grid["formation_days"] == int(formation_days))
         & (np.isclose(grid["entry_z"], float(entry_z)))
         & (grid["max_pairs"] == int(max_pairs)))
    return grid[m].sort_values("cost_bps_per_side").reset_index(drop=True)


def grid_summary(grid: pd.DataFrame) -> dict:
    """How many of the declared cells clear zero, and what the best one assumes."""
    positive = grid[grid["net_sharpe"] > 0]
    best = grid.loc[grid["net_sharpe"].idxmax()]
    return {
        "n_cells": int(len(grid)),
        "n_positive_net_sharpe": int(len(positive)),
        "best_net_sharpe": float(best["net_sharpe"]),
        "best_cell": {
            "formation_days": int(best["formation_days"]),
            "entry_z": float(best["entry_z"]),
            "max_pairs": int(best["max_pairs"]),
            "cost_bps_per_side": float(best["cost_bps_per_side"]),
        },
    }


def grid_display_table(selected: pd.DataFrame) -> pd.DataFrame:
    """One configuration's grid row, formatted for display. Strings, not floats."""
    return pd.DataFrame({
        "cost (bp/side)": selected["cost_bps_per_side"].map("{:.0f}".format),
        "gross Sharpe": selected["gross_sharpe"].map("{:+.4f}".format),
        "net Sharpe": selected["net_sharpe"].map("{:+.4f}".format),
        "net ann. return": selected["net_ann_return"].map(lambda v: f"{100*v:+.2f}%"),
        "net max drawdown": selected["net_max_drawdown"].map(lambda v: f"{100*v:.1f}%"),
        "ann. turnover": selected["ann_turnover"].map("{:.2f}x".format),
    }).reset_index(drop=True)


def funnel_counts(per_fold: pd.DataFrame, facts: dict) -> pd.DataFrame:
    """The multiplicity funnel: tests run down to pairs that were really cointegrated."""
    rows = [
        ("Engle-Granger tests run",
         float(facts["universe"]["n_engle_granger_tests"]),
         "780 pairs x 37 windows"),
        ("False rejections from chance alone",
         float(facts["universe"]["expected_false_rejections_at_5pct"]),
         "5% of every test"),
        ("Rejections observed at 5%",
         float(per_fold["n_pairs_p_below_alpha"].sum()),
         "1.44x what noise produces"),
        ("Pairs selected and traded",
         float(per_fold["n_selected"].sum()),
         "top 20 per window"),
        ("Surviving Benjamini-Hochberg",
         float(per_fold["n_bh_fdr"].sum()),
         "across all 37 windows"),
        ("Still cointegrated while traded",
         float(facts["persistence"]["n_still_cointegrated_when_traded"]),
         "re-tested in the trading window"),
    ]
    return pd.DataFrame(rows, columns=["stage", "count", "note"])


def exemplar_events(ex: pd.DataFrame) -> dict:
    """Split the exemplar pair's window into the pieces the second panel draws."""
    formation = ex[ex["phase"] == "formation"]
    trading = ex[ex["phase"] == "trading"]
    return {
        "formation": formation,
        "trading": trading,
        "entries": trading[trading["event"] == "entry"],
        "exits": trading[trading["event"] == "exit"],
        "stops": trading[trading["event"] == "stop"],
        "window_end": trading[trading["event"] == "window_end"],
        "held": trading[trading["held_position"] != 0],
    }


def held_spans(ex: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """Contiguous (start, end, direction) runs where a position was actually held."""
    trading = ex[ex["phase"] == "trading"].reset_index(drop=True)
    spans, i, n = [], 0, len(trading)
    while i < n:
        pos = trading.loc[i, "held_position"]
        if pos == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and trading.loc[j + 1, "held_position"] == pos:
            j += 1
        spans.append((trading.loc[i, "date"], trading.loc[j, "date"], float(pos)))
        i = j + 1
    return spans


# ==========================================================================
# PLOTTING -- each takes computed data and returns a Figure
# ==========================================================================
def _pct(x: float, dp: int = 1) -> str:
    return f"{100.0 * x:.{dp}f}%"


def fig_equity(daily: pd.DataFrame, cost_bps: float, borrow_bps: float,
               net_perf: dict, gross_perf: dict) -> plt.Figure:
    with plt.rc_context(PLOT_STYLE):
        gross_curve = equity_curve(daily["gross"])
        net_curve = equity_curve(net_returns(daily, cost_bps, borrow_bps))
        fig, ax = plt.subplots(figsize=(6.4, 3.9), constrained_layout=True)

        ax.axhline(1.0, color=C["ink"], lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.plot(gross_curve.index, gross_curve.values, color=C["gross"], lw=1.5,
                label=f"gross, no costs at all  (Sharpe {gross_perf['sharpe']:+.2f})")
        ax.plot(net_curve.index, net_curve.values, color=C["net"], lw=1.8,
                label=(f"net @ {cost_bps:g} bp/side + {borrow_bps:g} bp/yr borrow"
                       f"  (Sharpe {net_perf['sharpe']:+.2f})"))
        ax.fill_between(net_curve.index, net_curve.values, gross_curve.values,
                        color=C["net"], alpha=0.10, lw=0)

        end_g, end_n = float(gross_curve.iloc[-1]), float(net_curve.iloc[-1])
        ax.annotate(f"{end_g:.3f}", (gross_curve.index[-1], end_g),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=C["gross"])
        ax.annotate(f"{end_n:.3f}", (net_curve.index[-1], end_n),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=C["net"], fontweight="bold")

        ax.set_title(
            f"Under water before a single basis point is paid\n"
            f"1.000 of committed capital becomes {end_g:.3f} gross, {end_n:.3f} net",
            fontsize=10.5)
        ax.set_xlabel(f"out-of-sample date  ({gross_perf['n_days']:,} trading days, "
                      "37 non-overlapping walk-forward windows)")
        ax.set_ylabel("value of 1.00 of committed capital")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.margins(x=0.02)
        ax.legend(loc="lower left")
        return fig


def fig_cost_curve(curve: pd.DataFrame, breakeven: float, cost_bps: float,
                   borrow_bps: float, net_sharpe: float) -> plt.Figure:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.5), constrained_layout=True)

        ax.axhline(0.0, color=C["ink"], lw=1.1, zorder=2)
        ax.axvspan(curve["cost_bps_per_side"].min(), 0.0, color=C["faint"], alpha=0.45, lw=0)
        ax.plot(curve["cost_bps_per_side"], curve["sharpe"], color=C["net"], lw=2.0, zorder=3)

        ax.axvline(breakeven, color=C["bench"], lw=1.2, ls=(0, (4, 3)), zorder=3)
        ax.annotate(f"break-even\n{breakeven:.1f} bp/side",
                    (breakeven, 0.0), xytext=(-7, -52), textcoords="offset points",
                    fontsize=8.5, color=C["bench"], ha="right", va="center",
                    arrowprops=dict(arrowstyle="-", color=C["bench"], lw=0.8))

        ax.plot([cost_bps], [net_sharpe], "o", color=C["net"], ms=7,
                mec="white", mew=1.2, zorder=5)
        ax.annotate(f"you are here: {cost_bps:g} bp -> {net_sharpe:+.3f}",
                    (cost_bps, net_sharpe), xytext=(10, -17),
                    textcoords="offset points", fontsize=8.5, color=C["net"])

        ax.axvline(0.0, color=C["muted"], lw=0.9)
        ax.text(0.0, ax.get_ylim()[0], " trading is free ", rotation=90, va="bottom",
                ha="left", fontsize=8, color=C["muted"])
        ax.set_title(
            f"The zero line sits at {breakeven:.1f} bp: a broker would have to pay you\n"
            f"Net annualised Sharpe against the cost assumption "
            f"(borrow {borrow_bps:g} bp/yr)",
            fontsize=10.5)
        ax.set_xlabel("transaction cost charged per side (basis points of notional traded)")
        ax.set_ylabel("net annualised Sharpe ratio")
        ax.margins(x=0.01)
        return fig


def fig_per_year(per_year: pd.DataFrame, cost_bps: float, borrow_bps: float) -> plt.Figure:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 2.9), constrained_layout=True)
        colours = [C["converge"] if v > 0 else C["net"] for v in per_year["net_return"]]
        ax.bar(per_year["year"], 100.0 * per_year["net_return"], color=colours, width=0.72)
        ax.axhline(0.0, color=C["ink"], lw=1.0)
        n_pos = int((per_year["net_return"] > 0).sum())
        ax.set_title(
            f"{n_pos} of {len(per_year)} calendar years are positive net "
            f"at {cost_bps:g} bp/side + {borrow_bps:g} bp/yr\n"
            "No regime rescues it",
            fontsize=10.5)
        ax.set_xlabel("calendar year")
        ax.set_ylabel("net return (%)")
        ax.set_xticks(per_year["year"][::2])
        ax.tick_params(axis="x", rotation=0)
        return fig


def fig_exemplar(ex: pd.DataFrame, meta: dict, entry_z: float = 2.0,
                 exit_z: float = 0.5, stop_z: float = 3.5) -> plt.Figure:
    parts = exemplar_events(ex)
    spans = held_spans(ex)
    with plt.rc_context(PLOT_STYLE):
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(6.4, 5.4), sharex=True, constrained_layout=True,
            gridspec_kw={"height_ratios": [1.0, 1.15]})

        form_end = parts["formation"]["date"].iloc[-1]
        mu, sigma = meta["mu"], meta["sigma"]

        # --- top: the spread itself, in log units -------------------------
        for ax in (ax0, ax1):
            ax.axvspan(ex["date"].iloc[0], form_end, color=C["faint"], alpha=0.40, lw=0)
            for s, e, d in spans:
                ax.axvspan(s, e, color=(C["converge"] if d > 0 else C["bench"]),
                           alpha=0.11, lw=0)

        ax0.axhline(mu, color=C["muted"], lw=1.0)
        for k, ls in ((entry_z, (0, (4, 3))), (-entry_z, (0, (4, 3)))):
            ax0.axhline(mu + k * sigma, color=C["bench"], lw=0.9, ls=ls)
        ax0.plot(ex["date"], ex["spread"], color=C["ink"], lw=1.2)
        ax0.set_ylabel("spread (log units)")
        ax0.set_title(
            f"{meta['pair']}: genuinely cointegrated, Engle-Granger "
            f"p = {meta['eg_pvalue']:.5f}\n"
            f"Spread = log {meta['y']} - {meta['beta']:.3f} log {meta['x']}, "
            "moments frozen on the shaded window",
            fontsize=10.5)
        ax0.annotate(f"+{entry_z:g}σ", (ex["date"].iloc[0], mu + entry_z * sigma),
                     xytext=(3, 3), textcoords="offset points", fontsize=8, color=C["bench"])
        ax0.annotate(f"-{entry_z:g}σ", (ex["date"].iloc[0], mu - entry_z * sigma),
                     xytext=(3, -11), textcoords="offset points", fontsize=8, color=C["bench"])

        # --- bottom: the z-score and the rule -----------------------------
        ax1.axhline(0.0, color=C["muted"], lw=1.0)
        for k in (entry_z, -entry_z):
            ax1.axhline(k, color=C["bench"], lw=1.0, ls=(0, (4, 3)))
        for k in (exit_z, -exit_z):
            ax1.axhline(k, color=C["converge"], lw=0.9, ls=(0, (1, 2)))
        for k in (stop_z, -stop_z):
            ax1.axhline(k, color=C["stop"], lw=0.9, ls=(0, (6, 3)))
        ax1.plot(ex["date"], ex["z"], color=C["ink"], lw=1.2)

        ax1.plot(parts["entries"]["date"], parts["entries"]["z"], "^",
                 color=C["entry"], ms=8, mec="white", mew=1.0, zorder=5)
        ax1.plot(parts["exits"]["date"], parts["exits"]["z"], "o",
                 color=C["converge"], ms=7, mec="white", mew=1.0, zorder=5)
        ax1.plot(parts["stops"]["date"], parts["stops"]["z"], "X",
                 color=C["stop"], ms=9, mec="white", mew=1.0, zorder=5)
        ax1.plot(parts["window_end"]["date"], parts["window_end"]["z"], "s",
                 color=C["muted"], ms=6, mec="white", mew=1.0, zorder=5)

        ax1.set_yticks([-stop_z, -entry_z, 0.0, entry_z, stop_z])
        ax1.set_yticklabels([f"-{stop_z:g}  stop", f"-{entry_z:g}  entry", "0",
                             f"{entry_z:g}  entry", f"{stop_z:g}  stop"])

        n_stop = len(parts["stops"])
        n_exit = len(parts["exits"])
        ax1.set_title(
            f"Out of sample: {meta['spread_std_ratio']:.2f}x as wide, half-life "
            f"{meta['half_life_oos']:.0f}d not {meta['half_life_formation']:.1f}d\n"
            f"{n_exit} converged, {n_stop} stopped out -- and this one made money",
            fontsize=10.5)
        ax1.set_ylabel("z-score (formation sigma)")
        ax1.set_xlabel("date")
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

        handles = [
            Line2D([], [], marker="^", color=C["entry"], ls="", ms=8, mec="white",
                   label=f"entry |z| >= {entry_z:g}"),
            Line2D([], [], marker="o", color=C["converge"], ls="", ms=7, mec="white",
                   label=f"exit |z| <= {exit_z:g}"),
            Line2D([], [], marker="X", color=C["stop"], ls="", ms=9, mec="white",
                   label=f"stop |z| >= {stop_z:g}"),
        ]
        ax1.legend(handles=handles, loc="lower center", ncol=3, columnspacing=1.4,
                   frameon=True, framealpha=0.9, edgecolor="none", facecolor="white")
        ax1.margins(x=0.01)
        return fig


def fig_funnel(funnel: pd.DataFrame) -> plt.Figure:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
        y = np.arange(len(funnel))[::-1]
        colours = [C["muted"], C["faint"], C["bench"], C["net"], C["gross"], C["converge"]]
        ax.barh(y, funnel["count"], color=colours[:len(funnel)], height=0.66)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{s}\n({n})" for s, n in zip(funnel["stage"], funnel["note"])],
                           fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(20, float(funnel["count"].max()) * 4.0)
        for yi, n in zip(y, funnel["count"]):
            ax.annotate(f"  {n:,.0f}", (n, yi), va="center", fontsize=9,
                        color=C["ink"], fontweight="bold")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("number of pairs (log scale)")
        ax.set_title(
            f"{funnel.loc[0, 'count']:,.0f} tests, {funnel.loc[2, 'count']:,.0f} "
            f"rejections; chance alone gives {funnel.loc[1, 'count']:,.0f}\n"
            "What survives each stage, pooled over all 37 windows",
            fontsize=10.5)
        return fig


def fig_grid_fan(grid: pd.DataFrame, selected: pd.DataFrame, label: str,
                 cost_bps: float) -> plt.Figure:
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
        cols = ["formation_days", "entry_z", "max_pairs"]
        for _, cfg in grid[cols].drop_duplicates().iterrows():
            sl = grid_slice(grid, cfg["formation_days"], cfg["entry_z"], cfg["max_pairs"])
            ax.plot(sl["cost_bps_per_side"], sl["net_sharpe"],
                    color=C["faint"], lw=0.9, zorder=1)
        ax.axhline(0.0, color=C["ink"], lw=1.1, zorder=2)
        ax.plot(selected["cost_bps_per_side"], selected["net_sharpe"],
                color=C["net"], lw=2.2, marker="o", ms=4.5, zorder=4, label=label)
        default = grid[grid["is_default"]]
        if len(default):
            ax.plot(default["cost_bps_per_side"], default["net_sharpe"], "*",
                    color=C["ink"], ms=13, zorder=5,
                    label="declared default cell (the headline)")
        if cost_bps <= float(grid["cost_bps_per_side"].max()):
            ax.axvline(cost_bps, color=C["muted"], lw=0.9, ls=(0, (2, 3)), zorder=3)

        n_pos = int((grid["net_sharpe"] > 0).sum())
        best = float(grid["net_sharpe"].max())
        ax.set_title(
            f"Only {n_pos} of {len(grid)} declared configurations clears zero, "
            f"at {best:+.3f}\n"
            "and it is the cell that assumes trading costs nothing",
            fontsize=10.5)
        ax.set_xlabel("transaction cost charged per side (basis points)")
        ax.set_ylabel("net annualised Sharpe ratio")
        ax.legend(loc="lower left")
        return fig


# ==========================================================================
# ---- streamlit ----  widgets in, functions called, results rendered
# ==========================================================================
def main() -> None:  # pragma: no cover - rendering only
    import streamlit as st

    st.set_page_config(page_title="A strategy that dies at 6 basis points",
                       layout="centered")

    d = data_dir()
    daily = st.cache_data(load_daily)(d)
    grid = st.cache_data(load_grid)(d)
    per_fold = st.cache_data(load_per_fold)(d)
    pairs = st.cache_data(load_selected_pairs)(d)
    ex = st.cache_data(load_exemplar)(d)
    facts = st.cache_data(load_facts)(d)

    # ---------------------------------------------------------------- intro
    st.title("A strategy that dies at 6 basis points")
    st.markdown(
        "Two funds track roughly the same thing, their prices wander apart, and if the "
        "gap is stationary then a wide gap is a forecast. I wanted the narrow version of "
        "that question, because the broad one is unanswerable with free daily data: "
        "**does a formation-window cointegration test carry any information about the six "
        "months that follow?** I tested "
        f"{facts['universe']['n_pairs_per_window']} pairs of "
        f"{facts['universe']['n_tickers']} liquid US ETFs in each of "
        f"{facts['universe']['n_folds']} non-overlapping walk-forward windows, traded the "
        "twenty that passed, and measured the "
        f"{facts['oos']['n_days']:,} out-of-sample days that followed "
        f"({facts['oos']['start']} to {facts['oos']['end']}). "
        "The controls below re-price that single backtest at any cost assumption you like.\n\n"
        "*The full analysis — code, tests and the executed notebook — lives in a private "
        "repository. This page ships only derived series: no price data is redistributed.*"
    )

    # ------------------------------------------------------------- controls
    with st.container(border=True):
        st.markdown("**What does it cost you to trade this?**")
        c1, c2 = st.columns(2)
        cost_bps = c1.slider(
            "Transaction cost per side (basis points of notional traded)",
            min_value=0.0, max_value=25.0, value=DEFAULT_COST_BPS, step=0.5,
            help="Half-spread plus commission plus slippage, charged on the absolute "
                 "change in each leg's notional every time the position moves. The "
                 "study's declared assumption is 5 bps.")
        borrow_bps = c2.slider(
            "Short borrow (basis points per year)",
            min_value=0.0, max_value=300.0, value=DEFAULT_BORROW_BPS, step=5.0,
            help="Accrued daily on whichever leg is actually short while a position "
                 "is open. The study's declared assumption is 50 bps/yr, which is a "
                 "general-collateral figure for large ETFs.")

    # -------------------------------------------------------------- compute
    net = net_returns(daily, cost_bps, borrow_bps)
    net_perf = performance(net, daily["turnover"])
    gross_perf = performance(daily["gross"], daily["turnover"])
    breakeven = breakeven_cost_bps(daily, borrow_bps)
    costs = cost_decomposition(daily, cost_bps, borrow_bps)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net Sharpe", f"{net_perf['sharpe']:+.4f}",
              f"{net_perf['sharpe'] - gross_perf['sharpe']:+.4f} vs gross",
              delta_color="inverse")
    m2.metric("Net annualised return", _pct(net_perf["ann_return"], 2),
              f"{_pct(net_perf['ann_return'] - gross_perf['ann_return'], 2)} vs gross",
              delta_color="inverse")
    m3.metric("Net max drawdown", _pct(net_perf["max_drawdown"]),
              f"{_pct(net_perf['max_drawdown'] - gross_perf['max_drawdown'])} vs gross",
              delta_color="inverse")
    m4.metric("Break-even cost", f"{breakeven:.1f} bp",
              "per side, to reach Sharpe 0", delta_color="off")

    st.caption(
        f"Gross, with no costs charged at all: Sharpe {gross_perf['sharpe']:+.4f}, "
        f"{_pct(gross_perf['ann_return'], 2)} a year, "
        f"{_pct(gross_perf['max_drawdown'])} max drawdown. "
        f"Newey-West *t* on the net series is {net_perf['nw_tstat']:.2f} "
        f"(*p* = {net_perf['nw_pvalue']:.3f}) against {gross_perf['nw_tstat']:.2f} "
        f"(*p* = {gross_perf['nw_pvalue']:.3f}) gross. "
        f"95% Sharpe interval, net: [{net_perf['sharpe_ci95'][0]:.2f}, "
        f"{net_perf['sharpe_ci95'][1]:.2f}]."
    )

    # -------------------------------------------------------- equity curve
    st.pyplot(fig_equity(daily, cost_bps, borrow_bps, net_perf, gross_perf))
    st.markdown(
        f"The grey line is the point. **The book loses money before costs.** A strategy "
        f"that is positive gross and negative net has a cost problem; this one has no "
        f"gross edge to lose. What costs do is move the loss from ambiguous to reliable — "
        f"gross I cannot distinguish it from zero (*t* = {gross_perf['nw_tstat']:.2f}), "
        f"at the declared 5 bp I can (*t* = {facts['headline_net']['nw_tstat']:.2f}). "
        f"At {cost_bps:g} bp/side and {borrow_bps:g} bp/yr the sample pays "
        f"{_pct(costs['transaction_cost_total'], 2)} of committed capital in transaction "
        f"costs and {_pct(costs['borrow_cost_total'], 2)} in borrow, on annualised turnover "
        f"of {costs['ann_turnover']:.2f}x."
    )
    st.caption(
        "Twenty pairs drawn at random each fold — no test, no filter, identical rule and "
        f"costs — finish at net Sharpe {facts['random_control_net']['sharpe']:+.4f} against "
        f"{facts['headline_net']['sharpe']:+.4f} for the cointegration-selected book, at "
        "the study's declared 5 bp/side. Three thousandths apart, with a standard error "
        "of 0.23 on each. Selecting on cointegration bought nothing."
    )

    # --------------------------------------------------------- cost curve
    st.subheader("Where the zero line is")
    curve = sharpe_vs_cost(daily, borrow_bps,
                           np.round(np.arange(-20.0, 25.01, 0.5), 2))
    st.pyplot(fig_cost_curve(curve, breakeven, cost_bps, borrow_bps, net_perf["sharpe"]))
    st.markdown(
        f"Costs are linear in turnover, so the break-even is solved rather than "
        f"interpolated: **{breakeven:.1f} basis points per side** at "
        f"{borrow_bps:g} bp/yr borrow. It is negative, which is the whole finding — "
        "there is no cost assumption charitable enough to make this work, only one "
        "generous enough to make the loss small. The shaded region left of zero is "
        "the part of the axis a real broker does not sell."
    )

    # ------------------------------------------------------------- by year
    per_year = per_year_table(daily, cost_bps, borrow_bps)
    st.pyplot(fig_per_year(per_year, cost_bps, borrow_bps))

    # ---------------------------------------------------------- exemplar
    st.subheader("One pair, followed all the way through")
    exm = facts["exemplar"]
    meta = {
        "pair": exm["pair"], "y": exm["pair"].split("/")[0], "x": exm["pair"].split("/")[1],
        "beta": exm["beta"], "mu": exm["mu"], "sigma": exm["sigma"],
        "eg_pvalue": exm["eg_pvalue"],
        "half_life_formation": exm["half_life_formation"],
        "half_life_oos": exm["half_life_oos"],
        "spread_std_ratio": float(
            pairs.loc[(pairs["pair"] == exm["pair"]) & (pairs["fold"] == exm["fold"]),
                      "oos_spread_std_vs_formation"].iloc[0]),
        "share_positive_pct": 100.0 * facts["mechanism"]["share_pairs_with_positive_oos_gross"],
    }
    st.markdown(
        f"**{exm['pair']}** — financials against consumer discretionary, rank "
        f"{exm['selection_rank']} in its formation window, chosen on economic sense "
        "before I looked at what it did out of sample. It is one of 740 selections and "
        "proves nothing on its own; the mechanics are simply much easier to see on one "
        "pair than on a book of twenty."
    )
    st.pyplot(fig_exemplar(ex, meta))
    st.markdown(
        "This is what the aggregate is made of. Ranking 780 pairs by *p*-value sorts "
        "them by **speed** — the Engle-Granger statistic gets large when the residual "
        "reverts fast, and the fastest residuals in a daily ETF panel are bid-ask "
        f"bounce, not economics. Median in-sample half-life across all selections is "
        f"{facts['mechanism']['insample_half_life_median']:.1f} days; out of sample it is "
        f"{facts['mechanism']['oos_half_life_median']:.1f}, and the realised spread "
        f"standard deviation averages "
        f"**{facts['mechanism']['mean_oos_spread_std_vs_formation']:.2f}x** the sigma "
        "frozen at formation. So “two sigma, time to enter” is really about "
        "1.1 sigma — an ordinary day. Across the book, "
        f"**{facts['trades']['exit_reason_counts']['stop']} trades stopped out against "
        f"{facts['trades']['exit_reason_counts']['exit']} converged**. The payoffs are "
        f"near symmetric ("
        f"{_pct(facts['trades']['mean_net_pnl_by_exit_reason']['exit'], 2)} against "
        f"{_pct(facts['trades']['mean_net_pnl_by_exit_reason']['stop'], 2)}), so 1.7 "
        "stops per convergence is the whole arithmetic of the loss."
    )

    # ------------------------------------------------------------- the grid
    st.subheader("Was it just the parameters?")
    st.markdown(
        "Every threshold was declared in one file before the out-of-sample results were "
        f"read, and all {facts['universe']['n_configurations_declared']} alternatives get "
        "run and reported. The headline is always the default cell, never the best one. "
        "The panel above re-prices the default rule at any cost; this one re-runs the "
        "whole study under other rules, at the six cost levels the study declared."
    )
    g1, g2, g3 = st.columns(3)
    formation_days = g1.selectbox("Formation window (trading days)",
                                  sorted(grid["formation_days"].unique()), index=1)
    entry_z = g2.selectbox("Entry threshold |z|",
                           sorted(grid["entry_z"].unique()), index=1)
    max_pairs = g3.selectbox("Pairs traded per window",
                             sorted(grid["max_pairs"].unique()), index=1)
    sel = grid_slice(grid, formation_days, entry_z, max_pairs)
    label = f"{formation_days}d formation, entry |z| >= {entry_z:g}, top {max_pairs}"
    st.pyplot(fig_grid_fan(grid, sel, label, cost_bps))

    summary = grid_summary(grid)
    st.dataframe(grid_display_table(sel), hide_index=True, use_container_width=True)
    st.caption(
        f"Exactly {summary['n_positive_net_sharpe']} of {summary['n_cells']} declared "
        f"cells finishes above zero, at {summary['best_net_sharpe']:+.3f} — a "
        f"{summary['best_cell']['formation_days']}-day formation, entry "
        f"|z| >= {summary['best_cell']['entry_z']:g}, top "
        f"{summary['best_cell']['max_pairs']}, and "
        f"{summary['best_cell']['cost_bps_per_side']:.0f} bps of transaction cost. "
        "That cell assumes trading is free."
    )

    # ----------------------------------------------------------- the funnel
    st.subheader("What the cointegration test was actually selecting")
    st.pyplot(fig_funnel(funnel_counts(per_fold, facts)))
    st.markdown(
        "A pair drawn at random is cointegrated in a given six-month window "
        f"{facts['persistence']['base_rate']:.2%} of the time. A pair that just rejected "
        f"on formation data: {facts['persistence']['rate_given_formation_significant']:.2%}. "
        "Selecting on formation significance is worth "
        f"**{100 * facts['persistence']['lift_over_base']:+.2f} percentage points** — it "
        "is not a weak signal, it points the wrong way. Rank correlation between a pair's "
        "formation *p*-value and its out-of-sample P&L is "
        f"{facts['mechanism']['pvalue_vs_oos_pnl_spearman']:+.3f} across "
        f"{len(pairs):,} selections, and only "
        f"{facts['mechanism']['share_pairs_with_positive_oos_gross']:.1%} of them made "
        "money gross.\n\n"
        "Corrected for multiplicity the strategy barely exists: Benjamini-Hochberg leaves "
        f"the book empty in {facts['bh_fdr_control']['n_folds_with_zero_selected']} of "
        f"{facts['universe']['n_folds']} windows and places "
        f"{facts['bh_fdr_control']['n_trades']} trades in 18.5 years. The uncorrected run "
        "is not a more powerful strategy — it is the same evidence with the standard of "
        "proof lowered until there is enough to trade on."
    )

    # ------------------------------------------------------------ the fine print
    with st.expander("Method, and what these numbers are not"):
        st.markdown(
            f"""
**Design.** 12-month formation / 6-month trading, rolled forward without overlap,
top 20 pairs — the design in Gatev, Goetzmann & Rouwenhorst (2006). Beta, alpha,
mu, sigma and the half-life are fitted on
{facts['config']['formation_days']} days and frozen; the next
{facts['config']['trading_days']} are scored with them and never touched.
Signals lag execution by {facts['config']['signal_lag_days']} day: z comes from
day-*t* closes, the position is held from *t+1*.

**Price basis.** Total-return index, never closing prices. Over this period
113.8% of TLT's total return was income and 34.6% of SPY's; a mean-reversion
rule on closes reads every ex-dividend step as a spread that widened.

**Sharpe convention.** A dollar-neutral long/short book is self-financing to
first order, so the spread P&L is already an excess return and no risk-free rate
is subtracted again. Confidence intervals use the Lo (2002) i.i.d. standard
error; significance should be read off the Newey-West *t*, which truncates at
the usual 4(T/100)^(2/9) lags.

**These are upper bounds, not estimates.** Market impact that scales with size,
borrow going special, failure to locate, and bid-ask crossing that widens in
stressed markets are all unmodelled, and all push the same way. Real costs are
higher than anything the slider above will charge you.

**Other limitations.** Survivorship bias is present and unquantified — the
universe is ETFs that still exist. The book is mostly idle: at least one pair is
on for {facts['oos']['share_of_days_with_any_position']:.1%} of days but the
average is {facts['oos']['mean_pairs_in_position']:.1f} of 20 committed slots, so
annualised volatility is under 2% and these Sharpe ratios are a small number over
a small number. Twenty pairs is not twenty bets — mean within-fold correlation
across pair P&Ls implies about 8.7 effective independent bets.

**Not a trading system.** Research exercise. No claim is made that this strategy
is profitable, deployable or production-ready, and nothing here is investment
advice.

**Data.** Prices from Yahoo Finance via `yfinance` (personal use, not
redistributable — none is shipped with this page); risk-free rate from FRED
DTB3, public domain. This page ships only my own derived series: the daily
out-of-sample return decomposition, per-fold and per-pair summary statistics, the
declared sensitivity grid, and one pair's fitted spread. Study executed
{facts['generated_at_utc'][:10]}.
"""
        )

    st.caption(
        f"Benchmarked over the same {facts['oos']['n_days']:,} days: SPY buy-and-hold "
        f"returned {_pct(facts['benchmarks']['SPY buy-and-hold']['ann_return'], 2)} a year "
        f"at Sharpe {facts['benchmarks']['SPY buy-and-hold']['sharpe']:+.2f} "
        f"(max drawdown {_pct(facts['benchmarks']['SPY buy-and-hold']['max_drawdown'])})."
    )


if __name__ == "__main__":
    main()
