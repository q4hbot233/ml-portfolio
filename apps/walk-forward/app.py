"""The backtest that only works if trading is free.

An interactive walk-forward explorer over the derived monthly series of a study
of 49-industry return prediction, 1963-2026.

Structure: every number on this page is produced by a plain function in the
COMPUTE section, which takes arguments and returns DataFrames / arrays / dicts
and contains no Streamlit call. The CHARTS section turns those into matplotlib
figures. The APP section at the bottom reads widgets, calls those functions and
renders. Nothing is hard-coded: the shipped CSVs are the derived output of the
private repo's executed backtest.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:                                     # notebook / REPL
    BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"

MONTHS_PER_YEAR = 12
PRIMARY = "rf-expanding-refit12m-top10"
DECIMALISATION = "2001-01"

INK = "#1c1c1c"
GRID = "#d9d9d9"
STRAT = "#1b4f72"        # the ML book
STRAT_FILL = "#aec9dd"
MARKET = "#8a8a8a"
MOMENTUM = "#c8722a"
ACCENT = "#a63603"       # the cost line / the bad news
TRAIN = "#dfe6ec"        # schedule: what the model could see
TEST = "#1b4f72"         # schedule: what it had to predict blind


# ==========================================================================
# COMPUTE  --  plain functions, no Streamlit, no globals except constants
# ==========================================================================

def load_bundle(data_dir: Path | str = DATA_DIR) -> dict:
    """Read every shipped artifact. Returns a dict of DataFrames plus results.json."""
    d = Path(data_dir)

    def _series(name: str) -> pd.DataFrame:
        df = pd.read_csv(d / name)
        df.index = pd.PeriodIndex(df["month"].astype(str), freq="M")
        return df.drop(columns=["month"])

    strat = _series("strategy_gross_excess.csv")
    return {
        "gross": strat,
        "turnover": _series("strategy_turnover.csv"),
        "bench_gross": _series("benchmark_gross_excess.csv"),
        "bench_turnover": _series("benchmark_turnover.csv"),
        "ic": _series("rank_ic.csv"),
        "schedule": pd.read_csv(d / "schedule.csv"),
        "results": json.loads((d / "results.json").read_text()),
        "configs": list(strat.columns),
    }


def parse_config(name: str) -> dict:
    """`rf-expanding-refit12m-top10` -> its four pre-registered axes."""
    model, window, refit, legs = name.split("-")
    return {
        "model": model,
        "window": window,
        "refit_every": int(refit.replace("refit", "").replace("m", "")),
        "n_legs": int(legs.replace("top", "")),
    }


def ic_column(config: str) -> str:
    """The rank-IC series belongs to the model/schedule, not to the leg count."""
    p = parse_config(config)
    return f"{p['model']}-{p['window']}-refit{p['refit_every']}m"


def apply_costs(gross: pd.Series, turnover: pd.Series, cost_bps: float) -> pd.Series:
    """Net monthly excess return.

    `turnover` is one-way (0.5 * sum|dw|), so a one-way cost is paid on both the
    sells and the buys: the charge is 2 * turnover * cost.
    """
    t = turnover.reindex(gross.index).fillna(0.0)
    return gross - 2.0 * t * (cost_bps / 1e4)


def sharpe(r: pd.Series) -> float:
    """Annualised Sharpe of an already-excess monthly series."""
    r = r.dropna()
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(MONTHS_PER_YEAR))


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    return float((1.0 + r).prod() ** (MONTHS_PER_YEAR / len(r)) - 1.0)


def annualised_vol(r: pd.Series) -> float:
    return float(r.dropna().std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def equity_curve(r: pd.Series) -> pd.Series:
    """Growth of 1 unit, financed at the realised bill rate."""
    return (1.0 + r.dropna()).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    eq = equity_curve(r)
    return eq / eq.cummax() - 1.0


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min())


def hit_rate(r: pd.Series) -> float:
    r = r.dropna()
    return float((r > 0).mean()) if len(r) else float("nan")


def breakeven_cost_bps(gross: pd.Series, turnover: pd.Series) -> float:
    """One-way cost in bp at which the mean excess return -- and so the Sharpe -- hits zero."""
    mean_turn = turnover.reindex(gross.index).fillna(0.0).mean()
    if mean_turn <= 0:
        return float("inf")
    return float(gross.mean() / (2.0 * mean_turn) * 1e4)


def summarise(gross: pd.Series, turnover: pd.Series, cost_bps: float) -> dict:
    """The metric block shown at the top of the page, at one cost assumption."""
    net = apply_costs(gross, turnover, cost_bps)
    return {
        "months": int(len(net.dropna())),
        "sharpe": sharpe(net),
        "cagr": cagr(net),
        "ann_vol": annualised_vol(net),
        "max_drawdown": max_drawdown(net),
        "hit_rate": hit_rate(net),
        "ann_turnover": float(turnover.mean() * MONTHS_PER_YEAR),
        "breakeven_cost_bps": breakeven_cost_bps(gross, turnover),
        "cum_return": float(equity_curve(net).iloc[-1] - 1.0),
        "cost_bps": float(cost_bps),
    }


def sharpe_vs_cost(gross: pd.Series, turnover: pd.Series,
                   costs: np.ndarray | None = None) -> pd.DataFrame:
    """Annualised net Sharpe across a grid of one-way cost assumptions."""
    if costs is None:
        costs = np.arange(0.0, 60.5, 1.0)
    rows = [{"cost_bps": float(c), "sharpe": sharpe(apply_costs(gross, turnover, c))}
            for c in costs]
    return pd.DataFrame(rows)


def crossover_cost(curve: pd.DataFrame, other: pd.DataFrame) -> float | None:
    """Cost at which `other`'s net Sharpe first overtakes `curve`'s, if it does."""
    d = curve["sharpe"].to_numpy() - other["sharpe"].to_numpy()
    c = curve["cost_bps"].to_numpy()
    sign = np.sign(d)
    idx = np.flatnonzero((sign[:-1] > 0) & (sign[1:] <= 0))
    if len(idx) == 0:
        return None
    i = int(idx[0])
    span = d[i] - d[i + 1]
    if span == 0:
        return float(c[i])
    return float(c[i] + (c[i + 1] - c[i]) * d[i] / span)


def grid_table(bundle: dict, cost_bps: float) -> pd.DataFrame:
    """All sixteen pre-registered configurations at one cost, sorted by net Sharpe."""
    rows = []
    for name in bundle["configs"]:
        g, t = bundle["gross"][name], bundle["turnover"][name]
        s = summarise(g, t, cost_bps)
        rows.append({
            "configuration": name,
            "gross Sharpe": round(sharpe(g), 2),
            f"net Sharpe @{cost_bps:.0f}bp": round(s["sharpe"], 2),
            "net CAGR": f"{s['cagr'] * 100:.2f}%",
            "max DD": f"{s['max_drawdown'] * 100:.1f}%",
            "turnover/yr": f"{s['ann_turnover']:.2f}x",
            "break-even bp": f"{s['breakeven_cost_bps']:,.0f}",
        })
    return (pd.DataFrame(rows)
            .sort_values(f"net Sharpe @{cost_bps:.0f}bp", ascending=False)
            .reset_index(drop=True))


def newey_west_tstat(x: pd.Series, lags: int = 6) -> float:
    """t-statistic for mean(x) != 0 with a Newey-West HAC standard error."""
    x = x.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < lags + 2:
        return float("nan")
    e = x - x.mean()
    var = float(e @ e / n)
    for L in range(1, lags + 1):
        var += 2.0 * (1.0 - L / (lags + 1.0)) * float(e[L:] @ e[:-L] / n)
    if var <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(var / n))


def ic_summary(ic: pd.Series, window: int = 36) -> dict:
    """Rank-IC statistics plus the rolling mean drawn on the chart."""
    ic = ic.dropna()
    return {
        "mean": float(ic.mean()),
        "sd": float(ic.std(ddof=1)),
        "ir": float(ic.mean() / ic.std(ddof=1)),
        "t_nw6": newey_west_tstat(ic, 6),
        "share_positive": float((ic > 0).mean()),
        "rolling": ic.rolling(window, min_periods=window).mean(),
        "window": window,
    }


def schedule_for(schedule: pd.DataFrame, config: str) -> pd.DataFrame:
    """The refit schedule of one configuration, with bounds as timestamps."""
    p = parse_config(config)
    s = schedule[(schedule["window"] == p["window"])
                 & (schedule["refit_every"] == p["refit_every"])].copy()
    for col in ("train_start", "train_end", "test_start", "test_end"):
        s[col + "_ts"] = pd.PeriodIndex(s[col].astype(str), freq="M").to_timestamp(how="start")
    # a bar should cover the whole final month, not stop at its first day
    for col in ("train_end", "test_end"):
        s[col + "_ts"] = pd.PeriodIndex(s[col].astype(str), freq="M").to_timestamp(how="end")
    return s.sort_values("fold").reset_index(drop=True)


def to_timestamps(idx) -> pd.DatetimeIndex:
    """Month labels -> timestamps, whatever form the index arrives in.

    Rebuilding a PeriodIndex from one that already carries period dtype raises
    NotImplementedError on some pandas builds, so convert only when needed.
    """
    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp(how="start")
    if isinstance(idx, pd.DatetimeIndex):
        return idx
    return pd.PeriodIndex(pd.Index(idx).astype(str), freq="M").to_timestamp(how="start")


# ==========================================================================
# CHARTS  --  take data, return a figure; no Streamlit, no data access
# ==========================================================================

def _style(ax, *, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=10, fontweight="semibold")
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a9a9a")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=9, colors=INK, length=3)


def fig_equity(series: dict[str, pd.Series], *, label: str, cost_bps: float,
               show_net: bool, months: int) -> mpl.figure.Figure:
    """Compounded excess return of the book against its benchmarks, log scale,
    with the strategy's drawdown beneath it."""
    strat = series["strategy"]
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 5.6), dpi=130, sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1.0], "hspace": 0.12},
    )

    order = [("strategy", STRAT, 2.0, "-"),
             ("market", MARKET, 1.3, "-"),
             ("12-2 momentum", MOMENTUM, 1.3, "--")]
    lo, hi = 1.0, 1.0
    for key, colour, lw, ls in order:
        if key not in series:
            continue
        eq = equity_curve(series[key])
        lo, hi = min(lo, float(eq.min())), max(hi, float(eq.max()))
        ax.plot(to_timestamps(eq.index), eq.to_numpy(), color=colour, linewidth=lw,
                linestyle=ls, label=f"{key} ({sharpe(series[key]):.2f})", zorder=3)

    ax.set_yscale("log")
    ticks = [t for t in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)
             if lo * 0.85 <= t <= hi * 1.2]
    ax.set_yticks(ticks)
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:,.0f}x" if v >= 1 else f"{v:g}x"))
    ax.axhline(1.0, color="#9a9a9a", linewidth=0.8, zorder=1)
    dec = pd.Timestamp(DECIMALISATION + "-01")
    ax.axvline(dec, color=ACCENT, linewidth=0.9, linestyle=":", zorder=2)
    ax.text(dec, lo * 0.9, " decimalisation, 2001", color=ACCENT, fontsize=8,
            va="bottom", ha="left")

    basis = "Gross of costs" if not show_net else f"Net of {cost_bps:.0f} bp one-way"
    end = float(equity_curve(strat).iloc[-1])
    ss, vs, ds = sharpe(strat), annualised_vol(strat) * 100, max_drawdown(strat) * 100
    if "market" in series:
        m = series["market"]
        line1 = f"{basis}: Sharpe {ss:.2f} against the market's {sharpe(m):.2f}"
        line2 = (f"{end:.2f}x in dollars against {float(equity_curve(m).iloc[-1]):.1f}x — "
                 f"at {vs:.1f}% vol against {annualised_vol(m) * 100:.1f}%")
    else:
        line1 = f"{basis}: Sharpe {ss:.2f} over {months} months"
        line2 = f"{end:.2f}x in dollars at {vs:.1f}% annualised volatility"
    _style(ax, title=f"{line1}\n{line2}", xlabel="",
           ylabel="growth of 1 unit, excess return (log)")
    ax.title.set_fontsize(10.0)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9,
                    title="annualised Sharpe", title_fontsize=8.5)
    leg._legend_box.align = "left"

    dd = drawdown(strat)
    ax2.fill_between(to_timestamps(dd.index), dd.to_numpy() * 100, 0.0,
                     color=STRAT_FILL, zorder=3)
    ax2.plot(to_timestamps(dd.index), dd.to_numpy() * 100, color=STRAT, linewidth=0.8, zorder=4)
    ax2.axvline(dec, color=ACCENT, linewidth=0.9, linestyle=":", zorder=2)
    _style(ax2, title="", xlabel="formation month", ylabel="drawdown (%)")
    ax2.set_ylim(min(-1.0, dd.min() * 105), 1.0)

    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.10, hspace=0.12)
    return fig


def fig_cost_curve(curve: pd.DataFrame, *, cost_bps: float, breakeven: float,
                   label: str, momentum: pd.DataFrame | None = None,
                   momentum_breakeven: float | None = None) -> mpl.figure.Figure:
    """Net Sharpe as a function of the assumed one-way cost."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=130)

    ax.axhline(0.0, color="#9a9a9a", linewidth=0.9, zorder=2)
    ax.plot(curve["cost_bps"], curve["sharpe"], color=STRAT, linewidth=2.1,
            label=label, zorder=5)
    if momentum is not None:
        ax.plot(momentum["cost_bps"], momentum["sharpe"], color=MOMENTUM, linewidth=1.5,
                linestyle="--", label="naive 12-2 momentum, top 10", zorder=4)

    here = float(np.interp(cost_bps, curve["cost_bps"], curve["sharpe"]))
    ax.axvline(cost_bps, color=ACCENT, linewidth=1.1, zorder=3)
    ax.plot([cost_bps], [here], "o", color=ACCENT, markersize=7, zorder=6)
    ax.annotate(f"{cost_bps:.0f} bp\nSharpe {here:.2f}",
                xy=(cost_bps, here), xytext=(6, 14), textcoords="offset points",
                fontsize=9, color=ACCENT, fontweight="semibold")

    if np.isfinite(breakeven) and breakeven <= curve["cost_bps"].max():
        ax.plot([breakeven], [0.0], "o", color=STRAT, markersize=6,
                markerfacecolor="white", markeredgewidth=1.6, zorder=6)
        ax.annotate(f"break-even {breakeven:.1f} bp",
                    xy=(breakeven, 0.0), xytext=(-4, -22), textcoords="offset points",
                    fontsize=9, color=STRAT, ha="right", fontweight="semibold")
    if momentum is not None and momentum_breakeven is not None \
            and np.isfinite(momentum_breakeven) and momentum_breakeven <= curve["cost_bps"].max():
        ax.plot([momentum_breakeven], [0.0], "o", color=MOMENTUM, markersize=5,
                markerfacecolor="white", markeredgewidth=1.4, zorder=6)

    ax.fill_between(curve["cost_bps"], curve["sharpe"], 0.0,
                    where=curve["sharpe"] < 0, color=ACCENT, alpha=0.10, zorder=1)

    line2 = "all the slippage the signal can absorb, and no more"
    if momentum is not None:
        cross = crossover_cost(curve, momentum)
        if cross is not None:
            ax.plot([cross], [float(np.interp(cross, curve["cost_bps"], curve["sharpe"]))],
                    "o", color=INK, markersize=4, zorder=6)
            ax.annotate(f"momentum overtakes\nat {cross:.0f} bp", xy=(cross, float(
                np.interp(cross, curve["cost_bps"], curve["sharpe"]))),
                xytext=(12, -34), textcoords="offset points", fontsize=8.5, color=INK)
            line2 = f"momentum, a worse forecaster, overtakes it at {cross:.0f} bp"

    _style(ax, title=f"Sharpe hits zero at {breakeven:.1f} bp one-way\n{line2}",
           xlabel="assumed one-way transaction cost (basis points)",
           ylabel="annualised net Sharpe ratio")
    ax.title.set_fontsize(10.0)
    ax.set_xlim(curve["cost_bps"].min(), curve["cost_bps"].max())
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def fig_schedule(sched: pd.DataFrame, *, label: str, window: str,
                 refit_every: int) -> mpl.figure.Figure:
    """The walk-forward schedule itself: one bar per refit, train then test."""
    n = len(sched)
    height = max(3.8, 0.125 * n + 2.2)
    fig, ax = plt.subplots(figsize=(7.2, height), dpi=130)

    for _, row in sched.iterrows():
        y = row["fold"]
        ax.barh(y, row["train_end_ts"] - row["train_start_ts"], left=row["train_start_ts"],
                height=0.8, color=TRAIN, linewidth=0, zorder=3)
        ax.barh(y, row["test_end_ts"] - row["test_start_ts"], left=row["test_start_ts"],
                height=0.8, color=TEST, linewidth=0, zorder=4)

    first_oos = sched["test_start_ts"].iloc[0]
    ax.axvline(first_oos, color=ACCENT, linewidth=0.9, linestyle=":", zorder=5)
    ax.text(first_oos, n - 0.2, f"  first prediction, {sched['test_start'].iloc[0]}",
            color=ACCENT, fontsize=8.5, va="bottom", ha="left")

    ax.invert_yaxis()
    ax.set_ylim(n + 0.6, -1.0)
    step = 1 if n <= 20 else 6
    ax.set_yticks(list(range(0, n, step)))

    _style(ax, title=(f"{n} refits, {refit_every} months apart\n"
                      f"no model ever sees a day to the right of its own pale bar"),
           xlabel="calendar time", ylabel="refit (fold)")
    ax.title.set_fontsize(10.5)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[
        Line2D([0], [0], color=TRAIN, linewidth=8,
               label=("training window — " + ("grows with every refit" if window == "expanding"
                                              else "fixed at 120 months, rolls forward"))),
        Line2D([0], [0], color=TEST, linewidth=8,
               label=f"held out: {refit_every} months predicted blind, then refit"),
    ], loc="upper right", ncol=1, frameon=True, framealpha=0.94, edgecolor="none",
       facecolor="white", fontsize=8.5, handlelength=1.8, borderpad=0.7, labelspacing=0.5)
    fig.tight_layout()
    return fig


def fig_ic(ic: pd.Series, stats: dict, *, label: str) -> mpl.figure.Figure:
    """Monthly rank IC with its rolling mean -- a real edge, and a small one."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=130)
    x = to_timestamps(ic.dropna().index)

    ax.vlines(x, 0.0, ic.dropna().to_numpy(), color=STRAT_FILL, linewidth=0.9, zorder=3,
              label="monthly rank IC")
    ax.axhline(0.0, color="#9a9a9a", linewidth=0.9, zorder=4)
    roll = stats["rolling"].dropna()
    ax.plot(to_timestamps(roll.index), roll.to_numpy(), color=STRAT, linewidth=1.8, zorder=5,
            label=f"{stats['window']}-month rolling mean")
    ax.axhline(stats["mean"], color=ACCENT, linewidth=1.2, linestyle="--", zorder=6,
               label=f"full-sample mean {stats['mean']:.3f}")

    _style(ax, title=(f"Real, and tiny: mean rank IC {stats['mean']:.3f}, "
                      f"Newey-West(6) t = {stats['t_nw6']:.1f}\n"
                      f"positive in {stats['share_positive'] * 100:.0f}% of months, "
                      f"where a perfect ranking is 1.0"),
           xlabel="formation month", ylabel="rank IC (Spearman rho), monthly")
    ax.title.set_fontsize(10.0)
    ax.set_ylim(-1.0, 1.0)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=3)
    fig.tight_layout()
    return fig


def fig_turnover(turnover: pd.Series, *, ann_turnover: float,
                 label: str, window: int = 36) -> mpl.figure.Figure:
    """One-way turnover per month -- the other half of the arithmetic."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=130)
    t = turnover.dropna()
    x = to_timestamps(t.index)

    ax.plot(x, t.to_numpy() * 100, color=STRAT_FILL, linewidth=0.9, zorder=3,
            label="monthly one-way turnover")
    roll = t.rolling(window, min_periods=window).mean()
    ax.plot(to_timestamps(roll.index), roll.to_numpy() * 100, color=STRAT, linewidth=1.9,
            zorder=5, label=f"{window}-month rolling mean")
    ax.axhline(t.mean() * 100, color=ACCENT, linewidth=1.2, linestyle="--", zorder=6,
               label=f"mean {t.mean() * 100:.1f}% a month")

    _style(ax, title=(f"The bill: {ann_turnover:.2f}x the book's notional traded a year\n"
                      f"{t.mean() * 100:.0f}% of the book replaced in an average month, "
                      f"since 1973"),
           xlabel="formation month", ylabel="one-way turnover, % of gross notional")
    ax.title.set_fontsize(10.0)
    ax.set_ylim(0, max(55.0, t.max() * 105))
    ax.legend(loc="lower left", frameon=False, fontsize=9, ncol=3)
    fig.tight_layout()
    return fig


# ==========================================================================
# APP  --  widgets in, figures out
# ==========================================================================

def _pct(x: float, nd: int = 2) -> str:
    return f"{x * 100:.{nd}f}%"


@st.cache_data(show_spinner=False)
def _bundle() -> dict:
    return load_bundle(DATA_DIR)


def main() -> None:
    st.set_page_config(page_title="The backtest that only works if trading is free",
                       layout="centered")
    mpl.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})

    b = _bundle()
    R = b["results"]

    st.title("The backtest that only works if trading is free")
    st.markdown(
        f"""
I asked whether a cross-sectional machine-learning forecast of next-month industry
returns survives being traded honestly — the model never seeing a day of data that had
not yet happened at the moment the portfolio was decided. On the 49 Fama-French industry
portfolios, **{R['n_scored']} out-of-sample months ({R['oos_start']} to {R['oos_end']})**, the answer
was neither of the two I expected: the forecast is real, small and outside a permutation
null at *p* = 0.001, and the strategy still fails, because harvesting it means turning
over nearly six times the book's notional every year. Move the cost slider and watch a
real edge become no edge.

*The full analysis — data pipeline, leakage audit, 142 tests — lives in a private repo.
This page runs on its derived monthly output.*
"""
    )

    # ---- controls --------------------------------------------------------
    with st.sidebar:
        st.header("Controls")
        configs = b["configs"]
        config = st.selectbox(
            "Configuration", configs, index=configs.index(PRIMARY),
            help=("All sixteen were frozen in code before anything was fitted. "
                  f"`{PRIMARY}` was nominated in advance as the headline, and is "
                  "reported as such whether or not it is the best."),
        )
        cost_bps = st.slider(
            "One-way transaction cost (bp)", min_value=0, max_value=60,
            value=int(R["cost_bps_headline"]), step=1,
            help="Charged on turnover, on both the sells and the buys, against drifted weights.",
        )
        basis = st.radio("Equity curve", ["Net of costs", "Gross of costs"], index=0,
                         horizontal=False)
        show_net = basis.startswith("Net")
        p = parse_config(config)
        st.caption(
            f"**{config}**  \n{p['model'].upper()} · {p['window']} window · refit every "
            f"{p['refit_every']} months · long top {p['n_legs']} / short bottom {p['n_legs']}, "
            "dollar-neutral at gross 1.0."
        )

    gross = b["gross"][config]
    turnover = b["turnover"][config]
    net = apply_costs(gross, turnover, cost_bps)
    s = summarise(gross, turnover, cost_bps)
    s0 = summarise(gross, turnover, 0.0)

    # ---- headline --------------------------------------------------------
    st.subheader(f"Net of {cost_bps} bp one-way")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualised Sharpe", f"{s['sharpe']:.3f}",
              f"{s['sharpe'] - s0['sharpe']:+.3f} vs gross", delta_color="inverse")
    c2.metric("CAGR of excess return", _pct(s["cagr"]),
              f"{(s['cagr'] - s0['cagr']) * 100:+.2f} pp vs gross", delta_color="inverse")
    c3.metric("Max drawdown", _pct(s["max_drawdown"], 1))
    c4.metric("Break-even cost", f"{s['breakeven_cost_bps']:.1f} bp",
              f"turnover {s['ann_turnover']:.2f}x/yr", delta_color="off")

    if cost_bps > s["breakeven_cost_bps"]:
        st.error(f"Past break-even. At {cost_bps} bp the book loses "
                 f"{_pct(abs(s['cagr']))} a year: the signal is entirely eaten by the trading.")
    elif cost_bps >= R["cost_bps_headline"]:
        st.info(f"Annualised volatility {_pct(s['ann_vol'])}, hit rate {_pct(s['hit_rate'], 1)}. "
                f"The cost charge alone is "
                f"{_pct(s0['cagr'] - s['cagr'])} a year against a gross {_pct(s0['cagr'])}.")
    else:
        st.info(f"Annualised volatility {_pct(s['ann_vol'])}, hit rate {_pct(s['hit_rate'], 1)}. "
                f"A cost this low is generous for a liquid basket today and indefensible for 1973.")

    # ---- equity ----------------------------------------------------------
    series = {"strategy": net if show_net else gross}
    for key, col in (("market", "market"), ("12-2 momentum", "momentum_top10")):
        bg, bt = b["bench_gross"][col], b["bench_turnover"][col]
        series[key] = apply_costs(bg, bt, cost_bps) if show_net else bg
    st.pyplot(fig_equity(series, label=config, cost_bps=cost_bps, show_net=show_net,
                         months=s["months"]), use_container_width=True)
    st.caption(
        "Excess return over the realised one-month bill rate, so this is the equity curve of a "
        "position financed at the bill. Gross of costs the book ends *below* a passive market in "
        "dollars and beats it on Sharpe only because it runs at a fraction of the volatility."
    )

    # ---- the bill --------------------------------------------------------
    st.subheader("The bill")
    curve = sharpe_vs_cost(gross, turnover)
    mom_g, mom_t = b["bench_gross"]["momentum_top10"], b["bench_turnover"]["momentum_top10"]
    st.pyplot(fig_cost_curve(
        curve, cost_bps=cost_bps, breakeven=s["breakeven_cost_bps"], label=config,
        momentum=sharpe_vs_cost(mom_g, mom_t),
        momentum_breakeven=breakeven_cost_bps(mom_g, mom_t),
    ), use_container_width=True)
    st.caption(
        f"Naive 12-2 momentum is a clearly worse forecaster — gross Sharpe "
        f"{sharpe(mom_g):.2f} against {s0['sharpe']:.2f} — but it turns over "
        f"{mom_t.mean() * 12:.2f}x a year instead of {s['ann_turnover']:.2f}x, so it breaks even at "
        f"{breakeven_cost_bps(mom_g, mom_t):.0f} bp and overtakes the model somewhere in the teens. "
        "The machine learning wins the forecasting contest and loses the trading one."
    )

    # ---- the schedule ----------------------------------------------------
    st.subheader("The schedule itself")
    st.markdown(
        "Every honest backtest is really a claim about a calendar, and the claim is almost never "
        "drawn. This is it. Each row is one refit: the model is fitted on the pale bar and then "
        "used, unchanged, to predict the blue one — which is what a desk actually does, and is "
        "strictly harder than refitting every month. Under a random 80/20 split of months instead, "
        "roughly half the training months would sit *after* the test month they help predict."
    )
    sched = schedule_for(b["schedule"], config)
    st.pyplot(fig_schedule(sched, label=config, window=p["window"],
                           refit_every=p["refit_every"]), use_container_width=True)
    st.caption(
        f"{len(sched)} refits covering {int(sched['test_months'].sum())} out-of-sample formation "
        f"months. Burn-in {R['initial_train_months']} months; training blocks run "
        f"{int(sched['train_months'].min())} to {int(sched['train_months'].max())} months. "
        f"Features end at the close of day T−1 and the book is set at the close of day T — one day "
        "more conservative than the usual same-close convention."
    )

    # ---- the edge --------------------------------------------------------
    st.subheader("A real edge, and how small it is")
    ic = b["ic"][ic_column(config)]
    ics = ic_summary(ic)
    st.pyplot(fig_ic(ic, ics, label=config), use_container_width=True)
    ic_c1, ic_c2, ic_c3, ic_c4 = st.columns(4)
    ic_c1.metric("Mean rank IC", f"{ics['mean']:.4f}")
    ic_c2.metric("IC information ratio", f"{ics['ir']:.3f}")
    ic_c3.metric("Newey-West(6) t", f"{ics['t_nw6']:.2f}")
    ic_c4.metric("Months with IC > 0", _pct(ics["share_positive"], 1))
    st.caption(
        f"Out-of-sample R² against a zero forecast is {R['oos_r2'] * 100:.3f}% for the primary "
        f"specification — the size the return-predictability literature reports when it is being "
        f"honest. A large positive number here would be evidence of a leak, not of skill."
    )

    st.pyplot(fig_turnover(turnover, ann_turnover=s["ann_turnover"], label=config),
              use_container_width=True)
    st.caption(
        f"Turnover is measured against **drifted** weights: over the holding month positions move "
        f"with returns, and only the difference between where they drifted to and where the new "
        f"ranking wants them is actually traded. A randomly re-ranked book trades "
        f"{R['null_monthly_turnover'] * 100:.0f}% a month against this strategy's "
        f"{R['monthly_turnover'] * 100:.0f}%."
    )

    # ---- the verdict -----------------------------------------------------
    st.subheader("The verdict")
    st.warning(R["verdict"])
    cond = pd.DataFrame(
        [{"pre-registered condition": k, "met": "yes" if v else "no"}
         for k, v in R["conditions_for_a_positive_claim"].items()]
    )
    st.dataframe(cond, hide_index=True, use_container_width=True)
    st.markdown(
        f"The deflated Sharpe is {R['dsr']['dsr']:.3f} against a 0.95 bar, deflating for "
        f"{R['dsr']['n_trials']} trials. Those trials are far from independent, so 16 is "
        f"conservative and the true figure is somewhat better — I state the direction and stop "
        f"there, because swapping the trial count for a friendlier one after seeing which "
        f"condition failed is exactly the move the pre-registration exists to prevent."
    )
    st.markdown(f"> {R['era_note']}")

    with st.expander(f"All sixteen pre-registered configurations at {cost_bps} bp"):
        st.dataframe(grid_table(b, cost_bps), hide_index=True, use_container_width=True)
        st.caption(
            "Every one is reported whatever it shows. The grid was frozen in code before anything "
            "was fitted, and a test fails if a seventeenth is added — the multiple-testing "
            "correction depends on the count being honest."
        )

    with st.expander("Method, data and what this is not"):
        st.markdown(f"""
**Data.** The 49 Fama-French value-weighted industry portfolios, daily and monthly, plus
the factor files for the market return and the realised one-month bill rate
([Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html),
CRSP vintage {R['crsp_vintage']}). French's portfolios are point-in-time and include firms
that later went to zero, so survivorship is handled at source rather than by me hoping.
The modelling panel is {R['n_rows']:,} eligible industry-months across {R['n_months']}
formation months ({R['sample_start']} to {R['sample_end']}), 13 features, cross-section
{R['uni_min']}–{R['uni_max']} industries.

**Protocol.** Expanding or rolling window, {R['initial_train_months']}-month burn-in, refit
on a fixed schedule, {R['n_fits']} model fits across the whole grid. Target is next-month
excess return, cross-sectionally demeaned, so the model is a pure relative-value ranker and
is never rewarded for forecasting a market level a dollar-neutral book does not trade.
Industries with fewer than five constituent firms are screened out at formation — not outlier
cleaning, refusing to call one stock an industry.

**Limitations.** The instruments are not tradable: Fama-French industry portfolios are research
constructs, and implementing this would mean industry ETFs that mostly did not exist before the
late 1990s. One constant cost is applied across all {R['n_scored']} months. Short borrow, market
impact and capacity are not modelled at all, and all three push optimistic. The edge sits in the
era where the cost model is least believable.

**This page.** Only derived series ship here — monthly returns, turnover, rank IC and the refit
schedule, all computed in the private repo and reproduced exactly. The app recomputes costs,
Sharpe, drawdowns and the cost curve from them in the browser.

This is a research exercise, not investment advice.
""")


if __name__ == "__main__":
    main()
