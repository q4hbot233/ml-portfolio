"""Is it the car or the driver? — an interactive telemetry-diagnosis explorer.

Everything above the `STREAMLIT` banner is plain Python: functions that take a
dataframe and return a dataframe or a matplotlib figure, with no Streamlit call
anywhere in them.  The Streamlit code lives in :func:`main`, which only reads
widgets, calls those functions and renders what comes back — so ``import app``
from an ordinary Python session gets the whole compute layer and renders
nothing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.collections import LineCollection

DATA_DIR = Path(__file__).parent / "data"

#: One colour per idea, held constant across every figure on the page.
C = {
    "subject": "#D62828",    # the car under investigation
    "teammate": "#0E7C86",   # teammate frame
    "self": "#E76F51",       # self frame
    "benchmark": "#343A40",  # the session's own best
    "model": "#6A4C93",      # model frame
    "field": "#ADB5BD",      # everyone else
    "loss": "#C1121F",       # time lost
    "gain": "#0E7C86",       # time gained
    "track": "#DEE2E6",
}

FRAME_TEAMMATE = "Teammate — the other car in the same garage"
FRAME_SELF = "Himself — the same driver's own earlier laps"

SEGMENT_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2}

#: A teammate gap smaller than this reads as "the two cars agree", and a driver
#: this far behind the field's own improvement reads as "lost ground".  Both are
#: judgement calls, stated on the page rather than buried here.
AGREEMENT_S = 0.20
BEHIND_FIELD_S = 0.05


# =========================================================================
#  COMPUTE LAYER — plain functions, no Streamlit, nothing global mutated
# =========================================================================
def load_tables(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Read every shipped artifact into one dict of dataframes."""
    tables = {key: pd.read_csv(data_dir / name) for key, name in {
        "laps": "laps.csv",
        "corners": "corner_summary.csv",
        "markers": "corner_markers.csv",
        "centreline": "track_centreline.csv",
        "best": "session_best_laps.csv",
        "evolution": "evolution_q1_q2.csv",
        "season": "season_gap_to_fastest.csv",
        "meta": "meta.csv",
        "ablation": "feature_ablation.csv",
        "corner_error": "error_by_corner.csv",
        "importance": "permutation_importance.csv",
    }.items()}
    # The profile is one column per lap and one row per grid point, so the shared
    # distance axis is stored once instead of 88 times.
    profile = pd.read_csv(data_dir / "lap_profile.csv")
    profile.columns = ["distance_m"] + [int(c) for c in profile.columns[1:]]
    tables["profile"] = profile
    return tables


def ordinal(n: int) -> str:
    """``1 -> '1st'``, ``11 -> '11th'``, ``22 -> '22nd'``."""
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }".replace(" ", "")


def format_lap_time(seconds: float) -> str:
    """``92.173 -> '1:32.173'`` — the way a lap time is written on a timing screen."""
    return f"{int(seconds // 60)}:{seconds % 60:06.3f}"


def team_of(laps: pd.DataFrame, driver: str) -> str:
    """The team the driver ran for in this session."""
    return str(laps.loc[laps["driver"] == driver, "team"].iloc[0])


def teammate_of(laps: pd.DataFrame, driver: str) -> str | None:
    """The other driver in the same garage, or ``None`` for a one-car entry."""
    mates = sorted(set(laps.loc[laps["team"] == team_of(laps, driver), "driver"]) - {driver})
    return mates[0] if mates else None


def driver_laps(laps: pd.DataFrame, driver: str) -> pd.DataFrame:
    """That driver's push laps, in the order he drove them."""
    return laps[laps["driver"] == driver].sort_values("clock_min").reset_index(drop=True)


def lap_label(row: pd.Series) -> str:
    """``Q2 lap 13 — 1:32.173``, how a lap is named everywhere in the UI."""
    return f"{row['q_segment']} lap {int(row['lap_number'])} — {format_lap_time(float(row['lap_time_s']))}"


def distance_spread_m(laps: pd.DataFrame) -> float:
    """How far apart the cars are about the length of the same lap.

    ``Distance`` is integrated from wheel speed, so a car that takes a wider line
    genuinely travels further and the integration drifts on top of that.  This
    number is why every comparison on the page normalises onto fraction-of-lap
    before subtracting: intersecting the raw distance ranges instead throws away
    the tail of the longer lap and charges the time spent there to nobody.
    """
    d = laps["distance_travelled_m"]
    return float(d.max() - d.min())


def choose_reference(laps: pd.DataFrame, frame: str, lap_id: int) -> dict:
    """Pick the lap the chosen frame compares against, and say out loud why.

    The whole page turns on this function, so it reports its reasoning rather
    than silently returning an id.  Either frame can come back unavailable — a
    one-car entry has no teammate, a driver's first run of the day has no
    earlier lap — and that is a finding, not an error: the frame you lose tends
    to be lost on exactly the weekend something went wrong.

    Teammate frame: the other car's best lap in the *same* qualifying segment,
    so the track state is as nearly shared as it gets.  If the teammate set no
    lap in that segment the comparison falls back to his best of the session,
    which is flagged, because Q1 and Q3 are not the same racetrack.

    Self frame: the driver's own best lap from *before* the current qualifying
    segment — the Q1 → Q2 comparison, generalised.  Failing that, his best
    earlier lap within the same run.
    """
    lap = laps.loc[laps["lap_id"] == lap_id].iloc[0]
    driver, seg = str(lap["driver"]), str(lap["q_segment"])
    blank = {"lap_id": None, "available": False, "same_segment": None,
             "reference_driver": None}

    if frame == FRAME_TEAMMATE:
        mate = teammate_of(laps, driver)
        if mate is None:
            return {**blank, "note": f"{driver} is a one-car entry in this session, so the "
                                     "teammate frame does not exist here."}
        mate_laps = driver_laps(laps, mate)
        if not len(mate_laps):
            return {**blank, "note": f"{mate} set no usable push lap in this session, so "
                                     f"{driver} has no teammate frame."}
        same = mate_laps[mate_laps["q_segment"] == seg]
        pool, same_segment = (same, True) if len(same) else (mate_laps, False)
        ref = pool.loc[pool["lap_time_s"].idxmin()]
        note = (f"{mate}'s best {seg} lap — same car, same spec, same track state."
                if same_segment else
                f"{mate} set no {seg} lap, so this falls back to his best "
                f"{ref['q_segment']} lap. That is a different point in the session, so part "
                "of any gap is track evolution rather than the two cars.")
        return {"lap_id": int(ref["lap_id"]), "available": True, "reference_driver": mate,
                "same_segment": same_segment, "note": note}

    own = driver_laps(laps, driver)
    earlier_segment = own[own["q_segment"].map(SEGMENT_ORDER) < SEGMENT_ORDER[seg]]
    if len(earlier_segment):
        ref = earlier_segment.loc[earlier_segment["lap_time_s"].idxmin()]
        gap_min = float(lap["clock_min"]) - float(ref["clock_min"])
        return {"lap_id": int(ref["lap_id"]), "available": True, "reference_driver": driver,
                "same_segment": False,
                "note": f"{driver}'s own best {ref['q_segment']} lap, {gap_min:.0f} minutes "
                        "earlier — same hands, same style; the car, the tyres and the track "
                        "surface are what moved."}

    earlier_run = own[own["clock_min"] < float(lap["clock_min"])]
    if len(earlier_run):
        ref = earlier_run.loc[earlier_run["lap_time_s"].idxmin()]
        return {"lap_id": int(ref["lap_id"]), "available": True, "reference_driver": driver,
                "same_segment": True,
                "note": f"{driver}'s own best earlier lap in {seg} — this is his first run "
                        "of the day, so there is no previous segment to fall back on."}
    return {**blank, "note": f"This is {driver}'s first push lap of the session, so he has "
                             "no earlier lap of his own and the self frame is not available."}


def corner_delta(corners: pd.DataFrame, lap_id: int, ref_id: int) -> pd.DataFrame:
    """Per-corner time and speed difference between one lap and its reference.

    ``delta_s`` is positive when the chosen lap lost time in that segment.  The
    segments telescope: because both laps were resampled onto the same
    fraction-of-lap axis before their clocks were read off, the per-segment
    differences sum to the lap-time gap with no residual to explain away.
    """
    a = corners[corners["lap_id"] == lap_id].set_index("segment").sort_index()
    b = corners[corners["lap_id"] == ref_id].set_index("segment").sort_index()
    return pd.DataFrame({
        "segment": a.index,
        "corner": a["corner"],
        "sector": a["sector"],
        "start_m": a["start_m"],
        "end_m": a["end_m"],
        "delta_s": a["seg_time_s"] - b["seg_time_s"],
        "d_v_min_kph": a["v_min_kph"] - b["v_min_kph"],
        "d_v_entry_kph": a["v_entry_kph"] - b["v_entry_kph"],
        "d_v_exit_kph": a["v_exit_kph"] - b["v_exit_kph"],
        "v_min_kph": a["v_min_kph"],
        "ref_v_min_kph": b["v_min_kph"],
    }).reset_index(drop=True)


def loss_profile(profile: pd.DataFrame, centreline: pd.DataFrame,
                 lap_id: int, ref_id: int) -> pd.DataFrame:
    """Running time delta along the lap, and the rate of loss per 100 m of track.

    The rate is what the map paints.  Normalising by the length of each stretch
    is not cosmetic: a long flat-out straight and a short slow corner are not
    comparable until the loss is expressed per metre of track.
    """
    d = profile["distance_m"].to_numpy(dtype=float)
    delta = profile[lap_id].to_numpy(dtype=float) - profile[ref_id].to_numpy(dtype=float)
    span = np.diff(d, prepend=d[0])
    step = np.diff(delta, prepend=delta[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        per_100 = np.where(span > 0, step / np.where(span > 0, span, 1.0) * 100.0, 0.0)
    out = centreline.copy()
    out["cum_delta_s"] = delta
    out["rate_s_per_100m"] = per_100
    return out


def field_evolution(best: pd.DataFrame, from_seg: str, to_seg: str) -> dict:
    """How much the field itself found between two qualifying segments.

    The self frame's baseline is not fixed — every car that goes round lays
    rubber and the whole grid gets quicker — so a driver's own improvement only
    means something next to what everyone else found in the same window.  With
    ``from_seg == to_seg`` there is no window and no correction to make.
    """
    if from_seg == to_seg:
        return {"available": False, "median_s": np.nan, "n_drivers": 0,
                "note": "Both laps come from the same run, so there is no window of track "
                        "evolution to subtract."}
    wide = best.pivot_table(index="driver", columns="q_segment",
                            values="lap_time_s", aggfunc="min")
    if from_seg not in wide or to_seg not in wide:
        return {"available": False, "median_s": np.nan, "n_drivers": 0,
                "note": "Too few drivers set a lap in both segments."}
    pair = wide[[from_seg, to_seg]].dropna()
    improvement = (pair[from_seg] - pair[to_seg]).sort_values()
    return {"available": True, "median_s": float(improvement.median()),
            "n_drivers": int(len(pair)), "improvements": improvement,
            "note": f"{len(pair)} drivers set a valid lap in both {from_seg} and {to_seg}."}


def frame_verdict(laps: pd.DataFrame, best: pd.DataFrame, corners: pd.DataFrame,
                  frame: str, lap_id: int, ref_id: int) -> dict:
    """The numbers one frame produces for one lap, and what that frame is blind to."""
    lap = laps.loc[laps["lap_id"] == lap_id].iloc[0]
    ref = laps.loc[laps["lap_id"] == ref_id].iloc[0]
    seg = corner_delta(corners, lap_id, ref_id)
    total = float(seg["delta_s"].sum())
    official = float(lap["lap_time_s"]) - float(ref["lap_time_s"])
    worst = seg.loc[seg["delta_s"].abs().idxmax()]

    out = {
        "frame": frame,
        "gap_s": official,
        "segments_sum_s": total,
        "reconciliation_ms": (total - official) * 1000.0,
        "worst_corner": str(worst["corner"]),
        "worst_delta_s": float(worst["delta_s"]),
        "worst_share": abs(float(worst["delta_s"])) / abs(total) if abs(total) > 1e-9 else np.nan,
        "n_segments": int(len(seg)),
        "lap_time_s": float(lap["lap_time_s"]),
        "ref_time_s": float(ref["lap_time_s"]),
        "reference": lap_label(ref),
    }

    if frame == FRAME_TEAMMATE:
        out["blind_to"] = ("anything wrong with **both** cars — a shared fault sits on both "
                           "sides of the subtraction and cancels exactly.")
    else:
        evo = field_evolution(best, str(ref["q_segment"]), str(lap["q_segment"]))
        improvement = -official        # positive when the newer lap is the quicker one
        out["improvement_s"] = improvement
        out["field_median_s"] = evo["median_s"]
        out["vs_field_s"] = (improvement - evo["median_s"]) if evo["available"] else np.nan
        out["evolution"] = evo
        out["blind_to"] = ("anything that moves the baseline — track evolution, fuel, tyres. "
                           "The reference is not a fixed point.")
    return out


def compare_frames(laps: pd.DataFrame, best: pd.DataFrame, corners: pd.DataFrame,
                   lap_id: int) -> dict:
    """Run both frames on the same lap and report whether they agree.

    This is the point of the page.  The frames are not rivals with a correct
    answer between them — they difference away different things, so a case where
    they disagree localises the problem to whatever one of them cancelled.
    """
    out: dict = {}
    for frame in (FRAME_TEAMMATE, FRAME_SELF):
        ref = choose_reference(laps, frame, lap_id)
        out[frame] = ({"available": False, "note": ref["note"]} if not ref["available"]
                      else {"available": True, "note": ref["note"],
                            **frame_verdict(laps, best, corners, frame, lap_id, ref["lap_id"])})
    tm, sf = out[FRAME_TEAMMATE], out[FRAME_SELF]
    if tm["available"] and sf["available"] and sf["evolution"]["available"]:
        out["disagree"] = bool(abs(tm["gap_s"]) < AGREEMENT_S
                               and sf["vs_field_s"] < -BEHIND_FIELD_S)
    else:
        out["disagree"] = None
    return out


def season_symptom(season: pd.DataFrame, driver: str, event: str) -> pd.DataFrame:
    """That driver's gap to the fastest lap of each qualifying session, all 2023.

    The widest self reference available: the driver is held constant while the
    car and the circuit vary, which is the only frame in which one bad Saturday
    can be told apart from an anomaly.
    """
    d = season[season["driver"] == driver].sort_values("event_round").copy()
    d["is_event"] = d["event_name"] == event
    d["short_name"] = d["event_name"].str.replace(" Grand Prix", "", regex=False)
    return d.reset_index(drop=True)


def season_headline(d: pd.DataFrame, event: str) -> dict:
    """How far the case-study round sticks out of that driver's own season."""
    here = d.loc[d["is_event"], "gap_s"]
    others = d.loc[~d["is_event"], "gap_s"]
    if here.empty or others.empty:
        return {"available": False}
    gap, nxt = float(here.iloc[0]), float(others.max())
    return {"available": True, "gap_s": gap, "median_s": float(d["gap_s"].median()),
            "next_worst_s": nxt, "ratio": gap / nxt if nxt > 0 else np.nan,
            "rank": int((d["gap_s"] > gap).sum()) + 1, "n_rounds": int(len(d))}


# =========================================================================
#  FIGURES — take dataframes, return a Figure; still no Streamlit
# =========================================================================
mpl.rcParams.update({
    "figure.dpi": 110, "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.labelsize": 9.5, "axes.grid": True,
    "grid.alpha": 0.22, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white",
})


def fig_season(d: pd.DataFrame, driver: str, head: dict, event: str):
    """One bar per round: this driver's gap to the fastest lap of that session."""
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    colours = [C["subject"] if f else C["field"] for f in d["is_event"]]
    ax.bar(d["event_round"], d["gap_s"], color=colours, width=0.74,
           edgecolor="white", linewidth=0.6)
    med = float(d["gap_s"].median())
    ax.axhline(med, color=C["benchmark"], linestyle="--", linewidth=1.1)
    ax.text(0.4, med, f" season median {med:.3f} s", va="bottom", ha="left",
            fontsize=8.5, color=C["benchmark"])

    short_event = event.replace(" Grand Prix", "")
    if head["available"]:
        row = d[d["is_event"]].iloc[0]
        # The ratio to the next-worst round only says anything when this round *is* the
        # worst; anywhere else in the ranking it is a number about a different weekend.
        if head["rank"] == 1:
            note = f"{head['gap_s']:.3f} s  —  {head['ratio']:.1f}× his next-worst round"
            title = f"{short_event} is {driver}'s worst qualifying of 2023, by {head['ratio']:.1f}×"
        else:
            note = f"{head['gap_s']:.3f} s  —  only his {ordinal(head['rank'])}-worst round"
            title = f"{short_event} barely stands out in {driver}'s own season"
        ax.annotate(note, xy=(float(row["event_round"]), head["gap_s"]),
                    xytext=(max(float(row["event_round"]) - 8.5, 0.5),
                            head["gap_s"] + 0.12 * float(d["gap_s"].max())),
                    fontsize=8.8, color=C["subject"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["subject"], lw=1.2))
    else:
        title = f"{driver} — gap to the fastest lap of each qualifying session"
    ax.set_title(title, loc="left")
    ax.set_xticks(d["event_round"])
    ax.set_xticklabels(d["short_name"], rotation=58, ha="right", fontsize=7.2)
    ax.set_xlabel("2023 round")
    ax.set_ylabel("Gap to session\nfastest lap (s)")
    ax.set_ylim(0, max(float(d["gap_s"].max()) * 1.32, 0.4))
    return fig


def fig_where(seg: pd.DataFrame, prof: pd.DataFrame, markers: pd.DataFrame,
              subject: str, reference: str, total_s: float, accent: str):
    """Three panels on one lap-distance axis: the running delta, then per corner."""
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.4), sharex=True,
                             constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.0, 1.25, 0.95]})
    worst = seg.loc[seg["delta_s"].abs().idxmax()]
    share = abs(float(worst["delta_s"])) / abs(total_s) if abs(total_s) > 1e-9 else np.nan

    # (a) the running delta — is the time lost at a point, or bled away everywhere?
    ax = axes[0]
    ax.plot(prof["distance_m"], prof["cum_delta_s"], color=accent, linewidth=1.9)
    ax.fill_between(prof["distance_m"], 0, prof["cum_delta_s"], color=accent, alpha=0.16)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative\ntime delta (s)")
    ax.set_title("Time bleeds away steadily — no single place to send the mechanics"
                 if share < 0.35 else
                 f"Most of the gap arrives in one place: {worst['corner']}", loc="left")

    # (b) per corner, in seconds — each bar spans the stretch of track it describes
    ax = axes[1]
    width = seg["end_m"] - seg["start_m"]
    ax.bar(seg["start_m"], seg["delta_s"], width=width, align="edge",
           color=[C["loss"] if v > 0 else C["gain"] for v in seg["delta_s"]],
           edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_ylabel("Time lost by\nthis lap (s)")
    ax.set_title(f"{subject} − {reference}:  Σ segments {total_s:+.3f} s  ·  biggest corner "
                 f"{worst['corner']} {float(worst['delta_s']):+.3f} s = {share:.0%} of it",
                 loc="left", fontsize=9.6)

    # (c) per corner, in km/h — a broken part shows up as a minimum speed, not a lap time
    ax = axes[2]
    ax.bar(seg["start_m"], seg["d_v_min_kph"], width=width, align="edge",
           color=[C["loss"] if v < 0 else C["gain"] for v in seg["d_v_min_kph"]],
           edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_ylabel("Minimum speed\ndifference (km/h)")
    ax.set_xlabel("Lap distance (m) — ticks at the corner markers")
    ax.set_title(f"Corner minimum speeds average {float(seg['d_v_min_kph'].mean()):+.1f} km/h "
                 "against this reference", loc="left")

    ax.set_xticks(markers["apex_distance_m"].to_numpy())
    ax.set_xticklabels(markers["corner"], fontsize=7.0, rotation=90)
    ax.set_xlim(float(seg["start_m"].iloc[0]), float(seg["end_m"].iloc[-1]))
    for a in axes:
        for x in markers["apex_distance_m"]:
            a.axvline(x, color="#E9ECEF", linewidth=0.7, zorder=0)
    return fig


def fig_map(prof: pd.DataFrame, markers: pd.DataFrame, subject: str,
            reference: str, total_s: float):
    """The same numbers painted onto the circuit: rate of loss per 100 m of track."""
    fig, ax = plt.subplots(figsize=(7.0, 6.4), constrained_layout=True)
    xy = prof[["x", "y"]].to_numpy(dtype=float)
    rate = prof["rate_s_per_100m"].to_numpy(dtype=float)
    lim = float(np.nanpercentile(np.abs(rate), 97)) or 1e-3

    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="RdBu_r", norm=plt.Normalize(-lim, lim),
                        linewidth=6.0, capstyle="round")
    lc.set_array(rate[1:])
    ax.plot(xy[:, 0], xy[:, 1], color=C["track"], linewidth=9.5, zorder=0,
            solid_capstyle="round")
    ax.add_collection(lc)

    for _, m in markers.iterrows():
        ax.text(m["x"], m["y"], str(int(m["number"])), fontsize=6.6, ha="center",
                va="center", color="#212529", zorder=5,
                bbox=dict(boxstyle="circle,pad=0.13", facecolor="white",
                          edgecolor="#ADB5BD", linewidth=0.5))
    ax.set_aspect("equal")
    ax.axis("off")
    hot = float(np.nanmax(rate))
    ax.set_title(f"{subject} − {reference} ({total_s:+.3f} s): the worst stretch of track "
                 f"costs {hot:.3f} s per 100 m", loc="left")
    cbar = fig.colorbar(lc, ax=ax, orientation="horizontal", fraction=0.042,
                        pad=0.02, shrink=0.72)
    cbar.set_label("Time lost per 100 m of track (s)  —  red = losing, blue = gaining",
                   fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    return fig


def fig_evolution(evo: pd.DataFrame, driver: str, teammate: str | None):
    """Q1 → Q2 improvement for the whole field, against the field's own median."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    e = evo.sort_values("improvement_s").reset_index(drop=True)
    focus = {driver} | ({teammate} if teammate else set())
    ax.barh(e["driver"], e["improvement_s"], edgecolor="white", linewidth=0.6,
            color=[C["subject"] if d in focus else C["field"] for d in e["driver"]])
    med = float(e["field_median_s"].iloc[0])
    ax.axvline(med, color=C["benchmark"], linestyle="--", linewidth=1.3)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.text(med + 0.02, e["driver"].iloc[0], f"field median {med:+.3f} s",
            ha="left", va="center", fontsize=8.5, color=C["benchmark"])
    for d in focus:
        row = e[e["driver"] == d]
        if row.empty:
            continue
        v = float(row["improvement_s"].iloc[0])
        x, ha = (v - 0.03, "right") if v < 0 else (v + 0.03, "left")
        ax.text(x, d, f"{float(row['vs_field_s'].iloc[0]):+.3f} s vs field", va="center",
                ha=ha, fontsize=8.2, color=C["subject"], fontweight="bold")
    for yi, d in enumerate(e["driver"]):
        if d in focus:
            ax.get_yticklabels()[yi].set_color(C["subject"])
            ax.get_yticklabels()[yi].set_fontweight("bold")
    ax.set_xlabel("Time found from Q1 to Q2 (s) — positive is an improvement")
    slowest = e.iloc[0]
    ax.set_title(f"The track rubbered in and the field found {med:+.3f} s — "
                 f"{slowest['driver']} went {abs(float(slowest['improvement_s'])):.3f} s "
                 "slower", loc="left", fontsize=9.8)
    ax.set_xlim(float(e["improvement_s"].min()) - 0.32, float(e["improvement_s"].max()) + 0.34)
    return fig


def fig_model_frame(ablation: pd.DataFrame, corner_error: pd.DataFrame, target_sd: float):
    """The third frame's own limits: what it needs to work, and where it is wrong."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.08, 1]})

    ax = axes[0]
    a = ablation.sort_values("mae_kph", ascending=False).reset_index(drop=True)
    y = np.arange(len(a))
    ax.barh(y, a["mae_kph"], edgecolor="white", linewidth=0.6,
            color=[C["model"] if f == "all features" else C["field"] for f in a["features"]])
    for yi, v in enumerate(a["mae_kph"]):
        ax.text(v + 0.9, yi, f"{v:.1f}", va="center", fontsize=7.8, color="#444")
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace(" ", "\n", 1) for f in a["features"]], fontsize=7.4)
    ax.set_xlabel("Mean absolute error on unseen circuits (km/h)")
    ax.set_xlim(0, float(a["mae_kph"].max()) * 1.18)
    ax.set_title(f"Entry telemetry is what transfers:\n{float(a['mae_kph'].min()):.1f} km/h "
                 f"against a {target_sd:.1f} km/h target sd", loc="left", fontsize=9.3)

    ax = axes[1]
    ce = corner_error.sort_values("bias_kph").reset_index(drop=True)
    ax.barh(np.arange(len(ce)), ce["bias_kph"], color=C["field"], height=0.92)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks([])
    ax.set_ylabel(f"the {len(ce)} worst-fitted corners")
    ax.set_xlabel("Mean prediction error at that corner (km/h)")
    spread = float(ce["bias_kph"].max() - ce["bias_kph"].min())
    ax.set_title(f"…but on a circuit it has never seen it is\nwrong by whole corners: "
                 f"{spread:.0f} km/h of bias", loc="left", fontsize=9.3)
    return fig


# =========================================================================
#  STREAMLIT — widgets in, figures out.  Nothing below runs on import.
# =========================================================================
@st.cache_data
def _tables() -> dict[str, pd.DataFrame]:
    return load_tables()


def main() -> None:                                        # noqa: C901 - it is a page
    st.set_page_config(page_title="Is it the car or the driver?", page_icon="🏁",
                       layout="centered")
    tables = _tables()
    laps, corners = tables["laps"], tables["corners"]
    meta = tables["meta"].iloc[0]
    event = str(meta["event"])

    st.title("Is it the car or the driver?")
    st.markdown(
        f"""
A driver says the car feels wrong and cannot tell you *what* is wrong. You have twenty
minutes before the next run, no way to put the car on a rig, and the only evidence is
telemetry. Almost everything written with F1 telemetry is performance analysis — who is
quick, where, and by how much — so I went after the question an engineer actually has to
answer between runs, and found it has no answer in the abstract: **a diagnosis only exists
relative to a reference frame, and the frame you pick decides which faults you can see at
all.** This page lets you change the frame and watch the verdict change with it.

*{int(meta['n_push_laps'])} push laps from {event} qualifying {int(meta['year'])},
{int(meta['n_drivers'])} drivers, cut into {int(meta['n_segments'])} corner segments. The
full analysis — all 22 rounds of 2023, {int(meta['model_dataset_rows']):,} corner
traversals and the modelling behind section 4 — lives in a private repository; what is
shipped here is the derived aggregate it produced.*
"""
    )

    # ------------------------------------------------------------ controls ---
    if "driver" not in st.session_state:
        st.session_state.update(driver="VER", frame=FRAME_TEAMMATE)

    with st.sidebar:
        st.header("Reference frame")
        frame = st.radio(
            "Compare this lap against…", [FRAME_TEAMMATE, FRAME_SELF], key="frame",
            help="The teammate holds the machinery constant, so a gap points at the driver. "
                 "The driver's own earlier laps hold the driving constant, so a gap points "
                 "at the car — or at the track.")

        st.header("Lap under investigation")
        driver = st.selectbox("Driver", sorted(laps["driver"].unique()), key="driver")
        own = driver_laps(laps, driver)
        options = [int(i) for i in own["lap_id"]]
        fastest = int(own.loc[own["lap_time_s"].idxmin(), "lap_id"])
        lap_id = st.selectbox(
            "Lap", options, index=options.index(fastest),
            format_func=lambda i: lap_label(laps.loc[laps["lap_id"] == i].iloc[0]),
            help="Every lap this driver got within 7% of the session's best — his real "
                 "attempts, with in- and out-laps removed.")
        st.caption(f"{team_of(laps, driver)} · {len(own)} push laps in this session")

    lap_row = laps.loc[laps["lap_id"] == lap_id].iloc[0]
    ref = choose_reference(laps, frame, lap_id)
    both = compare_frames(laps, tables["best"], corners, lap_id)
    mate = teammate_of(laps, driver)

    # --------------------------------------------------- 1. the symptom ------
    st.subheader("1. The symptom, and why this session")
    st.markdown(
        "Spotting that something *was* wrong is itself a reference-frame problem. Inside one "
        "session, a driver who qualifies eleventh had a bad day and that is all you can say. "
        "The widest frame available holds the driver constant and lets the car and the "
        "circuit vary: his gap to the fastest lap of every qualifying session of the season.")
    season_d = season_symptom(tables["season"], driver, event)
    head = season_headline(season_d, event)
    st.pyplot(fig_season(season_d, driver, head, event), use_container_width=True)
    if head["available"]:
        st.caption(
            f"{driver}: median gap {head['median_s']:.3f} s across {head['n_rounds']} rounds; "
            f"{event.replace(' Grand Prix', '')} is {head['gap_s']:.3f} s, his "
            f"{ordinal(head['rank'])}-worst round of the year. A session only becomes an "
            "anomaly against a frame wider than itself.")

    # -------------------------------------------------- 2. the diagnosis -----
    st.subheader("2. Where did the time go — against *this* reference?")

    if not ref["available"]:
        st.warning(ref["note"])
        st.markdown(
            "Worth sitting with rather than clicking past. **The frames are not always "
            "available, and the one you lose is not random** — it tends to go missing on "
            "exactly the weekend something went wrong. Pérez's 2023 Australian qualifying "
            "ended in the gravel, so that weekend has no teammate frame at all.")
    else:
        v = frame_verdict(laps, tables["best"], corners, frame, lap_id, ref["lap_id"])
        seg = corner_delta(corners, lap_id, ref["lap_id"])
        prof = loss_profile(tables["profile"], tables["centreline"], lap_id, ref["lap_id"])
        ref_row = laps.loc[laps["lap_id"] == ref["lap_id"]].iloc[0]
        subject_label = f"{driver} {lap_row['q_segment']}"
        ref_label = f"{ref_row['driver']} {ref_row['q_segment']}"
        accent = C["teammate"] if frame == FRAME_TEAMMATE else C["self"]

        st.info(f"**Reference: {lap_label(ref_row)}.** {ref['note']}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Lap-time gap", f"{v['gap_s']:+.3f} s",
                  help="This lap minus the reference lap, from official timing.")
        c2.metric("Biggest single corner", f"{v['worst_delta_s']:+.3f} s",
                  delta=f"{v['worst_corner']} · {v['worst_share']:.0%} of the gap",
                  delta_color="off")
        c3.metric("Segments add back up to", f"{v['segments_sum_s']:+.3f} s",
                  delta=f"{v['reconciliation_ms']:+.1f} ms vs official timing",
                  delta_color="off")

        st.pyplot(fig_where(seg, prof, tables["markers"], subject_label, ref_label,
                            v["segments_sum_s"], accent), use_container_width=True)
        st.caption(
            f"Both laps are resampled onto a common fraction-of-lap axis before their clocks "
            f"are subtracted, so the {v['n_segments']} segments telescope: they sum to the "
            f"official gap to within {abs(v['reconciliation_ms']):.1f} ms. A decomposition "
            "whose parts do not add back up to the whole is a decomposition of something "
            "else — and the naive version, which just intersects the two laps' raw distance "
            f"ranges, gets it wrong by tenths, because the cars in this session disagree by "
            f"{distance_spread_m(laps):.0f} m about how long the same lap is. Distance is "
            "integrated from wheel speed, so a wider line really is further.")

        st.markdown("**The same numbers painted onto the circuit.** A mechanical fault with a "
                    "location shows up as one hot patch. A car-wide state does not.")
        st.pyplot(fig_map(prof, tables["markers"], subject_label, ref_label,
                          v["segments_sum_s"]), use_container_width=True)

        with st.expander("Per-corner detail, as a table"):
            table = seg[["corner", "sector", "delta_s", "d_v_min_kph", "d_v_entry_kph",
                         "d_v_exit_kph", "v_min_kph", "ref_v_min_kph"]].copy()
            table.columns = ["Corner", "Sector", "Time lost (s)", "Δ min speed (km/h)",
                             "Δ entry speed (km/h)", "Δ exit speed (km/h)",
                             "Min speed (km/h)", "Reference min speed (km/h)"]
            st.dataframe(table.round(3), hide_index=True, use_container_width=True)

    # ----------------------------------------------- 3. the two frames -------
    st.subheader("3. What each frame can and cannot see")
    tm, sf = both[FRAME_TEAMMATE], both[FRAME_SELF]
    left, right = st.columns(2)

    with left:
        st.markdown("##### Teammate frame")
        if tm["available"]:
            st.metric(f"{driver} − {mate}", f"{tm['gap_s']:+.3f} s")
            st.markdown(
                "Holds the **machinery** constant: same chassis, same spec, same track state. "
                "A gap here points at the driver, or at one side of the garage.\n\n"
                f"**Blind to** {tm['blind_to']}")
        else:
            st.markdown(f":grey[{tm['note']}]")

    with right:
        st.markdown("##### Self frame")
        if sf["available"]:
            st.metric(f"{driver} vs his own earlier lap", f"{sf['improvement_s']:+.3f} s",
                      delta=(f"{sf['vs_field_s']:+.3f} s vs the field"
                             if sf["evolution"]["available"] else None))
            extra = ("" if not sf["evolution"]["available"] else
                     f" The field's median driver found {sf['field_median_s']:+.3f} s over "
                     f"the same window ({sf['evolution']['n_drivers']} drivers).")
            st.markdown(
                "Holds the **driving** constant: same hands, same style. A gap here points at "
                f"the car, the tyres or the track.{extra}\n\n"
                f"**Blind to** {sf['blind_to']}")
        else:
            st.markdown(f":grey[{sf['note']}]")

    if both["disagree"] is True:
        st.error(
            f"**The two frames disagree about this lap, and the disagreement is the finding.** "
            f"The teammate frame sees {abs(tm['gap_s']):.3f} s between the two cars and calls "
            f"the weekend unremarkable. The self frame, once the track's own evolution is "
            f"subtracted, has {driver} {abs(sf['vs_field_s']):.3f} s behind what the field "
            f"found over the same window. They are not contradicting each other — they are "
            f"answering different questions. A problem affecting **both** cars is exactly the "
            f"thing the teammate frame cancels out and the self frame exposes. Run only the "
            f"comparison every telemetry project runs first, and you conclude the car is fine.")
    elif both["disagree"] is False:
        st.success(
            f"**Both frames tell the same story here.** The teammate gap is "
            f"{tm['gap_s']:+.3f} s, and against the field's own improvement over the same "
            f"window {driver} is {sf['vs_field_s']:+.3f} s. Nothing is being cancelled out of "
            f"one frame and into the other — which is what an ordinary Saturday looks like, "
            f"and why the case where they split is worth going and finding.")

    if not (driver == "VER" and str(lap_row["q_segment"]) == "Q2"):
        st.caption("The case this project was built around is Verstappen's Q2 lap, where the "
                   "two frames split. Pick **VER** and his Q2 lap in the sidebar to see it.")

    st.markdown("**The self frame's baseline moves under you.** Every car that goes round lays "
                "rubber, so a driver's own improvement means nothing until it is put next to "
                "what the rest of the grid found in the same window.")
    st.pyplot(fig_evolution(tables["evolution"], driver, mate), use_container_width=True)

    # ---------------------------------------------- 4. the third frame -------
    st.subheader("4. A model as a third frame — and its own blind spot")
    st.markdown(
        "Every frame above compares a lap to another *lap*. A fairer question is: given how "
        "this car arrived at this corner — its speed through the entry, its braking, its "
        "gear, the tyre and the track temperature — how fast **should** it have left? That is "
        f"a regression over {int(meta['model_dataset_rows']):,} corner traversals from "
        f"{int(meta['model_dataset_corners'])} distinct corners, and its residual reads as "
        "*slower than this car should have been given how it arrived*.")
    st.pyplot(fig_model_frame(tables["ablation"], tables["corner_error"],
                              float(meta["model_target_sd_kph"])), use_container_width=True)
    st.caption(
        f"Left: the same gradient-boosted model refitted on each block of features, scored on "
        f"circuits held out by date ({meta['model_test_events']}). Corner identity is worth "
        "nothing on a circuit the model has never seen — only entry telemetry and geometry "
        "transfer. Right: which is why the residual has to be de-biased per corner before two "
        "cars are compared, or you end up ranking drivers by the model's ignorance of the "
        "track. **And this frame is blind here too:** it conditions on entry speed, so a car "
        "that is slow everywhere arrives slow and is correctly predicted to leave slow. A "
        "car-wide deficit is already inside the input.")

    with st.expander("What the model actually leans on"):
        imp = tables["importance"].head(10).copy()
        imp.columns = ["Feature", "MAE increase when shuffled (km/h)", "sd"]
        st.dataframe(imp.round(2), hide_index=True, use_container_width=True)
        st.caption("Held-out permutation importance. The speed 25 m before the corner marker "
                   "carries the model almost single-handedly; corner radius is the only "
                   "geometry term that matters much.")

    # --------------------------------------------------- 5. the point --------
    st.subheader("5. What I take from it")
    st.markdown(
        """
* **Never diagnose from one lap.** Pick the frame before looking at the data, and write down
  what that frame cannot see.
* **When two frames disagree, that is data, not an error.** The disagreement localises the
  problem to whatever one of them differenced away.
* **Make the decomposition reconcile.** If the parts do not sum to the lap gap, they are parts
  of something else.
* **Run a control pair.** Two drivers in the same car — Leclerc against Sainz here — tell you
  how large a difference means nothing. That control killed my first attribution idea:
  splitting the lap by driving phase charged 92% of the control pair's gap to "braking", when
  all that differs between them is *where* they brake.

Everything on this page is stated as **consistent with**, never as a proven mechanical fault.
Telemetry records what the car did, not why. What it is genuinely good at is *eliminating* —
ruling out the stories that would each have left a fingerprint, and giving whatever is left a
shape. Setup sheets, damper traces and tyre temperatures would settle it, and none of them
are public.
"""
    )

    st.divider()
    st.caption(
        "Telemetry, timing and corner markers from "
        "[FastF1](https://github.com/theOehrly/Fast-F1) (MIT), which wraps the public F1 "
        "timing and car-telemetry feeds. No telemetry is redistributed here: the smallest "
        "unit shipped to this page is a 25 m stretch of one lap, derived from FastF1 output "
        f"at analysis time (fastf1 {meta['fastf1_version']}). Push laps are laps FastF1 rates "
        "accurate, not deleted, and within 7% of the session's best. Unofficial and not "
        "associated with the Formula 1 companies; F1, FORMULA 1 and related marks are trade "
        "marks of Formula One Licensing B.V."
    )


if __name__ == "__main__":
    main()
