"""Cloud classification — an interactive feature-separability explorer.

Public showcase app. It ships a derived slice of a private repository: the parsed
UCI *Cloud* table (10 raw columns plus the four features my pipeline engineers) and
the separability ladder that pipeline produced.

Everything above ``main()`` is plain Python: functions that take arrays and frames
and return arrays, frames and matplotlib figures, with no Streamlit calls inside
them. ``main()`` reads the widgets, calls those functions and renders. That split is
deliberate — the compute layer is testable without a browser or a Streamlit runtime.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # the module stays importable in plain Python, for testing
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - only hit outside Streamlit
    st = None

matplotlib.use("Agg")

FEATURES_FILE = "clouds_features.csv"
SEPARABILITY_FILE = "raw-column-separability.csv"

TARGET = "class"


def data_dir() -> Path:
    """Locate the shipped data, whether running from disk or from stlite's virtual FS.

    Under stlite the app is mounted somewhere the host chooses and ``__file__`` is not
    guaranteed, so the directory next to this module is tried first and the working
    directory second, rather than being assumed at import time.
    """
    here = globals().get("__file__")
    candidates = []
    if here:
        candidates.append(Path(here).resolve().parent / "data")
    candidates += [Path.cwd() / "data", Path("data"), Path.cwd()]
    for candidate in candidates:
        if (candidate / FEATURES_FILE).is_file():
            return candidate
    return candidates[0]

# The three features the random forest is actually fed, from the pipeline config.
PIPELINE_FEATURES = ("log_asm", "IR_norm_range", "entropy_x_contrast")

# One colour language, the same one the notebook uses: blue = image 1 / class 0,
# orange = image 2 / class 1, slate = a raw column, amber = a column I engineered.
CLASS0, CLASS1 = "#2f6fae", "#d4682b"
RAW_C, ENG_C = "#6b8496", "#c98a1f"
GOOD, BAD = "#3f8f5f", "#bf3f38"
INK, MUTED, GRID = "#1b1f23", "#5f6b76", "#dde2e6"

# Axis labels with units. The file is un-normalised 1989 AVHRR imagery, so the
# brightness columns are raw sensor counts; the texture columns are dimensionless
# except entropy, which is in nats (the ASM/entropy pair satisfies H >= -ln(ASM)).
AXIS_LABELS = {
    "visible_min": "visible_min — darkest pixel in the super-pixel (AVHRR counts)",
    "visible_max": "visible_max — brightest pixel in the super-pixel (AVHRR counts)",
    "visible_mean": "visible_mean — mean visible brightness (AVHRR counts)",
    "visible_mean_distribution": "visible_mean_distribution — spread of the visible mean (unitless)",
    "visible_contrast": "visible_contrast — visible contrast (squared AVHRR counts)",
    "visible_second_angular_momentum": "visible_second_angular_momentum — ASM (unitless, 0–1]",
    "visible_entropy": "visible_entropy — Shannon entropy of the visible super-pixel (nats)",
    "IR_min": "IR_min — coldest pixel in the super-pixel (AVHRR counts)",
    "IR_max": "IR_max — warmest pixel in the super-pixel (AVHRR counts)",
    "IR_mean": "IR_mean — mean infrared brightness (AVHRR counts)",
    "log_asm": "log_asm — log(angular second moment) (log units)",
    "entropy_x_contrast": "entropy_x_contrast — entropy × contrast (nats × squared counts)",
    "IR_range": "IR_range — IR_max − IR_min (AVHRR counts)",
    "IR_norm_range": "IR_norm_range — (IR_max − IR_min) / IR_mean (unitless ratio)",
}

CLASS_NAMES = {0: "image 1 (class 0)", 1: "image 2 (class 1)"}


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_features(path: Path | str | None = None) -> pd.DataFrame:
    """Load the parsed super-pixel table.

    One row per super-pixel: the ten raw columns from the file, the four features
    the pipeline engineers, and the ``class`` label (which of the two source images
    the super-pixel came from).
    """
    path = Path(path) if path is not None else data_dir() / FEATURES_FILE
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"{path} has no '{TARGET}' column")
    frame[TARGET] = frame[TARGET].astype(int)
    return frame


def load_separability(path: Path | str | None = None) -> pd.DataFrame:
    """Load the separability ladder produced by the pipeline's own notebook run."""
    path = Path(path) if path is not None else data_dir() / SEPARABILITY_FILE
    frame = pd.read_csv(path)
    expected = {"column", "kind", "stump_cv_accuracy", "fraction_overlapping"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    return frame.sort_values("stump_cv_accuracy", ascending=False).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Every modellable column, in file order: raw columns first, then engineered."""
    return [c for c in frame.columns if c != TARGET]


def axis_label(column: str) -> str:
    """Axis label with units for a column, falling back to the bare name."""
    return AXIS_LABELS.get(column, column)


# ---------------------------------------------------------------------------
# the decision stump: one threshold on one column
# ---------------------------------------------------------------------------

def stump_accuracy(values: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Score the better of the two one-threshold rules on a single column.

    A stump splits at ``threshold`` and must then choose which side predicts class 1.
    Both choices are evaluated and the better one is returned; because the labels are
    binary, flipping the side flips every prediction, so the two accuracies sum to 1.

    Returns a dict with the accuracy, the direction ('above' means "predict image 2
    when the value exceeds the threshold"), the predicted labels and the 2x2 counts.
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)

    predict_above = (values > threshold).astype(int)
    accuracy_above = float((predict_above == labels).mean())

    if accuracy_above >= 0.5:
        direction, predicted, accuracy = "above", predict_above, accuracy_above
    else:
        direction, predicted, accuracy = "below", 1 - predict_above, 1.0 - accuracy_above

    matrix = np.zeros((2, 2), dtype=int)
    for true_class in (0, 1):
        for predicted_class in (0, 1):
            matrix[true_class, predicted_class] = int(
                ((labels == true_class) & (predicted == predicted_class)).sum()
            )

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "direction": direction,
        "n_correct": int(round(accuracy * labels.size)),
        "n_total": int(labels.size),
        "predicted": predicted,
        "confusion_matrix": matrix,
    }


def best_stump(values: np.ndarray, labels: np.ndarray) -> dict:
    """Find the accuracy-optimal single threshold on one column, exactly.

    Sorts once and sweeps every midpoint between consecutive distinct values with a
    cumulative count, so this is O(n log n) and exhaustive — not a search. The
    returned dict has the same shape as :func:`stump_accuracy`.
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)

    order = np.argsort(values, kind="mergesort")
    sorted_values, sorted_labels = values[order], labels[order]
    n_total = values.size
    n_ones = int(sorted_labels.sum())
    n_zeros = n_total - n_ones

    boundary = sorted_values[1:] > sorted_values[:-1]
    if not boundary.any():  # a constant column admits no split at all
        majority = max(n_zeros, n_ones) / n_total
        result = stump_accuracy(values, labels, float(sorted_values[0]))
        result["accuracy"] = max(result["accuracy"], majority)
        return result

    left_ones = np.cumsum(sorted_labels)[:-1]
    left_size = np.arange(1, n_total)
    left_zeros = left_size - left_ones

    # "left predicts 0, right predicts 1", and its mirror image.
    correct_above = left_zeros + (n_ones - left_ones)
    correct_below = left_ones + (n_zeros - left_zeros)
    correct = np.maximum(correct_above, correct_below)

    correct = np.where(boundary, correct, -1)
    best = int(np.argmax(correct))
    threshold = float((sorted_values[best] + sorted_values[best + 1]) / 2.0)
    return stump_accuracy(values, labels, threshold)


def class_overlap(values: np.ndarray, labels: np.ndarray) -> dict:
    """The value range the two classes share, and how many rows sit inside it.

    This is the notebook's definition: the interval between the higher of the two
    class minima and the lower of the two class maxima. When the classes do not
    touch at all the interval is empty, and the gap between them is reported
    instead — any threshold inside that gap is a perfect classifier.
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    class0, class1 = values[labels == 0], values[labels == 1]

    low = max(class0.min(), class1.min())
    high = min(class0.max(), class1.max())
    separated = low > high

    if separated:
        n_inside = int(((values > high) & (values < low)).sum())
        gap = (float(high), float(low))
    else:
        n_inside = int(((values >= low) & (values <= high)).sum())
        gap = None

    return {
        "low": float(low),
        "high": float(high),
        "separated": bool(separated),
        "gap": gap,
        "gap_width": float(low - high) if separated else 0.0,
        "n_rows_inside": n_inside,
        "fraction_inside": n_inside / values.size,
        "class0_range": (float(class0.min()), float(class0.max())),
        "class1_range": (float(class1.min()), float(class1.max())),
    }


def threshold_range(values: np.ndarray) -> dict:
    """Slider bounds and step for one column.

    The bounds are the plotted window (:func:`display_window`), not the raw
    min/max: ``IR_norm_range`` spans 987 units because a handful of rows have a
    near-zero denominator, and a slider over that range would move in useless
    jumps while the interesting thresholds all sit between -2 and +2. Thresholds
    beyond the window classify almost every row the same way, so nothing
    informative is out of reach.
    """
    values = np.asarray(values, dtype=float)
    window = display_window(values)
    low, high = window["low"], window["high"]
    span = high - low
    if span <= 0:
        return {"low": low, "high": high, "step": 1.0, "trimmed": window["trimmed"]}

    raw = span / 400.0
    magnitude = 10.0 ** np.floor(np.log10(raw))
    step = float(magnitude * min(m for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw))
    return {
        "low": float(np.floor(low / step) * step),
        "high": float(np.ceil(high / step) * step),
        "step": step,
        "trimmed": window["trimmed"],
    }


def default_threshold(values: np.ndarray, labels: np.ndarray) -> float:
    """A sensible starting threshold: the accuracy-optimal one, on the slider's grid."""
    bounds = threshold_range(values)
    optimal = best_stump(values, labels)["threshold"]
    snapped = round_to_step(optimal, bounds["step"])
    return float(np.clip(snapped, bounds["low"], bounds["high"]))


def round_to_step(value: float, step: float) -> float:
    """Snap a value onto the slider's grid, so the widget and the maths agree."""
    if step <= 0:
        return float(value)
    return float(round(value / step) * step)


# ---------------------------------------------------------------------------
# display windows — heavy tails would otherwise squash every plot
# ---------------------------------------------------------------------------

def display_window(values: np.ndarray, keep: float | None = None,
                   fence: float = 3.0) -> dict:
    """Pick axis limits that show the body of a distribution, not just its tails.

    ``entropy_x_contrast`` runs 0 to 14,037 with a median of 294, and
    ``IR_norm_range`` spans -380 to +607 with a median of 0.24, so on their full
    range both render as a single spike. The window is Tukey's far-out fence,
    ``[q25 - 3·IQR, q75 + 3·IQR]``, clamped to the data — a rule that leaves
    well-behaved columns untouched (``IR_mean``, which is bimodal by construction,
    keeps its full range) and trims only genuinely long tails. Rows outside the
    window are counted and reported; the histogram stacks them into the end bins
    rather than dropping them. ``keep`` (the current threshold) is always inside.
    """
    values = np.asarray(values, dtype=float)
    data_low, data_high = float(values.min()), float(values.max())

    q25, q75 = (float(q) for q in np.quantile(values, [0.25, 0.75]))
    iqr = q75 - q25
    low = max(data_low, q25 - fence * iqr)
    high = min(data_high, q75 + fence * iqr)
    trimmed = (low > data_low) or (high < data_high)

    if high <= low:  # a degenerate column: fall back to the full range
        low, high, trimmed = data_low, data_high, False

    n_outside = int(((values < low) | (values > high)).sum())

    if keep is not None and np.isfinite(keep):
        low, high = min(low, float(keep)), max(high, float(keep))

    pad = 0.04 * (high - low) if high > low else 1.0
    return {
        "low": low,
        "high": high,
        "xlim": (low - pad, high + pad),
        "trimmed": bool(trimmed),
        "n_outside": n_outside,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def wrap_title(text: str, width: int = 62) -> str:
    """Wrap a headline so it fits a ~700px figure instead of being cut off.

    matplotlib does not wrap titles, and a truncated headline is worse than a
    two-line one. Width is in characters, tuned for bold text at ~11pt on a
    7.2-inch figure.
    """
    return "\n".join(
        textwrap.fill(line, width=width) for line in text.split("\n")
    )


def _style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def distribution_figure(frame: pd.DataFrame, column: str, threshold: float) -> plt.Figure:
    """Two class histograms of one column, with the stump threshold drawn on top."""
    values = frame[column].to_numpy(dtype=float)
    labels = frame[TARGET].to_numpy(dtype=int)

    scored = stump_accuracy(values, labels, threshold)
    overlap = class_overlap(values, labels)
    window = display_window(values, keep=threshold)

    bins = np.linspace(window["low"], window["high"], 61)
    clipped = np.clip(values, window["low"], window["high"])

    fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=110)
    fig.patch.set_facecolor("white")
    _style(ax)

    for cls, colour in ((0, CLASS0), (1, CLASS1)):
        ax.hist(clipped[labels == cls], bins=bins, color=colour, alpha=0.62,
                edgecolor="white", linewidth=0.3, label=CLASS_NAMES[cls])

    if overlap["separated"]:
        ax.axvspan(overlap["gap"][0], overlap["gap"][1], color=GOOD, alpha=0.18, zorder=0)
    else:
        ax.axvspan(max(overlap["low"], window["low"]), min(overlap["high"], window["high"]),
                   color=BAD, alpha=0.11, zorder=0)

    top = ax.get_ylim()[1] * 1.28
    ax.set_ylim(0, top)
    ax.axvline(threshold, color=INK, lw=1.8, ls="--")
    ax.annotate(f"threshold {threshold:,.6g}", xy=(threshold, top * 0.97),
                xytext=(4, 0), textcoords="offset points",
                fontsize=9, color=INK, weight="bold", va="top")

    side = "above" if scored["direction"] == "above" else "at or below"
    verdict_colour = GOOD if scored["accuracy"] >= 0.99 else (
        BAD if scored["accuracy"] < 0.75 else INK)

    ax.set_title(
        wrap_title(
            f"One threshold on {column} classifies "
            f"{scored['n_correct']:,} of {scored['n_total']:,} super-pixels correctly "
            f"({scored['accuracy']:.1%})"
        ),
        loc="left", fontsize=11.5, weight="bold", color=verdict_colour, pad=10,
    )
    caption = f"rule: predict image 2 when {column} is {side} the threshold"
    if window["trimmed"]:
        caption += (f"  ·  long tail trimmed: {window['n_outside']} row(s) outside "
                    f"this window are stacked into the end bins")
    ax.set_xlabel(f"{axis_label(column)}\n{wrap_title(caption, width=84)}",
                  fontsize=9.5, color=INK)
    ax.set_ylabel("super-pixels", fontsize=9.5, color=INK)
    ax.set_xlim(*window["xlim"])
    ax.legend(fontsize=9, frameon=False, loc="upper right")

    fig.tight_layout()
    return fig


def scatter_figure(frame: pd.DataFrame, x_column: str, y_column: str,
                   threshold: float) -> plt.Figure:
    """Two features against each other, coloured by class, with the threshold drawn."""
    labels = frame[TARGET].to_numpy(dtype=int)
    x_values = frame[x_column].to_numpy(dtype=float)
    y_values = frame[y_column].to_numpy(dtype=float)

    x_window = display_window(x_values, keep=threshold)
    y_window = display_window(y_values)
    hidden = int(((x_values < x_window["low"]) | (x_values > x_window["high"])
                  | (y_values < y_window["low"]) | (y_values > y_window["high"])).sum())

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=110)
    fig.patch.set_facecolor("white")
    _style(ax)

    for cls, colour in ((0, CLASS0), (1, CLASS1)):
        mask = labels == cls
        ax.scatter(x_values[mask], y_values[mask], s=12, alpha=0.45, color=colour,
                   edgecolors="none", label=CLASS_NAMES[cls])

    x_overlap = class_overlap(x_values, labels)
    if x_overlap["separated"]:
        ax.axvspan(x_overlap["gap"][0], x_overlap["gap"][1], color=GOOD, alpha=0.16,
                   zorder=0)
        ax.annotate(f"no super-pixel of either image\nfalls in this "
                    f"{x_overlap['gap_width']:,.6g}-unit corridor",
                    xy=(float(np.mean(x_overlap["gap"])), y_window["xlim"][0]),
                    xytext=(0, 8), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=GOOD, weight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85,
                          "pad": 2})

    ax.axvline(threshold, color=INK, lw=1.6, ls="--")
    ax.annotate(f"threshold on {x_column}", xy=(threshold, y_window["xlim"][1]),
                xytext=(5, -6), textcoords="offset points",
                fontsize=8.5, color=INK, va="top")

    if x_column == y_column:
        headline = f"{x_column} against itself — pick a second feature to see the plane"
    else:
        headline = (f"{x_column} separates the two images; "
                    f"{y_column} mostly does not"
                    if _separates(x_values, labels) and not _separates(y_values, labels)
                    else f"Where the two images sit in the {x_column} × {y_column} plane")

    ax.set_title(wrap_title(headline), loc="left", fontsize=11.5, weight="bold",
                 color=INK, pad=10)
    note = f"  ·  {hidden} row(s) outside this window are not drawn" if hidden else ""
    ax.set_xlabel(wrap_title(f"{axis_label(x_column)}{note}", width=82),
                  fontsize=9.5, color=INK)
    ax.set_ylabel(axis_label(y_column), fontsize=9.5, color=INK)
    ax.set_xlim(*x_window["xlim"])
    ax.set_ylim(*y_window["xlim"])
    ax.legend(fontsize=9, frameon=False, loc="best", markerscale=1.8)

    fig.tight_layout()
    return fig


def _separates(values: np.ndarray, labels: np.ndarray) -> bool:
    """True when one threshold on this column gets essentially everything right."""
    return best_stump(values, labels)["accuracy"] >= 0.99


def ladder_figure(separability: pd.DataFrame, highlight: str | None = None) -> plt.Figure:
    """The separability ladder, with the currently selected column called out."""
    ladder = separability.sort_values("stump_cv_accuracy").reset_index(drop=True)
    positions = np.arange(len(ladder))
    colours = [ENG_C if kind == "engineered" else RAW_C for kind in ladder["kind"]]
    edges = [BAD if col in PIPELINE_FEATURES else "none" for col in ladder["column"]]
    widths = [1.8 if col in PIPELINE_FEATURES else 0.0 for col in ladder["column"]]

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=110)
    fig.patch.set_facecolor("white")
    _style(ax)

    ax.barh(positions, ladder["stump_cv_accuracy"], xerr=ladder["stump_cv_std"],
            color=colours, edgecolor=edges, linewidth=widths,
            error_kw={"ecolor": MUTED, "lw": 1.0, "capsize": 3})

    if highlight in set(ladder["column"]):
        row = int(ladder.index[ladder["column"] == highlight][0])
        ax.axhspan(row - 0.48, row + 0.48, facecolor="#eceff2", edgecolor="none",
                   zorder=0)
        ax.barh([row], [ladder.loc[row, "stump_cv_accuracy"]], color="none",
                edgecolor=INK, linewidth=2.2, zorder=4)

    ax.axvline(0.5, color=MUTED, ls="--", lw=1.1)
    ax.text(0.505, len(ladder) - 0.35, "chance (classes are 1024 / 1024)",
            fontsize=8, color=MUTED, va="center")
    ax.axvline(1.0, color=GOOD, ls=":", lw=1.3)

    for y, (value, spread, column) in enumerate(
            zip(ladder["stump_cv_accuracy"], ladder["stump_cv_std"], ladder["column"])):
        ax.text(value + spread + 0.014, y, f"{value:.3f}", va="center", fontsize=8.5,
                color=INK if column == highlight else (
                    BAD if column in PIPELINE_FEATURES else MUTED),
                weight="bold" if column in (highlight, *PIPELINE_FEATURES) else "normal")

    ax.set_yticks(positions)
    ax.set_yticklabels(ladder["column"], fontsize=9)
    for label, column in zip(ax.get_yticklabels(), ladder["column"]):
        if column == highlight:
            label.set_fontweight("bold")
            label.set_color(INK)
    ax.set_xlim(0.44, 1.10)
    ax.set_ylim(-0.8, len(ladder) - 0.2)
    ax.set_xlabel("5-fold cross-validated accuracy of a depth-1 decision tree\n"
                  "(one threshold, one column — from the pipeline's own run, seed 42)",
                  fontsize=9.5, color=INK)
    fig.suptitle(
        wrap_title("Two raw infrared columns already score a perfect 1.000, and every "
                   "feature I engineered ranks below five columns the model never sees",
                   width=72),
        x=0.012, ha="left", fontsize=11, weight="bold", color=INK)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=RAW_C, label="raw column from the file"),
        plt.Rectangle((0, 0), 1, 1, fc=ENG_C, label="feature I engineered"),
        plt.Rectangle((0, 0), 1, 1, fc="white", ec=BAD, lw=1.8, label="fed to the random forest"),
        plt.Rectangle((0, 0), 1, 1, fc="white", ec=INK, lw=2.2, label="selected above"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit layer — widgets in, figures out, no computation of its own
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover - exercised by the browser, not by tests
    st.set_page_config(page_title="Cloud classification — separability explorer",
                       layout="centered")

    frame = load_features()
    separability = load_separability()
    columns = feature_columns(frame)
    labels = frame[TARGET].to_numpy(dtype=int)

    st.title("A pipeline you can actually trust")
    st.markdown(
        "The 1989 UCI *Cloud* dataset is two satellite scenes, each diced into 1,024 "
        "super-pixels and summarised by ten texture statistics; the label is which of "
        "the two scenes a super-pixel came from. Before fitting anything I wanted a "
        "floor — how much of that target does **one threshold on one column** already "
        "explain? All of it, as it turns out. The two scenes were calibrated "
        "differently, so the interesting work here is the engineering, not the "
        "modelling.\n\n"
        "Pick a column below and move the threshold yourself. The full pipeline, its "
        "config, its 125 tests and the walkthrough notebook live in a private repository."
    )

    st.divider()

    left, right = st.columns(2)
    with left:
        feature = st.selectbox(
            "Feature to threshold", columns,
            index=columns.index("IR_mean") if "IR_mean" in columns else 0,
            help="Ten raw columns from the file, then the four features my pipeline builds.",
        )
    with right:
        partners = [c for c in columns if c != feature]
        preferred = "entropy_x_contrast" if "entropy_x_contrast" in partners else partners[0]
        second = st.selectbox(
            "Second feature, for the 2-D view", partners,
            index=partners.index(preferred),
        )

    values = frame[feature].to_numpy(dtype=float)
    bounds = threshold_range(values)

    state_key = f"threshold::{feature}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_threshold(values, labels)

    help_text = ("Everything on one side of the threshold is called image 1, everything "
                 "on the other image 2.")
    if bounds["trimmed"]:
        help_text += (" The slider covers the plotted window; this column has a long "
                      "tail, and thresholds out there classify almost every row the "
                      "same way.")
    threshold = st.slider(
        f"Decision-stump threshold on {feature}",
        min_value=float(bounds["low"]), max_value=float(bounds["high"]),
        step=float(bounds["step"]), key=state_key, help=help_text,
    )

    scored = stump_accuracy(values, labels, threshold)
    best = best_stump(values, labels)
    overlap = class_overlap(values, labels)
    ladder_row = separability.loc[separability["column"] == feature]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy at this threshold", f"{scored['accuracy']:.3f}",
                          f"{scored['n_correct']:,} / {scored['n_total']:,} rows")
    metric_cols[1].metric("Best this column can do", f"{best['accuracy']:.3f}",
                          f"at {best['threshold']:,.6g}")
    if not ladder_row.empty:
        metric_cols[2].metric("5-fold CV accuracy (pipeline run)",
                              f"{float(ladder_row['stump_cv_accuracy'].iloc[0]):.3f}",
                              f"± {float(ladder_row['stump_cv_std'].iloc[0]):.3f}",
                              delta_color="off")
    metric_cols[3].metric("Rows in the class overlap", f"{overlap['n_rows_inside']:,}",
                          f"{overlap['fraction_inside']:.1%} of 2,048", delta_color="off")

    st.pyplot(distribution_figure(frame, feature, threshold))

    if overlap["separated"]:
        st.success(
            f"**The two classes do not touch on {feature}.** Class 0 runs "
            f"{overlap['class0_range'][0]:,.6g} to {overlap['class0_range'][1]:,.6g}, "
            f"class 1 runs {overlap['class1_range'][0]:,.6g} to "
            f"{overlap['class1_range'][1]:,.6g}, and the "
            f"{overlap['gap_width']:,.6g}-unit corridor between them holds "
            f"{overlap['n_rows_inside']} of 2,048 rows. Any threshold inside that "
            f"corridor is a perfect classifier."
        )
    else:
        st.info(
            f"**The classes overlap on {feature}** between "
            f"{overlap['low']:,.6g} and {overlap['high']:,.6g} — "
            f"{overlap['n_rows_inside']:,} of 2,048 rows "
            f"({overlap['fraction_inside']:.0%}) sit inside the shared range, so no "
            f"single threshold can separate them."
        )

    st.pyplot(scatter_figure(frame, feature, second, threshold))
    st.pyplot(ladder_figure(separability, highlight=feature))

    st.markdown(
        "### What this page is actually saying\n"
        "The top two bars are **raw columns, at exactly 1.000**. `IR_mean` and `IR_max` "
        "have zero rows in the value range the two classes share — the classes do not "
        "merely separate well, they do not touch. The three features the random forest "
        "is fed (red outlines) rank below five columns it never sees: my feature "
        "engineering did not extract signal, it threw the clean signal away and handed "
        "the forest a harder problem than the one sitting in the file. That is a "
        "negative result about my own pipeline, and it stays in.\n\n"
        "So the honest headline is not the forest's 0.9988 test accuracy. It is that "
        "this dataset's target is *which of two differently calibrated photographs a "
        "tile came from*, and that measures the data, not the model. What was worth "
        "building was the machinery around it: parsing a 1989 email that fights back, "
        "catching that the file's own documentation lists the columns in the wrong "
        "order (2,045 of 2,048 rows violate `min ≤ mean ≤ max` as documented), fixing "
        "an off-by-one that silently dropped a super-pixel, and proving by running it "
        "that the pipeline reproduces."
    )

    with st.expander("Data, provenance and what is shipped here"):
        st.markdown(
            "- **Dataset** — Collard, P. (1989). *Cloud* [Dataset]. UCI Machine Learning "
            "Repository. <https://doi.org/10.24432/C5359Z>. Licensed **CC BY 4.0**, so "
            "redistribution is permitted with attribution.\n"
            "- **`clouds_features.csv`** — the raw file parsed by the pipeline's own "
            "`create_dataset` and `generate_features` steps: 2,048 super-pixels "
            "(1,024 per class) × 10 raw columns + 4 engineered features. The column "
            "*names* are my correction, not the file's preamble.\n"
            "- **`raw-column-separability.csv`** — the separability ladder, copied "
            "verbatim from the pipeline's committed report. The CV figures on this page "
            "come from that file; the live accuracy and the optimal threshold are "
            "recomputed in your browser from the shipped table.\n"
            "- Nothing on this page is illustrative. Every number is computed from those "
            "two files."
        )
        st.dataframe(separability, hide_index=True)


if __name__ == "__main__":
    main()
