"""
vincentile_analysis.py
======================
Reads one or more participant CSV files and produces a compact 3x3 figure:

  Row 1 — SRT (Gaze Reaction Time) histograms  + pooled KDE + peak annotation
  Row 2 — HRT (Hand Reaction Time) histograms  + pooled KDE + peak annotation
  Row 3 — Vincentile plots of HRT minus SRT    (group mean ± 1 SD only)

  Column 1 = 0 deg/s   |   Column 2 = 75 deg/s   |   Column 3 = 150 deg/s
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines
import matplotlib.patches
import seaborn as sns
from pathlib import Path


# =============================================================================
# GLOBAL STYLE SETTINGS
# =============================================================================

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})


# =============================================================================
# CONFIGURATION
# =============================================================================

SPEEDS       = [0, 75, 150]
SPEED_LABELS = ["0 deg/s", "75 deg/s", "150 deg/s"]

# Number of vincentile bins — finer resolution of the RT distribution
N_BINS    = 20

# Number of histogram bars
HIST_BINS = 80

# X-axis limits (ms) — identical for SRT and HRT so rows are visually comparable
RT_RANGE_SRT = (0, 800)
RT_RANGE_HRT = (0, 800)

# Colours: blue = SRT, red = HRT, purple = HRT-SRT difference
MEAN_COLOR = {"srt": "#2E86AB", "hrt": "#E84855", "diff": "#8E44AD"}

# Group mean line appearance
MEAN_LW = 2.2
MEAN_MS = 5


# =============================================================================
# HELPER: shorten_label
# "CMT009_MASTER_Summary" -> "CMT009"
# =============================================================================

def shorten_label(raw_label):
    import re
    return re.sub(r"_MASTER_Summary\d*$", "", raw_label, flags=re.IGNORECASE)


# =============================================================================
# HELPER: make_participant_colors
# Returns n visually distinct colours — used only for the legend entries.
# Scales automatically: tab20 for <=20 participants, hsv rainbow for more.
# =============================================================================

def make_participant_colors(n):
    if n <= 20:
        cmap = matplotlib.colormaps.get_cmap("tab20")
    else:
        cmap = matplotlib.colormaps.get_cmap("hsv")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


# =============================================================================
# CORE MATHS: compute_vincentiles
# =============================================================================

def compute_vincentiles(rt_array, n_bins=N_BINS):
    """
    1. Remove NaN.
    2. Sort smallest to largest.
    3. Split into n_bins equal-membership groups.
    4. Return the mean of each group.
    Returns all-NaN if there are fewer valid trials than bins.
    """
    rt_clean = np.sort(rt_array[~np.isnan(rt_array)])
    if len(rt_clean) < n_bins:
        return np.full(n_bins, np.nan)
    return np.array([b.mean() for b in np.array_split(rt_clean, n_bins)])


# =============================================================================
# CORE MATHS: compute_hist_density
# =============================================================================

def compute_hist_density(rt_array, edges):
    """
    Normalised histogram using fixed shared bin edges.
    density=True makes participants with different trial counts comparable.
    """
    rt_clean = rt_array[~np.isnan(rt_array)]
    counts, _ = np.histogram(rt_clean, bins=edges, density=True)
    return counts


# =============================================================================
# HELPER: load_participant
# =============================================================================

def load_participant(path):
    return pd.read_csv(path)


# =============================================================================
# MAIN DATA PIPELINE: aggregate_participants
# =============================================================================

def aggregate_participants(file_list):
    """
    Loops over every CSV, computes per-participant statistics, then aggregates
    across participants. Returns results[speed][rt_type] = { statistics }.

    Scales automatically to any number of files — no changes needed here when
    adding or removing participants.
    """

    rt_cols   = {"srt": "GazeSRT_ms", "hrt": "HandRT_ms"}
    rt_ranges = {"srt": RT_RANGE_SRT,  "hrt": RT_RANGE_HRT}

    # Fixed bin edges shared by all participants so bar heights can be averaged
    hist_edges = {
        rt: np.linspace(lo, hi, HIST_BINS + 1)
        for rt, (lo, hi) in rt_ranges.items()
    }

    # Accumulators — grow automatically as files are loaded
    hist_store = {spd: {rt: [] for rt in rt_cols} for spd in SPEEDS}
    diff_store = {spd: []       for spd in SPEEDS}
    pool_store = {spd: {rt: [] for rt in rt_cols} for spd in SPEEDS}
    labels     = []

    n_loaded = 0
    for fpath in file_list:
        try:
            df = load_participant(fpath)
        except Exception as exc:
            warnings.warn(f"Could not load {fpath}: {exc}")
            continue

        n_loaded += 1
        labels.append(shorten_label(Path(fpath).stem))

        for spd in SPEEDS:
            sub = df[df["Speed_deg_per_s"] == spd].copy()

            for rt_type, col in rt_cols.items():
                rt_vals  = sub[col].values.astype(float)
                rt_clean = rt_vals[~np.isnan(rt_vals)]

                # Per-participant histogram density (for group-mean bars)
                hist_store[spd][rt_type].append(
                    compute_hist_density(rt_vals, hist_edges[rt_type])
                )

                # Pool all valid RTs together across participants for the KDE
                pool_store[spd][rt_type].extend(rt_clean.tolist())

            # HRT-SRT difference — paired trials only (both values must be present)
            paired    = sub.dropna(subset=["GazeSRT_ms", "HandRT_ms"])
            diff_vals = (paired["HandRT_ms"].values.astype(float)
                         - paired["GazeSRT_ms"].values.astype(float))
            diff_store[spd].append(compute_vincentiles(diff_vals))

    if n_loaded == 0:
        raise ValueError("No valid files were loaded.")

    print(f"  Loaded {n_loaded} participant file(s).")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    results = {}
    for spd in SPEEDS:
        results[spd] = {}

        for rt_type in rt_cols:
            hist_mat = np.array(hist_store[spd][rt_type])  # (n_participants, HIST_BINS)
            edges    = hist_edges[rt_type]
            centres  = (edges[:-1] + edges[1:]) / 2

            # Group-mean bar heights — one value per bin
            hist_mean = np.nanmean(hist_mat, axis=0)

            # Peak RT: the bin centre where the group-mean bar is tallest.
            # This is the most commonly occurring RT across the group.
            peak_idx = int(np.argmax(hist_mean))
            peak_rt  = float(centres[peak_idx])
            peak_val = float(hist_mean[peak_idx])

            results[spd][rt_type] = {
                "hist_x":    centres,
                "hist_mean": hist_mean,
                "peak_rt":   peak_rt,    # RT at tallest bar — annotated on the plot
                "peak_val":  peak_val,   # height of tallest bar — used to position annotation
                "pooled_rt": np.array(pool_store[spd][rt_type]),
                "labels":    labels,
                "n":         hist_mat.shape[0],
            }

        # HRT-SRT difference vincentiles
        diff_mat = np.array(diff_store[spd])   # (n_participants, N_BINS)
        n        = diff_mat.shape[0]
        ddof     = max(1, n - 1)

        results[spd]["diff"] = {
            "vinc_mean":  np.nanmean(diff_mat, axis=0),
            # SD across participants — shows spread of individual values around the mean
            "vinc_sd":    np.nanstd(diff_mat, axis=0, ddof=ddof),
            "vinc_x":     np.arange(1, N_BINS + 1),
            "labels":     labels,
            "n":          n,
        }

    return results


# =============================================================================
# PLOTTING: plot_histogram
# Group-mean bars + pooled KDE + peak RT annotation.
# No individual participant traces.
# =============================================================================

def plot_histogram(ax, data, rt_type, rt_range):
    """
    Draws:
      zorder 1 — pooled KDE (smooth curve over all participants combined)
      zorder 2 — group-mean histogram bars
      zorder 3 — peak RT annotation (text + vertical dashed line)
    """
    x        = data["hist_x"]
    y_mean   = data["hist_mean"]
    peak_rt  = data["peak_rt"]
    peak_val = data["peak_val"]
    mcol     = MEAN_COLOR[rt_type]
    width    = (x[1] - x[0]) * 0.85 if len(x) > 1 else 20

    # Pooled KDE — one smooth curve representing the whole group
    pooled = data["pooled_rt"]
    if len(pooled) >= 2:
        sns.kdeplot(
            pooled, ax=ax,
            color=mcol, linewidth=2.0, linestyle="--",
            alpha=0.85, cut=0, bw_adjust=1.0,
            zorder=1,
        )

    # Group-mean bars
    ax.bar(x, y_mean, width=width, color=mcol, alpha=0.65,
           edgecolor="none", zorder=2)

    # ── Peak RT annotation ─────────────────────────────────────────────────────
    # Draw a thin vertical dashed line at the peak bin and place the RT value
    # just above the bar as a text label.
    # xytext offset pushes the label slightly above the bar top so it doesn't
    # sit on top of the bar and become unreadable.
    ann = ax.annotate(
        f"{peak_rt:.0f} ms",
        xy=(peak_rt, peak_val),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center", va="bottom",
        fontsize=8.5, fontweight="bold",
        color=mcol,
        zorder=3,
        annotation_clip=True,   # keeps the label inside the axes boundaries
    )
    ann.set_clip_on(True)
    # Thin vertical line at the peak for easy reading
    ax.axvline(peak_rt, color=mcol, linewidth=0.8, linestyle=":", alpha=0.6, zorder=1)

    ax.set_xlim(rt_range)
    ax.set_xlabel("RT (ms)")
    ax.set_ylabel("Density")
    # Add 20% headroom above the tallest bar so the peak label sits
    # comfortably inside the panel without touching the top edge.
    ax.set_ylim(0, peak_val * 1.20)


# =============================================================================
# PLOTTING: plot_vincentile
# Group mean ± 1 SD only — no individual participant traces.
# =============================================================================

def plot_vincentile(ax, data_diff):
    """
    Draws:
      zorder 1 — dashed zero reference line
      zorder 5 — group mean ± 1 SD (bold line + error bars)

    Individual participant traces have been removed so the focus stays on
    the overall group trend across the RT distribution.
    """
    x    = data_diff["vinc_x"]
    mcol = MEAN_COLOR["diff"]

    # Dashed zero line — above = hand slower than eyes, below = hand faster
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.4, zorder=1)

    # Group mean ± 1 SD
    ax.errorbar(
        x,
        data_diff["vinc_mean"],
        yerr=data_diff["vinc_sd"],
        fmt="o-",
        color=mcol, ecolor=mcol,
        elinewidth=1.4, capsize=4,
        markersize=MEAN_MS, linewidth=MEAN_LW,
        zorder=5,
    )

    ax.set_xticks(np.arange(1, N_BINS + 1))
    ax.set_xlabel("Vincentile bin")
    ax.set_ylabel("HRT \u2212 SRT (ms)")


# =============================================================================
# FIGURE ASSEMBLY: build_figure
# =============================================================================

def build_figure(results, n_participants, title_suffix=""):
    """
    Assembles the compact 3x3 figure.

    Scaling behaviour (automatic, no code changes needed):
      - Colour palette grows with n_participants (tab20 up to 20, hsv beyond)
      - Legend columns adjust to fit participant count
      - All statistics (mean, SD) are computed over however many files were loaded
    """
    colors = make_participant_colors(n_participants)

    # Very compact figure — minimal whitespace
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        f"RT Distributions & Vincentile Plots{title_suffix}",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # Minimal GridSpec margins — panels sit close together.
    # bottom=0.10 is enough for the 3-entry legend.
    # top=0.91 leaves a slim strip for the column headers.
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.30, wspace=0.28,
        top=0.91, bottom=0.13, left=0.07, right=0.97,
    )

    row_labels = ["SRT (Gaze RT)", "HRT (Hand RT)", "HRT \u2212 SRT\nVincentile"]

    # ── Draw all nine panels ─────────────────────────────────────────────────
    for col_idx, (spd, spd_label) in enumerate(zip(SPEEDS, SPEED_LABELS)):
        d = results[spd]

        ax0 = fig.add_subplot(gs[0, col_idx])
        plot_histogram(ax0, d["srt"], "srt", RT_RANGE_SRT)
        ax0.set_facecolor("#F0F6FF")

        ax1 = fig.add_subplot(gs[1, col_idx])
        plot_histogram(ax1, d["hrt"], "hrt", RT_RANGE_HRT)
        ax1.set_facecolor("#FFF0F0")
        # X-axis info removed from HRT row — same scale as SRT row above it.
        # Removing it saves space and avoids repeating identical tick labels.
        ax1.set_xlabel("")
        ax1.tick_params(axis="x", labelbottom=False)

        ax2 = fig.add_subplot(gs[2, col_idx])
        plot_vincentile(ax2, d["diff"])
        ax2.set_facecolor("#F5F0FF")

    # ── Speed labels — once above the top row, centred on each column ─────────
    # Axes are added column-major (col 0 rows 0-2, then col 1 rows 0-2, etc.)
    # so the top-row panels are at indices 0, 3, 6.
    for col_idx, spd_label in enumerate(SPEED_LABELS):
        ax_top = fig.axes[col_idx * 3]
        bbox   = ax_top.get_position()
        x_mid  = (bbox.x0 + bbox.x1) / 2
        y_top  = bbox.y1 + 0.012

        fig.text(x_mid, y_top, spd_label,
                 ha="center", va="bottom",
                 fontsize=13, fontweight="bold", color="#1A1A2E")

    # ── Rotated row labels on the far left ───────────────────────────────────
    # y positions tuned to sit at the vertical midpoint of each row.
    # These update naturally if figure height changes because they're expressed
    # in figure-fraction coordinates derived from the GridSpec positions.
    row_ys = []
    for row_idx in range(3):
        # Grab the left-most panel in this row and use its vertical midpoint
        ax_ref = fig.axes[row_idx]                    # col 0, rows 0/1/2
        bbox   = ax_ref.get_position()
        row_ys.append((bbox.y0 + bbox.y1) / 2)

    for label, y in zip(row_labels, row_ys):
        fig.text(0.01, y, label,
                 va="center", ha="center",
                 fontsize=10, fontweight="bold",
                 rotation=90, color="#333333")

    # ── Shared legend at the bottom — summary entries only ───────────────────
    # Individual participant colours are omitted: the focus is the group average.
    legend_handles = [
        matplotlib.patches.Patch(color=MEAN_COLOR["srt"], alpha=0.8,
                                 label="SRT group mean"),
        matplotlib.patches.Patch(color=MEAN_COLOR["hrt"], alpha=0.8,
                                 label="HRT group mean"),
        matplotlib.lines.Line2D([0], [0], color=MEAN_COLOR["srt"],
                                 lw=2.0, linestyle="--",
                                 label="Pooled KDE (SRT)"),
        matplotlib.lines.Line2D([0], [0], color=MEAN_COLOR["hrt"],
                                 lw=2.0, linestyle="--",
                                 label="Pooled KDE (HRT)"),
        matplotlib.lines.Line2D([0], [0], color=MEAN_COLOR["diff"],
                                 lw=2.2, marker="o", ms=6,
                                 label="HRT\u2212SRT mean \u00b1 1 SD"),
    ]

    n = results[SPEEDS[0]]["diff"]["n"]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,                        # all 5 entries on one row
        fontsize=9,
        frameon=True, framealpha=0.9, edgecolor="#cccccc",
        title=f"n={n} participants",
        title_fontsize=9,
    )

    return fig


# =============================================================================
# ENTRY POINT: run_analysis
# =============================================================================

def run_analysis(file_list, output="vincentile_results.png"):
    print(f"\n{'='*55}")
    print("  Vincentile & RT Distribution Analysis")
    print(f"{'='*55}")
    print(f"  Input files    : {len(file_list)}")
    print(f"  Vincentile bins: {N_BINS}")
    print(f"  Histogram bins : {HIST_BINS}")

    results = aggregate_participants(file_list)
    n       = results[SPEEDS[0]]["diff"]["n"]

    suffix = (f" \u2014 {shorten_label(Path(file_list[0]).stem)}" if len(file_list) == 1
              else f" \u2014 {n} participants")

    fig = build_figure(results, n_participants=n, title_suffix=suffix)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved -> {output}")
    print(f"{'='*55}\n")


# =============================================================================
# SCRIPT ENTRY
# =============================================================================

if __name__ == "__main__":

    INPUT_FOLDER = r"C:\Users\Rishv\Downloads\Participant Data"  

    OUTPUT_FILE = r"C:\Users\Rishv\Downloads\vincentile_results_Standard _Deviation.png"

    # Collect every .csv in the folder, sorted alphabetically by filename
    files = sorted(glob.glob(str(Path(INPUT_FOLDER) / "*.csv")))

    if not files:
        print(f"No CSV files found in: {INPUT_FOLDER}")
        print("Check that the folder path is correct and contains .csv files.")
        sys.exit(1)

    run_analysis(files, output=OUTPUT_FILE)
