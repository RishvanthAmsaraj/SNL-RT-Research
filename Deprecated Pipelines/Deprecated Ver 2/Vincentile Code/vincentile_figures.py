"""
vincentile_analysis.py  -- PDF edition
======================================
Produces THREE PDF figures:

  Figure 1 -- KDE overlay (row 0) + combined vincentile all-3-speeds (row 1)
  Figure 2 -- SRT histogram (row 0) + HRT histogram (row 1)
  Figure 3 -- Individual vincentile plots, one per speed condition (with SD bars)

Colour scheme (consistent across all figures):
  0  deg/s -> RGB(191,230,191)  pale green
  75 deg/s -> RGB(245,191,191)  pale red/pink
  150deg/s -> RGB(191,214,249)  pale blue

Font: Arial, enlarged throughout.
PDF output: dpi=300, pdf.fonttype=42 (editable text in Illustrator/Inkscape).
"""

import sys, glob, warnings, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches
import seaborn as sns
from pathlib import Path


# =============================================================================
# GLOBAL STYLE
# =============================================================================

matplotlib.rcParams.update({
    "font.family":       "Arial",
    "font.size":         13,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    14,
    "axes.labelsize":    13,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   12,
    "figure.titlesize":  15,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})


# =============================================================================
# CONFIGURATION
# =============================================================================

SPEEDS       = [0, 75, 150]
SPEED_LABELS = ["0 deg/s", "75 deg/s", "150 deg/s"]
N_BINS       = 20
HIST_BINS    = 40

# Physiological RT filters -- must match ddm_fit_final.py exactly
SRT_MIN, SRT_MAX = 80, 600    # ms  (express saccade floor = 80ms)
HRT_MIN, HRT_MAX = 150, 800   # ms

RT_RANGE_SRT = (SRT_MIN, SRT_MAX)
RT_RANGE_HRT = (HRT_MIN, HRT_MAX)

# Condition colours RGB(0-255) -> (0-1)
SPEED_COLOR = {
    0:   tuple(v/255 for v in (191, 230, 191)),
    75:  tuple(v/255 for v in (245, 191, 191)),
    150: tuple(v/255 for v in (191, 214, 249)),
}
# Darker version for lines/markers (multiply by 0.55)
SPEED_COLOR_DARK = {
    spd: tuple(min(1.0, v * 0.55) for v in col)
    for spd, col in SPEED_COLOR.items()
}

MEAN_COLOR = {"srt": "#2E86AB", "hrt": "#E84855"}
MEAN_LW    = 2.2
MEAN_MS    = 5


# =============================================================================
# HELPERS
# =============================================================================

def shorten_label(raw_label):
    import re
    return re.sub(r"_MASTER_Summary\d*$", "", raw_label, flags=re.IGNORECASE)


def compute_vincentiles(rt_array, n_bins=N_BINS):
    rt_clean = np.sort(rt_array[~np.isnan(rt_array)])
    if len(rt_clean) < n_bins:
        return np.full(n_bins, np.nan)
    return np.array([b.mean() for b in np.array_split(rt_clean, n_bins)])


# =============================================================================
# DATA AGGREGATION
# =============================================================================

def aggregate_participants(file_list):
    pool_store = {spd: {"srt": [], "hrt": []} for spd in SPEEDS}
    diff_store = {spd: [] for spd in SPEEDS}
    labels     = []
    missing    = {spd: {"srt": 0, "hrt": 0} for spd in SPEEDS}
    paired     = {spd: 0 for spd in SPEEDS}
    n_loaded   = 0

    for fpath in file_list:
        try:
            df = pd.read_csv(fpath)
        except Exception as exc:
            warnings.warn(f"Could not read {fpath}: {exc}")
            continue

        df_i = df[df["BlockType"] == "I"].copy()
        if df_i.empty:
            continue

        label = shorten_label(Path(fpath).stem)
        labels.append(label)
        n_loaded += 1

        for spd in SPEEDS:
            sub     = df_i[df_i["Speed_deg_per_s"] == spd]
            srt_all = sub["GazeSRT_ms"].values.astype(float)
            hrt_all = sub["HandRT_ms"].values.astype(float)

            # Apply physiological filters -- same as ddm_fit_final.py
            srt_mask = (~np.isnan(srt_all) &
                        (srt_all >= SRT_MIN) & (srt_all <= SRT_MAX))
            hrt_mask = (~np.isnan(hrt_all) &
                        (hrt_all >= HRT_MIN) & (hrt_all <= HRT_MAX))

            pool_store[spd]["srt"].append(srt_all[srt_mask])
            pool_store[spd]["hrt"].append(hrt_all[hrt_mask])
            missing[spd]["srt"] += int((~srt_mask).sum())
            missing[spd]["hrt"] += int((~hrt_mask).sum())

            # Paired difference: both must pass their respective filters
            paired_mask = srt_mask & hrt_mask
            paired[spd] += int(paired_mask.sum())
            diff = hrt_all[paired_mask] - srt_all[paired_mask]
            diff_store[spd].append(compute_vincentiles(diff))

    print(f"\n{'Condition':<12}{'SRT valid':>10}{'SRT miss':>10}"
          f"{'HRT valid':>10}{'HRT miss':>10}{'Paired':>8}")
    print("-" * 60)
    for spd in SPEEDS:
        sv = sum(len(a) for a in pool_store[spd]["srt"])
        hv = sum(len(a) for a in pool_store[spd]["hrt"])
        print(f"{spd:>3} deg/s   {sv:>9} {missing[spd]['srt']:>9} "
              f"{hv:>9} {missing[spd]['hrt']:>9} {paired[spd]:>7}")

    results = {}
    for spd in SPEEDS:
        results[spd] = {}
        for rt_type in ("srt", "hrt"):
            pooled = np.concatenate(pool_store[spd][rt_type])
            pooled = pooled[~np.isnan(pooled)]
            # FIX: range was hard-coded to RT_RANGE_SRT for BOTH rt types, so the
            # HRT histogram was binned over (80,600) instead of (150,800) -- it
            # dropped the >600 ms tail and placed bars on the wrong grid. Use the
            # range matching each rt type.
            hist_range = RT_RANGE_SRT if rt_type == "srt" else RT_RANGE_HRT
            counts, edges = np.histogram(pooled, bins=HIST_BINS,
                                         range=hist_range, density=True)
            centres = (edges[:-1] + edges[1:]) / 2
            results[spd][rt_type] = {
                "hist_x":    centres,
                "hist_mean": counts,
                "median_rt": float(np.median(pooled)),
                "pooled_rt": pooled,
                "labels":    labels,
                "n":         n_loaded,
            }

        diff_mat = np.array(diff_store[spd])
        n        = diff_mat.shape[0]
        results[spd]["diff"] = {
            "vinc_mean": np.nanmean(diff_mat, axis=0),
            # FIX: ddof=max(1, n-1) made numpy divide SS by (n - (n-1)) = 1,
            # inflating every between-subject SD by sqrt(n-1) ~ 3.87x for n=16.
            # Sample SD across participants uses ddof=1.
            "vinc_sd":   np.nanstd(diff_mat, axis=0, ddof=1),
            "vinc_x":    np.arange(1, N_BINS + 1),
            "labels":    labels,
            "n":         n,
        }

    return results


# =============================================================================
# PLOT HELPERS
# =============================================================================

def plot_histogram(ax, data, rt_type, rt_range):
    x, y_hist = data["hist_x"], data["hist_mean"]
    median_rt = data["median_rt"]
    mcol      = MEAN_COLOR[rt_type]
    width     = (x[1] - x[0]) * 0.85 if len(x) > 1 else 20
    tallest   = float(np.max(y_hist))

    ax.bar(x, y_hist, width=width, color=mcol, alpha=0.65,
           edgecolor="none", zorder=2)
    pooled = data["pooled_rt"]
    if len(pooled) >= 2:
        sns.kdeplot(pooled, ax=ax, color=mcol, linewidth=2.0,
                    linestyle="--", alpha=0.85, cut=0, bw_adjust=1.0, zorder=1)

    ax.axvline(median_rt, color=mcol, linewidth=1.4, linestyle=":", alpha=0.9, zorder=3)
    ax.text(median_rt, tallest * 1.10,
            f"median\n{median_rt:.0f} ms",
            ha="center", va="bottom", fontsize=12,
            fontweight="bold", color=mcol, zorder=4, clip_on=True)

    ax.set_xlim(rt_range)
    ax.set_xlabel("RT (ms)")
    ax.set_ylabel("Density")
    ax.set_ylim(0, tallest * 1.30)
    ax.axvline(rt_range[0], color='#aaaaaa', lw=0.5, ls='--', alpha=0.5)  # filter boundary


def plot_kde_overlay(ax, srt_pooled, hrt_pooled, spd):
    if len(srt_pooled) >= 2:
        sns.kdeplot(srt_pooled, ax=ax, color=MEAN_COLOR["srt"],
                    fill=True, alpha=0.45, linewidth=1.8,
                    cut=0, bw_adjust=1.0, label="SRT")
    if len(hrt_pooled) >= 2:
        sns.kdeplot(hrt_pooled, ax=ax, color=MEAN_COLOR["hrt"],
                    fill=True, alpha=0.45, linewidth=1.8,
                    cut=0, bw_adjust=1.0, label="HRT")

    ax.set_xlim((SRT_MIN - 20, SRT_MAX + 20))   # filtered SRT range with small margin
    ax.set_xlabel("RT (ms)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=12, frameon=False, loc="upper right")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.22)

    if len(srt_pooled) >= 2:
        med = np.median(srt_pooled)
        ax.axvline(med, color=MEAN_COLOR["srt"], linestyle=":", linewidth=1.5)
        ax.text(med, ax.get_ylim()[1] * 0.99,
                f"median  {med:.0f} ms",
                color=MEAN_COLOR["srt"], fontsize=12, fontweight="bold",
                ha="center", va="top")

    if len(hrt_pooled) >= 2:
        med = np.median(hrt_pooled)
        ax.axvline(med, color=MEAN_COLOR["hrt"], linestyle=":", linewidth=1.5)
        ax.text(med, ax.get_ylim()[1] * 0.87,
                f"median  {med:.0f} ms",
                color=MEAN_COLOR["hrt"], fontsize=12, fontweight="bold",
                ha="center", va="top")


def plot_combined_vincentile(ax, results):
    """All 3 conditions on one panel, no error bars, filled with condition colour."""
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.4, zorder=1)
    for spd, label in zip(SPEEDS, SPEED_LABELS):
        x    = results[spd]["diff"]["vinc_x"]
        mean = results[spd]["diff"]["vinc_mean"]
        col  = SPEED_COLOR[spd]
        dark = SPEED_COLOR_DARK[spd]
        ax.fill_between(x, mean, 0, color=col, alpha=0.50, zorder=2)
        ax.plot(x, mean, "o-", color=dark,
                markersize=4, linewidth=2.0, zorder=5, label=label)

    ax.set_xticks([1, 5, 10, 15, 20])   # explicit ticks including 20
    ax.set_xlabel("Vincentile bin")
    ax.set_ylabel("HRT \u2212 SRT (ms)")
    ax.legend(fontsize=12, frameon=False, loc="upper left")


def plot_vincentile_single(ax, data_diff, spd, show_errorbars=True):
    """Vincentile for one speed condition with condition colour."""
    x    = data_diff["vinc_x"]
    col  = SPEED_COLOR[spd]
    dark = SPEED_COLOR_DARK[spd]
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.4, zorder=1)
    if show_errorbars:
        ax.errorbar(x, data_diff["vinc_mean"], yerr=data_diff["vinc_sd"],
                    fmt="o-", color=dark, ecolor=dark,
                    elinewidth=1.4, capsize=4,
                    markersize=MEAN_MS, linewidth=MEAN_LW, zorder=5)
        ax.fill_between(x, data_diff["vinc_mean"], 0,
                        color=col, alpha=0.40, zorder=2)
    else:
        ax.plot(x, data_diff["vinc_mean"], "o-", color=dark,
                markersize=MEAN_MS, linewidth=MEAN_LW, zorder=5)
        ax.fill_between(x, data_diff["vinc_mean"], 0,
                        color=col, alpha=0.40, zorder=2)
    ax.set_xticks([1, 5, 10, 15, 20])   # explicit ticks including 20
    ax.set_xlabel("Vincentile bin")
    ax.set_ylabel("HRT \u2212 SRT (ms)")


# =============================================================================
# FIGURE BUILDERS
# =============================================================================

def build_figure_1(results, n_participants, title_suffix=""):
    """Figure 1: KDE overlay only -- 3 panels side by side."""
    n   = results[SPEEDS[0]]["diff"]["n"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"RT Distributions (SRT & HRT Overlay){title_suffix}",
                 fontsize=15, fontweight="bold")

    for col_idx, (spd, spd_label) in enumerate(zip(SPEEDS, SPEED_LABELS)):
        d    = results[spd]
        left = (col_idx == 0)
        ax   = axes[col_idx]
        plot_kde_overlay(ax, d["srt"]["pooled_rt"], d["hrt"]["pooled_rt"], spd)
        ax.set_facecolor(SPEED_COLOR[spd] + (0.25,))
        ax.set_title(spd_label, fontsize=14, fontweight="bold", pad=6)
        if not left:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
            leg = ax.get_legend()
            if leg: leg.remove()

    # Legend with SRT/HRT entries only
    handles = [
        matplotlib.patches.Patch(color=MEAN_COLOR["srt"], alpha=0.45, label="SRT KDE"),
        matplotlib.patches.Patch(color=MEAN_COLOR["hrt"], alpha=0.45, label="HRT KDE"),
    ]
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.32,
                        wspace=0.22)
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.08), ncol=2, fontsize=11,
               frameon=True, framealpha=0.95, edgecolor="#cccccc",
               title=f"n={n} participants", title_fontsize=11)
    return fig


def build_figure_4(results, n_participants, title_suffix=""):
    """Figure 4: combined vincentile all 3 speeds -- squared-off single panel."""
    n   = results[SPEEDS[0]]["diff"]["n"]
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"HRT \u2212 SRT Vincentile (All Speeds){title_suffix}",
                 fontsize=15, fontweight="bold")

    plot_combined_vincentile(ax, results)
    ax.set_facecolor("#EEEEEE")

    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.20)
    fig.text(0.5, 0.04, f"Group mean  |  n={n} participants",
             ha="center", va="bottom", fontsize=11, color="#555555")
    return fig


def build_figure_2(results, n_participants, title_suffix=""):
    """Figure 2: SRT histogram row + HRT histogram row."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                             gridspec_kw={"hspace": 0.12, "wspace": 0.18})
    fig.suptitle(f"RT Histograms{title_suffix}",
                 fontsize=15, fontweight="bold", y=0.99)

    for col_idx, (spd, spd_label) in enumerate(zip(SPEEDS, SPEED_LABELS)):
        d    = results[spd]
        left = (col_idx == 0)

        ax0 = axes[0, col_idx]
        plot_histogram(ax0, d["srt"], "srt", RT_RANGE_SRT)
        ax0.set_facecolor(SPEED_COLOR[spd] + (0.30,))
        ax0.set_xlabel("")
        ax0.tick_params(axis="x", labelbottom=False)
        ax0.set_title(spd_label, fontsize=14, fontweight="bold", pad=6)
        if not left:
            ax0.set_ylabel("")
            ax0.tick_params(axis="y", labelleft=False)

        ax1 = axes[1, col_idx]
        plot_histogram(ax1, d["hrt"], "hrt", RT_RANGE_HRT)
        ax1.set_facecolor(SPEED_COLOR[spd] + (0.20,))
        if not left:
            ax1.set_ylabel("")
            ax1.tick_params(axis="y", labelleft=False)

    n = results[SPEEDS[0]]["diff"]["n"]
    handles = [
        matplotlib.patches.Patch(color=MEAN_COLOR["srt"], alpha=0.65, label="SRT histogram"),
        matplotlib.patches.Patch(color=MEAN_COLOR["hrt"], alpha=0.65, label="HRT histogram"),
        matplotlib.lines.Line2D([0], [0], color=MEAN_COLOR["srt"], lw=2.0,
                                linestyle="--", label="KDE (SRT)"),
        matplotlib.lines.Line2D([0], [0], color=MEAN_COLOR["hrt"], lw=2.0,
                                linestyle="--", label="KDE (HRT)"),
    ]

    # Use subplots_adjust so we control space precisely -- tight_layout alone
    # doesn't reserve room for a figure-level legend
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.20, hspace=0.12, wspace=0.18)
    leg = fig.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.5, 0.03), ncol=4, fontsize=11,
                     frameon=True, framealpha=0.95, edgecolor="#cccccc",
                     title=f"n={n} participants", title_fontsize=11)

    for row_idx, label in enumerate(["SRT (Gaze RT)", "HRT (Hand RT)"]):
        bbox = axes[row_idx, 0].get_position()
        fig.text(0.01, (bbox.y0 + bbox.y1)/2, label,
                 va="center", ha="center", fontsize=12,
                 fontweight="bold", rotation=90, color="#333333")
    return fig


def build_figure_3(results, n_participants, title_suffix=""):
    """Figure 3: individual vincentile per speed -- matching OG panel style."""
    n   = results[SPEEDS[0]]["diff"]["n"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f"HRT \u2212 SRT Vincentile by Speed{title_suffix}",
        fontsize=15, fontweight="bold")

    for col_idx, (spd, spd_label) in enumerate(zip(SPEEDS, SPEED_LABELS)):
        ax = axes[col_idx]
        plot_vincentile_single(ax, results[spd]["diff"], spd, show_errorbars=True)
        ax.set_facecolor(SPEED_COLOR[spd] + (0.30,))
        ax.set_title(spd_label, fontsize=14, fontweight="bold", pad=8)
        # Show all 20 bins on x-axis, rotated so they don't crowd
        ax.set_xticks(range(1, N_BINS + 1))
        ax.set_xticklabels([str(i) for i in range(1, N_BINS + 1)],
                           fontsize=9, rotation=0)
        if col_idx > 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18,
                        wspace=0.28)
    fig.text(0.5, 0.03, f"Group mean \u00b1 1 SD  |  n={n} participants",
             ha="center", va="bottom", fontsize=11, color="#555555")
    return fig


# =============================================================================
# CSV EXPORT
# =============================================================================

def save_pooled_csv(file_list, output_path):
    frames = []
    for fpath in file_list:
        try:
            df = pd.read_csv(fpath)
            df.insert(0, "Participant", shorten_label(Path(fpath).stem))
            frames.append(df)
        except Exception as exc:
            warnings.warn(f"Could not load {fpath}: {exc}")
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(output_path, index=False)
        print(f"  Pooled CSV  -> {output_path}  ({len(combined)} rows, {len(frames)} participants)")


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_analysis(file_list, output="vincentile_results.pdf"):
    print(f"\n{'='*55}")
    print("  Vincentile & RT Distribution Analysis")
    print(f"{'='*55}")
    print(f"  Input files : {len(file_list)}")

    csv_output = str(Path(output).parent / "pooled_data.csv")
    save_pooled_csv(file_list, csv_output)

    results = aggregate_participants(file_list)
    n       = results[SPEEDS[0]]["diff"]["n"]
    base    = Path(output)

    for fig_fn, suffix in [
        (build_figure_1, "_fig1_kde_overlay"),
        (build_figure_2, "_fig2_histograms"),
        (build_figure_3, "_fig3_vincentile_by_speed"),
        (build_figure_4, "_fig4_combined_vincentile"),
    ]:
        out = str(base.parent / (base.stem + suffix + base.suffix))
        fig = fig_fn(results, n)
        fig.savefig(out, dpi=300, bbox_inches="tight", format="pdf")
        plt.close(fig)
        print(f"  Saved -> {out}")

    print(f"{'='*55}\n")


if __name__ == "__main__":

    # =========================================================================
    # EDIT THIS LINE ONLY
    # Windows : r"C:\Users\Rishv\Downloads\ParticipantData"
    # Mac/Linux: "/Users/Rishv/Downloads/ParticipantData"
    # =========================================================================

    INPUT_FOLDER = r"C:\Users\Rishv\Desktop\SNL Lab\Participant Data"  

    OUTPUT_FILE = str(Path(INPUT_FOLDER) / "vincentile_results.pdf")

    file_list = sorted(
        f for f in glob.glob(str(Path(INPUT_FOLDER) / "*.csv"))
        if Path(f).name != "pooled_data.csv"
    )

    if not file_list:
        print(f"ERROR: No CSV files found in:\n  {INPUT_FOLDER}")
        sys.exit(1)

    print(f"Found {len(file_list)} participant files:")
    for f in file_list:
        print(f"  {Path(f).name}")

    run_analysis(file_list, OUTPUT_FILE)
