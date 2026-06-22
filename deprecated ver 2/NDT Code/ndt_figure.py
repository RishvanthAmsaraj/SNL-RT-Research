"""
ndt_pdf.py
==========
Non-Decision Time (t0) bar chart for both HRT and SRT.

Reads pre-computed parameter tables produced by ddm_fit_final.py:
  hrt_fits.csv  -- per-participant HRT DDM parameters
  srt_fits.csv  -- per-participant SRT DDM parameters

This ensures t0 values are IDENTICAL to those used in all other figures
and analyses (correct contamination model, bounded L-BFGS-B polishing,
3rd-percentile t0 ceiling, proper RT filters).

Previous version re-fit from scratch using outdated methodology (pure Wald
MLE, min(RT) t0 bound, v capped at 20, unbounded polish) -- inconsistent
with everything else.

Output: ndt_barchart.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

matplotlib.rcParams.update({
    "font.family":  "Arial",
    "font.size":    13,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

# =============================================================================
# CONFIG
# =============================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
HRT_FITS_CSV = os.path.join(SCRIPT_DIR, 'DDM_hrt_fits.csv')
SRT_FITS_CSV = os.path.join(SCRIPT_DIR, 'DDM_srt_fits.csv')
OUTPUT_PATH  = os.path.join(SCRIPT_DIR, 'ndt_barchart.pdf')

SPEEDS       = [0, 75, 150]
SPEED_LABELS = ['0 deg/s', '75 deg/s', '150 deg/s']
SPEED_COLOR  = {
    0:   tuple(v/255 for v in (191, 230, 191)),
    75:  tuple(v/255 for v in (245, 191, 191)),
    150: tuple(v/255 for v in (191, 214, 249)),
}
SPEED_COLOR_DARK = {
    spd: tuple(min(1.0, v * 0.50) for v in col)
    for spd, col in SPEED_COLOR.items()
}
C_DARK = '#1A1A2E'

# =============================================================================
# LOAD PRE-COMPUTED FITS
# =============================================================================
df_hrt = pd.read_csv(HRT_FITS_CSV)
df_srt = pd.read_csv(SRT_FITS_CSV)

# t0 is stored in milliseconds in the CSVs
n_participants = df_hrt['pid'].nunique()
print(f"Loaded fits for {n_participants} participants.")

# =============================================================================
# PRINT SUMMARY TABLE
# =============================================================================
print(f"\n{'Participant':<12}", end="")
for rt in ['HRT', 'SRT']:
    for spd in SPEEDS:
        print(f"  {rt}@{spd}", end="")
print()
print("-" * (12 + 10 * 6))

for pid in sorted(df_hrt['pid'].unique()):
    print(f"{pid:<12}", end="")
    for df_fits in [df_hrt, df_srt]:
        for spd in SPEEDS:
            row = df_fits[(df_fits['pid'] == pid) &
                          (df_fits['spd'] == spd)]['t0']
            val = row.iloc[0] if len(row) else np.nan
            print(f"  {val:6.1f}" if not np.isnan(val) else "     NaN", end="")
    print()

# =============================================================================
# COMPUTE MEANS AND SDs
# =============================================================================
hrt_means = [df_hrt[df_hrt['spd'] == spd]['t0'].mean() for spd in SPEEDS]
hrt_sds   = [df_hrt[df_hrt['spd'] == spd]['t0'].std(ddof=1) for spd in SPEEDS]
srt_means = [df_srt[df_srt['spd'] == spd]['t0'].mean() for spd in SPEEDS]
srt_sds   = [df_srt[df_srt['spd'] == spd]['t0'].std(ddof=1) for spd in SPEEDS]

# Friedman test for speed effects on t0
hrt_vals = [df_hrt[df_hrt['spd'] == spd]['t0'].values for spd in SPEEDS]
srt_vals = [df_srt[df_srt['spd'] == spd]['t0'].values for spd in SPEEDS]
hrt_stat, hrt_p = stats.friedmanchisquare(*hrt_vals)
srt_stat, srt_p = stats.friedmanchisquare(*srt_vals)

print(f"\nHRT t0 Friedman: chi2={hrt_stat:.3f}  p={hrt_p:.4f}"
      f"{'  *' if hrt_p < 0.05 else ''}")
print(f"SRT t0 Friedman: chi2={srt_stat:.3f}  p={srt_p:.4f}"
      f"{'  *' if srt_p < 0.05 else ''}")

# =============================================================================
# PLOT -- side-by-side HRT and SRT t0
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
fig.suptitle('Non-Decision Time ($t_0$) by Target Speed\n'
             f'Group mean +/- 1 SD  (n = {n_participants} participants)',
             fontsize=14, fontweight='bold', y=0.99)

for ax, means, sds, vals_list, stat, p, rt_label, t0_min in [
    (axes[0], hrt_means, hrt_sds, hrt_vals, hrt_stat, hrt_p, 'HRT', 100),
    (axes[1], srt_means, srt_sds, srt_vals, srt_stat, srt_p, 'SRT',  40),
]:
    x     = np.arange(len(SPEEDS))
    bar_w = 0.45

    for i, spd in enumerate(SPEEDS):
        # Bar
        ax.bar(x[i], means[i], width=bar_w,
               color=SPEED_COLOR[spd],
               edgecolor=SPEED_COLOR_DARK[spd],
               linewidth=1.5, zorder=3)
        # Error bar
        ax.errorbar(x[i], means[i], yerr=sds[i],
                    fmt='none', color=C_DARK,
                    capsize=9, capthick=2.2, elinewidth=2.2, zorder=5)
        # Value label
        ax.text(x[i], means[i] + sds[i] + 3,
                f'{means[i]:.0f} +/- {sds[i]:.0f} ms',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=C_DARK)

        # Individual data points (jittered)
        np.random.seed(42)
        jitter = np.random.uniform(-0.10, 0.10, len(vals_list[i]))
        ax.scatter(x[i] + jitter, vals_list[i],
                   color=SPEED_COLOR_DARK[spd], s=30,
                   alpha=0.55, zorder=4, linewidths=0)

    # Physiological minimum reference line
    ax.axhline(t0_min, color='#999999', lw=1.2, ls=':', alpha=0.8)
    ax.text(len(SPEEDS) - 0.6, t0_min + 2,
            f'Physiol. min\n({t0_min} ms)',
            ha='right', va='bottom', fontsize=9, color='#777777', style='italic')

    # Friedman annotation
    sig_str = ('*' if p < 0.05 else 'n.s.')
    ax.set_title(f'{rt_label} Non-Decision Time\n'
                 f'Friedman p = {p:.3f} ({sig_str})',
                 fontsize=13, fontweight='bold', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(SPEED_LABELS, fontsize=12, fontweight='bold')
    ax.set_ylabel('$t_0$ (ms)', fontsize=13)
    ax.set_xlim(-0.5, len(SPEEDS) - 0.5)
    ax.set_ylim(0, max(means) + max(sds) + 35)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.11, wspace=0.30)
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight',
            facecolor='white', format='pdf')
plt.close(fig)
print(f"\nSaved: {OUTPUT_PATH}")
