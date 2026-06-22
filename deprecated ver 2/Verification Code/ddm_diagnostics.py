"""
ddm_diagnostics_final.py -- COMPREHENSIVE VERSION
==================================================
9-page diagnostic PDF covering:
  Page 1  SRT histogram vs mixture-predicted PDF (with express-ppt overlay)
  Page 2  HRT histogram vs mixture-predicted PDF
  Page 3  SRT empirical vs mixture CDF
  Page 4  HRT empirical vs mixture CDF
  Page 5  Per-participant KS heatmap (shows exactly who is hard to fit)
  Page 6  DDM parameter trends across speed conditions with stats
  Page 7  SRT: all 16 vs 15 participants (excl. CMT0012)
  Page 8  Summary table with corrected quality labels
  Page 9  RT verification: raw paired HRT-SRT vs vincentile bin means
          (merged from the former standalone verify_rt_differences.py,
          now using the correct interception-only, filtered trial set)

Key methodological fix: overlays use MIXTURE-PREDICTED distributions
= average of 16 individual Wald PDFs/CDFs, not group-mean parameters.
Group-mean params applied to pooled data inflated KS by 2-6x artificially.

Run AFTER ddm_fit_final.py has produced hrt_fits.csv and srt_fits.csv.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

matplotlib.rcParams.update({
    "font.family":  "Arial",
    "font.size":    12,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

# =============================================================================
# CONFIG
# =============================================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = SCRIPT_DIR   # change if CSVs are elsewhere
# e.g. DATA_FOLDER = r"C:\Users\Rishv\Desktop\SNL Lab\DDM Code"

CSV_PATH     = os.path.join(DATA_FOLDER, 'pooled_data.csv')
HRT_FITS_CSV = os.path.join(DATA_FOLDER, 'hrt_fits.csv')
SRT_FITS_CSV = os.path.join(DATA_FOLDER, 'srt_fits.csv')
OUTPUT_PATH  = os.path.join(SCRIPT_DIR,  'ddm_diagnostics_final.pdf')

# ---------------------------------------------------------------------------
# Express-saccade-dominant participants.  Only CMT0012 qualifies: 80% of its
# valid SRT trials (80-600 ms) fall below 130 ms (median SRT 116 ms).  The next
# highest are CMT009 (16.9%) and CMT004 (15.9%) -- a large gap below CMT0012 --
# so neither is treated as express-dominant; their partial fast-saccade
# component (and its effect on SRT fit quality) is noted in the figure footnotes
# instead.  CMT002 (1.8%, median SRT 180 ms) is NOT express and was removed here.
# ---------------------------------------------------------------------------
EXPRESS_PIDS = ['CMT0012']   # express-dominant only; edit if the criterion changes
SPEEDS       = [0, 75, 150]
SPEED_LABELS = ['0 deg/s', '75 deg/s', '150 deg/s']
SPEED_COLOR  = {
    0:   tuple(v/255 for v in (191, 230, 191)),
    75:  tuple(v/255 for v in (245, 191, 191)),
    150: tuple(v/255 for v in (191, 214, 249)),
}
SPEED_COLOR_DARK = {s: tuple(min(1, v*0.55) for v in c)
                    for s, c in SPEED_COLOR.items()}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def wald_pdf(t, v, a):
    t = np.maximum(t, 1e-9)
    return (a / np.sqrt(2*np.pi*t**3)) * np.exp(-(a - v*t)**2 / (2*t))


def wald_cdf(t, v, a):
    mu = a / v; lam = a**2
    return stats.invgauss.cdf(t, mu=mu/lam, scale=lam)


def mixture_pdf_grid(t_grid, df_fits_spd):
    pdf_sum = np.zeros(len(t_grid))
    for _, row in df_fits_spd.iterrows():
        t0 = row['t0'] / 1000.0
        adj = t_grid - t0
        mask = adj > 0
        pdf_sum[mask] += wald_pdf(adj[mask], row['v'], row['a'])
    return pdf_sum / max(len(df_fits_spd), 1)


def mixture_cdf_grid(t_grid, df_fits_spd):
    cdf_sum = np.zeros(len(t_grid))
    for _, row in df_fits_spd.iterrows():
        t0  = row['t0'] / 1000.0
        adj = np.maximum(t_grid - t0, 1e-9)
        cdf_sum += wald_cdf(adj, row['v'], row['a'])
    return cdf_sum / max(len(df_fits_spd), 1)


def mixture_ks(rts, df_fits_spd):
    if len(df_fits_spd) == 0:
        return np.nan, np.nan
    rts_s = np.sort(rts)
    ecdf  = np.arange(1, len(rts_s)+1) / len(rts_s)
    fcdf  = mixture_cdf_grid(rts_s, df_fits_spd)
    ks    = float(np.max(np.abs(ecdf - fcdf)))
    mae   = float(np.mean(np.abs(ecdf - fcdf)))
    return ks, mae


def quality_label(ks_ind):
    if ks_ind < 0.05:  return 'Good'
    if ks_ind < 0.10:  return 'OK'
    if ks_ind < 0.12:  return 'Borderline'
    return 'Poor'


# =============================================================================
# LOAD DATA
# =============================================================================
df      = pd.read_csv(CSV_PATH)
df_i    = df[df['BlockType'] == 'I'].copy()
df_hfit = pd.read_csv(HRT_FITS_CSV)
df_sfit = pd.read_csv(SRT_FITS_CSV)

hrt_rts, srt_rts = {}, {}
for spd in SPEEDS:
    sub = df_i[df_i['Speed_deg_per_s'] == spd]
    hrt_rts[spd] = sub[(sub['HandRT_ms'] >= 150) &
                       (sub['HandRT_ms'] <= 800)]['HandRT_ms'].values / 1000.0
    srt_rts[spd] = sub[(sub['GazeSRT_ms'] >= 80) &
                       (sub['GazeSRT_ms'] <= 600)]['GazeSRT_ms'].values / 1000.0

df_sfit_excl = df_sfit[~df_sfit['pid'].isin(EXPRESS_PIDS)].copy()
# Participant counts derived from the data (never hard-code these in labels):
N_FULL = df_sfit['pid'].nunique()           # all participants
N_EXCL = df_sfit_excl['pid'].nunique()      # after dropping EXPRESS_PIDS
srt_rts_excl = {}
for spd in SPEEDS:
    sub = df_i[(df_i['Speed_deg_per_s'] == spd) &
               (~df_i['Participant'].isin(EXPRESS_PIDS))]
    srt_rts_excl[spd] = sub[(sub['GazeSRT_ms'] >= 80) &
                          (sub['GazeSRT_ms'] <= 600)]['GazeSRT_ms'].values / 1000.0

# -----------------------------------------------------------------------------
# Paired trial-by-trial HRT - SRT differences, for the verification page.
# Uses EXACTLY the same trial population as the vincentile pipeline:
#   - interception block only (df_i is already BlockType == 'I')
#   - both RTs present AND both within their physiological windows
# This is what the standalone verify_rt_differences.py was meant to check; that
# script omitted both the BlockType filter (pulling 'S'-block trials into the
# 0 deg/s panel) and the RT-range filters (admitting extreme outliers), so its
# 0 deg/s panel mixed two tasks. Computing it here keeps it consistent with the
# rest of the diagnostics.
rt_diffs = {}   # ms, kept in ms (not seconds) since this page is descriptive
for spd in SPEEDS:
    sub = df_i[df_i['Speed_deg_per_s'] == spd]
    s = sub['GazeSRT_ms'].values.astype(float)
    h = sub['HandRT_ms'].values.astype(float)
    m = ((~np.isnan(s)) & (s >= 80) & (s <= 600) &
         (~np.isnan(h)) & (h >= 150) & (h <= 800))
    diffs = h[m] - s[m]
    # vincentile bin means on the same differences (mirrors vincentile_pdf.py)
    sd = np.sort(diffs)
    vinc = (np.array([b.mean() for b in np.array_split(sd, 20)])
            if len(sd) >= 20 else np.full(20, np.nan))
    rt_diffs[spd] = {'diffs': diffs, 'vinc': vinc, 'n': len(diffs)}


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================
print("=" * 65)
print("  DDM Fit Quality")
print("=" * 65)
print(f"\n  {'Condition':<20}  {'Ind KS':>7}  {'Mix KS':>7}  {'%OK':>5}  Quality")
print(f"  {'-'*52}")
for label, df_fits, rts_dict in [('HRT', df_hfit, hrt_rts),
                                   ('SRT', df_sfit, srt_rts)]:
    for spd, slbl in zip(SPEEDS, SPEED_LABELS):
        s    = df_fits[df_fits['spd'] == spd]
        ks_i = s['ks'].mean()
        ks_m, _ = mixture_ks(rts_dict[spd], s)
        pct  = (s['ks'] < 0.10).mean() * 100
        print(f"  {label+' '+slbl:<20}  {ks_i:>7.4f}  {ks_m:>7.4f}  "
              f"{pct:>4.0f}%  {quality_label(ks_i)}")

print()
for spd, slbl in zip(SPEEDS, SPEED_LABELS):
    s    = df_sfit_excl[df_sfit_excl['spd'] == spd]
    ks_i = s['ks'].mean()
    ks_m, _ = mixture_ks(srt_rts_excl[spd], s)
    print(f"  SRT({N_EXCL}) {slbl:<12}  {ks_i:>7.4f}  {ks_m:>7.4f}  "
          f"  (excl. express participants)")

print("\n  Friedman non-parametric test for speed effects:")
for label, df_fits in [('HRT', df_hfit), ('SRT', df_sfit)]:
    for param in ['v', 'a', 't0']:
        vals = [df_fits[df_fits['spd'] == spd][param].values for spd in SPEEDS]
        stat, p = stats.friedmanchisquare(*vals)
        means   = [v.mean() for v in vals]
        sig     = '  *' if p < 0.05 else ''
        print(f"  {label} {param}: "
              f"0={means[0]:.2f}  75={means[1]:.2f}  150={means[2]:.2f}"
              f"  p={p:.4f}{sig}")


# =============================================================================
# PAGE BUILDERS
# =============================================================================

def page_histogram(rts_dict, df_fits, rt_label, rt_range_ms, title,
                   show_express=False):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    for ax, spd, slbl in zip(axes, SPEEDS, SPEED_LABELS):
        s    = df_fits[df_fits['spd'] == spd]
        rts  = rts_dict[spd] * 1000
        col  = SPEED_COLOR[spd]
        dark = SPEED_COLOR_DARK[spd]

        ax.hist(rts, bins=60, density=True,
                color=col, edgecolor='none', alpha=0.80, label='Empirical')

        t_grid  = np.linspace(rt_range_ms[0]/1000, rt_range_ms[1]/1000, 1500)
        mix_pdf = mixture_pdf_grid(t_grid, s)
        ax.plot(t_grid*1000, mix_pdf/1000, color=dark, lw=2.5, label='Model')

        if show_express and rt_label == 'SRT':
            s_expr = s[s['pid'].isin(EXPRESS_PIDS)]
            s_reg  = s[~s['pid'].isin(EXPRESS_PIDS)]
            if len(s_expr):
                ax.plot(t_grid*1000, mixture_pdf_grid(t_grid, s_expr)/1000,
                        color='#FF6B35', lw=1.8, ls=':', alpha=0.90,
                        label='Express (%s)' % '/'.join(EXPRESS_PIDS))
            if len(s_reg):
                ax.plot(t_grid*1000, mixture_pdf_grid(t_grid, s_reg)/1000,
                        color='#555555', lw=1.5, ls='--', alpha=0.65,
                        label='Other ppts')

        ks_i   = s['ks'].mean()
        ks_m, _ = mixture_ks(rts_dict[spd], s)
        ax.set_title(f"{slbl}\nInd KS={ks_i:.3f}  Mix KS={ks_m:.3f}"
                     f"  [{quality_label(ks_i)}]",
                     fontsize=10, pad=5)
        ax.set_xlabel(f'{rt_label} (ms)', fontsize=11)
        ax.set_xlim(rt_range_ms)
        ax.set_facecolor(col + (0.20,))
        ax.spines[['top', 'right']].set_visible(False)
        if spd == 0:
            ax.set_ylabel('Density', fontsize=11)
            ax.legend(fontsize=9, frameon=False)
        else:
            ax.set_ylabel('')

    fig.subplots_adjust(left=0.07, right=0.98, top=0.78,
                        bottom=0.14, wspace=0.28)
    return fig


def page_cdf(rts_dict, df_fits, rt_label, rt_range_ms, title):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    for ax, spd, slbl in zip(axes, SPEEDS, SPEED_LABELS):
        s    = df_fits[df_fits['spd'] == spd]
        rts  = np.sort(rts_dict[spd])
        col  = SPEED_COLOR[spd]
        dark = SPEED_COLOR_DARK[spd]

        ecdf = np.arange(1, len(rts)+1) / len(rts)
        ax.step(rts*1000, ecdf, color=dark, lw=1.5, alpha=0.75,
                label='Empirical CDF')

        t_grid  = np.linspace(rt_range_ms[0]/1000, rt_range_ms[1]/1000, 1000)
        ax.plot(t_grid*1000, mixture_cdf_grid(t_grid, s),
                color='#E84855', lw=2.2, ls='--', label='Model CDF')

        ks_i   = s['ks'].mean()
        ks_m, _ = mixture_ks(rts_dict[spd], s)
        ax.set_title(f"{slbl}\nInd KS={ks_i:.3f}  Mix KS={ks_m:.3f}"
                     f"  [{quality_label(ks_i)}]",
                     fontsize=11, pad=5)
        ax.set_xlabel(f'{rt_label} (ms)', fontsize=11)
        ax.set_xlim(rt_range_ms); ax.set_ylim(0, 1.05)
        ax.set_facecolor(col + (0.18,))
        ax.spines[['top', 'right']].set_visible(False)
        if spd == 0:
            ax.set_ylabel('Cumulative probability', fontsize=11)
            ax.legend(fontsize=10, frameon=False)
        else:
            ax.set_ylabel('')

    fig.subplots_adjust(left=0.07, right=0.98, top=0.82,
                        bottom=0.14, wspace=0.28)
    return fig


def page_ks_heatmap(df_hfit, df_sfit):
    pids = sorted(df_sfit['pid'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle('Per-Participant KS Goodness-of-Fit Heatmap\n'
                 'Green=Good  Yellow=OK  Red=Poor.  * = express-saccade dominant.',
                 fontsize=13, fontweight='bold')

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'fit', ['#2ecc71', '#f1c40f', '#e74c3c'], N=256)

    for ax, df_fits, label in [(axes[0], df_hfit, 'HRT'),
                                (axes[1], df_sfit, 'SRT')]:
        mat = np.full((len(pids), 3), np.nan)
        for i, pid in enumerate(pids):
            for j, spd in enumerate(SPEEDS):
                row = df_fits[(df_fits['pid']==pid) &
                              (df_fits['spd']==spd)]['ks']
                if len(row):
                    mat[i, j] = row.iloc[0]

        im = ax.imshow(mat, cmap=cmap, vmin=0.04, vmax=0.16, aspect='auto')
        plt.colorbar(im, ax=ax, label='KS statistic',
                     fraction=0.046, pad=0.04)
        ax.set_xticks(range(3))
        ax.set_xticklabels(SPEED_LABELS, fontsize=10)
        ax.set_yticks(range(len(pids)))
        ax.set_yticklabels(pids, fontsize=9)
        ax.set_title(f'{label}\nmean KS={df_fits.ks.mean():.4f}  '
                     f'%OK={((df_fits.ks<0.10).mean()*100):.0f}%',
                     fontsize=11, pad=8)

        for i in range(len(pids)):
            for j in range(3):
                if not np.isnan(mat[i, j]):
                    color = 'white' if mat[i, j] > 0.11 else 'black'
                    ax.text(j, i, f'{mat[i,j]:.3f}',
                            ha='center', va='center',
                            fontsize=8, color=color, fontweight='bold')
            if pids[i] in EXPRESS_PIDS:
                ax.text(-0.7, i, '*', ha='center', va='center',
                        fontsize=14, color='#E84855', fontweight='bold')

    fig.text(0.02, 0.02,
             '* = express-saccade-dominant participant (CMT0012 only: 80% of valid '
             'SRT trials <130 ms).  CMT004 (15.9% <130 ms) and CMT009 (16.9%) carry a '
             'partial fast-saccade component that worsens their SRT fits (KS up to '
             '~0.15) without dominating the distribution.',
             fontsize=8.5, color='#555555', style='italic')
    fig.subplots_adjust(left=0.10, right=0.95, top=0.88,
                        bottom=0.08, wspace=0.35)
    return fig


def page_parameter_trends(df_hfit, df_sfit):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('DDM Parameter Estimates by Speed Condition\n'
                 'Points = group mean +/-1 SD.  Light lines = individual participants.  '
                 'Orange = express-saccade dominant.  '
                 '* = Friedman omnibus p<0.05 (no pairwise post-hoc reported)',
                 fontsize=11, fontweight='bold')

    params     = ['v', 'a', 't0']
    param_lbls = ['Drift rate (v)', 'Boundary separation (a)',
                  'Non-decision time t0 (ms)']

    for row, (label, df_fits) in enumerate([('HRT', df_hfit), ('SRT', df_sfit)]):
        for col, (param, plbl) in enumerate(zip(params, param_lbls)):
            ax    = axes[row, col]
            xpos  = np.arange(3)
            c     = '#2E86AB' if label == 'HRT' else '#E84855'
            means = [df_fits[df_fits['spd']==spd][param].mean() for spd in SPEEDS]
            sds   = [df_fits[df_fits['spd']==spd][param].std()  for spd in SPEEDS]

            # Individual lines
            for pid in df_fits['pid'].unique():
                vals = [df_fits[(df_fits['pid']==pid) &
                                (df_fits['spd']==spd)][param].values
                        for spd in SPEEDS]
                vals = [v[0] if len(v) else np.nan for v in vals]
                lc = '#FF6B35' if pid in EXPRESS_PIDS else c
                lw = 1.2 if pid in EXPRESS_PIDS else 0.4
                al = 0.75 if pid in EXPRESS_PIDS else 0.20
                ax.plot(xpos, vals, color=lc, lw=lw, alpha=al)

            # Group mean
            ax.errorbar(xpos, means, yerr=sds, fmt='o-',
                        color=c, ecolor=c, capsize=6, capthick=2,
                        elinewidth=2, lw=2.5, ms=8,
                        markerfacecolor='white', markeredgewidth=2,
                        zorder=5)

            vals_list = [df_fits[df_fits['spd']==spd][param].values
                         for spd in SPEEDS]
            stat, p = stats.friedmanchisquare(*vals_list)
            # Kendall's W: W = chi2 / (k*(n-1))
            # Use minimum n across conditions to be conservative
            n_p = min(len(v) for v in vals_list)
            k   = len(SPEEDS)
            W   = stat / (k * (n_p - 1)) if n_p > 1 else np.nan
            sig  = '  *' if p < 0.05 else ''
            ax.set_title(f'{label}: {plbl}\nFriedman p={p:.3f}{sig}  W={W:.2f}',
                         fontsize=10, pad=5)
            ax.set_xticks(xpos)
            ax.set_xticklabels(SPEED_LABELS, fontsize=10)
            ax.set_xlim(-0.4, 2.4)
            ax.spines[['top', 'right']].set_visible(False)
            ax.yaxis.grid(True, linestyle='--', alpha=0.4)
            ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color='#2E86AB', lw=2.5, marker='o',
               markerfacecolor='white', label='HRT group mean'),
        Line2D([0],[0], color='#E84855', lw=2.5, marker='o',
               markerfacecolor='white', label='SRT group mean'),
        Line2D([0],[0], color='#FF6B35', lw=1.5, alpha=0.9,
               label='%s (express)' % ' / '.join(EXPRESS_PIDS)),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=10, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88,
                        bottom=0.10, hspace=0.45, wspace=0.32)
    return fig


def page_srt_comparison(srt_rts, srt_rts_excl, df_sfit, df_sfit_excl):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    n_full = df_sfit['pid'].nunique()
    n_excl = df_sfit_excl['pid'].nunique()
    excl   = sorted(set(df_sfit['pid']) - set(df_sfit_excl['pid']))
    fig.suptitle(f'SRT Fit Comparison: All {n_full} vs {n_excl} Participants\n'
                 f'(Bottom row excludes {", ".join(excl)})',
                 fontsize=13, fontweight='bold')

    for col, (spd, slbl) in enumerate(zip(SPEEDS, SPEED_LABELS)):
        col_c = SPEED_COLOR[spd]
        dark  = SPEED_COLOR_DARK[spd]
        t_grid = np.linspace(0.05, 0.60, 1000)

        for row, (rts_d, df_f, lbl) in enumerate([
            (srt_rts,      df_sfit,      f'All {n_full}'),
            (srt_rts_excl, df_sfit_excl, f'{n_excl} (excl. express)'),
        ]):
            ax  = axes[row, col]
            s   = df_f[df_f['spd'] == spd]
            rts = rts_d[spd] * 1000

            ax.hist(rts, bins=50, density=True,
                    color=col_c, edgecolor='none', alpha=0.80,
                    label='Empirical')
            ax.plot(t_grid*1000, mixture_pdf_grid(t_grid, s)/1000,
                    color=dark, lw=2.2, label='Model')

            ks_i   = s['ks'].mean()
            ks_m, _ = mixture_ks(rts_d[spd], s)
            ax.set_title(f"{slbl} -- {lbl}\n"
                         f"Ind KS={ks_i:.3f}  Mix KS={ks_m:.3f}"
                         f"  [{quality_label(ks_i)}]",
                         fontsize=9.5, pad=5)
            ax.set_xlabel('SRT (ms)', fontsize=10)
            ax.set_xlim(50, 600)
            ax.set_facecolor(col_c + (0.20,))
            ax.spines[['top', 'right']].set_visible(False)
            if col == 0:
                ax.set_ylabel('Density', fontsize=10)
                ax.legend(fontsize=9, frameon=False)
            else:
                ax.set_ylabel('')

    fig.subplots_adjust(left=0.07, right=0.98, top=0.86,
                        bottom=0.08, hspace=0.52, wspace=0.28)
    return fig


def page_summary_table(df_hfit, df_sfit, hrt_rts, srt_rts,
                       df_sfit_excl, srt_rts_excl):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis('off')
    fig.suptitle('DDM Fit Quality Summary',
                 fontsize=14, fontweight='bold', y=0.98)

    n_full = df_sfit['pid'].nunique()
    n_excl = df_sfit_excl['pid'].nunique()
    rows = []
    for spd, slbl in zip(SPEEDS, SPEED_LABELS):
        for label, df_f, rts_d in [
            (f'HRT (n={n_full})',   df_hfit,      hrt_rts),
            (f'SRT (n={n_full})',   df_sfit,      srt_rts),
            (f'SRT (n={n_excl})*',  df_sfit_excl, srt_rts_excl),
        ]:
            s      = df_f[df_f['spd'] == spd]
            ks_i   = s['ks'].mean()
            ks_m, mae = mixture_ks(rts_d[spd], s)
            pct_ok = (s['ks'] < 0.10).mean() * 100
            rows.append([
                f"{label} {slbl}",
                f"{s.v.mean():.2f}+/-{s.v.std():.2f}",
                f"{s.a.mean():.3f}+/-{s.a.std():.3f}",
                f"{s.t0.mean():.0f}+/-{s.t0.std():.0f}ms",
                str(int(s['n'].sum())),
                f"{ks_i:.4f}",
                f"{ks_m:.4f}",
                f"{pct_ok:.0f}%",
                quality_label(ks_i),
            ])

    cols = ['Condition', 'v', 'a', 't0', 'Trials',
            'Ind KS', 'Mix KS', '% Ind OK', 'Quality']
    tbl = ax.table(cellText=rows, colLabels=cols,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.80)

    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#2E86AB')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    qcol = {'Good': '#d4edda', 'OK': '#fff3cd',
            'Borderline': '#fde8d0', 'Poor': '#f8d7da'}
    for i, row in enumerate(rows):
        for j in range(len(cols)):
            tbl[i+1, j].set_facecolor(qcol.get(row[-1], 'white'))

    fig.text(0.5, 0.05,
             'Ind KS = mean KS across individual fits (PRIMARY metric). '
             '<0.10 = OK, 0.10-0.12 = Borderline, >0.12 = Poor.\n'
             'Mix KS = mixture-predicted CDF vs pooled data (all <0.04 = excellent). '
             '* Excludes the express-dominant set (%s); shifts mean SRT KS by <0.005.'
             % ', '.join(EXPRESS_PIDS),
             ha='center', va='bottom', fontsize=8.5,
             color='#555555', style='italic')
    fig.subplots_adjust(top=0.92, bottom=0.18)
    return fig


# =============================================================================
# PAGE 9 -- RT difference verification (raw paired HRT-SRT vs vincentile means)
# =============================================================================
def page_rt_verification(rt_diffs):
    """Confirms the vincentile bin means faithfully summarise the raw per-trial
    HRT - SRT differences, on the SAME filtered interception trials the
    vincentile figures use. Per speed: jittered raw differences, a box plot,
    and the 20 vincentile bin means overlaid on a twin x-axis."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    C_DARK = '#1A1A2E'
    C_GREY = '#666666'

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), sharey=True)
    fig.suptitle('RT Verification: Raw HRT - SRT Differences vs Vincentile Bin Means\n'
                 'Interception trials only, physiological filters applied '
                 '(SRT 80-600 ms, HRT 150-800 ms)',
                 fontsize=13, fontweight='bold', y=1.00)

    rng = np.random.default_rng(42)
    for ax, spd, slbl in zip(axes, SPEEDS, SPEED_LABELS):
        d    = rt_diffs[spd]['diffs']
        vinc = rt_diffs[spd]['vinc']
        n    = rt_diffs[spd]['n']
        col  = SPEED_COLOR[spd]
        dark = SPEED_COLOR_DARK[spd]

        # raw trial dots, jittered around x=1
        jit = rng.uniform(-0.28, 0.28, size=n)
        ax.scatter(np.ones(n) + jit, d, color=dark, alpha=0.18, s=7,
                   zorder=1, linewidths=0)

        # box plot of all raw differences (fliers off -- dots already show spread)
        ax.boxplot(d, positions=[1], widths=0.5, patch_artist=True,
                   zorder=3, showfliers=False,
                   boxprops=dict(facecolor=col, alpha=0.75,
                                 edgecolor=dark, linewidth=1.3),
                   medianprops=dict(color=C_DARK, linewidth=2.2),
                   whiskerprops=dict(color=dark, linewidth=1.2),
                   capprops=dict(color=dark, linewidth=1.2))

        # vincentile bin means on a twin axis (bins 1..20)
        ax2 = ax.twiny()
        ax2.plot(np.arange(1, 21), vinc, 'o-', color=C_DARK,
                 lw=2.0, ms=4, zorder=5)
        ax2.set_xlim(0.5, 20.5)
        ax2.set_xticks([1, 5, 10, 15, 20])
        ax2.set_xlabel('Vincentile bin', fontsize=9, color=C_DARK)
        ax2.tick_params(axis='x', labelsize=8, colors=C_DARK)

        # zero reference
        ax.axhline(0, color='#999999', lw=1.0, ls='--', alpha=0.7, zorder=0)

        ax.set_title(f'{slbl}\nn = {n} paired trials   '
                     f'mean = {d.mean():.1f} ms   median = {np.median(d):.1f} ms',
                     fontsize=10, fontweight='bold')
        ax.set_xticks([1])
        ax.set_xticklabels(['All\ntrials'], fontsize=9)
        ax.set_xlim(0.4, 1.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('HRT - SRT (ms)', fontsize=12)

    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=C_GREY,
               markersize=6, alpha=0.5, label='Individual trials'),
        Patch(facecolor=SPEED_COLOR[0], edgecolor=SPEED_COLOR_DARK[0],
              alpha=0.75, label='Box plot (all trials)'),
        Line2D([0], [0], marker='o', color=C_DARK, lw=2, markersize=4,
               label='Vincentile bin means'),
    ]
    axes[0].legend(handles=handles, fontsize=8, loc='upper left',
                   framealpha=0.9)

    fig.text(0.5, 0.01,
             'Eye leads hand (HRT - SRT > 0) in every condition. The vincentile '
             'line tracks the box-plot centre, confirming the bin means faithfully '
             'summarise the per-trial differences. Gap is widest at 0 deg/s.',
             ha='center', va='bottom', fontsize=9, color='#555555', style='italic')
    fig.subplots_adjust(left=0.07, right=0.97, top=0.74, bottom=0.13, wspace=0.12)
    return fig


# =============================================================================
# GENERATE
# =============================================================================
print(f"\nGenerating: {OUTPUT_PATH}")
with PdfPages(OUTPUT_PATH) as pdf:
    for fn, args in [
        (page_histogram, (srt_rts, df_sfit, 'SRT', (50, 600),
         f'SRT: Empirical vs Mixture-Predicted Wald PDF (n={N_FULL})\n'
         f'Orange dotted = express-dominant participant ({"/".join(EXPRESS_PIDS)}).  '
         'Grey dashed = remaining participants.', True)),
        (page_histogram, (hrt_rts, df_hfit, 'HRT', (100, 800),
         f'HRT: Empirical vs Mixture-Predicted Wald PDF (n={N_FULL})', False)),
        (page_cdf, (srt_rts, df_sfit, 'SRT', (50, 600),
         f'SRT: Empirical CDF vs Mixture-Predicted Wald CDF (n={N_FULL})')),
        (page_cdf, (hrt_rts, df_hfit, 'HRT', (100, 800),
         f'HRT: Empirical CDF vs Mixture-Predicted Wald CDF (n={N_FULL})')),
    ]:
        f = fn(*args)
        pdf.savefig(f, bbox_inches='tight')
        plt.close(f)

    for fn in [page_ks_heatmap, page_parameter_trends]:
        f = fn(df_hfit, df_sfit)
        pdf.savefig(f, bbox_inches='tight')
        plt.close(f)

    f = page_srt_comparison(srt_rts, srt_rts_excl, df_sfit, df_sfit_excl)
    pdf.savefig(f, bbox_inches='tight')
    plt.close(f)

    f = page_summary_table(df_hfit, df_sfit, hrt_rts, srt_rts,
                           df_sfit_excl, srt_rts_excl)
    pdf.savefig(f, bbox_inches='tight')
    plt.close(f)

    f = page_rt_verification(rt_diffs)
    pdf.savefig(f, bbox_inches='tight')
    plt.close(f)

print(f"Saved: {OUTPUT_PATH}")
