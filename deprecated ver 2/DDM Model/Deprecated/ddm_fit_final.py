"""
ddm_fit_final.py
================
Corrected 3-parameter DDM (= Wald first-passage) fitting.
Runs per-participant x speed, then averages for group-level figures.

Fixes vs previous versions
---------------------------
1. v upper bound 20 -> 100  (tight distributions need high v)
2. a upper bound  3 -> 10
3. t0 lower bound: HRT=100ms, SRT=40ms (physiological minimums)
4. t0 upper bound: 3rd percentile per participant per speed
   (not min(RT) -- robust to individual fast trials)
5. 3 DE seeds + bounded L-BFGS-B polish (not unbounded Nelder-Mead)
6. Contamination mixture (Ratcliff & Tuerlinckx 2002, 5% uniform)
7. SRT filter: 80-600ms; HRT filter: 150-800ms
8. Per-participant fitting -> group mean +/- SD on figures
9. Input validation: checks required columns before fitting

Output
------
- hrt_fits.csv        per-participant x speed HRT parameter table
- srt_fits.csv        per-participant x speed SRT parameter table
- ddm_hrt_*.pdf       HRT conceptual figures (3 speeds)
- ddm_srt_*.pdf       SRT conceptual figures (3 speeds)
- ddm_fit_summary.txt full per-participant KS table
"""

import os, sys, io, contextlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy import stats

matplotlib.rcParams.update({
    "font.family":  "Arial",
    "font.size":    13,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

# -----------------------------------------------------------------------------
# CONFIG -- change CSV_PATH to your local pooled_data.csv
# -----------------------------------------------------------------------------
CSV_PATH     = r'C:\Users\Rishv\Desktop\SNL Lab\Verification Code\pooled_data.csv'
OUTPUT_FOLDER = os.path.dirname(os.path.abspath(__file__))
P_CONTAM     = 0.05   # Ratcliff & Tuerlinckx (2002) contamination proportion
SPEEDS       = [0, 75, 150]
SPEED_LABELS = {0: '0 deg/s', 75: '75 deg/s', 150: '150 deg/s'}

SPEED_COLOR = {
    0:   tuple(v/255 for v in (191, 230, 191)),
    75:  tuple(v/255 for v in (245, 191, 191)),
    150: tuple(v/255 for v in (191, 214, 249)),
}
SPEED_COLOR_DARK = {
    s: tuple(min(1.0, v*0.55) for v in c)
    for s, c in SPEED_COLOR.items()
}


# -----------------------------------------------------------------------------
# CORE WALD FUNCTIONS
# -----------------------------------------------------------------------------
def wald_pdf(t, v, a):
    """First-passage density of single-boundary DDM (= Wald distribution)."""
    t = np.maximum(t, 1e-9)
    return (a / np.sqrt(2*np.pi*t**3)) * np.exp(-(a - v*t)**2 / (2*t))


def wald_cdf(t, v, a):
    """Wald CDF via scipy inverse-Gaussian parametrisation."""
    mu  = a / v
    lam = a ** 2
    return stats.invgauss.cdf(t, mu=mu/lam, scale=lam)


def fit_ddm(rts, t0_min, t0_max_pct=3):
    """
    Fit 3-parameter DDM to rts (in seconds).

    Returns (v, a, t0) or None if fit fails.

    Bounds
    ------
    v  : 0.1 - 100   (raised from 20; necessary for participants with tight SDs)
    a  : 0.05 - 10   (raised from 3)
    t0 : t0_min - 3rd-percentile of rts (condition-specific, not global min)
    """
    if len(rts) < 15:
        return None

    T_range = rts.max() - rts.min()
    if T_range < 0.001:
        return None

    # Robust t0 upper bound: 3rd percentile - 2ms
    t0_max = np.percentile(rts, t0_max_pct) - 0.002
    if t0_max < t0_min + 0.008:
        t0_max = np.percentile(rts, 5) - 0.002   # fallback
    if t0_max < t0_min + 0.004:
        t0_max = rts.min() * 0.88                 # last resort
    if t0_max <= t0_min:
        return None

    def nll(params):
        v, a, t0 = params
        adj = rts - t0
        if np.any(adj <= 0):
            return 1e10
        w = wald_pdf(adj, v, a)
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            return 1e10
        mixture = (1 - P_CONTAM) * w + P_CONTAM / T_range
        return -np.sum(np.log(mixture))

    bounds = [(0.1, 100.0), (0.05, 10.0), (t0_min, t0_max)]

    best = None
    # Three DE seeds -- polish=False to prevent unbounded escape
    for seed in [42, 7, 13]:
        res = differential_evolution(
            nll, bounds,
            seed=seed, maxiter=1500, tol=1e-9,
            popsize=15, polish=False, init='latinhypercube',
            mutation=(0.5, 1.5), recombination=0.7
        )
        if res.fun < 1e9 and (best is None or res.fun < best.fun):
            best = res

    if best is None:
        return None

    # Polish with L-BFGS-B which RESPECTS bounds -- fixes the escape problem
    res2 = minimize(nll, best.x, method='L-BFGS-B', bounds=bounds,
                    options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 2000})
    if res2.fun < best.fun and res2.success:
        candidate = res2.x
    else:
        candidate = best.x

    # Final safety clip to enforce bounds (guards against numerical edge cases)
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    candidate = np.clip(candidate, lo, hi)
    return candidate


def ks_quality(rts, v, a, t0):
    """KS statistic and qualitative label.
    Good: KS < 0.05, OK: 0.05-0.10, Borderline: 0.10-0.12, Poor: >0.12
    """
    adj = rts - t0
    adj = adj[adj > 0]
    ks, p = stats.kstest(adj, lambda x: wald_cdf(x, v, a))
    if ks < 0.05:   label = 'Good'
    elif ks < 0.10: label = 'OK'
    elif ks < 0.12: label = 'Borderline'
    else:           label = 'Poor'
    return ks, p, label


# -----------------------------------------------------------------------------
# FIT ALL PARTICIPANTS x SPEEDS
# -----------------------------------------------------------------------------
df   = pd.read_csv(CSV_PATH)
df_i = df[df['BlockType'] == 'I'].copy()

# -- Input validation ---------------------------------------------------------
REQUIRED_COLS = {'Participant', 'BlockType', 'Speed_deg_per_s',
                 'HandRT_ms', 'GazeSRT_ms'}
missing_cols = REQUIRED_COLS - set(df.columns)
if missing_cols:
    raise ValueError(f"CSV is missing required columns: {missing_cols}\n"
                     f"Found columns: {list(df.columns)}")
if df_i.empty:
    raise ValueError("No rows with BlockType='I' found in the CSV.")

participants = sorted(df_i['Participant'].unique())
n_pids       = len(participants)
n_fits       = n_pids * len(SPEEDS) * 2   # HRT + SRT

print(f"Participants: {n_pids}  |  "
      f"Total interception trials: {len(df_i)}")
print(f"Running {n_fits} fits total ({n_pids} ppts x 3 speeds x 2 RT types)...")
print(f"This typically takes 3-6 minutes on a modern CPU.")
print(f"{'-'*70}")

rows_hrt, rows_srt = [], []
fit_count = 0

for pid in participants:
    print(f"  {pid} ({participants.index(pid)+1}/{n_pids})...", end=' ', flush=True)
    for spd in SPEEDS:
        sub = df_i[(df_i['Participant'] == pid) &
                   (df_i['Speed_deg_per_s'] == spd)]

        # -- HRT --------------------------------------------------------------
        hrt_raw = sub[(sub['HandRT_ms'] >= 150) &
                      (sub['HandRT_ms'] <= 800)]['HandRT_ms'].values / 1000.0
        r = fit_ddm(hrt_raw, t0_min=0.10)
        if r is not None:
            v, a, t0 = r
            ks, p, lbl = ks_quality(hrt_raw, v, a, t0)
            rows_hrt.append(dict(pid=pid, spd=spd, v=v, a=a, t0=t0*1000,
                                 n=len(hrt_raw), ks=ks, p=p, quality=lbl))

        # -- SRT --------------------------------------------------------------
        srt_raw = sub[(sub['GazeSRT_ms'] >= 80) &
                      (sub['GazeSRT_ms'] <= 600)]['GazeSRT_ms'].values / 1000.0
        r = fit_ddm(srt_raw, t0_min=0.040)
        if r is not None:
            v, a, t0 = r
            ks, p, lbl = ks_quality(srt_raw, v, a, t0)
            rows_srt.append(dict(pid=pid, spd=spd, v=v, a=a, t0=t0*1000,
                                 n=len(srt_raw), ks=ks, p=p, quality=lbl))
    print("done")

df_hrt = pd.DataFrame(rows_hrt)
df_srt = pd.DataFrame(rows_srt)

# Save tables
df_hrt.to_csv(os.path.join(OUTPUT_FOLDER, 'hrt_fits.csv'), index=False)
df_srt.to_csv(os.path.join(OUTPUT_FOLDER, 'srt_fits.csv'), index=False)

# -----------------------------------------------------------------------------
# PRINT SUMMARY
# -----------------------------------------------------------------------------
def print_summary(df_fits, label):
    print(f"\n{'='*70}")
    print(f"  {label} -- Per-Participant Fit Summary")
    print(f"{'='*70}")
    print(f"  Overall: mean KS={df_fits.ks.mean():.4f}  "
          f"% Good (<0.05): {(df_fits.ks<0.05).mean()*100:.0f}%  "
          f"% OK (<0.10): {(df_fits.ks<0.10).mean()*100:.0f}%")
    print(f"\n  {'Speed':<8}  {'v':>12}  {'a':>12}  {'t0':>14}  {'KS':>7}")
    print(f"  {'-'*60}")
    for spd in SPEEDS:
        s = df_fits[df_fits.spd == spd]
        print(f"  {spd} deg/s  "
              f"  {s.v.mean():.2f}+/-{s.v.std():.2f}  "
              f"  {s.a.mean():.3f}+/-{s.a.std():.3f}  "
              f"  {s.t0.mean():.0f}+/-{s.t0.std():.0f} ms  "
              f"  {s.ks.mean():.4f}")
    print(f"\n  Per-participant detail:")
    print(f"  {'PID':<12}  {'Spd':>5}  {'v':>6}  {'a':>6}  "
          f"{'t0':>7}  {'n':>4}  {'KS':>6}  Qual")
    print(f"  {'-'*68}")
    for _, row in df_fits.iterrows():
        print(f"  {row.pid:<12}  {row.spd:>5}  {row.v:>6.2f}  {row.a:>6.3f}  "
              f"{row.t0:>5.0f}ms  {row.n:>4}  {row.ks:>6.4f}  {row.quality}")

print_summary(df_hrt, 'HRT')
print_summary(df_srt, 'SRT')

# Save summary to text file using explicit writes (stdout redirect
# is fragile on Windows due to console encoding)
summary_path = os.path.join(OUTPUT_FOLDER, 'ddm_fit_summary.txt')
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    print_summary(df_hrt, 'HRT')
    print_summary(df_srt, 'SRT')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(buf.getvalue())
print(f"\nSummary saved: {summary_path}")


# -----------------------------------------------------------------------------
# GROUP MEANS FOR FIGURES
# -----------------------------------------------------------------------------
def group_means(df_fits, spd):
    s = df_fits[df_fits.spd == spd]
    return {
        'v':    s.v.mean(),   'v_sd':  s.v.std(),
        'a':    s.a.mean(),   'a_sd':  s.a.std(),
        't0':   s.t0.mean()/1000,  # back to seconds for figure
        't0_sd': s.t0.std()/1000,
    }

fitted_hrt = {spd: group_means(df_hrt, spd) for spd in SPEEDS}
fitted_srt = {spd: group_means(df_srt, spd) for spd in SPEEDS}

print(f"\n{'-'*70}")
print("HRT group means (for figures):")
for spd in SPEEDS:
    p = fitted_hrt[spd]
    print(f"  {spd} deg/s: v={p['v']:.2f}+/-{p['v_sd']:.2f}  "
          f"a={p['a']:.3f}+/-{p['a_sd']:.3f}  "
          f"t0={p['t0']*1000:.0f}+/-{p['t0_sd']*1000:.0f}ms")
print("SRT group means (for figures):")
for spd in SPEEDS:
    p = fitted_srt[spd]
    print(f"  {spd} deg/s: v={p['v']:.2f}+/-{p['v_sd']:.2f}  "
          f"a={p['a']:.3f}+/-{p['a_sd']:.3f}  "
          f"t0={p['t0']*1000:.0f}+/-{p['t0_sd']*1000:.0f}ms")


# -----------------------------------------------------------------------------
# CONCEPTUAL FIGURE
# -----------------------------------------------------------------------------
def sim_one(v, a, s=1.0, dt=0.001, T=0.50, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    n = int(T / dt)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i-1] + v*dt + s*np.sqrt(dt)*rng.standard_normal()
        if x[i] >= a:
            x[i:] = a
            return x, i*dt, 'hit'
        x[i] = max(x[i], -a * 0.6)
    return x, T, 'miss'


def collect_paths(a, T_dec, n_hit=5, v_disp=1.0):
    """Simulate illustrative accumulation paths with a low *display* drift so
    they fan out legibly. Earlier this used a time window (T_dec) scaled to the
    FITTED drift while the paths moved at v_disp=1; for high-boundary conditions
    (e.g. SRT 75/150 deg/s, a~1.6-1.7) no path could climb to threshold within
    that short window, so zero paths were drawn. We size the simulation window
    to the display drift instead: a v_disp path reaches a in ~a/v_disp seconds,
    so a window of ~3x that (plus headroom) guarantees crossings for any a."""
    rng    = np.random.default_rng(7)
    T_sim  = max(T_dec, 3.0 * a / v_disp + 0.3)   # always long enough to cross
    hits   = []
    for _ in range(15_000):
        if len(hits) >= n_hit:
            break
        x, ht, hb = sim_one(v=v_disp, a=a, s=1.0, dt=0.001, T=T_sim, rng=rng)
        if hb == 'hit' and ht >= 0.01:
            hits.append((x, ht))
    return hits


def draw_ddm_figure(fitted_params, speed, speed_label, rt_type, out_path):
    p    = fitted_params[speed]
    v    = p['v'];    v_sd  = p['v_sd']
    a    = p['a'];    a_sd  = p['a_sd']
    t0   = p['t0'];   t0_sd = p['t0_sd']

    dt    = 0.001
    T_dec = 0.45
    T_tot = t0 + T_dec

    # -- Wald PDF (fitted v, a) ------------------------------------------------
    t_pdf    = np.linspace(0.001, T_dec * 0.97, 2000)
    pdf      = wald_pdf(t_pdf, v, a)
    pk_idx   = np.argmax(pdf)
    t_mode   = t_pdf[pk_idx]

    dist_h   = a * 0.75
    pdf_disp = pdf / pdf.max() * dist_h

    # Display paths (low visual drift, not the fitted v). The hit times are on
    # the display-drift timescale; they are rescaled at plot time to fill the
    # decision region up to the distribution mode, so the visual is consistent
    # across conditions regardless of boundary height.
    hits = collect_paths(a=a, T_dec=T_dec, n_hit=5)
    path_span = min(T_dec, t_mode + 0.01)   # visual width of the squiggles

    # Slope of display paths for arrow direction
    t_d  = np.linspace(0.001, 0.8, 5000)
    pd_d = wald_pdf(t_d, 1.0, a)
    t_md = t_d[np.argmax(pd_d)]
    slope = a / t_md

    C_FILL = SPEED_COLOR[speed]
    C_LINE = SPEED_COLOR_DARK[speed]
    C_NDT  = '#E8E8E8'
    C_BASE = '#BBBBBB'
    C_DARK = '#1A1A2E'
    C_GREY = '#666666'

    z = 0.0   # single-boundary model: accumulation starts at the baseline

    y_axis  = -a * 0.50
    y_brace = -a * 0.88
    y_bot   = -a * 1.32
    y_top   =  a + dist_h * 1.60

    x_left  = -t0 * 0.65
    x_right =  T_tot + 0.16

    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bot, y_top)
    ax.axis('off')

    # NDT shading
    ax.fill_betweenx([y_bot*0.70, y_top*0.97], 0, t0,
                     color=C_NDT, alpha=0.75, zorder=0)

    # Threshold
    ax.hlines(a, 0, T_tot, colors=C_LINE, linewidths=2.5, zorder=3)
    ax.text(T_tot+0.008, a+dist_h*0.08,
            f'Response threshold ($a$ = {a:.2f})',
            ha='left', va='bottom', fontsize=11,
            color=C_LINE, fontweight='bold')

    # Baseline
    ax.hlines(0, 0, T_tot, colors=C_BASE, linewidths=1.0,
              linestyles='--', zorder=2, alpha=0.7)
    ax.text(T_tot+0.008, -dist_h*0.06, 'Baseline',
            ha='left', va='top', fontsize=10,
            color=C_BASE, fontstyle='italic')

    # RT Distribution
    t_abs = t_pdf + t0
    ax.fill_between(t_abs, a, a+pdf_disp, color=C_FILL, alpha=0.55, zorder=2)
    ax.plot(t_abs, a+pdf_disp, color=C_LINE, lw=2.2, zorder=3)

    lbl_t       = t0 + T_dec*0.80
    lbl_pdf_idx = np.argmin(np.abs(t_pdf - T_dec*0.80))
    lbl_y       = a + pdf_disp[lbl_pdf_idx]*0.5 + dist_h*0.35
    ax.text(lbl_t, lbl_y,
            f'{rt_type} Distribution\n(Interception)',
            ha='center', va='bottom', fontsize=11,
            color=C_LINE, fontweight='bold', linespacing=1.3)

    # Paths (vector -- downsampled every 3rd point). Each path's own hit time
    # is rescaled to path_span so the squiggles fill the decision region to the
    # distribution mode, independent of the display-drift timescale.
    for i, (path, rt) in enumerate(hits):
        n  = min(int(rt/dt)+1, len(path))
        tp = np.linspace(0, path_span, n) + t0
        xp = path[:n] + z
        idx = list(range(0, n, 3))
        if idx[-1] != n-1: idx.append(n-1)
        ax.plot(tp[idx], xp[idx], color=C_LINE,
                lw=2.0 if i==0 else 0.9,
                alpha=0.88 if i==0 else 0.30,
                zorder=4, rasterized=False)

    # Starting point (inside NDT zone)
    ax.plot(t0, z, 'o', color=C_DARK, ms=9, zorder=8,
            markerfacecolor=C_DARK, markeredgewidth=0)
    ax.text(t0*0.45, a*0.18, 'Starting\npoint ($z$)',
            ha='center', va='center', fontsize=11,
            color=C_DARK, linespacing=1.3)

    # Drift rate arrow -- slope matches display paths
    ta0  = t0 + 0.012
    ta1  = ta0 + min(T_dec*0.32, t_mode*0.75)
    xa0  = z
    xa1  = min(xa0 + (ta1-ta0)*slope, a-0.02)
    ax.annotate("", xy=(ta1, xa1), xytext=(ta0, xa0),
                arrowprops=dict(arrowstyle='-|>', color=C_DARK,
                                lw=2.5, mutation_scale=18), zorder=9)
    ax.text(ta1+0.005, xa1,
            f'Drift rate ($v$ = {v:.2f})',
            ha='left', va='center', fontsize=11,
            fontweight='bold', color=C_DARK)

    # Threshold height arrow
    x_sep = x_left*0.60
    ax.annotate("", xy=(x_sep, a), xytext=(x_sep, 0),
                arrowprops=dict(arrowstyle='<->', color=C_DARK,
                                lw=1.8, mutation_scale=12))
    ax.text(x_sep-0.006, a/2, 'Threshold\nheight ($a$)',
            ha='right', va='center', fontsize=11,
            color=C_DARK, linespacing=1.3)

    # Time axis
    ax.annotate("", xy=(T_tot, y_axis), xytext=(0, y_axis),
                arrowprops=dict(arrowstyle='-|>', color='#444444',
                                lw=1.6, mutation_scale=12))
    ax.text(T_tot*0.50, y_axis-a*0.18, 'Time (ms)',
            ha='center', va='top', fontsize=12, color='#444444')

    tick_h   = a*0.055
    T_tot_ms = int(T_tot*1000)
    for t_ms in range(0, T_tot_ms+1, 100):
        t_s      = t_ms/1000.0
        col_tick = '#333333' if t_ms==0 else '#888888'
        ax.plot([t_s,t_s], [y_axis, y_axis-tick_h],
                color=col_tick, lw=1.2, zorder=5)
        ax.text(t_s, y_axis-tick_h-a*0.05, f'{t_ms}',
                ha='center', va='top', fontsize=9, color=col_tick)

    ax.plot([t0,t0], [y_axis, y_axis-tick_h*1.8],
            color=C_LINE, lw=2.0, zorder=6)
    ax.text(t0, y_axis-tick_h*1.8-a*0.05,
            f'{t0*1000:.0f} ms\n($t_0$)',
            ha='center', va='top', fontsize=9,
            color=C_LINE, fontweight='bold')

    # NDT bracket
    ax.annotate("", xy=(t0, y_brace), xytext=(0, y_brace),
                arrowprops=dict(arrowstyle='<->', color=C_GREY,
                                lw=1.4, mutation_scale=9))
    ax.text(t0/2, y_brace-a*0.08,
            f'Non-decision time ($t_0$) = {t0*1000:.0f} ms',
            ha='center', va='top', fontsize=10,
            color=C_GREY, fontstyle='italic')

    # Decision time bracket
    ax.annotate("", xy=(T_tot*0.85, y_brace), xytext=(t0, y_brace),
                arrowprops=dict(arrowstyle='<->', color='#AAAAAA',
                                lw=1.2, mutation_scale=8))
    ax.text((t0+T_tot*0.85)/2, y_brace-a*0.08, 'Decision time',
            ha='center', va='top', fontsize=10,
            color='#AAAAAA', fontstyle='italic')

    # Title with group mean +/- SD
    ax.set_title(
        f'DDM (Single Boundary, {rt_type}) -- {speed_label}\n'
        f'Group mean:  '
        f'$v$ = {v:.2f} +/- {v_sd:.2f},   '
        f'$a$ = {a:.2f} +/- {a_sd:.2f},   '
        f'$t_0$ = {t0*1000:.0f} +/- {t0_sd*1000:.0f} ms',
        fontsize=13, fontweight='bold', pad=12, linespacing=1.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='pdf')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# -----------------------------------------------------------------------------
# GENERATE FIGURES
# -----------------------------------------------------------------------------
print(f"\n{'-'*70}")
print("Generating HRT figures...")
for spd in SPEEDS:
    draw_ddm_figure(fitted_hrt, spd, SPEED_LABELS[spd], 'HRT',
                    os.path.join(OUTPUT_FOLDER, f'ddm_hrt_{spd}_degs.pdf'))

print("Generating SRT figures...")
for spd in SPEEDS:
    draw_ddm_figure(fitted_srt, spd, SPEED_LABELS[spd], 'SRT',
                    os.path.join(OUTPUT_FOLDER, f'ddm_srt_{spd}_degs.pdf'))

print(f"\nDone. All outputs saved to: {OUTPUT_FOLDER}")
