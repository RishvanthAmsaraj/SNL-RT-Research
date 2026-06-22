"""
ddm_srt_fixed.py  — SRT DDM, corrected
=======================================
Key fixes vs original:
  1. Filter: 80–600ms (not 150ms) — 150ms wrongly excluded real saccades
  2. Fitting: PER-PARTICIPANT then group mean — pooled fitting was distorted
     by CMT0012/CMT002 who are express-saccade dominant (~110ms median)
  3. Likelihood: Wald + 5% uniform contamination (Ratcliff & Tuerlinckx 2002)
     — keeps all data, mathematically down-weights outlier trials
  4. t0 bound: 3rd percentile per participant (robust to individual fast trials)
  5. t0 minimum: 50ms (physiological minimum for saccadic sensorimotor delay)

The DDM figures use group-mean parameters. Individual fits are printed for
reference so you can see the spread across participants.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import differential_evolution
import os

matplotlib.rcParams.update({
    "font.family":  "Arial",
    "font.size":    13,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION COLOURS
# ─────────────────────────────────────────────────────────────────────────────
SPEED_COLOR = {
    0:   tuple(v/255 for v in (191, 230, 191)),
    75:  tuple(v/255 for v in (245, 191, 191)),
    150: tuple(v/255 for v in (191, 214, 249)),
}
SPEED_COLOR_DARK = {
    spd: tuple(min(1.0, v * 0.55) for v in col)
    for spd, col in SPEED_COLOR.items()
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  FIT PARAMETERS  (per-participant → group mean)
# ─────────────────────────────────────────────────────────────────────────────
df   = pd.read_csv(r"C:\Users\Rishv\Desktop\SNL Lab\Participant Data\pooled_data.csv")   
df_i = df[(df['BlockType'] == 'I') &
          (df['GazeSRT_ms'] >= 80) &
          (df['GazeSRT_ms'] <= 600)].copy()
df_i['rt_s'] = df_i['GazeSRT_ms'] / 1000.0

P_CONTAM  = 0.05    # contamination proportion (Ratcliff & Tuerlinckx 2002)
T0_MIN    = 0.050   # 50ms physiological minimum for saccadic sensorimotor delay
SPEEDS    = [0, 75, 150]
SPEED_LBL = {0: '0 deg/s', 75: '75 deg/s', 150: '150 deg/s'}
participants = sorted(df_i['Participant'].unique())


def fit_wald_contamination(rts):
    """Fit Wald + uniform contamination model. Returns (v, a, t0) or None."""
    if len(rts) < 15:
        return None
    T_range = rts.max() - rts.min()
    if T_range < 0.001:
        return None
    t0_max = max(np.percentile(rts, 3) - 0.002, T0_MIN + 0.001)
    if t0_max <= T0_MIN:
        t0_max = np.percentile(rts, 5) - 0.002
    if t0_max <= T0_MIN:
        return None

    def neg_ll(params):
        v, a, t0 = params
        adj = rts - t0
        if np.any(adj <= 0): return 1e10
        wald = (a / np.sqrt(2*np.pi*adj**3)) * np.exp(-(a - v*adj)**2 / (2*adj))
        if not np.all(np.isfinite(wald)): return 1e10
        mixture = (1 - P_CONTAM) * wald + P_CONTAM / T_range
        if np.any(mixture <= 0): return 1e10
        return -np.sum(np.log(mixture))

    res = differential_evolution(neg_ll,
                                 [(0.1, 20.0), (0.1, 3.0), (T0_MIN, t0_max)],
                                 seed=42, maxiter=1500, tol=1e-8,
                                 popsize=15, polish=True)
    return res.x if res.fun < 1e9 else None


# Fit each participant × speed, collect results
print(f"\n{'Participant':<12}  {'Speed':>6}  {'v':>7}  {'a':>7}  {'t0':>8}  {'n':>5}")
print("-" * 56)

per_participant = {spd: [] for spd in SPEEDS}
for pid in participants:
    for spd in SPEEDS:
        rts = df_i[(df_i['Participant'] == pid) &
                   (df_i['Speed_deg_per_s'] == spd)]['rt_s'].values
        result = fit_wald_contamination(rts)
        if result is not None:
            v, a, t0 = result
            per_participant[spd].append({'v': v, 'a': a, 't0': t0, 'pid': pid})
            print(f"{pid:<12}  {spd:>6}  {v:>7.3f}  {a:>7.3f}  {t0*1000:>6.0f}ms  {len(rts):>5}")

# Group mean parameters (used for the conceptual figure)
fitted = {}
print(f"\n{'Speed':>8}  {'mean v':>8}  {'SD v':>6}  {'mean a':>8}  {'SD a':>6}  "
      f"{'mean t0':>9}  {'SD t0':>7}  {'n':>4}")
print("-" * 68)
for spd in SPEEDS:
    p   = per_participant[spd]
    vs  = np.array([x['v']  for x in p])
    as_ = np.array([x['a']  for x in p])
    t0s = np.array([x['t0'] for x in p])
    v_m, a_m, t0_m = vs.mean(), as_.mean(), t0s.mean()
    fitted[spd] = {'v': v_m, 'a': a_m, 't0': t0_m,
                   'v_sd': vs.std(), 'a_sd': as_.std(), 't0_sd': t0s.std()}
    print(f"{spd:>5} deg/s  {v_m:>8.3f}  {vs.std():>6.3f}  "
          f"{a_m:>8.3f}  {as_.std():>6.3f}  "
          f"{t0_m*1000:>7.1f}ms  {t0s.std()*1000:>5.1f}ms  {len(p):>4}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SIMULATION  (single absorbing boundary at a)
# ─────────────────────────────────────────────────────────────────────────────
def sim_one(v, a, s=1.0, dt=0.001, T=0.45, rng=None):
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


def collect_paths(v, a, s, T=0.45, dt=0.001,
                  min_t=0.01, max_t=0.43, n_hit=5):
    rng  = np.random.default_rng(7)
    hits = []
    for _ in range(10_000):
        if len(hits) >= n_hit:
            break
        x, ht, hb = sim_one(v, a, s, dt, T, rng)
        if hb == 'hit' and min_t <= ht <= max_t:
            hits.append((x, ht))
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# 3.  WALD PDF
# ─────────────────────────────────────────────────────────────────────────────
def wald_pdf(t, v, a):
    t = np.maximum(t, 1e-9)
    return (a / np.sqrt(2*np.pi*t**3)) * np.exp(-(a - v*t)**2 / (2*t))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DRAW ONE FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def draw_ddm(v, a, t0, v_sd, a_sd, t0_sd, speed_label, out_path, speed):

    dt    = 0.001
    T_dec = 0.44
    T_tot = t0 + T_dec

    # Wald PDF
    t_pdf  = np.linspace(0.001, T_dec * 0.97, 2000)
    pdf    = wald_pdf(t_pdf, v, a)
    pk_idx = np.argmax(pdf)
    t_mode = t_pdf[pk_idx]

    dist_h   = a * 0.75
    pdf_disp = pdf / pdf.max() * dist_h

    hits = collect_paths(v=1.0, a=a, s=1.0, T=T_dec,
                         min_t=0.01, max_t=t_mode, n_hit=5)

    C_FILL = SPEED_COLOR[speed]
    C_LINE = SPEED_COLOR_DARK[speed]
    C_NDT  = '#E8E8E8'
    C_BASE = '#BBBBBB'
    C_DARK = '#1A1A2E'
    C_GREY = '#666666'

    z = a / 2

    y_axis  = -a * 0.50
    y_brace = -a * 0.88
    y_bot   = -a * 1.30
    y_top   =  a + dist_h * 1.55

    x_left  = -t0 * 0.65
    x_right =  T_tot + 0.16

    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bot, y_top)
    ax.axis('off')

    # NDT shading
    ax.fill_betweenx([y_bot * 0.70, y_top * 0.97],
                     0, t0, color=C_NDT, alpha=0.75, zorder=0)

    # Threshold
    ax.hlines(a, 0, T_tot, colors=C_LINE, linewidths=2.5, zorder=3)
    ax.text(T_tot + 0.008, a + dist_h * 0.08,
            f'Response threshold ($a$ = {a:.3f})',
            ha='left', va='bottom', fontsize=11, color=C_LINE, fontweight='bold')

    # Baseline
    ax.hlines(0, 0, T_tot, colors=C_BASE, linewidths=1.0,
              linestyles='--', zorder=2, alpha=0.7)
    ax.text(T_tot + 0.008, -dist_h * 0.06,
            'Baseline', ha='left', va='top', fontsize=10,
            color=C_BASE, fontstyle='italic')

    # RT Distribution
    t_abs = t_pdf + t0
    ax.fill_between(t_abs, a, a + pdf_disp, color=C_FILL, alpha=0.55, zorder=2)
    ax.plot(t_abs, a + pdf_disp, color=C_LINE, lw=2.2, zorder=3)

    lbl_t       = t0 + T_dec * 0.80
    lbl_pdf_idx = np.argmin(np.abs(t_pdf - T_dec * 0.80))
    lbl_y       = a + pdf_disp[lbl_pdf_idx] * 0.5 + dist_h * 0.35
    ax.text(lbl_t, lbl_y, 'SRT Distribution\n(Interception)',
            ha='center', va='bottom', fontsize=11,
            color=C_LINE, fontweight='bold', linespacing=1.3)

    # Paths
    for i, (p, rt) in enumerate(hits):
        n  = min(int(rt / dt) + 1, len(p))
        tp = np.arange(n) * dt + t0
        xp = p[:n] + z - a/2
        idx = list(range(0, n, 3))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        ax.plot(tp[idx], xp[idx], color=C_LINE,
                lw=2.0 if i == 0 else 0.9,
                alpha=0.88 if i == 0 else 0.30,
                zorder=4, rasterized=False)

    # Starting point
    ax.plot(t0, z, 'o', color=C_DARK, ms=9, zorder=8,
            markerfacecolor=C_DARK, markeredgewidth=0)
    ax.text(t0 * 0.50, z, 'Starting\npoint ($z$)',
            ha='center', va='center', fontsize=11,
            color=C_DARK, linespacing=1.3)

    # Drift rate arrow — slope matches display paths
    t_disp   = np.linspace(0.001, 0.8, 5000)
    pdf_d    = wald_pdf(t_disp, 1.0, a)
    t_mode_d = t_disp[np.argmax(pdf_d)]
    slope    = a / t_mode_d

    ta0  = t0 + 0.012
    ta1  = ta0 + min(T_dec * 0.30, t_mode * 0.80)
    xa0  = z
    xa1  = min(xa0 + (ta1 - ta0) * slope, a - 0.02)
    ax.annotate("", xy=(ta1, xa1), xytext=(ta0, xa0),
                arrowprops=dict(arrowstyle='-|>', color=C_DARK,
                                lw=2.5, mutation_scale=18), zorder=9)
    ax.text(ta1 + 0.005, xa1, f'Drift rate ($v$ = {v:.2f})',
            ha='left', va='center', fontsize=11,
            fontweight='bold', color=C_DARK)

    # Threshold height arrow
    x_bsep = x_left * 0.60
    ax.annotate("", xy=(x_bsep, a), xytext=(x_bsep, 0),
                arrowprops=dict(arrowstyle='<->', color=C_DARK,
                                lw=1.8, mutation_scale=12))
    ax.text(x_bsep - 0.006, a / 2,
            'Threshold\nheight ($a$)',
            ha='right', va='center', fontsize=11,
            color=C_DARK, linespacing=1.3)

    # Time axis
    ax.annotate("", xy=(T_tot, y_axis), xytext=(0, y_axis),
                arrowprops=dict(arrowstyle='-|>', color='#444444',
                                lw=1.6, mutation_scale=12))
    ax.text(T_tot * 0.50, y_axis - a * 0.18,
            'Time (ms)', ha='center', va='top', fontsize=12, color='#444444')

    tick_h   = a * 0.055
    T_tot_ms = int(T_tot * 1000)
    for t_ms in range(0, T_tot_ms + 1, 100):
        t_s      = t_ms / 1000.0
        col_tick = '#333333' if t_ms == 0 else '#888888'
        ax.plot([t_s, t_s], [y_axis, y_axis - tick_h],
                color=col_tick, lw=1.2, zorder=5)
        ax.text(t_s, y_axis - tick_h - a * 0.05,
                f'{t_ms}', ha='center', va='top', fontsize=9, color=col_tick)

    ax.plot([t0, t0], [y_axis, y_axis - tick_h * 1.8],
            color=C_LINE, lw=2.0, zorder=6)
    ax.text(t0, y_axis - tick_h * 1.8 - a * 0.05,
            f'{t0*1000:.0f} ms\n($t_0$)', ha='center', va='top',
            fontsize=9, color=C_LINE, fontweight='bold')

    # NDT bracket
    ax.annotate("", xy=(t0, y_brace), xytext=(0, y_brace),
                arrowprops=dict(arrowstyle='<->', color=C_GREY,
                                lw=1.4, mutation_scale=9))
    ax.text(t0 / 2, y_brace - a * 0.08,
            f'Non-decision time ($t_0$) = {t0*1000:.0f} ms',
            ha='center', va='top', fontsize=10,
            color=C_GREY, fontstyle='italic')

    # Decision time bracket
    ax.annotate("", xy=(T_tot * 0.85, y_brace), xytext=(t0, y_brace),
                arrowprops=dict(arrowstyle='<->', color='#AAAAAA',
                                lw=1.2, mutation_scale=8))
    ax.text((t0 + T_tot * 0.85) / 2, y_brace - a * 0.08,
            'Decision time', ha='center', va='top', fontsize=10,
            color='#AAAAAA', fontstyle='italic')

    # Title — group mean with ±SD
    ax.set_title(
        f'DDM (Single Boundary, SRT) — {speed_label}\n'
        f'Group mean:  $v$ = {v:.2f} ± {v_sd:.2f},   '
        f'$a$ = {a:.3f} ± {a_sd:.3f},   '
        f'$t_0$ = {t0*1000:.0f} ± {t0_sd*1000:.0f} ms',
        fontsize=13, fontweight='bold', pad=12, linespacing=1.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='pdf')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GENERATE
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_FOLDER = os.path.dirname(os.path.abspath(__file__))

for speed in SPEEDS:
    p   = fitted[speed]
    out = os.path.join(OUTPUT_FOLDER, f'ddm_srt_{speed}_degs.pdf')
    draw_ddm(p['v'], p['a'], p['t0'],
             p['v_sd'], p['a_sd'], p['t0_sd'],
             SPEED_LBL[speed], out, speed)
