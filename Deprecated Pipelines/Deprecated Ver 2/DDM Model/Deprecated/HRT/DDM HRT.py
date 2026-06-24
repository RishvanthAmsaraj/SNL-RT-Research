"""
DDM Conceptual Figures — PDF edition
Single-boundary (Wald) model, three speed conditions.

Changes vs previous version:
  - Condition colours: 0→green, 75→red/pink, 150→blue (RGB as specified)
  - Font: Arial throughout, enlarged
  - Diffusion paths are full VECTORS (rasterized=False) — downsampled to
    every 3rd point for manageable file size while remaining fully editable
  - Paths stop at the x-position of the RT distribution PEAK (mode), not
    at an arbitrary time, so they align visually with the curve above
  - Drift rate arrow is also a vector
  - Dead whitespace minimised
  - pdf.fonttype=42: all text editable in Illustrator/Inkscape
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
# CONDITION COLOURS  (RGB 0-255 → 0-1)
# ─────────────────────────────────────────────────────────────────────────────
SPEED_COLOR = {
    0:   tuple(v/255 for v in (191, 230, 191)),   # pale green
    75:  tuple(v/255 for v in (245, 191, 191)),   # pale red/pink
    150: tuple(v/255 for v in (191, 214, 249)),   # pale blue
}
SPEED_COLOR_DARK = {
    spd: tuple(min(1.0, v * 0.50) for v in col)
    for spd, col in SPEED_COLOR.items()
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  FIT PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
df   = pd.read_csv(r"C:\Users\Rishv\Desktop\SNL Lab\Participant Data\pooled_data.csv")  
df_i = df[(df['BlockType'] == 'I') &
          (df['HandRT_ms'] >= 150) &
          (df['HandRT_ms'] <= 800)].copy()
df_i['rt_s'] = df_i['HandRT_ms'] / 1000.0

# Contamination proportion (Ratcliff & Tuerlinckx 2002):
# 5% of trials treated as uniformly distributed contaminants.
# This keeps ALL data but mathematically down-weights genuine outliers.
P_CONTAM = 0.05

fitted = {}
for speed in [0, 75, 150]:
    rts = df_i[df_i['Speed_deg_per_s'] == speed]['rt_s'].values
    T_range = rts.max() - rts.min()   # range for uniform contaminant density

    def neg_ll(params, rts=rts, T_range=T_range):
        v, a, t0 = params
        adj = rts - t0
        if np.any(adj <= 0): return 1e10
        # Wald (first-passage) density
        wald = (a / np.sqrt(2*np.pi*adj**3)) * np.exp(-(a - v*adj)**2 / (2*adj))
        if not np.all(np.isfinite(wald)): return 1e10
        # Mixture: (1 - p_c) * Wald + p_c * Uniform
        mixture = (1 - P_CONTAM) * wald + P_CONTAM / T_range
        if np.any(mixture <= 0): return 1e10
        return -np.sum(np.log(mixture))

    # Robust t0_max: use 3rd percentile so a handful of fast trials
    # don't collapse the search window identically across conditions
    t0_max = np.percentile(rts, 3) - 0.002
    t0_min = 0.10   # 100ms physiological minimum for hand RT

    res = differential_evolution(neg_ll,
                                 [(0.1, 20.0), (0.2, 3.0), (t0_min, t0_max)],
                                 seed=42, maxiter=2000, tol=1e-9,
                                 popsize=20, polish=True)
    v_f, a_f, t0_f = res.x
    fitted[speed] = {'v': v_f, 'a': a_f, 't0': t0_f}
    print(f"Speed {speed:3d}:  v={v_f:.3f}  a={a_f:.3f}  t0={t0_f*1000:.1f} ms  "
          f"(n={len(rts)}, t0 window: {t0_min*1000:.0f}-{t0_max*1000:.0f}ms)")


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
        x[i] = max(x[i], -a * 0.6)   # soft display floor
    return x, T, 'miss'


def collect_paths(v, a, s, T=0.45, dt=0.001,
                  min_t=0.03, max_t=0.43, n_hit=5):
    rng   = np.random.default_rng(7)
    hits  = []
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
def draw_ddm(v, a, t0, speed_label, out_path, speed):

    dt    = 0.001
    T_dec = 0.44
    T_tot = t0 + T_dec

    # ── Wald PDF (using FITTED v, a) ─────────────────────────────────────────
    t_pdf    = np.linspace(0.001, T_dec * 0.97, 2000)
    pdf      = wald_pdf(t_pdf, v, a)
    pk_idx   = np.argmax(pdf)
    t_mode   = t_pdf[pk_idx]          # decision time at distribution peak
    t_mode_abs = t0 + t_mode          # absolute time of peak on figure

    dist_h   = a * 0.75              # height of RT distribution display
    pdf_disp = pdf / pdf.max() * dist_h

    # ── Collect paths that reach threshold BEFORE t_mode ─────────────────────
    # Only keep hits where rt <= t_mode so paths naturally end at boundary a
    hits = collect_paths(v=1.0, a=a, s=1.0, T=T_dec,
                         min_t=0.01, max_t=t_mode, n_hit=5)

    # ── Colours ───────────────────────────────────────────────────────────────
    C_FILL = SPEED_COLOR[speed]
    C_LINE = SPEED_COLOR_DARK[speed]
    C_NDT  = '#E8E8E8'
    C_BASE = '#BBBBBB'
    C_DARK = '#1A1A2E'
    C_GREY = '#666666'

    z = a / 2   # starting point

    # ── Y layout ─────────────────────────────────────────────────────────────
    y_axis  = -a * 0.50
    y_brace = -a * 0.88
    y_bot   = -a * 1.30
    y_top   =  a + dist_h * 1.55    # extra top room for distribution label

    # ── X layout ─────────────────────────────────────────────────────────────
    x_left  = -t0 * 0.65
    x_right =  T_tot + 0.16

    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bot, y_top)
    ax.axis('off')

    # ── NDT shading ───────────────────────────────────────────────────────────
    ax.fill_betweenx([y_bot * 0.70, y_top * 0.97],
                     0, t0, color=C_NDT, alpha=0.75, zorder=0)

    # ── Response threshold ────────────────────────────────────────────────────
    ax.hlines(a, 0, T_tot, colors=C_LINE, linewidths=2.5, zorder=3)
    # Label to the right, above the line so it doesn't clash with baseline label
    ax.text(T_tot + 0.008, a + dist_h * 0.08,
            f'Response threshold ($a$ = {a:.3f})',
            ha='left', va='bottom', fontsize=11,
            color=C_LINE, fontweight='bold')

    # ── Baseline reference ────────────────────────────────────────────────────
    ax.hlines(0, 0, T_tot, colors=C_BASE, linewidths=1.0,
              linestyles='--', zorder=2, alpha=0.7)
    ax.text(T_tot + 0.008, -dist_h * 0.06,
            'Baseline',
            ha='left', va='top', fontsize=10,
            color=C_BASE, fontstyle='italic')

    # ── RT Distribution above threshold ──────────────────────────────────────
    t_abs = t_pdf + t0
    ax.fill_between(t_abs, a, a + pdf_disp,
                    color=C_FILL, alpha=0.55, zorder=2)
    ax.plot(t_abs, a + pdf_disp, color=C_LINE, lw=2.2, zorder=3)

    # Label on the RIGHT side of the distribution (past the tail at 80% of T_dec)
    # This keeps it far from the title and the threshold label
    lbl_t = t0 + T_dec * 0.80
    # Find pdf height at that x-position for vertical placement
    lbl_pdf_idx = np.argmin(np.abs(t_pdf - T_dec * 0.80))
    lbl_y = a + pdf_disp[lbl_pdf_idx] * 0.5 + dist_h * 0.35
    ax.text(lbl_t, lbl_y,
            'RT Distribution\n(Interception)',
            ha='center', va='bottom', fontsize=11,
            color=C_LINE, fontweight='bold', linespacing=1.3)

    # ── Diffusion paths — vector, paths that hit BEFORE t_mode ───────────────
    for i, (p, rt) in enumerate(hits):
        n  = min(int(rt / dt) + 1, len(p))
        tp = np.arange(n) * dt + t0
        xp = p[:n] + z - a/2        # shift so start = z

        # Downsample every 3rd point, always keep last (at threshold)
        idx = list(range(0, n, 3))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        ax.plot(tp[idx], xp[idx], color=C_LINE,
                lw=2.0 if i == 0 else 0.9,
                alpha=0.88 if i == 0 else 0.30,
                zorder=4, rasterized=False)

    # ── Starting point ────────────────────────────────────────────────────────
    ax.plot(t0, z, 'o', color=C_DARK, ms=9, zorder=8,
            markerfacecolor=C_DARK, markeredgewidth=0)
    # Label inside NDT zone (to the left) — never overlaps paths or arrow
    ax.text(t0 * 0.50, z,
            'Starting\npoint ($z$)',
            ha='center', va='center', fontsize=11, color=C_DARK,
            linespacing=1.3)

    # ── Drift rate arrow — slope matched to display path direction ────────────
    # Display paths use v=1.0. Their mean slope in data coords = a / t_mode_display.
    # Compute t_mode_display from Wald(v=1.0, a=a_fitted).
    t_disp     = np.linspace(0.001, 0.8, 5000)
    pdf_disp   = wald_pdf(t_disp, 1.0, a)
    t_mode_d   = t_disp[np.argmax(pdf_disp)]      # time at which display paths peak
    path_slope = a / t_mode_d                      # data-coord slope matching paths

    arrow_dur = min(T_dec * 0.30, t_mode * 0.70)  # keep arrow within paths zone
    ta0 = t0 + 0.012
    ta1 = ta0 + arrow_dur
    xa0 = z
    xa1 = xa0 + arrow_dur * path_slope             # slope matches paths visually
    # Clamp so arrow stays below threshold
    xa1 = min(xa1, a - 0.02)

    ax.annotate("", xy=(ta1, xa1), xytext=(ta0, xa0),
                arrowprops=dict(arrowstyle='-|>', color=C_DARK,
                                lw=2.5, mutation_scale=18), zorder=9)
    # Label: placed to the RIGHT of the arrowhead, at the same y, so it
    # never overlaps the paths or starting-point label
    ax.text(ta1 + 0.005, xa1,
            f'Drift rate ($v$ = {v:.2f})',
            ha='left', va='center', fontsize=11,
            fontweight='bold', color=C_DARK)

    # ── Threshold height arrow — left of figure ───────────────────────────────
    x_bsep = x_left * 0.60
    ax.annotate("", xy=(x_bsep, a), xytext=(x_bsep, 0),
                arrowprops=dict(arrowstyle='<->', color=C_DARK,
                                lw=1.8, mutation_scale=12))
    ax.text(x_bsep - 0.006, a / 2,
            'Threshold\nheight ($a$)',
            ha='right', va='center', fontsize=11,
            color=C_DARK, linespacing=1.3)

    # ── Time axis ─────────────────────────────────────────────────────────────
    ax.annotate("", xy=(T_tot, y_axis), xytext=(0, y_axis),
                arrowprops=dict(arrowstyle='-|>', color='#444444',
                                lw=1.6, mutation_scale=12))
    ax.text(T_tot * 0.50, y_axis - a * 0.18,
            'Time (ms)', ha='center', va='top', fontsize=12, color='#444444')

    # Tick marks every 100 ms
    tick_h   = a * 0.055
    T_tot_ms = int(T_tot * 1000)
    for t_ms in range(0, T_tot_ms + 1, 100):
        t_s      = t_ms / 1000.0
        col_tick = '#333333' if t_ms == 0 else '#888888'
        ax.plot([t_s, t_s], [y_axis, y_axis - tick_h],
                color=col_tick, lw=1.2, zorder=5)
        ax.text(t_s, y_axis - tick_h - a * 0.05,
                f'{t_ms}', ha='center', va='top', fontsize=9, color=col_tick)

    # t0 tick highlighted in condition colour
    ax.plot([t0, t0], [y_axis, y_axis - tick_h * 1.8],
            color=C_LINE, lw=2.0, zorder=6)
    ax.text(t0, y_axis - tick_h * 1.8 - a * 0.05,
            f'{t0*1000:.0f} ms\n($t_0$)', ha='center', va='top',
            fontsize=9, color=C_LINE, fontweight='bold')

    # ── NDT bracket ───────────────────────────────────────────────────────────
    ax.annotate("", xy=(t0, y_brace), xytext=(0, y_brace),
                arrowprops=dict(arrowstyle='<->', color=C_GREY,
                                lw=1.4, mutation_scale=9))
    ax.text(t0 / 2, y_brace - a * 0.08,
            f'Non-decision time ($t_0$) = {t0*1000:.0f} ms',
            ha='center', va='top', fontsize=10,
            color=C_GREY, fontstyle='italic')

    # ── Decision time bracket ─────────────────────────────────────────────────
    ax.annotate("", xy=(T_tot * 0.85, y_brace), xytext=(t0, y_brace),
                arrowprops=dict(arrowstyle='<->', color='#AAAAAA',
                                lw=1.2, mutation_scale=8))
    ax.text((t0 + T_tot * 0.85) / 2, y_brace - a * 0.08,
            'Decision time', ha='center', va='top', fontsize=10,
            color='#AAAAAA', fontstyle='italic')

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        f'DDM (Single Boundary) — {speed_label}\n'
        f'$v$ = {v:.2f},   $a$ = {a:.3f},   $t_0$ = {t0*1000:.0f} ms',
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

labels = {0: '0 deg/s', 75: '75 deg/s', 150: '150 deg/s'}
for speed in [0, 75, 150]:
    p   = fitted[speed]
    out = os.path.join(OUTPUT_FOLDER, f'ddm_{speed}_degs.pdf')
    draw_ddm(p['v'], p['a'], p['t0'], labels[speed], out, speed)
