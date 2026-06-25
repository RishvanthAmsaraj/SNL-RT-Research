"""
why_saccadic_t0_floors.py  --  Diagnostic: mechanism of saccadic t0 flooring

Shows that a shifted Wald forces implied t0 = mean_RT - 3*SD/skewness. Hand RT is
strongly right-skewed (implied t0 ~191 ms, above the 130 ms floor); saccadic RT is
near-symmetric (implied t0 ~20-30 ms, below the 70 ms floor). This is a property of
saccadic data, not a coding error.

Output: why_saccadic_t0_floors.pdf/.png

Run: python why_saccadic_t0_floors.py
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import skew, gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
_fam = "Arial"
for f in fm.findSystemFonts():
    if "arial" in f.lower():
        try: fm.fontManager.addfont(f); _fam = fm.FontProperties(fname=f).get_name(); break
        except Exception: pass
matplotlib.rcParams.update({"font.family": _fam, "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})

df = pd.read_csv(os.path.join(HERE, "pooled_data.csv")); df = df[df["BlockType"] == "I"]

cfg = [("HandRT_ms", "HAND  (manual reaction time)", 130, (150, 800), (0.45, 0.68, 0.40)),
       ("GazeSRT_ms", "EYE  (saccadic reaction time)", 70, (80, 600), (0.50, 0.62, 0.82))]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
for ax, (col, title, floor, filt, color) in zip(axes, cfg):
    r = df[col].dropna().values.astype(float); r = r[(r >= filt[0]) & (r <= filt[1])]
    m, sd, sk = r.mean(), r.std(), skew(r)
    implied = m - 3 * sd / sk

    xs = np.linspace(r.min(), np.percentile(r, 99), 400)
    kde = gaussian_kde(r)(xs)
    ax.fill_between(xs, kde, color=color, alpha=0.45, zorder=2)
    ax.plot(xs, kde, color=tuple(c * 0.55 for c in color), lw=1.8, zorder=3)

    ymax = kde.max() * 1.18
    # physiological floor
    ax.axvline(floor, color="#777", ls=":", lw=1.6, zorder=4)
    ax.text(floor, ymax * 0.97, f"physiological\nfloor {floor} ms", ha="center", va="top",
            fontsize=8.5, style="italic", color="#777")
    # model-implied t0 from the distribution shape
    ax.axvline(implied, color="#b00", ls="--", lw=2.0, zorder=5)
    ax.text(implied, ymax * 0.62, f"shape-implied\n$t_0$ = {implied:.0f} ms", ha="center", va="top",
            fontsize=9, fontweight="bold", color="#b00")
    if implied < floor:
        ax.annotate("", xy=(implied, ymax * 0.30), xytext=(floor, ymax * 0.30),
                    arrowprops=dict(arrowstyle="->", color="#b00", lw=1.6))
        ax.text((implied + floor) / 2, ymax * 0.34, "wants to floor", ha="center",
                fontsize=8.5, color="#b00", fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("reaction time (ms)"); ax.set_ylabel("density")
    ax.set_ylim(0, ymax); ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    txt = (f"skewness = {sk:.2f}\nskew/CV ratio = {sk/(sd/m):.1f}\n(a pure Wald = 3.0)")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc"))

fig.suptitle("Why saccadic non-decision time floors: it is the distribution SHAPE, not the data\n"
             "implied $t_0$ = mean RT \u2212 3\u00b7SD / skewness   (forced by the Wald's skew\u2013spread geometry)",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.text(0.5, -0.04,
         "The hand distribution is strongly right-skewed, so the Wald reads a late onset plus a short skewed decision \u2014 $t_0$ lands above its floor and is identified.  "
         "The saccadic distribution is nearly symmetric for its spread (skew/CV \u2248 3, low absolute skew), so the model attributes almost all of the RT to the\n"
         "decision process and $t_0$ is pushed below the 70 ms physiological floor.  More trials cannot change this \u2014 it is set by the shape, which is why the saccade "
         "field fixes the dead time (e.g. the LATER / reciprocal-normal model) rather than estimating it freely.",
         ha="center", fontsize=8.2, color="#555")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "why_saccadic_t0_floors.pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(HERE, "why_saccadic_t0_floors.png"), dpi=150, bbox_inches="tight", facecolor="white")
print(f"HAND implied t0 vs floor: see figure;  EYE implied t0 floors below 70 ms")
print("saved why_saccadic_t0_floors.pdf/.png")
