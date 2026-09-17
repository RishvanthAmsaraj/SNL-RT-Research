"""
LATER_analysis.py  --  Saccade-native LATER model (complementary analysis)

Fits the LATER model (Carpenter & Williams 1995) to saccadic RTs. LATER has no
separate non-decision-time parameter (the t0 identifiability problem does not exist).
Results are not directly comparable to the Wald hand model (different parameter space).

Outputs: LATER_reciprobit.pdf/.png, LATER_fits.csv

Run: python LATER_analysis.py  (needs pooled_data.csv)
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm, matplotlib.ticker
from scipy.stats import norm, linregress, invgauss, kstest

HERE = os.path.dirname(os.path.abspath(__file__))
_fam = "Arial"
for f in fm.findSystemFonts():
    if "arial" in f.lower():
        try: fm.fontManager.addfont(f); _fam = fm.FontProperties(fname=f).get_name(); break
        except Exception: pass
matplotlib.rcParams.update({"font.family": _fam, "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})

SC = {0: (0.45, 0.68, 0.40), 75: (0.85, 0.55, 0.55), 150: (0.50, 0.62, 0.82)}
df = pd.read_csv(os.path.join(HERE, "pooled_data.csv")); df = df[df["BlockType"] == "I"]
SPMAP = {1: 0, 2: 75, 3: 150}; df["spd"] = df["SpeedCode"].map(SPMAP)
g = df.dropna(subset=["GazeSRT_ms"]); g = g[(g["GazeSRT_ms"] >= 80) & (g["GazeSRT_ms"] <= 600)]

def recipro(rt_ms):
    rate = np.sort(1000.0 / rt_ms)            # 1/s, ascending
    n = len(rate); p = (np.arange(1, n + 1) - 0.5) / n
    return rate, norm.ppf(p)

def fit_line(rate, z, lo=0.10, hi=0.90):       # central population only (avoid express tail)
    n = len(rate); a, b = int(lo * n), int(hi * n)
    sl, ic, r, _, _ = linregress(rate[a:b], z[a:b])
    return sl, ic, r ** 2

# ---- per-cell LATER fit table ----
rows = []
for (pid, spd), sub in g.groupby(["Participant", "spd"]):
    rt = sub["GazeSRT_ms"].values.astype(float)
    if len(rt) < 40: continue
    rate = 1000.0 / rt; mu, sd = rate.mean(), rate.std()
    r_s, z_s = recipro(rt); _, _, r2 = fit_line(r_s, z_s)
    ks = kstest(rate, lambda q: norm.cdf(q, mu, sd))[0]
    rows.append(dict(pid=pid, spd=spd, mu_rate=mu, sd_rate=sd, median_lat_ms=1000.0 / mu,
                     reciprobit_r2=r2, ks=ks, express_frac=(rt < 130).mean()))
L = pd.DataFrame(rows); L.to_csv(os.path.join(HERE, "LATER_fits.csv"), index=False)

# choose a clean regular participant (high r2, low express) and the express-dominant one
agg = L.groupby("pid").agg(r2=("reciprobit_r2", "mean"), exp=("express_frac", "mean")).reset_index()
regular = agg[agg.exp < 0.05].sort_values("r2", ascending=False).iloc[0]["pid"]
express = agg.sort_values("exp", ascending=False).iloc[0]["pid"]

xt_lat = [500, 300, 200, 150, 120, 100]; xt_pos = [1000.0 / t for t in xt_lat]

def reciprobit_panel(ax, pid, title, two_line=False):
    sub = g[g.Participant == pid]
    for spd in [0, 75, 150]:
        rt = sub[sub.spd == spd]["GazeSRT_ms"].values.astype(float)
        if len(rt) < 20: continue
        rate, z = recipro(rt)
        ax.scatter(rate, z, s=14, color=SC[spd], alpha=0.55, edgecolor="none", zorder=3, label=f"{spd} deg/s")
        sl, ic, r2 = fit_line(rate, z)
        xs = np.linspace(rate.min(), np.percentile(rate, 90), 50)
        ax.plot(xs, ic + sl * xs, color=tuple(c * 0.55 for c in SC[spd]), lw=1.8, zorder=4)
    if two_line:  # express early line on the fast tail (pooled)
        rt = sub["GazeSRT_ms"].values.astype(float); rate, z = recipro(rt)
        n = len(rate); fast = rate[int(0.80 * n):]; zf = z[int(0.80 * n):]
        if len(fast) > 5:
            sl, ic, _, _, _ = linregress(fast, zf)
            xs = np.linspace(np.percentile(rate, 75), rate.max(), 30)
            ax.plot(xs, ic + sl * xs, color="#b00", lw=1.8, ls="--", zorder=5)
            ax.text(rate.max(), ic + sl * rate.max(), " express\n early line", color="#b00",
                    fontsize=8, fontweight="bold", va="center")
    ax.set_xticks(xt_pos); ax.set_xticklabels(xt_lat)
    ax.set_xlabel("saccadic latency (ms, reciprocal scale)")
    ax.set_ylabel("cumulative probability (probit)")
    yt = [norm.ppf(p) for p in [.05, .25, .5, .75, .95]]
    ax.set_yticks(yt); ax.set_yticklabels(["5%", "25%", "50%", "75%", "95%"])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.text(0.03, 0.96, "straight line =\nLATER (reciprocal-normal)", transform=ax.transAxes,
            va="top", fontsize=8, style="italic", color="#666")

fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
reciprobit_panel(ax[0], regular, f"Regular saccades ({regular})\nlatencies fall on a straight line")
reciprobit_panel(ax[1], express, f"Express-dominant ({express})\ntwo populations: express + regular", two_line=True)

# Panel C: LATER median latency by speed (per-participant), point + dots
rng = np.random.default_rng(0)
for i, spd in enumerate([0, 75, 150]):
    vals = L[L.spd == spd]["median_lat_ms"].values
    m, sd = vals.mean(), vals.std(ddof=1)
    ax[2].scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=24, color=SC[spd],
                  alpha=0.55, edgecolor="#555", linewidth=0.4, zorder=3)
    ax[2].errorbar(i, m, yerr=sd, fmt="o", ms=12, color=SC[spd], mec="#222", mew=1.4,
                   ecolor="#222", capsize=6, lw=2.0, zorder=5)
    ax[2].text(i + 0.22, m, f"{m:.0f} ms", ha="left", va="center", fontsize=10, fontweight="bold")
ax[2].set_xticks(range(3)); ax[2].set_xticklabels([f"{s} deg/s" for s in [0, 75, 150]]); ax[2].set_xlim(-0.5, 2.8)
ax[2].set_ylabel("LATER median latency (ms)")
ax[2].set_title("Saccadic latency by speed (LATER)\nno non-decision parameter \u2014 nothing to floor", fontsize=11, fontweight="bold")
ax[2].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
ax[2].grid(True, axis="y", ls="--", alpha=0.3); ax[2].spines[["top", "right"]].set_visible(False)

fig.suptitle("LATER model for saccades \u2014 the saccade-native alternative to the shifted-Wald",
             fontsize=13.5, fontweight="bold", y=1.02)
fig.text(0.5, -0.03,
         f"Across participants the reciprobit is highly linear (median r\u00b2 = {L.reciprobit_r2.median():.2f}): saccadic latencies are reciprocal-normal, exactly the LATER prediction. "
         "LATER models the whole latency as a rise to threshold, so there is no separate non-decision-time parameter and therefore no floor to hit. "
         "The trade-off: LATER's rate/threshold parameters are not comparable to the Wald's drift/boundary/non-decision, so this is a complementary saccade analysis, not a drop-in replacement for the cross-effector comparison.",
         ha="center", fontsize=8.2, color="#555")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "LATER_reciprobit.pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(HERE, "LATER_reciprobit.png"), dpi=150, bbox_inches="tight", facecolor="white")
print(f"regular={regular}, express={express}; median reciprobit r2={L.reciprobit_r2.median():.3f}")
print("LATER median latency by speed:", {s: round(L[L.spd==s].median_lat_ms.mean()) for s in [0,75,150]})
print("saved LATER_reciprobit.pdf/.png and LATER_fits.csv")
