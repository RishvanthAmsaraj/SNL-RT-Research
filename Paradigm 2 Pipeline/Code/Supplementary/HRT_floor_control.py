"""
HRT_floor_control.py  --  Paradigm 2 (supplementary): hand negative control for the floor sweep

SRT_identifiability_check.py refits saccade cells at a range of imposed floors; a slope of
fitted t0 on floor near 1 means t0 is set by the floor, not the data. The matching HAND
control (Paradigm 1: v3_mode_and_control.py -> hand median slope 0.00 vs saccade 0.975)
is what turns "saccadic t0 floors" into an identifiability dissociation.

Method: identical fitter to SRT_identifiability_check.py (fit_t0, copied verbatim below:
differential evolution, seeds 42/7, maxiter 300, 5% contamination). Hand floors
90-140 ms (bracketing the 130 ms floor), saccade floors 40-90 ms -- the Paradigm 1 v3
sets. Only cells whose fastest RT sits above every imposed floor (+2 ms) are eligible,
as in v3; the saccade side is read from SRT_identifiability.csv.

Inputs : pooled_data_P2.csv, DDM_hrt_fits.csv, SRT_identifiability.csv
Outputs: HRT_floor_control.csv, HRT_floor_control.pdf/.png
Run    : python HRT_floor_control.py   (after SRT_identifiability_check.py)
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import differential_evolution
import matplotlib.font_manager as fm
_fam = "Arial" if "Arial" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"
matplotlib.rcParams.update({"font.family": _fam, "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
HERE = os.path.dirname(os.path.abspath(__file__))
def _need(f):
    p = os.path.join(HERE, f)
    if not os.path.exists(p): sys.exit(f"ERROR: {f} not found. Run DDM_fit.py and SRT_identifiability_check.py first.")
    return p
FL_HAND = [0.090, 0.100, 0.110, 0.120, 0.130, 0.140]

# ---- verbatim from SRT_identifiability_check.py ----
def wald_pdf(t,v,a):
    t=np.maximum(t,1e-9); return (a/np.sqrt(2*np.pi*t**3))*np.exp(-(a-v*t)**2/(2*t))
def fit_t0(rts, floor, contam=0.05):
    Tr=rts.max()-rts.min()
    def nll(p):
        v,a,t0=p; adj=rts-t0
        if np.any(adj<=0): return 1e10
        w=wald_pdf(adj,v,a)
        if np.any(w<=0) or not np.all(np.isfinite(w)): return 1e10
        d=(1-contam)*w+contam/Tr
        if np.any(d<=0): return 1e10
        return -np.sum(np.log(d))
    b=[(0.1,20),(0.05,2.5),(floor,max(np.percentile(rts,3)-0.002,floor+1e-3))]
    best=None
    for s in [42,7]:
        r=differential_evolution(nll,b,seed=s,maxiter=300,tol=1e-9,popsize=12,polish=True)
        if best is None or r.fun<best.fun: best=r
    return best.x[2]*1000
# ----------------------------------------------------

def main():
    dfi = pd.read_csv(_need("pooled_data_P2.csv")); dfi = dfi[dfi.BlockType == "P2"]
    h = pd.read_csv(_need("DDM_hrt_fits.csv")); eye = pd.read_csv(_need("SRT_identifiability.csv"))
    rows = []
    for _, r in h.iterrows():
        z = dfi[(dfi.Participant == r.pid) & (dfi.Speed_deg_per_s == r.spd)]
        x = z["HandRT_ms"].values.astype(float); x = x[(~np.isnan(x)) & (x >= 150) & (x <= 800)] / 1000
        eligible = bool(len(x) >= 15 and x.min() > max(FL_HAND) + 0.002)
        t0s = [fit_t0(x, fl) for fl in FL_HAND] if eligible else [np.nan] * len(FL_HAND)
        slope = float(np.polyfit([f * 1000 for f in FL_HAND], t0s, 1)[0]) if eligible else np.nan
        rows.append(dict(effector="HRT", pid=r.pid, spd=int(r.spd), n=len(x), min_rt_ms=round(x.min() * 1000, 1),
                         eligible=eligible, slope=round(slope, 3) if eligible else np.nan,
                         tracks_floor=bool(slope > 0.7) if eligible else np.nan,
                         ceiling_bound=bool(eligible and max(abs(t - x.min() * 1000) for t in t0s) < 2.0),
                         **{f"t0_at_{int(f*1000)}": (round(t, 1) if eligible else np.nan) for f, t in zip(FL_HAND, t0s)}))
    hand = pd.DataFrame(rows)
    e = eye.assign(effector="SRT").rename(columns={"min_srt_ms": "min_rt_ms"})
    out = pd.concat([hand, e], ignore_index=True, sort=False)
    out.to_csv(os.path.join(HERE, "HRT_floor_control.csv"), index=False)
    cb = lambda t: t.ceiling_bound.fillna(False).astype(bool)          # stuck at the fastest RT: slope ~0 but NOT identified
    he, ee = hand[hand.eligible & ~cb(hand)], e[e.eligible & ~cb(e)]
    print(f"excluded as ceiling-bound: hand {int((hand.eligible & cb(hand)).sum())}, saccade {int((e.eligible & cb(e)).sum())}")
    print(f"HAND  eligible {len(he)}/{len(hand)}: median slope {he.slope.median():.3f}, tracking (slope>0.7) {int(he.tracks_floor.sum())}/{len(he)}")
    print(f"EYE   eligible {len(ee)}/{len(e)}: median slope {ee.slope.median():.3f}, tracking {int(ee.tracks_floor.sum())}/{len(ee)}"
          f"   (all {len(e)} cells: median {e.slope.median():.3f}, tracking {int(e.tracks_floor.sum())})")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    xf = [f * 1000 for f in FL_HAND]
    for _, r in he.iterrows():
        ax[0].plot(xf, [r[f"t0_at_{int(f*1000)}"] for f in FL_HAND], "-o", ms=3, lw=1, alpha=0.55,
                   color="#C0392B" if r.tracks_floor else "#27AE60")
    ax[0].plot([90, 140], [90, 140], "k--", lw=1.5, label="$t_0$ = floor (unidentified)")
    ax[0].set_xlabel("imposed hand non-decision floor (ms)"); ax[0].set_ylabel("fitted hand $t_0$ (ms)")
    ax[0].set_title(f"A.  HAND $t_0$ vs imposed floor ({len(he)} eligible cells)\nred = tracks floor; green = stable (identified)",
                    fontsize=11, fontweight="bold")
    ax[0].legend(fontsize=9); ax[0].spines[["top", "right"]].set_visible(False); ax[0].grid(True, ls="--", alpha=0.3)
    bins = np.linspace(-0.1, 1.1, 25)
    ax[1].hist(he.slope, bins=bins, color="#27AE60", alpha=0.6, label=f"hand (n={len(he)}, median {he.slope.median():.2f})")   # ceiling-bound cells excluded
    ax[1].hist(ee.slope, bins=bins, color="#C0392B", alpha=0.6, label=f"saccade (n={len(ee)}, median {ee.slope.median():.2f})")
    ax[1].axvline(0.7, color="#555", ls=":", lw=1.4)
    ax[1].set_xlabel("slope of fitted $t_0$ on imposed floor  (1 = set by the floor)"); ax[1].set_ylabel("cells")
    ax[1].set_title("B.  Floor-tracking slope: hand vs saccade\n(fastest RT above every floor; cells stuck at the fastest RT excluded)", fontsize=11, fontweight="bold")
    ax[1].legend(fontsize=9); ax[1].spines[["top", "right"]].set_visible(False); ax[1].grid(True, axis="y", ls="--", alpha=0.3)
    fig.suptitle("Floor-sweep negative control (Paradigm 2): does fitted $t_0$ follow the imposed floor?", fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "HRT_floor_control.pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(HERE, "HRT_floor_control.png"), dpi=140, bbox_inches="tight", facecolor="white")
    print("saved HRT_floor_control.csv, HRT_floor_control.pdf/.png")

if __name__ == "__main__":
    main()
