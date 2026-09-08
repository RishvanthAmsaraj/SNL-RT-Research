#!/usr/bin/env python
"""
q2_later_vs_wald.py -- the professor's directive, done on a fair footing.

    python q2_later_vs_wald.py [--B 200]

"I think you can just try the LATER model on the arm data, and compare the goodness
of fit of the two models to the RT distribution. If it is better for LATER this is
probably a good enough reason."

Right instinct. Three things have to be right for the comparison to mean anything,
and the current pipeline gets none of them right:

  1. Your LATER fit is OLS on reciprobit plotting positions, not a likelihood. It
     cannot be compared to an MLE-fit Wald by AIC, BIC, or a likelihood ratio. Here
     LATER is fit by proper MLE (closed form).
  2. Plain LATER has 2 parameters, the shifted Wald has 3. So SHIFTED LATER
     (RT = t0 + 1/r) is fitted too -- 3 parameters, like-for-like.
  3. Your Wald likelihood carries 5% contamination; LATER as normally fitted does
     not. Everything here uses pure likelihoods for both so neither model gets a
     handicap.

WHAT IT REPORTS PER CELL
------------------------
  Vuong z       the proper non-nested test. z > 0 favours the Wald, z < 0 LATER.
  CV log-lik    5-fold cross-validated, the robust cross-check
  AIC / BIC     for the 2-vs-3 parameter comparisons
  bootstrap KS  each model against ITS OWN null -- absolute adequacy
  shift LRT     does this cell's distribution demand a non-decision term at all?

THE OUTCOME TO PREPARE FOR
--------------------------
On data generated from a shifted Wald, the Wald and shifted LATER come out
INDISTINGUISHABLE (Vuong p ~ 0.8, near-identical KS). These two distributions are
genuinely hard to tell apart at n ~ 110-160. So the single most likely result is a
tie, and "if it is better for LATER this is probably a good enough reason" then has
no trigger. Decide NOW what a tie means, before seeing the numbers:

  - a tie on the HAND means the Wald's extra structure buys nothing there, and
    LATER-for-both becomes attractive on parsimony and comparability grounds
  - a tie on SACCADES was already your finding (KS ~ 0.12 each), and the argument
    for LATER there was never about fit -- it is that LATER has no non-decision
    parameter to floor

THE MORE INFORMATIVE TEST
-------------------------
The shift LRT re-asks your central question inside the LATER family, where no Wald
assumption is involved. If hand cells demand t0 > 0 under LATER and saccade cells do
not, you have replicated the entire dissociation in a second model family. That is a
stronger result than any goodness-of-fit horse race, and it comes free here.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))

import later as L
import compare as C
from wald_fast import (fit_cell, wald_pdf, ks_statistic, HRT_FLOOR, SRT_FLOOR)

OUT = os.path.join(HERE, "q2_later_vs_wald.csv")


def _need(f):
    for p in (os.path.join(HERE, f), os.path.join(ROOT, f), f):
        if os.path.exists(p):
            return p
    sys.exit(f"ERROR: {f} not found.")


def wald_spec(floor):
    return dict(
        name="Wald", k=3,
        fit=lambda r: fit_cell(r, floor, contam=0.0),
        logpdf=lambda r, f: np.log(np.maximum(
            wald_pdf(np.maximum(r - f["t0"], 1e-12), f["v"], f["a"]), 1e-300)),
        ks=lambda rs, f: ks_statistic(rs, f["v"], f["a"], f["t0"]),
        rvs=lambda f, n, rg: f["t0"] + stats.invgauss.rvs(
            (f["a"] / f["v"]) / f["a"] ** 2, scale=f["a"] ** 2, size=n, random_state=rg))


LATER_SPEC = dict(
    name="LATER", k=2,
    fit=lambda r: L.later_mle(r),
    logpdf=lambda r, f: L.later_logpdf(r, f["mu"], f["sigma"], 0.0),
    ks=lambda rs, f: L.later_ks(rs, f),
    rvs=lambda f, n, rg: L.later_rvs(f, n, rg))

SLATER_SPEC = dict(
    name="sLATER", k=3,
    fit=lambda r: L.shifted_later_mle(r, 0.0),
    logpdf=lambda r, f: L.later_logpdf(r, f["mu"], f["sigma"], f["t0"]),
    ks=lambda rs, f: L.later_ks(rs, f),
    rvs=lambda f, n, rg: L.later_rvs(f, n, rg))


def load_cell(df, pid, spd, col, lo, hi):
    s = df[(df.Participant == pid) & (df.Speed_deg_per_s == spd)]
    x = s[col].values.astype(float)
    return x[(~np.isnan(x)) & (x >= lo) & (x <= hi)] / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(_need("pooled_data.csv"))
    df = df[df["BlockType"] == "I"]

    rows = []
    for tag, col, lo, hi, floor in [("HRT", "HandRT_ms", 150, 800, HRT_FLOOR),
                                    ("SRT", "GazeSRT_ms", 80, 600, SRT_FLOOR)]:
        W = wald_spec(floor)
        for pid in sorted(df.Participant.unique()):
            for spd in sorted(df.Speed_deg_per_s.unique()):
                rt = load_cell(df, pid, int(spd), col, lo, hi)
                if len(rt) < 20:
                    continue
                seed = int(abs(hash((tag, pid, int(spd)))) % 10 ** 6)

                # like-for-like, 3 vs 3
                r33 = C.compare_models(rt, W, SLATER_SPEC, B=args.B, seed=seed)
                # as the professor framed it, 3 vs 2
                r32 = C.compare_models(rt, W, LATER_SPEC, B=args.B, seed=seed)
                # does this cell demand a shift, inside the LATER family?
                lrt = L.shift_lrt(rt, floor=0.0)
                fL = L.later_mle(rt)
                trunc = L.check_truncation(fL)

                rows.append(dict(
                    effector=tag, pid=pid, spd=int(spd), n=len(rt),
                    # 3 vs 3
                    ll_wald=round(r33["ll1"], 3), ll_slater=round(r33["ll2"], 3),
                    vuong_z_33=round(r33["vuong_z"], 3),
                    vuong_p_33=round(r33["vuong_p"], 4),
                    verdict_33=r33["verdict"],
                    cv_wald=round(r33["cv_ll1"], 3), cv_slater=round(r33["cv_ll2"], 3),
                    ks_wald=round(r33["ks1"], 4), ks_wald_p=round(r33["ks1_p"], 4),
                    ks_slater=round(r33["ks2"], 4), ks_slater_p=round(r33["ks2_p"], 4),
                    # 3 vs 2
                    ll_later=round(r32["ll2"], 3),
                    aic_wald=round(r32["aic1"], 2), aic_later=round(r32["aic2"], 2),
                    vuong_z_32=round(r32["vuong_z"], 3),
                    vuong_p_32=round(r32["vuong_p"], 4),
                    verdict_32=r32["verdict"],
                    ks_later=round(r32["ks2"], 4), ks_later_p=round(r32["ks2_p"], 4),
                    # the informative one
                    shift_D=round(lrt["D"], 3), shift_p=round(lrt["p"], 5),
                    shift_t0_ms=round(lrt["t0_ms"], 1),
                    demands_shift=lrt["demands_shift"],
                    # diagnostics
                    mu_over_sigma=round(trunc["mu_over_sigma"], 2),
                    p_rate_neg=trunc["p_rate_below_zero"],
                    trunc_needed=trunc["truncation_needed"],
                    reciprobit_r2=round(L.reciprobit_r2(rt), 4)))
                print(f"  {tag} {pid}@{int(spd):<4d} n={len(rt):3d}  "
                      f"Vuong(3v3) z={r33['vuong_z']:+6.2f} -> {r33['verdict']:20s} "
                      f"shift D={lrt['D']:6.2f} p={lrt['p']:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print("\n" + "=" * 76)
    print("Q2  LATER vs WALD")
    print("=" * 76)
    for tag in ("HRT", "SRT"):
        s = out[out.effector == tag]
        if not len(s):
            continue
        print(f"\n{tag}  ({len(s)} cells)")
        print("  --- like-for-like: Wald (3) vs shifted LATER (3) ---")
        w = (s.verdict_33 == "Wald").sum()
        l = (s.verdict_33 == "sLATER").sum()
        t = len(s) - w - l
        print(f"    Wald wins {w}   sLATER wins {l}   indistinguishable {t}")
        print(f"    median Vuong z = {s.vuong_z_33.median():+.3f}  "
              f"(positive favours Wald)")
        print(f"    CV log-lik favours Wald in "
              f"{(s.cv_wald > s.cv_slater).sum()}/{len(s)} cells")
        print("  --- as asked: Wald (3) vs plain LATER (2) ---")
        print(f"    median AIC difference (Wald - LATER) = "
              f"{(s.aic_wald - s.aic_later).median():+.2f}  "
              f"(negative favours Wald)")
        print(f"    AIC favours LATER in {(s.aic_later < s.aic_wald).sum()}/{len(s)} cells")
        print("  --- absolute adequacy (bootstrap KS p) ---")
        print(f"    Wald   median p = {s.ks_wald_p.median():.3f}   "
              f"rejected {(s.ks_wald_p < 0.05).sum()}/{len(s)}")
        print(f"    LATER  median p = {s.ks_later_p.median():.3f}   "
              f"rejected {(s.ks_later_p < 0.05).sum()}/{len(s)}")
        print(f"    sLATER median p = {s.ks_slater_p.median():.3f}   "
              f"rejected {(s.ks_slater_p < 0.05).sum()}/{len(s)}")
        print("  --- THE KEY TEST: does this effector demand a non-decision term? ---")
        print(f"    cells demanding a shift (LRT p < 0.05): "
              f"{s.demands_shift.sum()}/{len(s)}")
        print(f"    median LRT D = {s.shift_D.median():.2f}")
        print(f"    median fitted shift = {s.shift_t0_ms.median():.1f} ms")
        print("  --- LATER diagnostics ---")
        print(f"    median mu/sigma = {s.mu_over_sigma.median():.2f}   "
              f"truncation needed in {s.trunc_needed.sum()}/{len(s)} cells")
        print(f"    median reciprobit r2 = {s.reciprobit_r2.median():.4f}   "
              f"(kept for continuity; NOT a fit statistic)")

    h, sr = out[out.effector == "HRT"], out[out.effector == "SRT"]
    if len(h) and len(sr):
        print("\n" + "=" * 76)
        print("THE DISSOCIATION, RE-DERIVED INSIDE THE LATER FAMILY")
        print("=" * 76)
        print("  A per-cell LRT at n ~ 110-160 has LOW POWER, so counting significant")
        print("  cells understates the effect. The dissociation is an EFFECTOR-level")
        print("  question and belongs at the effector level.\n")
        print(f"  {'':22s} {'HRT':>12s} {'SRT':>12s}")
        print(f"  {'cells':22s} {len(h):12d} {len(sr):12d}")
        print(f"  {'median LRT D':22s} {h.shift_D.median():12.3f} "
              f"{sr.shift_D.median():12.3f}")
        print(f"  {'mean LRT D':22s} {h.shift_D.mean():12.3f} "
              f"{sr.shift_D.mean():12.3f}")
        print(f"  {'cells with p<0.05':22s} {h.demands_shift.sum():12d} "
              f"{sr.demands_shift.sum():12d}")
        print(f"  {'median fitted shift':22s} {h.shift_t0_ms.median():11.1f}m "
              f"{sr.shift_t0_ms.median():11.1f}m")

        # Direct two-sample test of the dissociation
        u = stats.mannwhitneyu(h.shift_D.values, sr.shift_D.values,
                               alternative="greater")
        print(f"\n  Mann-Whitney (HRT D > SRT D): U = {u.statistic:.1f}, "
              f"p = {u.pvalue:.5f}")

        # Fisher's combined p within each effector
        for tag, s in (("HRT", h), ("SRT", sr)):
            pv = np.clip(s.shift_p.values, 1e-12, 1.0)
            chi = -2 * np.sum(np.log(pv))
            pf = stats.chi2.sf(chi, 2 * len(pv))
            print(f"  Fisher combined, {tag}: chi2 = {chi:.1f} on {2*len(pv)} df, "
                  f"p = {pf:.3e}")

        strong = (u.pvalue < 0.05 and h.shift_D.median() > sr.shift_D.median())
        if strong:
            print("\n  >>> REPLICATED at the effector level. Hand RT distributions demand")
            print("      a non-decision term more than saccade distributions do, in a")
            print("      model family with no Wald assumptions in it. This is independent")
            print("      evidence for the central claim and does not require the Wald to")
            print("      be the correct model.")
            print("      Report the Mann-Whitney and the median D, NOT the per-cell count.")
        else:
            print("\n  >>> The pattern does NOT separate at the effector level.")
            print("      That would be a genuine problem for the Wald-based version too,")
            print("      so investigate before proceeding.")
    print(f"\nWrote {os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
