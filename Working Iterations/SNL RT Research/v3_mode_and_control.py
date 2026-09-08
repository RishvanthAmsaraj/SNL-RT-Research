#!/usr/bin/env python
"""
v3_mode_and_control.py -- two independent checks in one pass.

    python v3_mode_and_control.py

CHECK A: is 'express_mode' the mode or the mean?
------------------------------------------------
DDM_fit.py writes  express_mode = t0e + ae/ve. That is the MEAN of the shifted
Wald, not the mode. Recomputes both for every mixture cell and reports:

  - the per-cell discrepancy in ms
  - whether the >= 30 ms separation rule flips when applied to modes
  - whether the "< 130 ms express territory" classification flips

WHAT TO CHECK
  1. If max discrepancy is < 5 ms across all cells, this is cosmetic -- just rename
     the columns to express_mean / reg_mean and move on.
  2. If any cell exceeds ~15 ms, or any classification flips, you must decide which
     quantity you mean and make the column name, the separation rule, and the
     < 130 ms test all use the SAME one.
  3. The express-saccade literature is about a modal peak near 100-130 ms, so the
     mode is usually the quantity you want.

CHECK B: the missing hand control on the identifiability sweep
---------------------------------------------------------------
SRT_identifiability_check.py sweeps the floor for SRT cells only. Without the same
sweep on HRT cells you have an observation, not a dissociation.

Runs the identical sweep on both:
    HRT cells, floors 90-140 ms (bracketing the 130 ms floor)
    SRT cells, floors 40-90 ms  (as in the existing script)

WHAT TO CHECK
  1. HRT median slope should be near 0 and n_tracking should be 0 (or near it).
     If HRT cells also track their floor, your central claim needs rewording.
  2. SRT median slope should be high (~0.9) with most cells tracking.
  3. The GAP between the two is the result. Report both panels side by side in
     SRT_identifiability.pdf -- currently only the SRT panel exists.

OUTPUT
------
    v3_mode_vs_mean.csv
    v3_floor_control.csv
"""
import os
import sys
import numpy as np
import pandas as pd

try:
    from wald_fast import wald_mean, wald_mode, floor_sweep, HRT_FLOOR, SRT_FLOOR
except ImportError:
    sys.exit("ERROR: wald_fast.py must sit in this folder.")

HERE = os.path.dirname(os.path.abspath(__file__))
FL_HAND = [0.090, 0.100, 0.110, 0.120, 0.130, 0.140]
FL_EYE = [0.040, 0.050, 0.060, 0.070, 0.080, 0.090]


def _need(f):
    p = os.path.join(HERE, f)
    if not os.path.exists(p):
        sys.exit(f"ERROR: {f} not found next to this script.")
    return p


def load_cell(dfi, pid, spd, col, lo, hi):
    sub = dfi[(dfi.Participant == pid) & (dfi.Speed_deg_per_s == spd)]
    x = sub[col].values.astype(float)
    return x[(~np.isnan(x)) & (x >= lo) & (x <= hi)] / 1000.0


def check_a(s):
    print("=" * 74)
    print("V3A  'express_mode' -- mean or mode?")
    print("=" * 74)
    mix = s[s.model == "mixture"].copy()
    if not len(mix):
        print("  No mixture cells in DDM_srt_fits.csv; skipping.")
        return pd.DataFrame()
    rows = []
    for _, r in mix.iterrows():
        ve, ae, t0e = float(r.ve), float(r.ae), float(r.t0e) / 1000.0
        vr, ar, t0r = float(r.vr), float(r.ar), float(r.t0r) / 1000.0
        em_mean, rm_mean = wald_mean(ve, ae, t0e) * 1000, wald_mean(vr, ar, t0r) * 1000
        em_mode, rm_mode = wald_mode(ve, ae, t0e) * 1000, wald_mode(vr, ar, t0r) * 1000
        rows.append(dict(
            pid=r.pid, spd=int(r.spd), pi=float(r.pi),
            reported=float(r.express_mode),
            express_mean=round(em_mean, 1), express_mode=round(em_mode, 1),
            express_diff=round(em_mean - em_mode, 1),
            reg_mean=round(rm_mean, 1), reg_mode=round(rm_mode, 1),
            sep_mean=round(rm_mean - em_mean, 1), sep_mode=round(rm_mode - em_mode, 1),
            sep_rule_flips=bool((rm_mean - em_mean >= 30) != (rm_mode - em_mode >= 30)),
            express_lt130_mean=bool(em_mean < 130), express_lt130_mode=bool(em_mode < 130),
            lt130_flips=bool((em_mean < 130) != (em_mode < 130))))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "v3_mode_vs_mean.csv"), index=False)

    print(f"  mixture cells             : {len(out)}")
    print(f"  max express mean-mode gap : {out.express_diff.max():.1f} ms "
          f"({out.loc[out.express_diff.idxmax(),'pid']}@"
          f"{int(out.loc[out.express_diff.idxmax(),'spd'])})")
    print(f"  median gap                : {out.express_diff.median():.1f} ms")
    print(f"  30 ms separation rule flips : {out.sep_rule_flips.sum()}/{len(out)}")
    print(f"  '<130 ms express' flips     : {out.lt130_flips.sum()}/{len(out)}")
    print(f"  express cells < 130 ms  by MEAN: {out.express_lt130_mean.sum()}   "
          f"by MODE: {out.express_lt130_mode.sum()}")

    if out.lt130_flips.sum() or out.sep_rule_flips.sum():
        print("\n  >>> CLASSIFICATION CHANGES -- these cells move:")
        for _, w in out[out.lt130_flips | out.sep_rule_flips].iterrows():
            print(f"      {w.pid}@{int(w.spd):<4d} mean={w.express_mean:.0f} "
                  f"mode={w.express_mode:.0f}  sep {w.sep_mean:.0f}->{w.sep_mode:.0f}")
    elif out.express_diff.max() < 5:
        print("\n  >>> Cosmetic. Rename the columns to express_mean / reg_mean.")
    else:
        print("\n  >>> No flips, but the gap is non-trivial. Rename OR recompute,")
        print("      and make the column name and both rules use the same quantity.")
    print("  Wrote v3_mode_vs_mean.csv\n")
    return out


def check_b(dfi, h, s):
    print("=" * 74)
    print("V3B  FLOOR-SWEEP CONTROL -- hand vs eye")
    print("=" * 74)
    rows = []
    for tag, tbl, col, lo, hi, floors in [
            ("HRT", h, "HandRT_ms", 150, 800, FL_HAND),
            ("SRT", s[s.model == "single"], "GazeSRT_ms", 80, 600, FL_EYE)]:
        for _, r in tbl.iterrows():
            rts = load_cell(dfi, r.pid, int(r.spd), col, lo, hi)
            if len(rts) < 15 or rts.min() <= max(floors) + 2e-3:
                continue
            sw = floor_sweep(rts, floors)
            rows.append(dict(effector=tag, pid=r.pid, spd=int(r.spd), n=len(rts),
                             slope=round(sw["slope"], 3),
                             tracks_floor=sw["tracks_floor"],
                             **{f"t0_at_{int(f*1000)}": round(t, 1)
                                for f, t in zip(floors, sw["t0_ms"])}))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "v3_floor_control.csv"), index=False)

    for tag in ("HRT", "SRT"):
        g = out[out.effector == tag]
        if not len(g):
            continue
        print(f"  {tag}: median slope = {g.slope.median():.2f}   "
              f"tracking (slope>0.7) = {g.tracks_floor.sum()}/{len(g)}")

    hrt, srt = out[out.effector == "HRT"], out[out.effector == "SRT"]
    print()
    if len(hrt) and len(srt):
        ok = hrt.tracks_floor.mean() < 0.2 and srt.tracks_floor.mean() > 0.5
        if ok:
            print("  >>> CONTROL HOLDS. Hand t0 does not track its floor; saccadic t0 does.")
            print("      Add the HRT panel to SRT_identifiability.pdf -- this is the")
            print("      negative control that turns the observation into a dissociation.")
        else:
            print("  >>> CONTROL DOES NOT CLEANLY HOLD. Inspect v3_floor_control.csv before")
            print("      making any claim about hand-vs-eye identifiability.")
    print("  Wrote v3_floor_control.csv")


def main():
    dfi = pd.read_csv(_need("pooled_data.csv"))
    dfi = dfi[dfi["BlockType"] == "I"]
    h = pd.read_csv(_need("DDM_hrt_fits.csv"))
    s = pd.read_csv(_need("DDM_srt_fits.csv"))
    check_a(s)
    check_b(dfi, h, s)


if __name__ == "__main__":
    main()
