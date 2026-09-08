#!/usr/bin/env python
"""
v4_intervals_and_AvB.py -- add uncertainty to Method A, then test A-vs-B properly.

    python v4_intervals_and_AvB.py

WHY
---
DDM_hrt_fits.csv and DDM_srt_fits.csv report v, a, t0 as bare point estimates with
no uncertainty. Method A vs Method B agreement is therefore assessed by eye against
Method B's credible intervals -- comparing a point to an interval.

The profile likelihood gives a 95% CI from the same computation at no extra cost.

WHAT IT DOES
------------
1. Per-cell profile-likelihood 95% CI for t0 (chi2_1, delta log-lik = 1.92).
2. If Bayesian_hrt_fits.csv / Bayesian_srt_fits.csv are present, compares Method A
   and Method B per cell: correlation, mean |difference|, and interval overlap.

WHAT TO CHECK
-------------
1. floor_open. TRUE means the profile is still improving at the floor, so t0 is not
   identified from below in that cell. Expect nearly all SRT cells to be floor_open.
   Expect SOME HRT cells to be floor_open too at the single-cell level -- that is
   normal and is exactly why the hierarchical model earns its keep. Do not panic;
   it is a statement about per-cell frequentist estimation, not about your group
   result.
2. Interval WIDTH. If hand per-cell CIs are ~40-60 ms wide while Method B intervals
   are ~25-30 ms, that difference is the value added by partial pooling. Say so
   explicitly in the paper -- it justifies the hierarchical model.
3. Method A vs B correlation. Expect r > 0.9 for hand. If it is lower, the two
   methods disagree more than a "convergent validity" claim can support.
4. Contamination mismatch. Method A uses 5% contamination, Method B uses 0%. Until
   you align them, any A-vs-B difference confounds frequentist-vs-Bayesian with
   contaminated-vs-pure. Re-run this after aligning them and compare the two runs.

OUTPUT
------
    v4_profile_intervals.csv
"""
import os
import sys
import numpy as np
import pandas as pd

try:
    from wald_fast import profile_ci, HRT_FLOOR, SRT_FLOOR
except ImportError:
    sys.exit("ERROR: wald_fast.py must sit in this folder.")

HERE = os.path.dirname(os.path.abspath(__file__))


def _need(f):
    p = os.path.join(HERE, f)
    if not os.path.exists(p):
        sys.exit(f"ERROR: {f} not found next to this script.")
    return p


def _opt(f):
    p = os.path.join(HERE, f)
    return p if os.path.exists(p) else None


def load_cell(dfi, pid, spd, col, lo, hi):
    sub = dfi[(dfi.Participant == pid) & (dfi.Speed_deg_per_s == spd)]
    x = sub[col].values.astype(float)
    return x[(~np.isnan(x)) & (x >= lo) & (x <= hi)] / 1000.0


def main():
    dfi = pd.read_csv(_need("pooled_data.csv"))
    dfi = dfi[dfi["BlockType"] == "I"]

    rows = []
    for tag, fits_file, bayes_file, col, lo, hi, floor in [
            ("HRT", "DDM_hrt_fits.csv", "Bayesian_hrt_fits.csv", "HandRT_ms", 150, 800, HRT_FLOOR),
            ("SRT", "DDM_srt_fits.csv", "Bayesian_srt_fits.csv", "GazeSRT_ms", 80, 600, SRT_FLOOR)]:
        pub = pd.read_csv(_need(fits_file))
        pub = pub[pub.model == "single"]
        bp = _opt(bayes_file)
        bay = pd.read_csv(bp) if bp else None

        for _, r in pub.iterrows():
            rts = load_cell(dfi, r.pid, int(r.spd), col, lo, hi)
            if len(rts) < 15:
                continue
            c = profile_ci(rts, floor)
            rec = dict(effector=tag, pid=r.pid, spd=int(r.spd), n=len(rts),
                       t0_A_ms=round(c["t0"] * 1000, 1),
                       t0_A_lo=round(c["lo"] * 1000, 1),
                       t0_A_hi=round(c["hi"] * 1000, 1),
                       width_ms=round(c["width_ms"], 1),
                       floor_open=c["floor_open"])
            if bay is not None:
                m = bay[(bay.pid == r.pid) & (bay.spd == int(r.spd))]
                if len(m) and pd.notna(m.iloc[0].get("t0")):
                    b = m.iloc[0]
                    rec["t0_B_ms"] = float(b.t0)
                    rec["t0_B_lo"] = float(b.get("t0_lo95", np.nan))
                    rec["t0_B_hi"] = float(b.get("t0_hi95", np.nan))
                    rec["AB_diff_ms"] = round(rec["t0_A_ms"] - rec["t0_B_ms"], 1)
                    rec["intervals_overlap"] = bool(
                        rec["t0_A_lo"] <= rec.get("t0_B_hi", np.inf)
                        and rec["t0_A_hi"] >= rec.get("t0_B_lo", -np.inf))
            rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "v4_profile_intervals.csv"), index=False)

    print("=" * 74)
    print("V4  PROFILE INTERVALS FOR METHOD A, AND METHOD A vs B")
    print("=" * 74)
    for tag in ("HRT", "SRT"):
        s = out[out.effector == tag]
        if not len(s):
            continue
        print(f"\n{tag}  ({len(s)} single cells)")
        print(f"    median 95% CI width      : {s.width_ms.median():.1f} ms")
        print(f"    CI spans the floor       : {s.floor_open.sum()}/{len(s)} cells")
        if "t0_B_ms" in s.columns and s.t0_B_ms.notna().any():
            k = s.dropna(subset=["t0_B_ms"])
            r = np.corrcoef(k.t0_A_ms, k.t0_B_ms)[0, 1] if len(k) > 2 else np.nan
            print(f"    Method A vs B correlation: r = {r:.3f}  (n = {len(k)})")
            print(f"    mean |A - B|             : {k.AB_diff_ms.abs().mean():.1f} ms")
            print(f"    max  |A - B|             : {k.AB_diff_ms.abs().max():.1f} ms")
            if "intervals_overlap" in k.columns:
                print(f"    intervals overlap        : "
                      f"{k.intervals_overlap.sum()}/{len(k)} cells")
            if len(k) and k.AB_diff_ms.abs().mean() > 15:
                print("    >>> A and B differ by more than 15 ms on average. Align the")
                print("        contamination setting (A=0.05, B=0.00) and re-run before")
                print("        presenting this as convergent validity.")
        else:
            print("    (Bayesian CSV not found -- A-vs-B comparison skipped)")

    print("\n" + "-" * 74)
    print("Wrote v4_profile_intervals.csv")
    print("Add t0_lo95 / t0_hi95 columns to the DDM fit tables from this output.")


if __name__ == "__main__":
    main()
