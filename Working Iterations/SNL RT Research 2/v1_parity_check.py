#!/usr/bin/env python
"""
v1_parity_check.py -- profile-likelihood fitter vs the published DDM fit tables.

Drop this next to pooled_data.csv, DDM_hrt_fits.csv, DDM_srt_fits.csv and run:

    python v1_parity_check.py

WHAT IT COMPARES
----------------
For every participant x speed cell, refits with the profile likelihood and compares
to the values already in your CSVs (which came from differential evolution).

WHAT "PASS" MEANS -- read this before interpreting the output
-------------------------------------------------------------
The primary criterion is the NLL, NOT the parameters.

The profile fitter is exact given t0, so it can only ever match or BEAT differential
evolution. If profile NLL < DE NLL, that is DE having undershot, not a parity failure.
So:

    nll_profile <= nll_DE + 1e-6   ->  PASS (profile is never worse)
    nll_profile >  nll_DE + 1e-6   ->  FAIL (investigate; should not happen)

Parameters can differ while the NLL matches. That is not an error -- it means the
likelihood surface is flat in that direction, i.e. the parameters are weakly
identified. Those cells are reported separately because they are a finding about
your data, not a bug in either fitter.

Cells pinned at v = 20 or a = 2.5 are expected to disagree: this fitter applies the
caps inside the M-step, DE applies them as box constraints. They are listed apart.

OUTPUT
------
    v1_parity_results.csv   per-cell comparison
    console summary with PASS / REVIEW / FAIL counts
"""
import os
import sys
import numpy as np
import pandas as pd

try:
    from wald_fast import fit_cell, wald_pdf, V_MAX, A_MAX, P_CONTAM, HRT_FLOOR, SRT_FLOOR
except ImportError:
    sys.exit("ERROR: wald_fast.py must sit in this folder.")

HERE = os.path.dirname(os.path.abspath(__file__))
NLL_TOL = 1e-6          # profile must never be worse than DE by more than this
T0_TOL_MS = 1.0         # your CSVs round t0 to whole ms
V_REL_TOL = 0.02        # 2% relative on drift
A_REL_TOL = 0.02        # 2% relative on boundary

# Your CSVs store v to 3 dp, a to 4 dp, t0 to whole ms. Re-evaluating the NLL at those
# ROUNDED values is always slightly worse than the true DE optimum, so a small negative
# d_nll is a rounding artifact, not a real improvement. Only |d_nll| beyond this counts.
NLL_MEANINGFUL = 0.5


def _need(f):
    p = os.path.join(HERE, f)
    if not os.path.exists(p):
        sys.exit(f"ERROR: {f} not found next to this script.")
    return p


def load_cell(dfi, pid, spd, col, lo, hi):
    sub = dfi[(dfi.Participant == pid) & (dfi.Speed_deg_per_s == spd)]
    x = sub[col].values.astype(float)
    return x[(~np.isnan(x)) & (x >= lo) & (x <= hi)] / 1000.0


def nll_at(rts, v, a, t0, contam=P_CONTAM):
    """NLL of the published parameters under the exact likelihood from DDM_fit.py."""
    Tr = rts.max() - rts.min()
    adj = rts - t0
    if np.any(adj <= 0):
        return np.inf
    w = wald_pdf(adj, v, a)
    d = (1 - contam) * w + (contam / Tr if contam > 0 else 0.0)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        return np.inf
    return float(-np.sum(np.log(d)))


def main():
    dfi = pd.read_csv(_need("pooled_data.csv"))
    dfi = dfi[dfi["BlockType"] == "I"]

    rows = []
    for tag, fits_file, col, lo, hi, floor in [
            ("HRT", "DDM_hrt_fits.csv", "HandRT_ms", 150, 800, HRT_FLOOR),
            ("SRT", "DDM_srt_fits.csv", "GazeSRT_ms", 80, 600, SRT_FLOOR)]:
        pub = pd.read_csv(_need(fits_file))
        pub = pub[pub.model == "single"]        # mixture cells are a separate comparison
        for _, r in pub.iterrows():
            rts = load_cell(dfi, r.pid, int(r.spd), col, lo, hi)
            if len(rts) < 15:
                continue
            v_pub, a_pub, t0_pub = float(r.v), float(r.a), float(r.t0) / 1000.0
            nll_pub = nll_at(rts, v_pub, a_pub, t0_pub)

            f = fit_cell(rts, floor)
            nll_new = f["nll"]

            d_nll = nll_new - nll_pub                     # negative = profile is better
            dt0 = abs(f["t0"] * 1000 - r.t0)
            dv = abs(f["v"] - v_pub) / max(abs(v_pub), 1e-9)
            da = abs(f["a"] - a_pub) / max(abs(a_pub), 1e-9)

            cap_hit = (v_pub >= V_MAX - 1e-6 or a_pub >= A_MAX - 1e-6
                       or f["v"] >= V_MAX - 1e-6 or f["a"] >= A_MAX - 1e-6)
            floor_hit = f["t0_at_floor"] or abs(r.t0 - floor * 1000) < 1.5

            if d_nll > NLL_TOL:
                verdict = "FAIL"
            elif cap_hit:
                verdict = "CAP"             # v or a pinned -- expected to differ
            elif dt0 <= T0_TOL_MS and dv <= V_REL_TOL and da <= A_REL_TOL:
                verdict = "FLOOR" if floor_hit else "PASS"
            else:
                verdict = "FLAT"        # NLL matches or improves but params moved

            rows.append(dict(
                effector=tag, pid=r.pid, spd=int(r.spd), n=len(rts),
                v_pub=v_pub, v_new=round(f["v"], 3),
                a_pub=a_pub, a_new=round(f["a"], 4),
                t0_pub_ms=int(r.t0), t0_new_ms=round(f["t0"] * 1000, 1),
                dt0_ms=round(f["t0"] * 1000 - r.t0, 2),
                nll_pub=round(nll_pub, 6), nll_new=round(nll_new, 6),
                d_nll=round(d_nll, 6),
                ks_pub=float(r.ks), ks_new=round(f["ks"], 4),
                at_bound=f["at_bound"], verdict=verdict))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "v1_parity_results.csv"), index=False)

    print("=" * 74)
    print("V1  PARITY: profile likelihood vs published differential-evolution fits")
    print("=" * 74)
    for tag in ("HRT", "SRT"):
        s = out[out.effector == tag]
        if not len(s):
            continue
        print(f"\n{tag}  ({len(s)} single-model cells)")
        for verdict, label in [("PASS", "match within tolerance"),
                               ("FLOOR", "matches, t0 pinned at the floor (expected for SRT)"),
                               ("FLAT", "NLL matches, params moved -> flat ridge, INSPECT"),
                               ("CAP", "v or a pinned at a cap -- expected to differ"),
                               ("FAIL", "profile WORSE than DE -- investigate")]:
            k = (s.verdict == verdict).sum()
            if k:
                print(f"    {verdict:6s} {k:3d}   {label}")

        better = s[s.d_nll < -NLL_MEANINGFUL]
        print(f"    profile beat DE by > {NLL_MEANINGFUL} log-lik in "
              f"{len(better)}/{len(s)} cells  (smaller gaps are CSV rounding)")
        if len(better):
            print("      cells where DE genuinely undershot (d_nll, dt0 ms):")
            for _, w in better.nsmallest(5, "d_nll").iterrows():
                print(f"        {w.pid}@{int(w.spd):<4d} {w.d_nll:+9.3f}  {w.dt0_ms:+7.2f}")
        ok = s[s.verdict.isin(["PASS", "FLOOR"])]
        if len(ok):
            print(f"    max |dt0| among PASS/FLOOR cells: {ok.dt0_ms.abs().max():.2f} ms")

    n_fail = (out.verdict == "FAIL").sum()
    print("\n" + "-" * 74)
    if n_fail == 0:
        print("RESULT: no FAIL cells. The profile fitter never did worse than DE.")
        print("        Safe to adopt. Review FLAT and CAP cells before publishing.")
    else:
        print(f"RESULT: {n_fail} FAIL cells -- do NOT adopt yet. Inspect v1_parity_results.csv")
        print(out[out.verdict == "FAIL"][["effector", "pid", "spd", "d_nll"]].to_string(index=False))
    print("-" * 74)
    print("Wrote v1_parity_results.csv")


if __name__ == "__main__":
    main()
