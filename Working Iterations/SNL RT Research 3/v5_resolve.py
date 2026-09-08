#!/usr/bin/env python
"""
v5_resolve.py -- three things the validation run left open.

    python v5_resolve.py                  # diagnose + BH only (seconds)
    python v5_resolve.py --refine         # also re-run borderline cells at high B

PART 1 -- diagnose the -inf cells
---------------------------------
d_nll = -inf means the PUBLISHED (v, a, t0) gives an infinite NLL on the trials the
validation loader selected. Three possible causes, and they have different fixes:

  A. published t0 >= min(RT)   -> the stored triple is infeasible; store more decimals
  B. a NaN in the published row -> that cell never fitted; it should not be in the table
  C. the loader and DDM_fit.py selected DIFFERENT trials
     -> the parity comparison for that cell was invalid, and possibly for others

C is the one that matters, because it would mean the whole V1 result rests on a
trial-selection assumption that has not been checked. The n column discriminates:
if n here != n in the published table, the two are not looking at the same data.

PART 2 -- Benjamini-Hochberg on the KS p-values
-----------------------------------------------
48 SRT cells tested at alpha = 0.05 means ~2.4 expected false rejections under a true
model. 28 is far beyond that so the finding is real, but the uncorrected COUNT
overstates. A reviewer will ask for the FDR-adjusted number; have it ready.

PART 3 -- targeted refinement (--refine)
-----------------------------------------
Cells pinned at p = 1/(B+1) do NOT need more replicates. Simulation: a cell with true
p <= 0.01 has P(p >= 0.05) = 0.0000 at B=200 already. Raising B there refines a
number that is already decisively below threshold and changes no decision.

The cells that DO need more replicates are the ones NEAR the threshold, where the
Monte Carlo error is comparable to the distance from 0.05:

    true p    B=200 SE    P(misclassified vs 0.05)
    0.03      0.0119      0.077
    0.08      0.0191      0.037

So --refine re-runs only cells with p in [0.015, 0.12] at high B. That is the
opposite of refining the pinned cells, and it is where the decisions actually live.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KS_CSV = os.path.join(HERE, "v2_ks_calibration.csv")
PAR_CSV = os.path.join(HERE, "v1_parity_results.csv")


def benjamini_hochberg(p, q=0.05):
    """Returns (n_surviving, adaptive cutoff, boolean mask in original order)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    ps = p[order]
    thresh = (np.arange(1, m + 1) / m) * q
    below = np.where(ps <= thresh)[0]
    mask = np.zeros(m, bool)
    if len(below) == 0:
        return 0, np.nan, mask
    k = below.max()
    cut = ps[k]
    mask[order[:k + 1]] = True
    return k + 1, float(cut), mask


def part1_diagnose():
    print("=" * 74)
    print("PART 1  --  what caused the -inf cells?")
    print("=" * 74)
    if not os.path.exists(PAR_CSV):
        print("  v1_parity_results.csv not found; skipping.\n")
        return
    par = pd.read_csv(PAR_CSV)
    bad = par[~np.isfinite(par.d_nll)]
    if not len(bad):
        print("  No non-finite d_nll rows. Nothing to diagnose.\n")
        return

    print(f"  {len(bad)} cell(s) with non-finite d_nll:\n")
    for _, r in bad.iterrows():
        pub_file = f"DDM_{r.effector.lower()}_fits.csv"
        pub_path = os.path.join(HERE, pub_file)
        note = []
        if os.path.exists(pub_path):
            pub = pd.read_csv(pub_path)
            m = pub[(pub.pid == r.pid) & (pub.spd == r.spd)]
            if len(m):
                row = m.iloc[0]
                n_pub = row.get("n", np.nan)
                # Route B: NaN in the published row
                if any(pd.isna(row.get(c)) for c in ("v", "a", "t0")):
                    note.append("ROUTE B: a published parameter is NaN")
                # Route C: trial-count mismatch
                if pd.notna(n_pub) and int(n_pub) != int(r.n):
                    note.append(f"ROUTE C: n mismatch -- published {int(n_pub)}, "
                                f"loader {int(r.n)}")
                elif pd.notna(n_pub):
                    note.append(f"n matches ({int(n_pub)}) -> NOT route C")
                # Route A
                if pd.notna(row.get("t0")):
                    note.append(f"published t0 = {row.t0} ms "
                                f"(compare against this cell's min RT)")
        print(f"    {r.effector} {r.pid}@{int(r.spd)}  n(loader)={int(r.n)}")
        for x in note:
            print(f"        {x}")
        print()

    print("  INTERPRETATION")
    print("    Route A -> store t0 with more precision in the fit tables.")
    print("    Route B -> that cell never fitted; drop it or mark it explicitly.")
    print("    Route C -> STOP. The loader and DDM_fit.py disagree about which trials")
    print("               belong to this cell, so the V1 verdict for it is meaningless.")
    print("               Check every cell, not just these:")
    print()
    print("      import pandas as pd")
    print("      par = pd.read_csv('v1_parity_results.csv')")
    print("      pub = pd.read_csv('DDM_hrt_fits.csv')")
    print("      m = par.merge(pub[['pid','spd','n']], on=['pid','spd'],")
    print("                    suffixes=('_loader','_pub'))")
    print("      print(m[m.n_loader != m.n_pub])       # must be EMPTY")
    print()


def part2_fdr():
    print("=" * 74)
    print("PART 2  --  multiple-comparison correction")
    print("=" * 74)
    if not os.path.exists(KS_CSV):
        print("  v2_ks_calibration.csv not found; skipping.\n")
        return None
    ks = pd.read_csv(KS_CSV)
    out = []
    for tag in ("HRT", "SRT"):
        s = ks[ks.effector == tag].copy()
        if not len(s):
            continue
        raw = int((s.ks_p < 0.05).sum())
        n5, c5, m5 = benjamini_hochberg(s.ks_p.values, 0.05)
        n10, c10, _ = benjamini_hochberg(s.ks_p.values, 0.10)
        s["bh_q05"] = m5
        out.append(s)
        print(f"\n  {tag}  ({len(s)} cells)")
        print(f"    uncorrected p < 0.05      : {raw}/{len(s)}")
        print(f"    BH FDR q = 0.05           : {n5}/{len(s)}   "
              f"(cutoff p <= {c5:.4f})" if np.isfinite(c5) else
              f"    BH FDR q = 0.05           : {n5}/{len(s)}")
        print(f"    BH FDR q = 0.10           : {n10}/{len(s)}   "
              f"(cutoff p <= {c10:.4f})" if np.isfinite(c10) else
              f"    BH FDR q = 0.10           : {n10}/{len(s)}")
        exp_false = 0.05 * len(s)
        print(f"    expected false rejections under a TRUE model: {exp_false:.1f}")

        # BH's tightest threshold is (1/m)*q. If the bootstrap p-floor exceeds the
        # threshold at the rank where cells actually sit, BH cannot reject ANYTHING
        # regardless of how strong the evidence is -- a resolution artifact, not a
        # statistical result.
        pf = float(s.p_floor.iloc[0]) if "p_floor" in s.columns else np.nan
        if np.isfinite(pf):
            k_ok = int(np.floor(pf * len(s) / 0.05))
            if n5 == 0 and raw > 0:
                print(f"    !! BH rejected NOTHING while {raw} cells are below 0.05.")
                print(f"       Your p floor is {pf:.4f}; BH needs p <= (k/{len(s)})*0.05,")
                print(f"       which the floor cannot satisfy until rank {max(k_ok,1)}.")
                print(f"       This is a RESOLUTION artifact -- raise B and re-run before")
                print(f"       reporting any FDR-corrected count.")
            elif pf > 0.05 / len(s):
                print(f"    note: p floor {pf:.4f} exceeds BH's tightest threshold "
                      f"{0.05/len(s):.4f};")
                print(f"          cells below rank {max(k_ok,1)} cannot be resolved at "
                      f"this B.")

        if raw > 3 * exp_false and n5 > 0:
            print(f"    -> {raw} is far beyond chance; the finding is real regardless")
            print(f"       of correction. Report the BH number as the headline count.")
    if out:
        allout = pd.concat(out)
        allout.to_csv(os.path.join(HERE, "v5_ks_fdr.csv"), index=False)
        print(f"\n  Wrote v5_ks_fdr.csv (adds a bh_q05 column)")
    print()
    return ks


def part3_refine(ks, B_hi, lo, hi):
    print("=" * 74)
    print(f"PART 3  --  targeted refinement at B = {B_hi}")
    print("=" * 74)
    try:
        from wald_fast import bootstrap_ks, HRT_FLOOR, SRT_FLOOR
    except ImportError:
        sys.exit("ERROR: wald_fast.py must sit in this folder.")

    dfi = pd.read_csv(os.path.join(HERE, "pooled_data.csv"))
    dfi = dfi[dfi["BlockType"] == "I"]

    target = ks[(ks.ks_p >= lo) & (ks.ks_p <= hi)]
    pinned = ks[ks.ks_p <= ks.p_floor + 1e-9]
    print(f"  cells in the uncertain band p in [{lo}, {hi}] : {len(target)}  <- refining")
    print(f"  cells pinned at the p floor                  : {len(pinned)}  <- SKIPPED")
    print(f"     (a pinned cell has P(p >= 0.05) = 0.0000 already; more replicates")
    print(f"      cannot change its verdict)")
    if not len(target):
        print("\n  Nothing in the uncertain band. Done.\n")
        return
    est = len(target) * B_hi / 200 * 15.6 / 60
    print(f"  estimated runtime: ~{est:.0f} min\n")

    OUT = os.path.join(HERE, "v5_refined.csv")
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.effector, prev.pid, prev.spd))
        print(f"  resuming: {len(done)} already done\n")
    header = not os.path.exists(OUT)

    cols = {"HRT": ("HandRT_ms", 150, 800, HRT_FLOOR),
            "SRT": ("GazeSRT_ms", 80, 600, SRT_FLOOR)}
    for _, r in target.iterrows():
        key = (r.effector, r.pid, int(r.spd))
        if key in done:
            continue
        col, lo_ms, hi_ms, floor = cols[r.effector]
        sub = dfi[(dfi.Participant == r.pid) & (dfi.Speed_deg_per_s == int(r.spd))]
        x = sub[col].values.astype(float)
        x = x[(~np.isnan(x)) & (x >= lo_ms) & (x <= hi_ms)] / 1000.0
        b = bootstrap_ks(x, floor, B=B_hi, seed=int(abs(hash(key)) % 10 ** 6))
        row = dict(effector=r.effector, pid=r.pid, spd=int(r.spd), n=len(x),
                   ks=round(b["ks"], 4), p_B200=r.ks_p,
                   p_refined=round(b["ks_p"], 5), B=B_hi,
                   rejected_B200=bool(r.ks_p < 0.05),
                   rejected_refined=b["rejected"],
                   flipped=bool((r.ks_p < 0.05) != b["rejected"]))
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=header, index=False)
        header = False
        flag = "  <-- FLIPPED" if row["flipped"] else ""
        print(f"  {r.effector} {r.pid}@{int(r.spd):<4d} "
              f"p: {r.ks_p:.4f} -> {b['ks_p']:.4f}{flag}", flush=True)

    ref = pd.read_csv(OUT)
    print(f"\n  refined {len(ref)} cells; {int(ref.flipped.sum())} changed verdict")
    if ref.flipped.any():
        print("  flipped cells:")
        for _, w in ref[ref.flipped].iterrows():
            print(f"    {w.effector} {w.pid}@{int(w.spd)}: "
                  f"{w.p_B200:.4f} -> {w.p_refined:.4f}")
    print("  Wrote v5_refined.csv\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--lo", type=float, default=0.015)
    ap.add_argument("--hi", type=float, default=0.12)
    args = ap.parse_args()

    part1_diagnose()
    ks = part2_fdr()
    if args.refine and ks is not None:
        part3_refine(ks, args.B, args.lo, args.hi)
    elif ks is not None:
        print("Run with --refine to re-run the borderline cells at high B.")


if __name__ == "__main__":
    main()
