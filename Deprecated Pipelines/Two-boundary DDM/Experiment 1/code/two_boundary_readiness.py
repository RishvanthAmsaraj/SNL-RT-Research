"""
two_boundary_readiness.py  --  can this dataset support a two-boundary DDM?

RUN THIS FIRST. A two-boundary DDM is fitted to the JOINT distribution of
(reaction time, which of two responses was made). The current pooled_data.csv has
Participant, Speed_deg_per_s, SpeedCode, BlockType, HandRT_ms and GazeSRT_ms --
reaction times and nothing else. There is no per-trial binary outcome in it, so
the two-boundary model cannot be fitted from that file as it stands. This script
tells you whether your RAW per-participant files contain something that can serve
as one, and if so whether there are enough minority responses to identify the
model per cell, only hierarchically, or not at all.

Three ways to supply the outcome:

  --choice-col NAME      a column that is already binary (0/1, L/R, hit/miss, ...)
  --sign-col NAME        a SIGNED continuous column; the outcome is its sign.
                         This is the recommended route for an interception task:
                         signed spatial error at interception (ahead of the target
                         = 1, behind = 0) is defined identically for the hand and
                         the eye, so the cross-effector comparison survives.
  --error-col NAME --error-threshold X
                         a magnitude column dichotomised at X (e.g. miss if the
                         interception error exceeds the target radius).

With no outcome argument the script runs in INSPECT mode: it prints every column
in the file with its type, unique-value count and a sample, so you can see what
is available to use.

Examples
--------
    python two_boundary_readiness.py --data raw_CMT001.csv
    python two_boundary_readiness.py --data pooled_raw.csv --sign-col HandErr_deg
    python two_boundary_readiness.py --data pooled_raw.csv --choice-col Direction \\
        --rt-col HandRT_ms --effector hand

Outputs: two_boundary_readiness.csv (per-cell table), two_boundary_readiness.pdf
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Identifiability thresholds. Both must hold for a cell to be fitted on its own.
# The minority-share threshold is the point below which per-cell fitting of the
# two-boundary DDM degrades sharply: with almost no responses at one barrier the
# likelihood carries little information about the boundary separation or the
# start point, and a/v trade off against each other almost freely.
MIN_MINORITY_SHARE = 0.05
MIN_MINORITY_COUNT = 10
SPEEDS = [0, 75, 150]
SPEED_FILL = {0: (0.749, 0.902, 0.749), 75: (0.961, 0.749, 0.749), 150: (0.749, 0.839, 0.976)}


def _line(s):
    return tuple(0.55 * c for c in SPEED_FILL[s])


# --------------------------------------------------------------------------- #
def inspect(df: pd.DataFrame) -> None:
    print(f"\n{len(df):,} rows x {df.shape[1]} columns\n")
    print(f"{'column':<28}{'dtype':<12}{'n_unique':>9}  {'n_missing':>9}  sample values")
    print("-" * 100)
    for c in df.columns:
        s = df[c]
        uniq = s.nunique(dropna=True)
        samp = ", ".join(str(x)[:14] for x in s.dropna().unique()[:4])
        print(f"{c:<28}{str(s.dtype):<12}{uniq:>9}  {s.isna().sum():>9}  {samp}")
    print("\nLooking for a two-boundary outcome, the useful shapes are:")
    print("  * n_unique == 2                     -> pass it to --choice-col")
    print("  * float, spans negative AND positive -> pass it to --sign-col")
    print("  * float, non-negative magnitude      -> pass it to --error-col with a threshold")
    print("\nCandidates in this file:")
    found = False
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if df[c].nunique(dropna=True) == 2:
            print(f"  --choice-col {c}   (2 unique values: "
                  f"{list(df[c].dropna().unique())})")
            found = True
        elif s.notna().sum() > 0.5 * len(s) and (s < 0).any() and (s > 0).any():
            neg = float((s < 0).mean())
            print(f"  --sign-col {c}     (signed float, {neg:.1%} negative)")
            found = True
    if not found:
        print("  none -- this file has no column that can act as a binary outcome.")


# --------------------------------------------------------------------------- #
def build_choice(df, args):
    """Return a 0/1 Series (1 = 'upper' barrier) plus a human-readable label."""
    if args.choice_col:
        raw = df[args.choice_col]
        vals = sorted(pd.Series(raw.dropna().unique()).astype(str))
        if len(vals) != 2:
            sys.exit(f"ERROR: --choice-col {args.choice_col} has {len(vals)} unique "
                     f"values ({vals[:6]}), needs exactly 2.")
        upper = args.upper_value if args.upper_value is not None else vals[1]
        ch = (raw.astype(str) == str(upper)).astype(float).where(raw.notna())
        return ch, f"{args.choice_col} == {upper}"
    if args.sign_col:
        x = pd.to_numeric(df[args.sign_col], errors="coerce")
        ch = (x > 0).astype(float).where(x.notna() & (x != 0))
        return ch, f"sign({args.sign_col}) > 0"
    if args.error_col:
        if args.error_threshold is None:
            sys.exit("ERROR: --error-col needs --error-threshold.")
        x = pd.to_numeric(df[args.error_col], errors="coerce")
        ch = (x <= args.error_threshold).astype(float).where(x.notna())
        return ch, f"{args.error_col} <= {args.error_threshold}"
    return None, None


def verdict(n, n_min):
    share = n_min / n if n else 0.0
    if n_min >= MIN_MINORITY_COUNT and share >= MIN_MINORITY_SHARE:
        return "per-cell"
    if n_min >= 3:
        return "hierarchical only"
    return "single-boundary only"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="trial-level CSV")
    ap.add_argument("--participant-col", default="Participant")
    ap.add_argument("--speed-col", default="Speed_deg_per_s")
    ap.add_argument("--blocktype-col", default="BlockType")
    ap.add_argument("--blocktype-keep", default="I")
    ap.add_argument("--rt-col", default=None,
                    help="RT column in ms (default: HandRT_ms for hand, GazeSRT_ms for eye)")
    ap.add_argument("--effector", default="hand", choices=["hand", "eye"])
    ap.add_argument("--choice-col", default=None)
    ap.add_argument("--upper-value", default=None,
                    help="which value of --choice-col counts as the upper barrier")
    ap.add_argument("--sign-col", default=None)
    ap.add_argument("--error-col", default=None)
    ap.add_argument("--error-threshold", type=float, default=None)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"ERROR: {args.data} not found.")
    df = pd.read_csv(args.data, sep=None, engine="python")

    if not (args.choice_col or args.sign_col or args.error_col):
        print("=" * 100)
        print("INSPECT MODE -- no outcome column given, so here is what the file contains.")
        print("=" * 100)
        inspect(df)
        return

    if args.blocktype_col in df.columns and args.blocktype_keep:
        df = df[df[args.blocktype_col].astype(str) == args.blocktype_keep]

    rt_col = args.rt_col or ("HandRT_ms" if args.effector == "hand" else "GazeSRT_ms")
    for c in (args.participant_col, args.speed_col, rt_col):
        if c not in df.columns:
            sys.exit(f"ERROR: column '{c}' is not in {args.data}. "
                     f"Available: {list(df.columns)}")

    lo, hi = (150.0, 800.0) if args.effector == "hand" else (80.0, 600.0)
    rt = pd.to_numeric(df[rt_col], errors="coerce")
    ch, label = build_choice(df, args)

    d = pd.DataFrame({
        "pid": df[args.participant_col].astype(str),
        "spd": pd.to_numeric(df[args.speed_col], errors="coerce"),
        "rt": rt, "choice": ch,
    })
    n_raw = len(d)
    d = d.dropna()
    d = d[(d.rt >= lo) & (d.rt <= hi)]
    d = d[d.spd.isin(SPEEDS)]

    print("=" * 100)
    print(f"TWO-BOUNDARY READINESS -- {args.effector}, outcome = {label}")
    print("=" * 100)
    print(f"trials: {n_raw:,} read -> {len(d):,} usable "
          f"(BlockType=={args.blocktype_keep}, {lo:.0f}-{hi:.0f} ms, outcome present)")
    print(f"overall split: {d.choice.mean():.3f} upper / {1 - d.choice.mean():.3f} lower\n")

    rows = []
    for (pid, spd), g in d.groupby(["pid", "spd"]):
        n = len(g)
        n_up = int(g.choice.sum())
        n_min = min(n_up, n - n_up)
        rows.append(dict(pid=pid, spd=int(spd), n=n, n_upper=n_up, n_lower=n - n_up,
                         p_upper=n_up / n, minority_n=n_min, minority_share=n_min / n,
                         verdict=verdict(n, n_min)))
    cells = pd.DataFrame(rows).sort_values(["spd", "pid"])

    print(f"{'speed':<9}{'cells':>7}{'median n':>10}{'median minority %':>20}"
          f"{'per-cell':>11}{'hier only':>12}{'no 2B':>8}")
    print("-" * 78)
    for s in SPEEDS:
        z = cells[cells.spd == s]
        if not len(z):
            continue
        print(f"{s:<9}{len(z):>7}{z.n.median():>10.0f}{100 * z.minority_share.median():>19.1f}%"
              f"{(z.verdict == 'per-cell').sum():>11}"
              f"{(z.verdict == 'hierarchical only').sum():>12}"
              f"{(z.verdict == 'single-boundary only').sum():>8}")

    n_ok = int((cells.verdict == "per-cell").sum())
    print("\n" + "-" * 78)
    print(f"{n_ok} / {len(cells)} cells clear the per-cell bar "
          f"(minority share >= {MIN_MINORITY_SHARE:.0%} and >= {MIN_MINORITY_COUNT} trials).")
    print(f"total minority responses in the dataset: {int(cells.minority_n.sum()):,} "
          f"-- this is what the hierarchical fit has to work with.")
    if n_ok >= 0.9 * len(cells):
        print("\nVERDICT: fit per cell OR hierarchically. Use Bayesian_2B_fit.py as-is.")
        if n_ok < len(cells):
            weak = cells[cells.verdict != "per-cell"]
            print(f"         {len(weak)} cell(s) fall short and will lean on the group: "
                  + ", ".join(f"{r.pid}@{r.spd}" for _, r in weak.iterrows()))
    elif cells.minority_n.sum() >= 200:
        print("\nVERDICT: do NOT fit per cell. Fit hierarchically (partial pooling across")
        print("         participants), which is what Bayesian_2B_fit.py does. Report the")
        print("         group-level parameters per speed; treat weak cells as shrunk.")
    else:
        print("\nVERDICT: too few minority responses for a two-boundary DDM at any level.")
        print("         The likelihood has almost no information about boundary separation")
        print("         or start point; a and v will trade off and the fit will reduce to a")
        print("         reparametrised single-boundary Wald. Stay single-boundary, and say")
        print("         so explicitly rather than reporting an unidentified two-boundary fit.")

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, f"two_boundary_readiness_{args.effector}.csv")
    cells.to_csv(csv_path, index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        fam = "Arial" if "Arial" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"
        matplotlib.rcParams.update({"font.family": fam, "font.size": 11,
                                    "pdf.fonttype": 42, "ps.fonttype": 42})
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        rng = np.random.default_rng(0)
        for i, s in enumerate(SPEEDS):
            z = cells[cells.spd == s]
            if not len(z):
                continue
            ax[0].scatter(i + rng.uniform(-0.16, 0.16, len(z)), 100 * z.minority_share,
                          s=34, color=SPEED_FILL[s], edgecolor=_line(s), zorder=3)
            ax[0].errorbar(i, 100 * z.minority_share.mean(), fmt="o", ms=11,
                           color=_line(s), mec="#222", mew=1.2, zorder=5)
            ax[1].scatter(z.n, 100 * z.minority_share, s=34, color=SPEED_FILL[s],
                          edgecolor=_line(s), label=f"{s} deg/s", zorder=3)
        ax[0].axhline(100 * MIN_MINORITY_SHARE, color="#b22", ls="--", lw=1.2)
        ax[0].text(2.35, 100 * MIN_MINORITY_SHARE + 0.6, "per-cell floor",
                   color="#b22", fontsize=8)
        ax[0].set_xticks(range(len(SPEEDS)))
        ax[0].set_xticklabels([f"{s} deg/s" for s in SPEEDS])
        ax[0].set_ylabel("minority-response share (%)")
        ax[0].set_title("How much choice information each cell carries", fontsize=11,
                        fontweight="bold")
        ax[1].axhline(100 * MIN_MINORITY_SHARE, color="#b22", ls="--", lw=1.2)
        ax[1].set_xlabel("trials in cell"); ax[1].set_ylabel("minority-response share (%)")
        ax[1].set_title("Cell size vs choice information", fontsize=11, fontweight="bold")
        ax[1].legend(fontsize=8, frameon=False)
        for a_ in ax:
            a_.spines[["top", "right"]].set_visible(False)
            a_.grid(True, axis="y", ls="--", alpha=0.3)
        fig.suptitle(f"Two-boundary DDM readiness -- {args.effector} "
                     f"(outcome: {label})", fontsize=12.5, fontweight="bold", y=1.02)
        fig.tight_layout()
        pdf_path = os.path.join(args.out, f"two_boundary_readiness_{args.effector}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight",
                    facecolor="white")
        print(f"\nwrote {csv_path}\nwrote {pdf_path} (+ .png)")
    except Exception as e:
        print(f"\nwrote {csv_path}  (figure skipped: {e})")


if __name__ == "__main__":
    main()
