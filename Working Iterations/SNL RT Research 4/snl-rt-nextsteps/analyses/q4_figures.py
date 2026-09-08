#!/usr/bin/env python
"""
q4_figures.py -- the diagnostics to put in front of the professor.

    python q4_figures.py

Produces:
  fig_reciprobit_by_effector.pdf   reciprobit plots, hand vs eye, LATER fit overlaid
  fig_model_qq.pdf                 Wald vs LATER Q-Q against the empirical quantiles
  fig_shift_lrt.pdf                the dissociation, inside the LATER family

The reciprobit panel is the direct answer to "is LATER on solid ground for the arm?"
A straight line means the Gaussian-rate assumption holds. Curvature or a swivel means
it does not, and the LATER fit should not be trusted for that effector regardless of
what any single fit statistic says.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))

import later as L
from wald_fast import fit_cell, wald_cdf, HRT_FLOOR, SRT_FLOOR

# One palette, defined once (audit finding S2: the pipeline currently has four).
SPEED_COLORS = {0: (0.30, 0.55, 0.20), 75: (0.85, 0.55, 0.15), 150: (0.55, 0.25, 0.55)}
EFF_LABEL = {"HRT": "Hand", "SRT": "Saccade"}


def _need(f):
    for p in (os.path.join(HERE, f), os.path.join(ROOT, f), f):
        if os.path.exists(p):
            return p
    sys.exit(f"ERROR: {f} not found.")


def load_cell(df, pid, spd, col, lo, hi):
    s = df[(df.Participant == pid) & (df.Speed_deg_per_s == spd)]
    x = s[col].values.astype(float)
    return x[(~np.isnan(x)) & (x >= lo) & (x <= hi)] / 1000.0


def fig_reciprobit(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (tag, col, lo, hi) in zip(axes, [("HRT", "HandRT_ms", 150, 800),
                                             ("SRT", "GazeSRT_ms", 80, 600)]):
        for spd in sorted(df.Speed_deg_per_s.unique()):
            pooled = []
            for pid in df.Participant.unique():
                pooled.append(load_cell(df, pid, int(spd), col, lo, hi))
            rt = np.concatenate([p for p in pooled if len(p)])
            if rt.size < 20:
                continue
            x, y = L.reciprobit_coords(rt)
            c = SPEED_COLORS.get(int(spd), (0.4, 0.4, 0.4))
            ax.plot(x, y, ".", ms=2.5, alpha=0.45, color=c,
                    label=f"{int(spd)} deg/s")
            f = L.later_mle(rt)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, (-xs - f["mu"]) / f["sigma"] * -1, "-", lw=1.6, color=c)
        ax.set_title(f"{EFF_LABEL[tag]}  —  reciprobit")
        ax.set_xlabel("-1 / RT  (s$^{-1}$)")
        ax.set_ylabel("probit(cumulative probability)")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Straight = LATER's Gaussian-rate assumption holds; "
                 "curvature = it does not", fontsize=9, y=1.0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(out)}")


def fig_qq(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (tag, col, lo, hi, floor) in zip(
            axes, [("HRT", "HandRT_ms", 150, 800, HRT_FLOOR),
                   ("SRT", "GazeSRT_ms", 80, 600, SRT_FLOOR)]):
        pooled = np.concatenate([load_cell(df, pid, int(spd), col, lo, hi)
                                 for pid in df.Participant.unique()
                                 for spd in df.Speed_deg_per_s.unique()])
        pooled = np.sort(pooled)
        n = pooled.size
        pp = (np.arange(1, n + 1) - 0.5) / n
        fw = fit_cell(pooled, floor, contam=0.0)
        q_w = fw["t0"] + stats.invgauss.ppf(
            pp, (fw["a"] / fw["v"]) / fw["a"] ** 2, scale=fw["a"] ** 2)
        fl = L.shifted_later_mle(pooled, 0.0)
        q_l = fl["t0"] + 1.0 / np.maximum(norm.ppf(1 - pp, fl["mu"], fl["sigma"]), 1e-9)
        ax.plot(pooled * 1000, q_w * 1000, ".", ms=2, alpha=0.5,
                color=(0.2, 0.35, 0.65), label="shifted Wald")
        ax.plot(pooled * 1000, q_l * 1000, ".", ms=2, alpha=0.5,
                color=(0.75, 0.35, 0.20), label="shifted LATER")
        lim = [pooled.min() * 1000 * 0.95, np.percentile(pooled, 99.5) * 1000]
        ax.plot(lim, lim, "k-", lw=0.8, alpha=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"{EFF_LABEL[tag]}  —  Q-Q, pooled")
        ax.set_xlabel("observed RT (ms)"); ax.set_ylabel("model quantile (ms)")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(out)}")


def fig_shift(out):
    p = os.path.join(HERE, "q2_later_vs_wald.csv")
    if not os.path.exists(p):
        print("  (skipping fig_shift_lrt.pdf — run q2 first)")
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    xs, labels = [], []
    for i, tag in enumerate(("HRT", "SRT")):
        s = d[d.effector == tag]
        if not len(s):
            continue
        jitter = np.random.default_rng(0).normal(0, 0.045, len(s))
        ax.plot(np.full(len(s), i) + jitter, s.shift_D, "o", ms=5, alpha=0.55,
                color=(0.2, 0.35, 0.65) if tag == "HRT" else (0.75, 0.35, 0.20))
        ax.plot([i - 0.2, i + 0.2], [s.shift_D.median()] * 2, "k-", lw=2.2)
        xs.append(i); labels.append(f"{EFF_LABEL[tag]}\n(n={len(s)})")
    ax.axhline(2.71, ls="--", lw=0.9, color="0.4")
    ax.text(1.45, 2.85, "p = 0.05", fontsize=8, color="0.4")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("LRT statistic for $t_0 > 0$  (LATER family)")
    ax.set_title("Does the RT distribution demand a non-decision term?", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(out)}")


def main():
    df = pd.read_csv(_need("pooled_data.csv"))
    df = df[df["BlockType"] == "I"]
    print("Q4  FIGURES")
    fig_reciprobit(df, os.path.join(HERE, "fig_reciprobit_by_effector.pdf"))
    fig_qq(df, os.path.join(HERE, "fig_model_qq.pdf"))
    fig_shift(os.path.join(HERE, "fig_shift_lrt.pdf"))


if __name__ == "__main__":
    main()
