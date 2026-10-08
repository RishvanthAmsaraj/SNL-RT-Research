"""
insights.py -- numbers that make each figure's explanation specific to the results being shown, and the "what this
comparison shows" notes for side by side. Everything is read from the tables the scripts wrote; nothing is modelled
here, and anything that cannot be computed is simply left out.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .engine import results


def _fmt(vals, unit=" ms"):
    return " / ".join(f"{v:.0f}" for v in vals) + unit


def _speeds(exp):
    return tuple(exp.speeds) if exp is not None else ()


def hand_bayes(rs, speeds):
    h = results.headline(rs, speeds)
    return h if "hand_t0" in h else {}


def method_a(rs, speeds):
    d = rs.table("DDM_hrt_fits.csv")
    if d is None or not {"spd", "t0"} <= set(d.columns): return {}
    return {"t0": {s: float(d[d.spd == s].t0.mean()) for s in speeds if (d.spd == s).any()},
            "floored": int((d.t0 <= 130.5).sum()), "cells": len(d)}


def saccade_models(rs):
    d = rs.table("DDM_srt_fits.csv")
    if d is None or "model" not in d: return {}
    out = {"cells": len(d), "mixture": int((d.model == "mixture").sum())}
    if "ks" in d: out["good"] = int((d.ks < 0.10).sum())
    return out


def floor_shares(rs):
    d = rs.table("HRT_floor_control.csv")
    if d is None or not {"effector", "eligible", "slope"} <= set(d.columns): return {}
    d = d[(d.eligible == True) & (d.get("ceiling_bound", False).fillna(False).astype(bool) == False)]
    out = {}
    for eff in ("HRT", "SRT"):
        v = d[d.effector == eff].slope
        if len(v): out[eff] = (int((v > 0.7).sum()), len(v))
    return out


def srt_sweep(rs):
    d = rs.table("SRT_identifiability.csv")
    if d is None or "slope" not in d: return {}
    ceil = d.get("ceiling_bound", pd.Series(False, index=d.index)).fillna(False).astype(bool)
    track = (d.slope > 0.7) & ~ceil
    return {"n": len(d), "track": int(track.sum()), "ceiling": int(ceil.sum()), "identified": int((~track & ~ceil).sum())}


def fixed_t0(rs):
    d = rs.table("SRT_fixedt0_sensitivity.csv")
    if d is None or not {"t0_fixed_ms", "mean_ks"} <= set(d.columns): return {}
    return {int(r.t0_fixed_ms): float(r.mean_ks) for r in d.itertuples()}


def _p(p) -> str:
    try: p = float(p)
    except (TypeError, ValueError): return "p unavailable"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def schematic(rs, table, speed):
    d = rs.table(table)
    if d is None or "spd" not in d: return {}
    if "model" in d: d = d[d.model == "single"]
    d = d[d.spd == speed]
    if not len(d) or not {"v", "a", "t0"} <= set(d.columns): return {}
    return {"v": float(d.v.mean()), "a": float(d.a.mean()), "t0": float(d.t0.mean()), "n": len(d)}


def later(rs):
    d = rs.table("LATER_fits.csv")
    if d is None or "reciprobit_r2" not in d: return {}
    out = {"r2": float(d.reciprobit_r2.median()), "cells": len(d)}
    if "express_frac" in d: out["express_cells"] = int((d.express_frac > 0.1).sum())
    return out


def readiness(rs):
    d = rs.table("two_boundary_readiness_hand.csv")
    if d is None or "minority_share" not in d: return {}
    return {"cells": len(d), "median_minority": float(d.minority_share.median())}


def step(rs, measure, paradigm, contrast):
    d = rs.table("P1_vs_P2_summary.csv")
    if d is None: return None
    r = d[(d.measure == measure) & (d.paradigm == paradigm) & (d.contrast.astype(str) == contrast)]
    return r.iloc[0] if len(r) else None


# ------------------------------------------------------------------ per-figure facts
def facts(stem: str, exp, rs) -> list[str]:
    sp = _speeds(exp); base = stem.split("__")[-1]; out = []
    if base == "NDT_barchart_bayesian":
        h = hand_bayes(rs, sp)
        if h:
            out.append(f"Hand t₀ is {_fmt(h['hand_t0'].values())} at {' / '.join(map(str, sp))} deg/s, with {h['hand_floored']} of "
                       f"{h['hand_cells']} cells on the 130 ms floor" + (f" (Friedman {_p(h['friedman_p'])})." if "friedman_p" in h else "."))
            if exp is not None and exp.id == "E1" and 0 in h["hand_t0"] and 75 in h["hand_t0"]:
                out.append(f"The stationary target's t₀ is {h['hand_t0'][0] - h['hand_t0'][75]:.1f} ms longer than at 75 deg/s.")
        s = results.headline(rs, sp).get("sacc")
        if s: out.append(f"Saccadic t₀: {s['floor']} of {s['n']} intervals reach the floor, {s['ceiling']} the ceiling, {s['free']} neither.")
    elif base == "Bayesian_srt_ndt":
        s = results.headline(rs, sp).get("sacc")
        if s:
            out.append(f"{s['floor']} of {s['n']} participants' intervals reach the 70 ms floor and {s['ceiling']} their fastest-saccade "
                       f"ceiling; {s['free']} {'sits' if s['free'] == 1 else 'sit'} clear of both. Estimates range {s['range'][0]}–{s['range'][1]} ms.")
    elif base == "Bayesian_summary":
        m, h = method_a(rs, sp), hand_bayes(rs, sp)
        if m and h: out.append(f"Method A puts {m['floored']} of {m['cells']} hand cells on the 130 ms floor; the Bayesian fit puts {h['hand_floored']}.")
        sm = saccade_models(rs)
        if sm: out.append(f"{sm['mixture']} of {sm['cells']} saccade cells needed the two-component (express/regular) model.")
    elif base in ("HRT_floor_control",):
        f = floor_shares(rs)
        if "HRT" in f: out.append(f"Hand: {f['HRT'][0]} of {f['HRT'][1]} eligible fits follow the floor.")
        if "SRT" in f: out.append(f"Saccade: {f['SRT'][0]} of {f['SRT'][1]} eligible fits follow the floor.")
    elif base == "SRT_identifiability":
        s = srt_sweep(rs)
        if s: out.append(f"Of {s['n']} single-component saccade cells, {s['track']} follow the floor, {s['ceiling']} "
                         f"{'is' if s['ceiling'] == 1 else 'are'} stuck at the fastest saccade and {s['identified']} "
                         f"{'is' if s['identified'] == 1 else 'are'} identified.")
    elif base == "SRT_fixedt0_sensitivity":
        k = fixed_t0(rs)
        if k: out.append("Mean KS with t₀ fixed at " + ", ".join(f"{t} ms: {v:.3f}" for t, v in sorted(k.items())) +
                         f" — a spread of {max(k.values()) - min(k.values()):.3f}.")
    elif base == "NDT_barchart" or base == "DDM_summary":
        m = method_a(rs, sp)
        if m: out.append(f"Method A hand t₀ is {_fmt(m['t0'].values())} by speed, with {m['floored']} of {m['cells']} cells on the floor.")
        sm = saccade_models(rs)
        if sm and base == "DDM_summary" and "good" in sm:
            out.append(f"{sm['good']} of {sm['cells']} saccade cells fit with KS below 0.10; {sm['mixture']} use the two-component model.")
    elif base == "LATER_reciprobit":
        l = later(rs)
        if l: out.append(f"Median reciprobit r² = {l['r2']:.3f} across {l['cells']} cells" +
                         (f"; {l['express_cells']} cells have more than 10% express saccades." if "express_cells" in l else "."))
    elif base == "two_boundary_readiness_hand":
        r = readiness(rs)
        if r: out.append(f"Median share of the less common response across {r['cells']} cells: {100 * r['median_minority']:.0f}%.")
    else:
        for kind, table in (("bayes_hrt_", "Bayesian_hrt_fits.csv"), ("bayes_srt_", "Bayesian_srt_fits.csv"),
                            ("ddm_hrt_", "DDM_hrt_fits.csv"), ("ddm_srt_", "DDM_srt_fits.csv")):
            if base.startswith(kind) and base.endswith("_degs"):
                try: speed = int(base[len(kind):-5])
                except ValueError: break
                g = schematic(rs, table, speed)
                if g: out.append(f"At {speed} deg/s the group means are v = {g['v']:.2f}, a = {g['a']:.2f} and t₀ = {g['t0']:.0f} ms "
                                 f"({g['n']} participants).")
                break
    if stem.startswith("P1_vs_P2_hand"):
        for par, con, label in (("P1", "75 - 0", "Experiment 1, stationary → 75"), ("P1", "150 - 75", "Experiment 1, 75 → 150"),
                                ("P2", "150 - 75", "Experiment 2, 75 → 150")):
            t, q = step(rs, "t0", par, con), step(rs, "hrt_p10", par, con)
            if t is not None and q is not None:
                out.append(f"{label}: model t₀ {t['mean']:+.1f} ms ({_p(t['wilcoxon_p'])}); fastest-10% RT {q['mean']:+.1f} ms "
                           f"({_p(q['wilcoxon_p'])}).")
    if stem.startswith("P1_vs_P2_saccade"):
        for par, label in (("P1", "Experiment 1"), ("P2", "Experiment 2")):
            r = step(rs, "srt_med", par, "150 - 75")
            if r is not None: out.append(f"{label}: median saccade RT {r['mean']:+.1f} ms from 75 to 150 deg/s ({_p(r['wilcoxon_p'])}).")
    return out


# ------------------------------------------------------------------ side by side
def compare(left: tuple, right: tuple) -> tuple[str, list[str]]:
    """left/right = (exp or None, ResultSet, stem, source label). Returns a heading and the notes."""
    (el, rl, sl, ll), (er, rr, sr, lr) = left, right
    bl, br = sl.split("__")[-1], sr.split("__")[-1]
    if sl == sr and ll == lr:
        return "Same figure in both panes", ["Both panes show the same figure; pick a different one on either side to compare."]
    if bl != br:
        return "Two different figures", ["These panes show different figures, so read each on its own terms; the notes under each "
                                          "image say what it shows. To compare the experiments, use one of the pairs above."]
    notes = []
    if el is not None and er is not None and el.id == er.id:
        notes.append(f"The same {el.name} figure from two sources ({ll} and {lr}). Differences reflect the data each was made from: "
                     "the scripts are identical.")
    names = (el.name if el else ll, er.name if er else lr)
    if bl == "NDT_barchart_bayesian":
        hl, hr = hand_bayes(rl, _speeds(el)), hand_bayes(rr, _speeds(er))
        if hl and hr:
            notes.append(f"Hand t₀: {names[0]} {_fmt(hl['hand_t0'].values())} at {' / '.join(map(str, el.speeds))} deg/s; "
                         f"{names[1]} {_fmt(hr['hand_t0'].values())} at {' / '.join(map(str, er.speeds))} deg/s.")
            shared = [s for s in hl["hand_t0"] if s in hr["hand_t0"]]
            if shared:
                notes.append("At the speeds both share: " + "; ".join(
                    f"{s} deg/s {hl['hand_t0'][s]:.0f} vs {hr['hand_t0'][s]:.0f} ms" for s in shared) + ".")
            notes.append("Different cohorts, so compare patterns rather than single values: Experiment 1's extra length is at the "
                         "stationary target, and between moving speeds both experiments are flat.")
        sl_, sr_ = results.headline(rl, _speeds(el)).get("sacc"), results.headline(rr, _speeds(er)).get("sacc")
        if sl_ and sr_:
            notes.append(f"Saccadic t₀: {names[0]} {sl_['floor']}/{sl_['n']} on the floor and {sl_['ceiling']} at the ceiling; "
                         f"{names[1]} {sr_['floor']}/{sr_['n']} on the floor and {sr_['ceiling']} at the ceiling — not identifiable in either.")
    elif bl == "Bayesian_srt_ndt":
        a, b = results.headline(rl, _speeds(el)).get("sacc"), results.headline(rr, _speeds(er)).get("sacc")
        if a and b:
            notes.append(f"{names[0]}: {a['floor']} of {a['n']} on the floor, {a['ceiling']} at the ceiling, {a['free']} clear. "
                         f"{names[1]}: {b['floor']} of {b['n']} on the floor, {b['ceiling']} at the ceiling, {b['free']} clear.")
            notes.append("Where one experiment's estimates press the ceiling and the other's the floor, both are still set by a bound: "
                         "neither experiment can place saccadic t₀, which is why it is fixed at 70 ms in both.")
    elif bl == "HRT_floor_control":
        a, b = floor_shares(rl), floor_shares(rr)
        for eff, word in (("HRT", "Hand"), ("SRT", "Saccade")):
            if eff in a and eff in b:
                notes.append(f"{word} fits following the floor: {names[0]} {a[eff][0]}/{a[eff][1]} ({100 * a[eff][0] / a[eff][1]:.0f}%), "
                             f"{names[1]} {b[eff][0]}/{b[eff][1]} ({100 * b[eff][0] / b[eff][1]:.0f}%).")
        notes.append("The same pattern in both experiments — hand t₀ comes from the data, saccadic t₀ largely from the floor — is what "
                     "makes the identifiability result general rather than a quirk of one cohort.")
    elif bl == "Bayesian_summary":
        for e, r, n in ((el, rl, names[0]), (er, rr, names[1])):
            m, h, smod = method_a(r, _speeds(e)), hand_bayes(r, _speeds(e)), saccade_models(r)
            if m and h: notes.append(f"{n}: Method A floors {m['floored']}/{m['cells']} hand cells, the Bayesian fit {h['hand_floored']}; "
                                     f"{smod.get('mixture', 0)} saccade cells need the two-component model.")
    elif bl == "SRT_identifiability":
        for r, n in ((rl, names[0]), (rr, names[1])):
            s = srt_sweep(r)
            if s: notes.append(f"{n}: {s['track']} of {s['n']} cells follow the floor, {s['ceiling']} stuck at the fastest saccade, "
                               f"{s['identified']} identified.")
    elif bl == "SRT_fixedt0_sensitivity":
        for r, n in ((rl, names[0]), (rr, names[1])):
            k = fixed_t0(r)
            if k: notes.append(f"{n}: KS spread across assumed t₀ values {max(k.values()) - min(k.values()):.3f} — the data cannot choose a t₀.")
    elif bl == "LATER_reciprobit":
        a, b = later(rl), later(rr)
        if a and b:
            notes.append(f"Median reciprobit r²: {names[0]} {a['r2']:.3f}, {names[1]} {b['r2']:.3f}. LATER describes both cohorts' "
                         "saccades well, but it remains deprecated because its parameters cannot be compared with the hand model.")
    elif bl.startswith(("bayes_hrt_", "bayes_srt_", "ddm_hrt_", "ddm_srt_")):
        table = {"bayes_hrt_": "Bayesian_hrt_fits.csv", "bayes_srt_": "Bayesian_srt_fits.csv", "ddm_hrt_": "DDM_hrt_fits.csv",
                 "ddm_srt_": "DDM_srt_fits.csv"}[next(k for k in ("bayes_hrt_", "bayes_srt_", "ddm_hrt_", "ddm_srt_") if bl.startswith(k))]
        try: speed = int(bl.split("_")[-2])
        except ValueError: speed = None
        a, b = (schematic(rl, table, speed), schematic(rr, table, speed)) if speed is not None else ({}, {})
        if a and b:
            notes.append(f"At {speed} deg/s: {names[0]} v = {a['v']:.2f}, a = {a['a']:.2f}, t₀ = {a['t0']:.0f} ms; "
                         f"{names[1]} v = {b['v']:.2f}, a = {b['a']:.2f}, t₀ = {b['t0']:.0f} ms.")
    elif bl.startswith("vincentile") or bl == "why_saccadic_t0_floors":
        notes.append("Model-free or shape-based figures for two cohorts of the same task: matching shapes mean the cohorts behave "
                     "alike, so differences in model results are not explained by very different data.")
    if not notes:
        notes.append("The same figure for both sides. Read the two with the same scale in mind; differences between cohorts are "
                     "between groups of people, not a within-person change.")
    return "What this comparison shows", notes
