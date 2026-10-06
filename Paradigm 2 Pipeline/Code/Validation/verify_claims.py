"""
verify_claims.py -- re-derives the key numbers quoted in the Paradigm 2 documents from the result tables and raw trials
(fresh code: pandas/scipy, not the pipeline's own functions) and checks that each one appears, as written, in every
document that quotes it. Run from anywhere inside the repo:  python verify_claims.py   ->  verify_claims_report.csv
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, fisher_exact
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P1C, P2C = os.path.join(REPO, "Current Pipeline", "Code"), os.path.join(REPO, "Paradigm 2 Pipeline", "Code")
DOCS = {k: os.path.join(REPO, "Paradigm 2 Pipeline", "Documents", f) for k, f in
        [("R", "P2_Results_and_P1_Comparison.md"), ("B", "P2_Technical_Breakdown.md"), ("G", "FIGURE_GUIDE.md"), ("V", "VERIFICATION_REPORT_P2.md")]}
TXT = {k: open(p).read().replace("\u2212", "-").replace("**", "") for k, p in DOCS.items()}
rows = []
def claim(cid, what, value, docs, alt=()):
    found = [k for k in docs if value in TXT[k] or any(a in TXT[k] for a in alt)]
    rows.append(dict(id=cid, claim=what, value=value, quoted_in=",".join(docs), found_in=",".join(found), status="PASS" if len(found) == len(docs) else "FAIL"))
f1 = lambda x: f"{x:.1f}"; s1 = lambda x: f"{x:+.1f}"; f0 = lambda x: f"{x:.0f}"
def boot(x):
    r = np.random.default_rng(0); b = [r.choice(x, len(x)).mean() for _ in range(3000)]; return np.percentile(b, [2.5, 97.5])
# ---------------- raw data
p1 = pd.read_csv(os.path.join(REPO, "Working Iterations", "SNL RT Research", "pooled_data.csv")); p1 = p1[p1.BlockType == "I"]
p2 = pd.read_csv(os.path.join(P2C, "pooled_data_P2.csv"), usecols=["Participant", "BlockType", "Speed_deg_per_s", "HandRT_ms", "GazeSRT_ms"])
raw = {"P1": p1, "P2": p2}
claim("D1", "P2 hand RTs kept", f"{int(p2.HandRT_ms.between(150, 800).sum()):,}", "RB"); claim("D2", "P2 saccade RTs kept", f"{int(p2.GazeSRT_ms.between(80, 600).sum()):,}", "RB")
claim("D3", "P1 interception trials", f"{len(p1):,}", "RV"); claim("D4", "P1 hand kept", f"{int(p1.HandRT_ms.between(150, 800).sum()):,}", "RV")
claim("D5", "P1 saccade kept", f"{int(p1.GazeSRT_ms.between(80, 600).sum()):,}", "RV")
cell = {}
for P, d in raw.items():
    h = d[d.HandRT_ms.between(150, 800)].groupby(["Participant", "Speed_deg_per_s"]).HandRT_ms; s = d[d.GazeSRT_ms.between(80, 600)].groupby(["Participant", "Speed_deg_per_s"]).GazeSRT_ms
    cell[P] = dict(med=h.median().unstack(), p10=h.quantile(0.10).unstack(), smed=s.median().unstack())
B = {P: pd.read_csv(os.path.join(P1C if P == "P1" else P2C, "Bayesian", "Bayesian_hrt_fits.csv")) for P in ["P1", "P2"]}
for P in B: B[P]["dt"] = 1000 * B[P].a / B[P].v
sp = {"P1": [0, 75, 150], "P2": [75, 100, 125, 150]}
claim("H1", "P2 Bayesian hand t0 by speed", " / ".join(f1(B["P2"][B["P2"].spd == s].t0.mean()) for s in sp["P2"]), "BG")
claim("H2", "P1 Bayesian hand t0 by speed", " -> ".replace("->", "\u2192").join(f1(B["P1"][B["P1"].spd == s].t0.mean()) for s in sp["P1"]), "G")
for P, k in [("P1", "H3"), ("P2", "H4")]:
    claim(k, f"{P} decision time a/v by speed", " / ".join(f1(B[P][B[P].spd == s].dt.mean()) for s in sp[P]), "G")
    j = " / " if P == "P2" else " \u2192 "
    claim(k + "m", f"{P} raw median HRT by speed", j.join(f1(cell[P]["med"][s].mean()) for s in sp[P]), "G")
    claim(k + "q", f"{P} raw 10th-pct HRT by speed", j.join(f1(cell[P]["p10"][s].mean()) for s in sp[P]), "G")
    claim(k + "s", f"{P} raw median SRT by speed", " \u2192 ".join(f1(cell[P]["smed"][s].mean()) for s in sp[P]), "G")
w2 = B["P2"].pivot_table(index="pid", columns="spd", values="t0"); w1 = B["P1"].pivot_table(index="pid", columns="spd", values="t0")
claim("H5", "P2 Bayesian hand Friedman p", f"{friedmanchisquare(*[w2[s] for s in sp['P2']]).pvalue:.3f}", "RBG")
claim("H6", "P1 Bayesian hand Friedman p", f"{friedmanchisquare(*[w1[s] for s in sp['P1']]).pvalue:.3f}", "G")
x = (np.array(sp["P2"]) - 75) / 25; slopes = np.array([np.polyfit(x, w2.loc[p, sp["P2"]].values, 1)[0] for p in w2.index])
claim("H7", "P2 per-participant slope Wilcoxon p", f"{wilcoxon(slopes).pvalue:.3f}", "RBG")
for P, a, b, tag in [("P1", 0, 75, "S1"), ("P1", 75, 150, "S2"), ("P2", 75, 150, "S3")]:
    wt = B[P].pivot_table(index="pid", columns="spd", values="t0"); wd = B[P].pivot_table(index="pid", columns="spd", values="dt")
    dt0, ddt = wt[b] - wt[a], wd[b] - wd[a]
    claim(tag + "t", f"{P} {a}->{b} t0 change", s1(dt0.mean()), "RG"); claim(tag + "d", f"{P} {a}->{b} decision-time change", s1(ddt.mean()), "RG")
    for nm, mm in [("median", cell[P]["med"]), ("p10", cell[P]["p10"])]:
        chg = (mm[b] - mm[a]); lo_, hi_ = boot(chg.values)
        claim(tag + nm, f"{P} {a}->{b} raw {nm} HRT change", s1(chg.mean()), "RG")
        if P == "P2" and nm == "p10": claim(tag + "p10ci", "P2 fast-end change with its interval", f"{s1(chg.mean())} [{f1(lo_)}, {s1(hi_)}]", "RG", alt=(f"{s1(chg.mean())} ms [{f1(lo_)}, {s1(hi_)}]",))
    lo, hi = boot((dt0).values); claim(tag + "ci", f"{P} {a}->{b} t0 change CI", f"[{s1(lo) if lo > 0 else f1(lo)}, {s1(hi) if hi > 0 else f1(hi)}]", "RV")
    if b == 150: claim(tag + "r", f"{P} corr(delta t0, delta decision)", f"{np.corrcoef(dt0, ddt)[0, 1]:.2f}", "RG")
lo, hi = boot((cell["P2"]["p10"][150] - cell["P2"]["p10"][75]).values); claim("S3pci", "P2 fast-end change CI", f"[{f1(lo)}, {s1(hi)}]", "RG")
for P, a, tag in [("P1", 75, "E1"), ("P2", 75, "E2")]:
    claim(tag, f"{P} 75->150 SRT median change", s1((cell[P]["smed"][150] - cell[P]["smed"][a]).mean()), "RG")
# ---------------- saccadic t0 (participant level) and bounds
nd = {P: pd.read_csv(os.path.join(P1C if P == "P1" else P2C, "Bayesian", "Bayesian_srt_ndt.csv")) for P in ["P1", "P2"]}
fl = {P: int((nd[P].t0_lo95 <= 71).sum()) for P in nd}; ce = {P: int((nd[P].t0_hi95 >= nd[P].min_srt_ms - 2).sum()) for P in nd}
claim("N1", "P2 participants at floor", f"{fl['P2']}/{len(nd['P2'])}", "RBG", alt=(f"{fl['P2']} of {len(nd['P2'])}",)); claim("N2", "P2 participants at ceiling", f"{ce['P2']}/{len(nd['P2'])}", "RBG")
claim("N3", "P1 participants at floor", f"{fl['P1']}/{len(nd['P1'])}", "RG", alt=(f"all {fl['P1']} at the floor",) if fl["P1"] == len(nd["P1"]) else ())
fr = nd["P2"][(nd["P2"].t0_lo95 > 71) & (nd["P2"].t0_hi95 < nd["P2"].min_srt_ms - 2)]
claim("N4", "the one free participant", f"{fr.pid.iloc[0]} ({int(fr.t0_ms.iloc[0])} ms [{int(fr.t0_lo95.iloc[0])}, {int(fr.t0_hi95.iloc[0])}])", "RB")
claim("N5", "ceiling-bind Fisher p", f"{fisher_exact([[ce['P1'], len(nd['P1']) - ce['P1']], [ce['P2'], len(nd['P2']) - ce['P2']]])[1]:.3f}", "RBG")
for P, k in [("P1", "F1"), ("P2", "F2")]:
    t = pd.read_csv(os.path.join(P2C, "Comparison", "P1_floor_sweep_rerun.csv") if P == "P1" else os.path.join(P2C, "Supplementary", "HRT_floor_control.csv"))
    t = t[(t.eligible == True) & (t.ceiling_bound.fillna(False).astype(bool) == False)]
    for eff in ["HRT", "SRT"]:
        v = t[t.effector == eff].slope.values; claim(f"{k}{eff}", f"{P} {eff} fits following the floor", f"{int((v > 0.7).sum())}/{len(v)}", "RG")
# ---------------- NDT charts (Bayesian and Method A)
for P, k in [("P1", "G1"), ("P2", "G2")]:
    s = pd.read_csv(os.path.join(P1C if P == "P1" else P2C, "Bayesian", "Bayesian_srt_fits.csv")); s = s[s.model == "single"]
    claim(k + "e", f"{P} Bayesian eye t0 by speed", " / ".join(f0(s[s.spd == q].t0.mean()) for q in sp[P]), "G")
    claim(k + "f", f"{P} Bayesian eye estimates on floor", f"{int(((s.t0 - 70) < 2).sum())}/{len(s)}", "G")
    h = pd.read_csv(os.path.join(P1C if P == "P1" else P2C, "DDM", "DDM_hrt_fits.csv"))
    claim(k + "a", f"{P} Method A hand t0 by speed", " / ".join(f0(h[h.spd == q].t0.mean()) for q in sp[P]), "G")
    claim(k + "af", f"{P} Method A hand cells on floor", f"{int((h.t0 <= 130.5).sum())}/{len(h)}", "G")
# ---------------- eye-hand lag and pooled distributions
for P, k in [("P1", "L1"), ("P2", "L2")]:
    d = raw[P]; d = d[d.HandRT_ms.between(150, 800) & d.GazeSRT_ms.between(80, 600)]
    lag = (d.HandRT_ms - d.GazeSRT_ms).groupby([d.Participant, d.Speed_deg_per_s]).mean().unstack()
    vals = [f1(lag[q].mean()) for q in sp[P]]
    claim(k, f"{P} mean eye-hand lag by speed", " \u2192 ".join(vals), "G", alt=((vals[0] + " (stationary) \u2192 " + " \u2192 ".join(vals[1:])),) if P == "P1" else ())
    for col, lo_, hi_ in [("HandRT_ms", 150, 800), ("GazeSRT_ms", 80, 600)]:
        for q in [75, 150]:
            z = raw[P][(raw[P].Speed_deg_per_s == q) & raw[P][col].between(lo_, hi_)][col]
            claim(f"{k}{col[:4]}{q}", f"{P} pooled {col} at {q}: median (n)", f"{np.median(z):.0f} (n = {len(z):,})", "G")
# ---------------- recovery and parity tables
rc = pd.read_csv(os.path.join(P2C, "Validation", "bayesian_recovery_P2_summary.csv")).set_index("quantity").value
claim("V1", "recovery: participants detecting the drop", f"{float(rc['delta t0 participants negative']):.0f}/16", "RV")
claim("V2", "recovery: recovered drop", s1(float(rc["delta t0 150-75, recovered (truth -10.0)"])), "RV")
claim("V3", "recovery: decision-time artefact", s1(float(rc["delta decision time a/v 150-75, recovered (truth 0)"])), "RV")
claim("V4", "recovery: noise-only correlation", f"{float(rc['corr(delta t0, delta decision time) with no true variation']):.2f}", "RVG")
claim("V5", "recovery: coverage", f"{float(rc['cell 95% interval coverage (%)']):.0f}%", "RVG")
pr = pd.read_csv(os.path.join(P2C, "Validation", "P1_parity_results.csv")).set_index("table")
claim("V6", "P1 parity Bayesian r", f"{pr.loc['Bayesian_hrt_fits.csv', 'corr_t0']:.4f}", "RB")
out = pd.DataFrame(rows); out.to_csv(os.path.join(HERE, "verify_claims_report.csv"), index=False)
print(f"{(out.status == 'PASS').sum()}/{len(out)} claims PASS")
if (out.status == "FAIL").any(): print(out[out.status == "FAIL"][["id", "claim", "value", "quoted_in", "found_in"]].to_string(index=False))
