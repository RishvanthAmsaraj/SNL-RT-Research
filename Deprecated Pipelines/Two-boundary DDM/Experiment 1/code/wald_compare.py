"""Fit the single-boundary shifted Wald and the two-boundary DDM to IDENTICAL trials,
so any t0 difference is attributable to the model and not to the trial set."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/two_boundary')
from scipy.optimize import differential_evolution
from wfpt import numpy_wfpt_logpdf_vec

FLOOR=0.130
def hi_t0(rt): return max(np.percentile(rt,3)-0.002, FLOOR+1e-3)

def fit_wald(rt, seed=42):
    def nll(p):
        v,a,t0=p; tau=rt-t0
        if np.any(tau<=0): return 1e10
        lp=np.log(a)-0.5*np.log(2*np.pi)-1.5*np.log(tau)-(a-v*tau)**2/(2*tau)
        return 1e10 if not np.all(np.isfinite(lp)) else -float(lp.sum())
    b=[(0.01,20.),(0.05,2.5),(FLOOR,hi_t0(rt))]
    best=None
    for s in (seed,seed+5):
        r=differential_evolution(nll,b,seed=s,maxiter=150,tol=1e-8,popsize=15,polish=True)
        if best is None or r.fun<best.fun: best=r
    return best.x

def fit_2b(rt, ch, seed=42):
    def nll(p):
        v,a,w,t0=p; tau=rt-t0
        if np.any(tau<=0): return 1e10
        lp=numpy_wfpt_logpdf_vec(tau,v,a,w,ch)
        return 1e10 if not np.all(np.isfinite(lp)) else -float(lp.sum())
    b=[(-20.,20.),(0.05,5.),(0.02,0.98),(FLOOR,hi_t0(rt))]
    best=None
    for s in (seed,seed+5):
        r=differential_evolution(nll,b,seed=s,maxiter=150,tol=1e-8,popsize=15,polish=True)
        if best is None or r.fun<best.fun: best=r
    return best.x

P=pd.read_csv('/mnt/user-data/uploads/pooled_data.csv'); I=P[P.BlockType=='I']
base=pd.DataFrame({'pid':I.Participant.astype(str),'spd':I.Speed_deg_per_s.astype(float),
                   'rt':I.HandRT_ms/1000.,'e':I.SignedError_deg})
allt=base.dropna(subset=['pid','spd','rt'])
allt=allt[(allt.rt>=.150)&(allt.rt<=.800)]                      # every usable hand trial
sub =allt.dropna(subset=['e']); sub=sub[sub.e!=0].copy()        # only those with an outcome
sub['choice']=(sub.e>0).astype(float)
print(f"ALL hand trials: {len(allt)}   |   SUBSET with SignedError: {len(sub)}",flush=True)

rows=[]
for (p,s),g in sub.groupby(['pid','spd']):
    ga=allt[(allt.pid==p)&(allt.spd==s)]
    w_sub=fit_wald(g.rt.values); w_all=fit_wald(ga.rt.values)
    b=fit_2b(g.rt.values,g.choice.values)
    rows.append(dict(pid=p,spd=int(s),n_sub=len(g),n_all=len(ga),
        wald_all_t0=w_all[2]*1000, wald_all_v=w_all[0], wald_all_a=w_all[1],
        wald_sub_t0=w_sub[2]*1000, wald_sub_v=w_sub[0], wald_sub_a=w_sub[1],
        b2_t0=b[3]*1000, b2_v=b[0], b2_a=b[1], b2_w=b[2],
        minrt_sub=g.rt.min()*1000, minrt_all=ga.rt.min()*1000))
    print(f"  {p:<9}{int(s):>4}  n {len(g):>4}/{len(ga):<4} "
          f"waldALL {w_all[2]*1000:6.1f}  waldSUB {w_sub[2]*1000:6.1f}  2B {b[3]*1000:6.1f}",flush=True)
pd.DataFrame(rows).to_csv('/home/claude/t0_decomposition.csv',index=False)
print("DONE",flush=True)
