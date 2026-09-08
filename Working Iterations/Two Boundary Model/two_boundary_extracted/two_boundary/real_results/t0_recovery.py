"""Decisive test: if the data really come from a two-boundary process, does the Wald
UNDER-estimate t0? And if they really come from a Wald, does the two-boundary model
OVER-estimate it? Simulate from each, fit both, compare."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/two_boundary')
from wfpt import sample_ddm
from wald_compare import fit_wald, fit_2b

rng=np.random.default_rng(5); REPS=20; N=120; T0=0.170
res=[]
print("A) TRUTH = two-boundary DDM (v=2.5, a=0.58, w=0.40, t0=170 ms)",flush=True)
for r in range(REPS):
    rt,ch=sample_ddm(N,2.5,0.58,0.40,T0,rng=rng)
    ok=np.isfinite(rt)&(rt>=.150)&(rt<=.800); rt,ch=rt[ok],ch[ok]
    if len(rt)<60 or min(ch.sum(),len(ch)-ch.sum())<5: continue
    res.append(dict(truth='2B', wald=fit_wald(rt)[2]*1000, twob=fit_2b(rt,ch.astype(float))[3]*1000))
d=pd.DataFrame([x for x in res if x['truth']=='2B'])
print(f"   Wald recovers t0 = {d.wald.mean():6.1f} ms (bias {d.wald.mean()-T0*1000:+6.1f}), floored {(d.wald<=131).sum()}/{len(d)}",flush=True)
print(f"   2B   recovers t0 = {d.twob.mean():6.1f} ms (bias {d.twob.mean()-T0*1000:+6.1f}), floored {(d.twob<=131).sum()}/{len(d)}",flush=True)

print("\nB) TRUTH = single-boundary Wald (v=9, a=0.83, t0=170 ms)",flush=True)
from scipy.stats import invgauss
res2=[]
for r in range(REPS):
    v,a=9.0,0.83
    tau=invgauss.rvs(mu=(a/v)/(a**2), scale=a**2, size=N, random_state=rng.integers(1e9))
    rt=tau+T0
    ok=(rt>=.150)&(rt<=.800); rt=rt[ok]
    if len(rt)<60: continue
    ch=(rng.random(len(rt))<0.62).astype(float)   # choice independent of RT
    res2.append(dict(wald=fit_wald(rt)[2]*1000, twob=fit_2b(rt,ch)[3]*1000))
d2=pd.DataFrame(res2)
print(f"   Wald recovers t0 = {d2.wald.mean():6.1f} ms (bias {d2.wald.mean()-T0*1000:+6.1f}), floored {(d2.wald<=131).sum()}/{len(d2)}",flush=True)
print(f"   2B   recovers t0 = {d2.twob.mean():6.1f} ms (bias {d2.twob.mean()-T0*1000:+6.1f}), floored {(d2.twob<=131).sum()}/{len(d2)}",flush=True)
pd.concat([d,d2.assign(truth='Wald')]).to_csv('/home/claude/t0_recovery.csv',index=False)
print("DONE",flush=True)
