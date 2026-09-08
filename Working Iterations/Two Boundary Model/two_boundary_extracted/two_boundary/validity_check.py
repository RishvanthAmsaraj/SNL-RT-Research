"""Is the head-to-head comparison biased against the two-boundary model?
Simulate data FROM the fitted 2B parameters, then run the identical comparison.
If 2B does not win on data it generated itself, the procedure is broken."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/two_boundary')
from wfpt import sample_ddm, numpy_wfpt_logpdf_vec
from wald_compare import fit_wald, fit_2b
R=pd.read_csv('/home/claude/real_2b_mle.csv'); rng=np.random.default_rng(17)
out=[]
for _,r in R.iterrows():
    rt,ch=sample_ddm(int(r.n), r.v, r.a, r.w, r.t0_ms/1000., rng=rng)
    ok=np.isfinite(rt)&(rt>=.150)&(rt<=.800); rt,ch=rt[ok],ch[ok].astype(float)
    if len(rt)<50 or min(ch.sum(),len(ch)-ch.sum())<3: continue
    e2=fit_2b(rt,ch); ew=fit_wald(rt)
    ll2=numpy_wfpt_logpdf_vec(rt-e2[3],e2[0],e2[1],e2[2],ch).sum()
    tau=rt-ew[2]
    llw=(np.log(ew[1])-0.5*np.log(2*np.pi)-1.5*np.log(tau)-(ew[1]-ew[0]*tau)**2/(2*tau)).sum()
    p=min(max(ch.mean(),1e-9),1-1e-9); llb=(ch*np.log(p)+(1-ch)*np.log(1-p)).sum()
    out.append(dict(pid=r.pid,spd=r.spd,n=len(rt),d=ll2-(llw+llb),
                    t0_true=r.t0_ms,t0_2b=e2[3]*1000,t0_wald=ew[2]*1000))
    print(f"  {r.pid:<9}{int(r.spd):>4}  dLL {ll2-(llw+llb):+8.2f}   t0 true {r.t0_ms:6.1f} -> 2B {e2[3]*1000:6.1f}  Wald {ew[2]*1000:6.1f}",flush=True)
O=pd.DataFrame(out); O.to_csv('/home/claude/validity_check.csv',index=False)
print(f"\n2B wins on its OWN simulated data in {(O.d>0).sum()}/{len(O)} cells, mean dLL {O.d.mean():+.2f}",flush=True)
print(f"t0 recovery on own data: 2B bias {(O.t0_2b-O.t0_true).mean():+.1f} ms, Wald bias {(O.t0_wald-O.t0_true).mean():+.1f} ms",flush=True)
print("DONE",flush=True)
