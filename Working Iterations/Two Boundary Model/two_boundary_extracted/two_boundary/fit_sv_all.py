import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude'); sys.path.insert(0,'/home/claude/two_boundary')
from ddm_sv import fit_sv, loglik_sv
R=pd.read_csv('real_2b_mle.csv'); D=pd.read_csv('t0_decomposition.csv')
P=pd.read_csv('/mnt/user-data/uploads/pooled_data.csv'); I=P[P.BlockType=='I']
b=pd.DataFrame({'pid':I.Participant.astype(str),'spd':I.Speed_deg_per_s.astype(float),
                'rt':I.HandRT_ms/1000.,'e':I.SignedError_deg}).dropna()
b=b[(b.rt>=.150)&(b.rt<=.800)&(b.e!=0)]; b['choice']=(b.e>0).astype(float)
out=[]
for _,r in R.iterrows():
    g=b[(b.pid==r.pid)&(b.spd==r.spd)]; rt,ch=g.rt.values,g.choice.values
    w=D[(D.pid==r.pid)&(D.spd==r.spd)].iloc[0]
    (v,sv,a,ww,t0),ll = fit_sv(rt,ch,(r.v,r.a,r.w,r.t0_ms/1000.))
    tau=rt-w.wald_sub_t0/1000.
    llw=(np.log(w.wald_sub_a)-.5*np.log(2*np.pi)-1.5*np.log(tau)
         -(w.wald_sub_a-w.wald_sub_v*tau)**2/(2*tau)).sum()
    p=min(max(ch.mean(),1e-9),1-1e-9); llb=(ch*np.log(p)+(1-ch)*np.log(1-p)).sum()
    ll2=loglik_sv(rt,ch,r.v,0.0,r.a,r.w,r.t0_ms/1000.)
    out.append(dict(pid=r.pid,spd=int(r.spd),n=len(rt),v=v,sv=sv,a=a,w=ww,t0_ms=t0*1000,
        ll_sv=ll, ll_2b=ll2, ll_wb=llw+llb,
        aic_sv=-2*ll+10, aic_2b=-2*ll2+8, aic_wb=-2*(llw+llb)+8,
        t0_2b=r.t0_ms, t0_wald=w.wald_sub_t0))
    o=out[-1]
    print(f"  {r.pid:<9}{int(r.spd):>4}  sv {sv:5.2f}  t0 {t0*1000:6.1f} (2B {r.t0_ms:6.1f}, Wald {w.wald_sub_t0:6.1f})  "
          f"AIC sv {o['aic_sv']:8.1f} | 2B {o['aic_2b']:8.1f} | W*B {o['aic_wb']:8.1f}",flush=True)
pd.DataFrame(out).to_csv('/home/claude/fit_sv_all.csv',index=False)
print("DONE",flush=True)
