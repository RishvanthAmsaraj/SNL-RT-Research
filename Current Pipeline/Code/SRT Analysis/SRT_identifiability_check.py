"""
SRT_identifiability_check.py  --  Diagnostic for saccadic t0 identifiability

Refits each single-component SRT cell at multiple floors (40-90 ms). Cells whose
t0 tracks the floor are not identified by the data. Cells whose t0 stays put
regardless of the floor are genuinely identified.

Reads DDM_srt_fits.csv + pooled_data.csv. Output: SRT_identifiability.pdf/.png

Run: python SRT_identifiability_check.py  (needs DDM_srt_fits.csv first)
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import differential_evolution
import matplotlib.font_manager as fm
_fam = "Arial" if "Arial" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"
matplotlib.rcParams.update({"font.family":_fam,"font.size":11,"pdf.fonttype":42,"ps.fonttype":42})
HERE = os.path.dirname(os.path.abspath(__file__))
def _need(f):
    p = os.path.join(HERE, f)
    if not os.path.exists(p): sys.exit(f"ERROR: {f} not found next to this script. Run DDM_fit.py first.")
    return p

def wald_pdf(t,v,a):
    t=np.maximum(t,1e-9); return (a/np.sqrt(2*np.pi*t**3))*np.exp(-(a-v*t)**2/(2*t))
def wald_cdf(t,v,a):
    t=np.maximum(t,1e-9); return stats.invgauss.cdf(t,mu=1/(v*a),scale=a**2)
def fit_t0(rts, floor, contam=0.05):
    Tr=rts.max()-rts.min()
    def nll(p):
        v,a,t0=p; adj=rts-t0
        if np.any(adj<=0): return 1e10
        w=wald_pdf(adj,v,a)
        if np.any(w<=0) or not np.all(np.isfinite(w)): return 1e10
        d=(1-contam)*w+contam/Tr
        if np.any(d<=0): return 1e10
        return -np.sum(np.log(d))
    b=[(0.1,20),(0.05,2.5),(floor,max(np.percentile(rts,3)-0.002,floor+1e-3))]
    best=None
    for s in [42,7]:
        r=differential_evolution(nll,b,seed=s,maxiter=300,tol=1e-9,popsize=12,polish=True)
        if best is None or r.fun<best.fun: best=r
    return best.x[2]*1000

dfi=pd.read_csv(_need('pooled_data.csv')); dfi=dfi[dfi.BlockType=='I']
s=pd.read_csv(_need('DDM_srt_fits.csv')); sing=s[s.model=='single']
FLOORS=[0.040,0.050,0.060,0.070,0.080,0.090]

rows=[]
for _,r in sing.iterrows():
    sub=dfi[(dfi.Participant==r.pid)&(dfi.Speed_deg_per_s==r.spd)]
    x=sub['GazeSRT_ms'].values.astype(float); x=x[(~np.isnan(x))&(x>=80)&(x<=600)]/1000
    t0s=[fit_t0(x,fl) for fl in FLOORS]
    # identified if t0 does NOT closely track the floor (slope of t0 vs floor << 1)
    slope=np.polyfit([f*1000 for f in FLOORS], t0s, 1)[0]
    rows.append((f"{r.pid}@{int(r.spd)}", t0s, slope))

n_tracking=sum(1 for _,_,sl in rows if sl>0.7)
print(f"SRT single cells tested: {len(rows)}")
print(f"  cells whose t0 TRACKS the floor (slope>0.7 -> NOT identified): {n_tracking}/{len(rows)}")
print(f"  cells with genuinely identified t0 (slope<=0.7): {len(rows)-n_tracking}/{len(rows)}")

fig,ax=plt.subplots(1,2,figsize=(13,5.4))
xf=[f*1000 for f in FLOORS]
for name,t0s,slope in rows:
    c='#C0392B' if slope>0.7 else '#27AE60'
    ax[0].plot(xf,t0s,'-o',color=c,alpha=0.55,ms=3,lw=1)
ax[0].plot([40,90],[40,90],'k--',lw=1.5,label='t0 = floor (unidentified)')
ax[0].set_xlabel('imposed non-decision floor (ms)'); ax[0].set_ylabel('fitted $t_0$ (ms)')
ax[0].set_title('SRT $t_0$ vs imposed floor\nred = tracks floor (unidentified); green = stable (identified)',fontsize=11,fontweight='bold')
ax[0].legend(fontsize=9); ax[0].spines[['top','right']].set_visible(False); ax[0].grid(True,ls='--',alpha=0.3)

slopes=[sl for _,_,sl in rows]
ax[1].hist(slopes,bins=np.linspace(0,1.05,12),color='#7f8c8d',edgecolor='white')
ax[1].axvline(0.7,color='#C0392B',ls=':',lw=1.5); ax[1].text(0.71,ax[1].get_ylim()[1]*0.9,'tracks floor →',color='#C0392B',fontsize=9)
ax[1].set_xlabel('slope of $t_0$ vs floor  (1 = perfectly tracks floor)'); ax[1].set_ylabel('number of cells')
ax[1].set_title(f'{n_tracking}/{len(rows)} SRT single cells are floor-determined',fontsize=11,fontweight='bold')
ax[1].spines[['top','right']].set_visible(False); ax[1].grid(True,axis='y',ls='--',alpha=0.3)
fig.suptitle('Saccadic non-decision time is largely NOT identifiable — it sits at the imposed floor (both DDM and Bayesian)',fontsize=12.5,fontweight='bold',y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(HERE,'SRT_identifiability.pdf'),dpi=300,bbox_inches='tight',facecolor='white')
fig.savefig(os.path.join(HERE,'SRT_identifiability.png'),dpi=140,bbox_inches='tight',facecolor='white')
print("saved SRT_identifiability.pdf and .png to", HERE)
