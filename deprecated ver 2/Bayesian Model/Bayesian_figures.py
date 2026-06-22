"""Bayesian_figures.py -- figures for the Bayesian model (HRT + SRT).
Reads Bayesian_hrt_fits.csv, DDM_hrt_fits.csv, Bayesian_srt_fits.csv, DDM_srt_fits.csv.
Produces Bayesian_summary.pdf/.png."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
_fam = "Arial" if "Arial" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"
matplotlib.rcParams.update({"font.family":_fam,"font.size":11,"pdf.fonttype":42,"ps.fonttype":42})
HERE = os.path.dirname(os.path.abspath(__file__))
def _need(f):
    p = os.path.join(HERE, f)
    if not os.path.exists(p): sys.exit(f"ERROR: {f} not found next to this script. Run DDM_fit.py, Bayesian_HRT_fit.py and Bayesian_SRT_fit.py first.")
    return p
SC={0:(0.30,0.55,0.20),75:(0.78,0.30,0.30),150:(0.20,0.35,0.62)}; SP=[0,75,150]
bh=pd.read_csv(_need('Bayesian_hrt_fits.csv')); dh=pd.read_csv(_need('DDM_hrt_fits.csv'))
bs=pd.read_csv(_need('Bayesian_srt_fits.csv')); ds=pd.read_csv(_need('DDM_srt_fits.csv'))
m=dh.merge(bh[['pid','spd','t0','t0_lo95','t0_hi95']],on=['pid','spd'],suffixes=('_ddm','_bayes'))

fig,ax=plt.subplots(1,3,figsize=(16,5.4))

# Panel A: HRT t0 — DDM (MLE) vs Bayesian, showing floor degeneracy resolved
for spd in SP:
    g=m[m.spd==spd]
    ax[0].errorbar(g.t0_ddm,g.t0_bayes,yerr=[g.t0_bayes-g.t0_lo95,g.t0_hi95-g.t0_bayes],
                   fmt='o',ms=6,color=SC[spd],ecolor=SC[spd],elinewidth=0.7,capsize=2,alpha=0.85,label=f'{spd} deg/s')
ax[0].axvline(100,color='#E84855',ls=':',lw=1.3); ax[0].text(101,188,'DDM floor (100ms)',color='#E84855',fontsize=8,rotation=90,va='top')
ax[0].plot([95,200],[95,200],color='#999',lw=1)
ax[0].set_xlabel('DDM (MLE) $t_0$ (ms)'); ax[0].set_ylabel('Bayesian $t_0$ (ms, mean ± 95% CI)')
ax[0].set_title('A.  HRT: Bayesian resolves the floor degeneracy\n(floored DDM cells lifted to realistic values)',fontsize=11,fontweight='bold')
ax[0].legend(fontsize=8.5,loc='lower right',title='speed'); ax[0].set_xlim(95,200); ax[0].set_ylim(95,200)
ax[0].spines[['top','right']].set_visible(False); ax[0].grid(True,ls='--',alpha=0.3)

# Panel B: SRT express fraction with 95% CI (reliability), sorted
mix=bs[bs.model=='mixture'].copy().sort_values('pi').reset_index(drop=True)
for i,r in mix.iterrows():
    ax[1].plot([r.pi_lo95,r.pi_hi95],[i,i],color=SC[r.spd],lw=2,alpha=0.8); ax[1].plot(r.pi,i,'o',color=SC[r.spd],ms=6)
ax[1].set_yticks(range(len(mix))); ax[1].set_yticklabels([f'{r.pid}@{int(r.spd)}' for _,r in mix.iterrows()],fontsize=7.5)
ax[1].set_xlabel('Express fraction π (Bayesian, 95% CI)'); ax[1].set_xlim(0,1); ax[1].axvline(0.5,color='#bbb',lw=0.8)
ax[1].set_title('B.  SRT express fraction & uncertainty\n(wide bars = bimodality not firmly identified)',fontsize=11,fontweight='bold')
ax[1].spines[['top','right']].set_visible(False); ax[1].grid(True,axis='x',ls='--',alpha=0.3)

# Panel C: SRT unimodal t0 by speed, Bayesian vs DDM (both respect floor)
bss=bs[bs.model=='single']; dss=ds[ds.model=='single']
x=np.arange(3); w=0.36
bm=[bss[bss.spd==s].t0.mean() for s in SP]; dm=[dss[dss.spd==s].t0.mean() for s in SP]
ax[2].bar(x-w/2,dm,w,color='#cccccc',edgecolor='#888',label='DDM (MLE)')
ax[2].bar(x+w/2,bm,w,color=[SC[s] for s in SP],edgecolor='#444',label='Bayesian')
ax[2].axhline(70,color='#E84855',ls=':',lw=1.3); ax[2].text(2.4,72,'floor 70ms',color='#E84855',fontsize=8,ha='right')
for i,v in enumerate(bm): ax[2].text(i+w/2,v+2,f'{v:.0f}',ha='center',fontsize=9,fontweight='bold')
ax[2].set_xticks(x); ax[2].set_xticklabels([f'{s} deg/s' for s in SP]); ax[2].set_ylabel('$t_0$ (ms)')
ax[2].set_title('C.  SRT non-decision time (unimodal cells)\n(both methods agree; physiological floor)',fontsize=11,fontweight='bold')
ax[2].legend(fontsize=8.5); ax[2].set_ylim(0,140); ax[2].spines[['top','right']].set_visible(False); ax[2].grid(True,axis='y',ls='--',alpha=0.3)

fig.suptitle('Bayesian Model — hierarchical DDM with partial pooling (HRT degeneracy resolved; SRT with credible intervals)',fontsize=12.5,fontweight='bold',y=1.00)
fig.tight_layout()
fig.savefig(os.path.join(HERE,'Bayesian_summary.pdf'),dpi=300,bbox_inches='tight',facecolor='white')
fig.savefig(os.path.join(HERE,'Bayesian_summary.png'),dpi=140,bbox_inches='tight',facecolor='white')
print("saved Bayesian_summary.pdf and Bayesian_summary.png to", HERE)
