"""Bayesian_figures.py -- figures for the Bayesian model (HRT + SRT).
Reads Bayesian_hrt_fits.csv, DDM_hrt_fits.csv, Bayesian_srt_fits.csv, DDM_srt_fits.csv, and (optionally) Bayesian_srt_ndt.csv for Panel C.
Produces Bayesian_summary.pdf/.png."""
import os, sys, pandas as pd, matplotlib
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
# per-participant estimated saccadic t0 (the resolution); optional so this still runs without it
_ndt_path = os.path.join(HERE, 'Bayesian_srt_ndt.csv')
NDT = pd.read_csv(_ndt_path) if os.path.exists(_ndt_path) else None
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

# Panel C: SRT non-decision time ESTIMATED per participant (the resolution; individual differences, no floor-piling)
if NDT is not None:
    nn = NDT.sort_values('t0_ms').reset_index(drop=True)
    for i, r in nn.iterrows():
        tight = r.ci_width_ms < 35
        c = '#2c7fb8' if tight else '#d95f0e'
        ax[2].plot([r.t0_lo95, r.t0_hi95], [i, i], color=c, lw=1.8, alpha=0.85)
        ax[2].plot(r.t0_ms, i, 'o', color=c, ms=4.5)
    ax[2].axvline(nn.t0_ms.mean(), color='#444', ls='--', lw=1.2)
    ax[2].set_yticks(range(len(nn))); ax[2].set_yticklabels(nn.pid, fontsize=6.5)
    ax[2].set_xlabel('$t_0$ (ms, mean ± 95% CI)')
    ax[2].set_title('C.  SRT non-decision time — estimated per participant\n(individual differences preserved; not floored)', fontsize=11, fontweight='bold')
    ax[2].legend(handles=[Line2D([0],[0],color='#2c7fb8',lw=2,label='tighter CI (<35 ms)'),
                          Line2D([0],[0],color='#d95f0e',lw=2,label='looser CI (≥35 ms)')], fontsize=8, loc='lower right')
else:
    ax[2].text(0.5,0.5,'Run Bayesian_SRT_ndt.py to add\nper-participant saccadic $t_0$ panel',
               ha='center',va='center',fontsize=10,color='#888',transform=ax[2].transAxes)
    ax[2].set_xticks([]); ax[2].set_yticks([])
    ax[2].set_title('C.  SRT non-decision time (per participant)', fontsize=11, fontweight='bold')
ax[2].spines[['top','right']].set_visible(False); ax[2].grid(True, axis='x', ls='--', alpha=0.3)

fig.suptitle('Bayesian Model — HRT non-decision-time degeneracy resolved; saccadic $t_0$ estimated per participant',fontsize=12.5,fontweight='bold',y=1.00)
fig.tight_layout()
fig.savefig(os.path.join(HERE,'Bayesian_summary.pdf'),dpi=300,bbox_inches='tight',facecolor='white')
fig.savefig(os.path.join(HERE,'Bayesian_summary.png'),dpi=140,bbox_inches='tight',facecolor='white')
print("saved Bayesian_summary.pdf and Bayesian_summary.png to", HERE)
