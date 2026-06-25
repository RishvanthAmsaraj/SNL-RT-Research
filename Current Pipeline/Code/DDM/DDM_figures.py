"""DDM_figures.py  --  DDM method comparison and diagnostic figures.

Produces summary figure panels from DDM fits.
Reads DDM_hrt_fits.csv, DDM_srt_fits.csv. Output: DDM_summary.pdf/.png.

Run: python DDM_figures.py  (needs DDM_fit.py outputs first)
"""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_fam = "Arial" if "Arial" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"
matplotlib.rcParams.update({"font.family":_fam,"font.size":11,"pdf.fonttype":42,"ps.fonttype":42})
HERE = os.path.dirname(os.path.abspath(__file__))
def _need(f):
    p = os.path.join(HERE, f)
    if not os.path.exists(p): sys.exit(f"ERROR: {f} not found next to this script. Run DDM_fit.py first.")
    return p
h=pd.read_csv(_need('DDM_hrt_fits.csv')); s=pd.read_csv(_need('DDM_srt_fits.csv'))
SC={0:(0.30,0.55,0.20),75:(0.78,0.30,0.30),150:(0.20,0.35,0.62)}
SP=[0,75,150]
rng=np.random.default_rng(0)
fig,ax=plt.subplots(1,3,figsize=(16,5.2))

for i,spd in enumerate(SP):
    d=h[h.spd==spd]
    ax[0].scatter(np.full(len(d),i)+rng.uniform(-.12,.12,len(d)),d.ks,color=SC[spd],s=28,alpha=0.7,edgecolor='white',linewidth=0.4)
ax[0].axhline(0.10,color='#888',ls='--',lw=1); ax[0].axhline(0.12,color='#E84855',ls=':',lw=1)
ax[0].text(2.4,0.102,'OK',fontsize=8,color='#666'); ax[0].text(2.4,0.122,'Poor',fontsize=8,color='#E84855')
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([f'{x} deg/s' for x in SP])
ax[0].set_ylabel('KS statistic'); ax[0].set_title('A.  HRT fit quality\n(single-Wald, 0% Poor)',fontsize=11,fontweight='bold')
ax[0].spines[['top','right']].set_visible(False); ax[0].set_ylim(0,0.18); ax[0].grid(True,axis='y',ls='--',alpha=0.3)

srt_s=s.sort_values('ks_single').reset_index(drop=True)
ax[1].plot(range(len(srt_s)),srt_s.ks_single,'o-',color='#aaa',ms=3,lw=0.8,label='single-Wald (all)')
mx=srt_s[srt_s.model=='mixture']
ax[1].scatter(mx.index,mx.ks,color='#C0392B',s=40,zorder=5,label='mixture (selected)')
for i in mx.index: ax[1].plot([i,i],[srt_s.loc[i,'ks_single'],srt_s.loc[i,'ks']],color='#C0392B',lw=0.7,alpha=0.5)
ax[1].axhline(0.10,color='#888',ls='--',lw=1); ax[1].axhline(0.12,color='#E84855',ls=':',lw=1)
ax[1].set_xlabel('SRT cell (sorted by single-Wald KS)'); ax[1].set_ylabel('KS statistic')
ax[1].set_title('B.  SRT fit quality\n(mixture rescues bimodal cells)',fontsize=11,fontweight='bold')
ax[1].legend(fontsize=8.5,loc='upper left'); ax[1].spines[['top','right']].set_visible(False)
ax[1].grid(True,axis='y',ls='--',alpha=0.3)

hm=[h[h.spd==spd].t0.mean() for spd in SP]; hsd=[h[h.spd==spd].t0.std() for spd in SP]
x=np.arange(3)
ax[2].bar(x,hm,yerr=hsd,capsize=6,color=[SC[spd] for spd in SP],edgecolor='#444',linewidth=1.2)
for i,(m,sd) in enumerate(zip(hm,hsd)): ax[2].text(i,m+sd+2,f'{m:.0f}',ha='center',fontsize=10,fontweight='bold')
ax[2].axhline(100,color='#E84855',ls=':',lw=1.2); ax[2].text(2.4,102,'floor 100ms',fontsize=8,color='#E84855',ha='right')
ax[2].set_xticks(x); ax[2].set_xticklabels([f'{spd} deg/s' for spd in SP]); ax[2].set_ylabel('$t_0$ (ms)')
ax[2].set_title('C.  HRT non-decision time\n(defensible bounds)',fontsize=11,fontweight='bold')
ax[2].spines[['top','right']].set_visible(False); ax[2].grid(True,axis='y',ls='--',alpha=0.3); ax[2].set_ylim(0,200)

fig.suptitle('DDM (Drift-Diffusion Model): literature-grounded bounds + SRT express/regular mixture',fontsize=13,fontweight='bold',y=1.00)
fig.tight_layout()
fig.savefig(os.path.join(HERE,'DDM_summary.pdf'),dpi=300,bbox_inches='tight',facecolor='white')
fig.savefig(os.path.join(HERE,'DDM_summary.png'),dpi=140,bbox_inches='tight',facecolor='white')
print("saved DDM_summary.pdf and DDM_summary.png to", HERE)
