"""Manuscript rendering of verified, multiplicity-weighted joint spectra.
Input: verified_joint.npz, made by audit_joint_spectrum.py from frozen archives.
Spatial ranks are frequency-specific SVD ranks, not spatial wavenumbers.
"""
from pathlib import Path
import sys
import argparse
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'verified_joint.npz');p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args()
sys.path.insert(0,str(Path(__file__).parent/'urbanflow_revision/plotdeps'))
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'paper/singlecolumn/figures'))
import fig2d_spectrum3d as F
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'pdf.fonttype':42,'ps.fonttype':42})
z=np.load(a.data);floor=-4.5;norm=mpl.colors.Normalize(floor,0)
fig=plt.figure(figsize=(7.16,5.05));xs=[.060,.355,.650];pw=.255
# Keep the original canvas height, but raise all previously sub-7-point text.
TICK=7.0; NOTE=7.2
summary={}
for row,field in enumerate(['source','routing']):
 for col,city in enumerate(['Porto','SF','Chengdu']):
  q=z[city+'_'+field];n=18 if city=='Chengdu' else 24;freq=np.arange(q.shape[0])/n
  ax=fig.add_axes([xs[col],.53 if row==0 else .105,pw,.395],projection='3d')
  ax.set(xlim=(-.03,.53),ylim=(8.5,.5),zlim=(floor,0))
  ax.set_xticks([0,.25,.5]);ax.set_xticklabels(['0','0.25','0.50'],fontsize=TICK)
  ax.set_yticks([1,4,8]);ax.set_yticklabels(['1','4','8'],fontsize=TICK)
  ax.set_zticks([-4,-3,-2,-1,0]);ax.set_zticklabels([])
  ax.tick_params(axis='x',pad=-6);ax.tick_params(axis='y',pad=-5)
  ax.view_init(elev=36,azim=-58);F.frame(ax,1.02);ax.computed_zorder=False
  cells=[(i,k) for i in range(q.shape[0]) for k in range(8)]
  M=ax.get_proj();cells.sort(key=lambda v:-float(proj3d.proj_transform(freq[v[0]],v[1]+1,floor,M)[2]))
  for order,(i,k) in enumerate(cells):
   height=max(float(np.log10(q[i,k])),floor)-floor
   ax.bar3d(freq[i]-.3/n,k+1-.20,floor,.6/n,.40,height,color=F.CMAP(norm(np.log10(q[i,k]))),shade=False,edgecolor='black',linewidth=.14 if k<2 else .08,zorder=10+order)
  ax.text2D(.0,.99,chr(97+row*3+col),transform=ax.transAxes,weight='bold',fontsize=9)
  ax.text2D(.86,.075,'$k$',transform=ax.transAxes,fontsize=8)
  ax.text2D(.05,.075,f'Shown: {q.sum():.1%}',transform=ax.transAxes,fontsize=TICK,color='#40576b')
  summary[city+'_'+field]={'shown':float(q.sum()),'below_floor':int((q<=10**floor).sum()),'minimum':float(q.min())}
for j,title in enumerate(['Porto','San Francisco','Chengdu']):
 fig.text(xs[j]+pw/2,.955,title,ha='center',fontsize=9)
 fig.text(xs[j]+pw/2,.925,'18-point DFT; 06–23 h' if j==2 else '24-point DFT; 00–23 h',ha='center',fontsize=NOTE,color='#40576b')
for yy,title in [(.727,'Source'),(.302,'Routing')]:
 fig.text(.021,yy,title,rotation=90,ha='center',va='center',fontsize=9)
cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=F.CMAP),cax=fig.add_axes([.915,.24,.012,.48]))
cb.set_ticks([-4,-3,-2,-1,0]);cb.set_ticklabels([r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$','1']);cb.ax.tick_params(labelsize=TICK,length=2);cb.outline.set_linewidth(.4)
cb.set_label('Energy share (log scale)',fontsize=7.5,labelpad=3)
fig.text(.51,.069,'Temporal frequency (cycles per observed hour)',ha='center',fontsize=8)
fig.text(.51,.028,'$k$: spatial mode rank within each frequency; Chengdu uses an 18-hour observed block',ha='center',fontsize=NOTE,color='#40576b')
a.out.mkdir(parents=True,exist_ok=True)
for ext in ['pdf','png']:fig.savefig(a.out/('fig03_joint_spectrum.'+ext),dpi=350)
import json
(a.out/'render_checks.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
