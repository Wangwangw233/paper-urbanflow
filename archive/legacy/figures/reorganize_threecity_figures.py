"""Regenerate three-city response plots from frozen outputs.

The default mode writes figures only. The historical --rewrite-manuscript mode
is retained solely for reproducing the 2026-10-30 development edit.
"""
from pathlib import Path
import sys,json,re,shutil,difflib
sys.path.insert(0,str(Path(__file__).parent/'urbanflow_revision/plotdeps'))
import redraw_urbanflow as R
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT=R.ROOT
OUT=Path(sys.argv[sys.argv.index('--out')+1]).resolve() if '--out' in sys.argv else R.OUT
R.OUT=OUT
PAPER=OUT.parent
BASE=ROOT/'reports/spatiotemporal_constitutive_spectra/results'
CITIES=['Porto','SF','Chengdu']
data={c:[R.js(BASE/f'obj_{c}{suffix}.json') for suffix in (['','_f1','_f2','_f3','_f4'] if c=='SF' else [''])] for c in CITIES}
for c,ds in data.items():
 for d in ds:
  assert d['meta']['name']==c
  assert d['meta']['kmax']==(4 if c=='Chengdu' else 12)
  for key in ['x','y']:
   shares=[v['share_'+key] for v in d['spectra']['by_k'].values()]
   assert abs(sum(shares)-1)<1e-7
summary={}

fig=plt.figure(figsize=(7,3.8))
for j,(key,title) in enumerate([('x','Source balance'),('y','Road-projected routing')]):
 ax=fig.add_axes([.02+.49*j,.16,.46,.72],projection='3d')
 xs=[];ys=[];zs=[];colors=[]
 for ci,c in enumerate(CITIES):
  ds=data[c];ks=sorted(int(k) for k in ds[0]['spectra']['by_k'])
  full=[]
  for d in ds:
   acct=d['spectra']['energy_accounting'][key]
   v=[d['spectra']['by_k'][str(k)]['share_'+key]*acct['fitted_share'] for k in ks]
   full.append(v+[acct['unmodelled_share']])
  v=np.mean(full,axis=0);assert abs(v.sum()-1)<1e-7
  summary.setdefault(c,{})[key]={'harmonics':ks,'raw_energy_shares':v[:-1].tolist(),'rest':float(v[-1])}
  for k,value in zip(ks+[14],v):
   if value<1e-9:continue
   xs.append(k-.32);ys.append(ci-.23);zs.append(value);colors.append(R.COL[c])
 ax.bar3d(xs,ys,np.full(len(xs),-3.),.64,.32,np.log10(zs)+3,color=colors,alpha=1,shade=True,edgecolor='white',linewidth=.16,zsort='average')
 ax.set(xlim=(-.6,14.8),ylim=(-.5,2.5),zlim=(-3,0),xticks=[0,2,4,8,12,14],yticks=[0,1,2],zticks=[-3,-2,-1,0])
 ax.set_xticklabels(['0','2','4','8','12','rest']);ax.set_yticklabels(['Porto','SF','Chengdu']);ax.set_zticklabels(['0.001','0.01','0.1','1'])
 ax.set_xlabel('Clock harmonic',labelpad=5)
 ax.tick_params(pad=0,labelsize=6.5);ax.view_init(elev=32,azim=-48);ax.set_box_aspect((1.6,1.1,.95));ax.set_title(title,pad=10)
 for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
  axis.pane.fill=False;axis._axinfo['grid']['color']=(.87,.87,.87,1);axis._axinfo['grid']['linewidth']=.4
 ax.text2D(.01,1.04,'ab'[j],transform=ax.transAxes,weight='bold',fontsize=10)
fig.text(.5,.96,'Full-field energy share (log scale)',ha='center',fontsize=8)
fig.text(.5,.045,'Chengdu: observed 06–23 h; harmonics 0–4 plus unresolved remainder.',ha='center',fontsize=8)
R.save(fig,'nc_fig3_threecity_3d')

chengdu18=R.nz(ROOT/'reports/threecity_redesign/results/chengdu_response18.npz')
np.testing.assert_array_equal(chengdu18['m'],np.arange(10))
np.testing.assert_allclose(chengdu18['freq_cycles_per_observed_hour'],chengdu18['m']/18,rtol=0,atol=1e-12)
assert np.all(np.isfinite(chengdu18['cos'])) and np.all(np.isfinite(chengdu18['rho']))
fig,axes=plt.subplots(2,3,figsize=(7,3.85));fig.subplots_adjust(left=.09,right=.98,bottom=.14,top=.86,hspace=.33,wspace=.22)
for j,c in enumerate(CITIES):
 if c=='Chengdu':ks=chengdu18['m']
 else:
  ds=data[c];ks=np.array(sorted(int(k) for k in ds[0]['coherent']))
 for row,key in enumerate(['cos','ratio']):
  if c=='Chengdu':
   mean=chengdu18['cos' if key=='cos' else 'rho'];arr=mean[None,:]
  else:
   arr=[]
   for d in ds:
    vals=[]
    for k in ks:
     t=d['coherent'][str(k)]['test']; vals.append(t['truth_vs_op']['cos'] if key=='cos' else t['gain_op']/t['gain_emp'])
    arr.append(vals)
   arr=np.array(arr);mean=arr.mean(0)
  ax=axes[row,j]
  ax.plot(ks,mean,'o-',color=R.COL[c],ms=3)
  if len(arr)>1:ax.fill_between(ks,arr.min(0),arr.max(0),color=R.COL[c],alpha=.15,lw=0)
  xmax=9 if c=='Chengdu' else 12
  ax.set(xlim=(-.3,xmax+.3),xticks=[0,3,6,9] if c=='Chengdu' else [0,4,8,12],ylim=(0,1.6 if row else 1.03));R.tidy(ax);R.tag(ax,'abcdef'[row*3+j])
  if row:
   ax.axhline(1,color=R.GRAY,ls='--',lw=.7)
   ax.set_xlabel('DFT bin, m; Freq m/18' if c=='Chengdu' else 'Clock harmonic')
  else:ax.set_title(R.FULL[c]+('\n18-point DFT' if c=='Chengdu' else ''),pad=5 if c=='Chengdu' else 12)
  if j==0:ax.set_ylabel('Output-direction alignment' if row==0 else 'Predicted / observed norm')
  else:ax.tick_params(labelleft=False)
  summary[c][key]={'mean':mean.tolist(),'folds':len(arr)}
R.save(fig,'fig04_frequency_response')
(OUT/'threecity_response_manifest.json').write_text(json.dumps({'inputs_sha256':R.inputs,'summary':summary,'normalization':'share within fitted subspace multiplied by fitted_share; rest shown separately. SF fold means. Porto/SF coherent response uses archived P-selected operators.','chengdu_response':'chengdu_response18.npz; P-selected operator; all ten single-sided 18-point DFT bins; frequency m/18 cycles per observed hour.'},indent=2),encoding='utf-8')
print('saved three-city response plots; manuscript editing is disabled in the public copy')
