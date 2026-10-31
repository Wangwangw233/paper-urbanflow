"""Publication plots from frozen outputs. No model fitting or experiment runs.
Run with numpy/matplotlib; optional --deps adds a local dependency directory.
"""
from pathlib import Path
import os
import sys, json, csv, hashlib, argparse
if '--deps' in sys.argv: sys.path.insert(0,sys.argv[sys.argv.index('--deps')+1])
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

REPO_ROOT=Path(__file__).resolve().parents[3]
ROOT=Path(os.environ.get('URBANFLOW_ROOT',REPO_ROOT))
OUT=REPO_ROOT/'paper/merged_manuscript/figures'
SUPP_OUT=REPO_ROOT/'paper/merged_manuscript/figures_supp'
BLUE='#315f85'; ORANGE='#a76539'; GREEN='#49735b'; GRAY='#777777'
COL={'Porto':BLUE,'SF':ORANGE,'Chengdu':GREEN}
FULL={'Porto':'Porto','SF':'San Francisco','Chengdu':'Chengdu'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9,
 'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
 'lines.linewidth':1.3,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
inputs={}; summary={}
def track(p):
 p=Path(p); inputs[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest(); return p
def js(p): return json.loads(track(p).read_text(encoding='utf-8'))
def rows(p): return list(csv.DictReader(track(p).open(encoding='utf-8-sig')))
def nz(p): return np.load(track(p),allow_pickle=True)
def tag(ax,s): ax.text(-.13,1.06,s,transform=ax.transAxes,weight='bold',fontsize=10,va='bottom')
def tidy(ax):
 ax.tick_params(width=.6,length=3); ax.grid(axis='y',color='#e5e5e5',lw=.5); ax.set_axisbelow(True)
def save(fig,name,out=None):
 folder=Path(out or OUT); folder.mkdir(parents=True,exist_ok=True)
 fig.savefig(folder/(name+'.pdf'),dpi=600)
 fig.savefig(folder/(name+'.png'),dpi=600)
 fig.savefig(folder/(name+'_preview.png'),dpi=150)
 plt.close(fig); print('saved',folder/name,flush=True)

def concept():
 fig,axs=plt.subplots(1,3,figsize=(7.0,1.9));fig.subplots_adjust(left=.05,right=.99,bottom=.08,top=.77,wspace=.3)
 a=[(0,0),(0,1),(0,2),(1,2),(2,2)]
 b=[(0,0),(1,0),(2,0),(2,1),(2,2)]
 for i,ax in enumerate(axs):
  for n in range(3):
   ax.plot([0,2],[n,n],c='#dddddd',lw=.65); ax.plot([n,n],[0,2],c='#dddddd',lw=.65)
  path=a if i==0 else b if i==1 else a+list(reversed(b))[1:]
  for u,v in zip(path[:-1],path[1:]):
   ax.annotate('',v,u,arrowprops=dict(arrowstyle='-|>',color=BLUE,lw=1.6,shrinkA=3,shrinkB=3,mutation_scale=10))
  if i<2:
   ax.scatter([0,2],[0,2],s=110,facecolor=['white',BLUE],edgecolor=BLUE,zorder=3)
   ax.text(0,-.32,'Source',ha='center',fontsize=8);ax.text(2,2.23,'Sink',ha='center',fontsize=8)
  ax.set(xlim=(-.35,2.35),ylim=(-.43,2.43),aspect='equal');ax.axis('off')
  ax.set_title(['Routing A','Routing B','Difference: zero divergence'][i],pad=10)
  ax.text(-.34 if i==2 else -.13,1.06,'abc'[i],transform=ax.transAxes,weight='bold',fontsize=10,va='bottom')
 save(fig,'fig01_concept')

def threecity():
 score=rows(ROOT/'reports/mainstream_mobility_comparison/crosscity_results.csv')
 sp=js(ROOT/'reports/crosscity_standard_replication/results/spectra.json')
 fig,axs=plt.subplots(2,3,figsize=(7.0,4.15));fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.88,hspace=.40,wspace=.25)
 summary['threecity']={}
 for j,city in enumerate(COL):
  ax=axs[0,j]; color=COL[city]
  for k,frame in enumerate(['P','D']):
   for support,dx,fill in [('ROAD',-.11,True),('FACE',.11,False)]:
    r=next(r for r in score if r['object']==city and r['frame']==frame and r['support']==support and r['model']=='OURS')
    val=float(r['pooled_R']);ax.plot(k+dx,val,'o',color=color,mfc=color if fill else 'white',ms=6)
  cl=next(r for r in score if r['object']==city and r['frame']=='P' and r['support']=='ROAD' and r['model']=='CLIM')
  clim=float(cl['pooled_R']);ax.plot([-0.24],[clim],'D',color='#6d757c',mfc='white',mew=1.0,ms=4.6)
  summary['threecity'].setdefault(city,{})['CLIM']=clim
  ax.set(xticks=[0,1],xticklabels=['P: raw','D: departure'],ylim=(0,1.0),xlim=(-.5,1.5),yticks=[0,.2,.4,.6,.8,1.0]);ax.set_title(FULL[city],pad=8);tidy(ax);tag(ax,'abc'[j])
  if j==0: ax.set_ylabel('Held-out routing score')
  else: ax.tick_params(labelleft=False)
  ss=[r for r in sp['rows'] if r['city']==city]; ax=axs[1,j]
  for key,ls,label in [('capture_train','--','TRAIN'),('capture_test','-','TEST')]:
   a=np.array([r[key] for r in ss]);m=a.mean(axis=0)
   ax.plot(sp['ranks'],m,ls,color=color,marker='o' if key=='capture_test' else None,ms=3,label=label)
   if len(a)>1:ax.fill_between(sp['ranks'],a.min(0),a.max(0),color=color,alpha=.1,lw=0)
   summary['threecity'].setdefault(city,{})[key]=m.tolist()
  ax.set(xscale='log',xticks=[1,4,16,64],xticklabels=['1','4','16','64'],ylim=(0,1.02),yticks=[0,.25,.5,.75,1],xlabel='Number of spatial modes');tidy(ax);tag(ax,'def'[j])
  ax.minorticks_off()
  if j==0: ax.set_ylabel('Routing energy captured')
  else:ax.tick_params(labelleft=False)
  if j==1:ax.legend(frameon=False,loc='lower right')
 fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#333333',label='ROAD'),Line2D([],[],marker='o',ls='',color='#333333',mfc='white',label='FACE'),Line2D([],[],marker='D',ls='',color='#6d757c',mfc='white',label='CLIM (source-free)')],loc='upper center',bbox_to_anchor=(.52,1.0),ncol=3,frameon=False)
 save(fig,'fig02_threecity_coupling')

def reconstruction():
 zgeom=nz(ROOT/'routing_urban_medium_constitutive/results/um_ctx.npz')
 frozen=nz(ROOT/'frozen/reconstruction.npz')
 fig=plt.figure(figsize=(7.0,4.5))
 rects=[(.08,.53,.27,.38),(.08,.09,.27,.38),(.43,.09,.275,.82),(.72,.09,.275,.82)]
 summary['snapshots']={}
 for j,city in enumerate(['Porto','SF']):
  nx=int(zgeom[f'{city}__nx']);ny=int(zgeom[f'{city}__ny'])
  p0=zgeom[f'{city}__face_pairs_p0'].astype(int);p1=zgeom[f'{city}__face_pairs_p1'].astype(int)
  ph=int(frozen['phase']);i=int(frozen[f'{city}__index'])
  obs=frozen[f'{city}__obs'];hat=frozen[f'{city}__hat'];err=float(frozen[f'{city}__date_error'])
  vmax=float(np.quantile(np.abs(obs),.99)); assert vmax>0
  # Face segments faithfully show signed face crossing direction; no cell-vector averaging.
  x0=(p0%nx+.5)*.25;y0=(p0//nx+.5)*.25;x1=(p1%nx+.5)*.25;y1=(p1//nx+.5)*.25
  xx=(x0+x1)/2;yy=(y0+y1)/2
  keep=((p0%nx)%3==0)&((p0//nx)%3==0)&(np.abs(obs)>.08*vmax)
  for k,v in enumerate([obs,hat]):
   ax=fig.add_axes(rects[j*2+k]);c=COL[city]
   norm=np.clip(np.abs(v)/vmax,0,1)
   seg=np.stack([np.c_[x0,y0],np.c_[x1,y1]],axis=1)
   rgba=np.tile(matplotlib.colors.to_rgba(c),(len(v),1));rgba[:,3]=.8*np.sqrt(norm)
   ax.add_collection(LineCollection(seg,colors=rgba,linewidths=.8))
   sign=np.sign(v[keep]);dx=(x1-x0)[keep]*sign;dy=(y1-y0)[keep]*sign
   ax.quiver(xx[keep],yy[keep],dx,dy,angles='xy',scale_units='xy',scale=.7,color=c,width=.0045,pivot='mid',headwidth=3.5)
   ax.set(xlim=(0,nx*.25),ylim=(0,ny*.25),aspect='equal');ax.set_xticks([]);ax.set_yticks([])
   for spine in ax.spines.values():spine.set_visible(False)
   ax.set_title(('Observed' if k==0 else 'Reconstructed'),fontsize=8,pad=7)
   ax.text(-.09,1.015,'abcd'[j*2+k],transform=ax.transAxes,weight='bold',fontsize=10,va='bottom')
   if k==0:
    ax.plot([.5,2.5],[.5,.5],color='#333333',lw=1.4);ax.text(1.5,.85,'2 km',ha='center',fontsize=7)
  date=str(np.datetime64('1970-01-01')+np.timedelta64(int(frozen[f'{city}__test_date']),'D'))
  summary['snapshots'][city]={'date':date,'phase':ph,'date_error':float(err),'index':i,'scale_p99':vmax,'selection':'closest-to-median archived per-date error; fixed 12:00'}
 fig.text(.215,.975,'Porto',ha='center',fontsize=10);fig.text(.71,.975,'San Francisco',ha='center',fontsize=10)
 fig.text(.215,.025,summary['snapshots']['Porto']['date']+' · 12:00',ha='center',fontsize=8)
 fig.text(.71,.025,summary['snapshots']['SF']['date']+' · 12:00',ha='center',fontsize=8)
 save(fig,'figS04_heldout_reconstruction',SUPP_OUT)

def source_spectrum(city):
 """Read the small frozen q matrix derived from the large aligned source cache."""
 with np.load(ROOT/'frozen/source_spectra.npz',allow_pickle=False) as z:
  return np.asarray(z[f'{city}__q'],dtype=np.float64)

def spectra_response():
 route=nz(ROOT/'paper/singlecolumn/figures/_f2_shares.npz')
 fig=plt.figure(figsize=(7.0,6.25));gs=fig.add_gridspec(3,2,left=.1,right=.86,bottom=.09,top=.94,hspace=.7,wspace=.35,height_ratios=[1,1,1.1])
 summary['spectra']={}
 for j,city in enumerate(['Porto','SF']):
  qs=source_spectrum(city);p=route[f'{city}__P'];qr=route[f'{city}__pk'][:,:8]*(p/p.sum())[:,None]
  expected=.7030 if city=='Porto' else .9692
  assert abs(qs.sum()-expected)<5e-4,(city,qs.sum())
  summary['spectra'][city]={'source_shown_share':float(qs.sum()),'routing_shown_share':float(qr.sum()),'source_h01_shown_share':float(qs[:2].sum()),'routing_h01_shown_share':float(qr[:2].sum())}
  for r,q in enumerate([qs,qr]):
   ax=fig.add_subplot(gs[r,j]);im=ax.imshow(q.T,origin='lower',aspect='auto',extent=(-.5,12.5,.5,8.5),cmap='Blues',norm=LogNorm(1e-4,1),interpolation='nearest')
   ax.set(xticks=[0,3,6,9,12],yticks=[1,4,8],xlabel='Temporal harmonic, h',ylabel='Spatial mode rank')
   ax.set_title(FULL[city]+' · '+('source' if r==0 else 'routing')+f'\nShown power: {q.sum():.1%}',pad=8,fontsize=8);tag(ax,'abcd'[r*2+j])
 cb=fig.colorbar(im,cax=fig.add_axes([.9,.47,.018,.42]));cb.set_label('Normalized spectral power',fontsize=8);cb.ax.tick_params(labelsize=7)
 ax=fig.add_subplot(gs[2,0]); rr=rows(ROOT/'structured_wiener/results/response_scalar.csv')
 for city in ['Porto','SF']:
  sub=[r for r in rr if r['city']==city]; folds=sorted(set(r['fold'] for r in sub));a=[]
  for fold in folds:
   s=sorted([r for r in sub if r['fold']==fold and int(float(r['h']))>0],key=lambda r:float(r['h']))
   ar=np.array([float(r['G_abs']) for r in s]);a.append(ar/ar.mean())
  a=np.array(a);ax.plot(np.arange(1,13),a.mean(0),color=COL[city],marker='o' if city=='Porto' else 's',ms=2.5,label=FULL[city])
  if len(a)>1:ax.fill_between(np.arange(1,13),a.min(0),a.max(0),color=COL[city],alpha=.15,lw=0)
 ax.axhline(1,c=GRAY,lw=.7,ls='--');ax.set(xlabel='Temporal harmonic, h',ylabel='Normalized response gain',xticks=[1,4,8,12]);tidy(ax);tag(ax,'e');ax.legend(frameon=False,fontsize=6.8,loc='lower left')
 ax=fig.add_subplot(gs[2,1]);r=rows(ROOT/'constitutive_spatial_characterization/results/cs_curve_Porto.csv')
 d=np.array([float(x['d_m'])/1000 for x in r]); vals=lambda k:np.array([float(x[k]) for x in r])
 ax.plot(d,vals('observed_equal'),color=BLUE,label='Porto response');ax.plot(d,vals('null_equal'),color=GRAY,ls='--',label='Spatial null')
 ax.fill_between(d,vals('null_lo'),vals('null_hi'),color=GRAY,alpha=.15,lw=0)
 ax.set(xlabel='Distance from probe (km)',ylabel='Cumulative energy',ylim=(0,1.02),xlim=(0,12));tidy(ax);tag(ax,'f');ax.legend(frameon=False,fontsize=6.8,loc='lower right')
 save(fig,'figS06_spectra_response',SUPP_OUT)

def anisotropy():
 z=nz(ROOT/'full_operator_spatial_anisotropy/results/fo4_fig5_data.npz')
 fig=plt.figure(figsize=(7.0,4.7));gs=fig.add_gridspec(2,3,left=.08,right=.98,bottom=.12,top=.91,wspace=.5,hspace=.65,width_ratios=[1,.55,1.35])
 # Maps use identical distance per axis and honest city aspect ratios.
 for j,city in enumerate(['Porto','SF']):
  ax=fig.add_subplot(gs[:,j]); nx=int(z[f'{city}_nx']);ny=int(z[f'{city}_ny']);cells=z[f'{city}_cells'].astype(int);a=z[f'{city}_A_C'];theta=z[f'{city}_theta_C']
  grid=np.full(nx*ny,np.nan);grid[cells]=a;grid=grid.reshape(ny,nx)
  im=ax.imshow(grid,origin='lower',extent=[0,nx*.25,0,ny*.25],vmin=0,vmax=1,cmap='Blues',interpolation='nearest',aspect='equal')
  keep=((cells%nx)%3==0)&((cells//nx)%3==0);x=(cells[keep]%nx+.5)*.25;y=(cells[keep]//nx+.5)*.25
  dx=.32*a[keep]*np.cos(theta[keep]);dy=.32*a[keep]*np.sin(theta[keep]);ax.add_collection(LineCollection(np.stack([np.c_[x-dx,y-dy],np.c_[x+dx,y+dy]],axis=1),colors='#333333',linewidths=.55))
  ax.set_title(FULL[city],pad=10);ax.set_xticks([]);ax.set_yticks([]);tag(ax,'ab'[j])
  for s in ax.spines.values():s.set_visible(False)
  ax.plot([.5,2.5],[.5,.5],c='#333333',lw=1.3);ax.text(1.5,1,'2 km',ha='center',fontsize=7)
 cb=fig.colorbar(im,cax=fig.add_axes([.1,.105,.35,.015]),orientation='horizontal');cb.set_label('Response anisotropy',fontsize=8);cb.set_ticks([0,.5,1])
 ax=fig.add_subplot(gs[0,2])
 for city in ['Porto','SF']:
  for key,ls in [('A_C','-'),('A_U','--')]:
   a=np.sort(z[f'{city}_{key}']);ax.plot(a,np.arange(1,len(a)+1)/len(a),color=COL[city],ls=ls)
 ax.set(xlabel='Local anisotropy',ylabel='Fraction of cells',xlim=(0,1),ylim=(0,1));tidy(ax);tag(ax,'c')
 ax.legend(handles=[Line2D([],[],color=COL[c],label=FULL[c]) for c in ['Porto','SF']]+[Line2D([],[],color=GRAY,ls='-',label='Response'),Line2D([],[],color=GRAY,ls='--',label='Road support')],frameon=False,fontsize=6.5,loc='lower right')
 ax=fig.add_subplot(gs[1,2])
 for city in ['Porto','SF']:
  r=rows(ROOT/f'results/kr_response_characteristics/kr_spectrum_{city}.csv');s=np.array([float(x['sigma_foldmean']) for x in r]);s=s[s>s[0]*1e-10]
  ax.semilogy(np.arange(1,len(s)+1),s/s[0],color=COL[city],label=FULL[city])
 ax.set(xlabel='Constitutive mode rank',ylabel='Relative singular value');tidy(ax);tag(ax,'d');ax.legend(frameon=False,fontsize=6.5)
 save(fig,'figS05_anisotropy_modes',SUPP_OUT)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--out',type=Path,default=OUT)
 parser.add_argument('--supp-out',type=Path,default=SUPP_OUT)
 parser.add_argument('--deps',type=Path)
 args,_=parser.parse_known_args()
 OUT,SUPP_OUT=args.out,args.supp_out
 concept();threecity()
 (OUT/'figure_manifest.json').write_text(json.dumps({'inputs_sha256':inputs,'summary':summary,'style':'7 inch vector PDF; embedded TrueType; 600 dpi PNG; common fonts and city colors; no smoothing'},indent=2),encoding='utf-8')
