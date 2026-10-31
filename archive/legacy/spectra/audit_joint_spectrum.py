from pathlib import Path
import json, sys, numpy as np
R=Path(__file__).resolve().parents[3]
O=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
O.mkdir(parents=True,exist_ok=True)
old=np.load(R/'paper/singlecolumn/figures/_f2_shares.npz')
data={};report={}
csv=np.genfromtxt(R/'reports/threecity_joint_spectrum/results/threecity_drawn.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
for city,splitname in [('Porto','porto'),('SF','cabspotting')]:
 z=np.load(R/f'empirical_transfer_round/artifacts/et_data_{city.lower()}.npz')
 sp=json.loads((R/f'reports/deep_baseline_training_record/splits/{splitname}_datefreeze.json').read_text())
 if city=='Porto': trains=[sp['split']['train']['dates']]
 else:
  blocks=[f['block'] for f in sp['split']['folds']]
  trains=[[d for d in sp['eligible_dates'] if d not in blocks[i]+blocks[(i+1)%5]] for i in range(5)]
 pos={int(d):i for i,d in enumerate(z['order'])};weights=np.r_[1,np.full(11,2),1]
 report[city]={}
 for field,key in [('source','X'),('routing','R')]:
  powers=[];errors=[]
  for tr in trains:
   a=z[key].reshape(len(pos),24,-1)[[pos[d] for d in tr]]
   C=np.fft.rfft(a,axis=1,norm='ortho');P=np.mean(np.sum(abs(C)**2,axis=2),axis=0)
   powers.append(P);errors.append(abs((P*weights).sum()/np.mean(np.sum(a*a,axis=(1,2)))-1))
  P=np.mean(powers,axis=0);unw=P/P.sum();weighted=P*weights/(P*weights).sum()
  rows=csv[(csv['city']==city)&(csv['field']==field)]
  q=rows['share'].reshape(13,8)
  if field=='routing':
   archived=old[city+'__P'];archshare=archived/archived.sum()
   report[city]['routing_archive_error_unweighted']=float(np.max(abs(archshare-unw)))
   report[city]['routing_archive_error_weighted']=float(np.max(abs(archshare-weighted)))
   assert np.allclose(archshare,unw,rtol=1e-8,atol=1e-10)
  q=q*(weighted/unw)[:,None]
  data[city+'_'+field]=q
  report[city][field]={'DC':float(weighted[0]),'kept':float(q.sum()),'parseval_error':max(errors)}
cd=np.load(R/'reports/threecity_joint_spectrum/results/chengdu_cycle18.npz')
report['Chengdu']={}
for field,tag in [('source','src'),('routing','rt')]:
 q=cd[tag+'_train__pk']*cd[tag+'_train__share'][:,None];data['Chengdu_'+field]=q
 report['Chengdu'][field]={'DC':float(cd[tag+'_train__share'][0]),'kept':float(q.sum())}
np.savez_compressed(O/'verified_joint.npz',**data)
(O/'audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
