#!/usr/bin/env python3
"""Reproducible NHANES MASLD-risk liver stiffness prediction and policy analysis.
NO data are generated here. The script requires the actual CDC files.
Run: python run_analysis.py --data data --output results --download
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit, logit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.inspection import permutation_importance

SEED=20261119
FEATURES=['age','sex','bmi','diabetes','hba1c','ast','alt','platelets','ggt']
LABELS={'FIB-4':'FIB-4','Logistic':'Penalized logistic regression','Gradient boosting':'Calibrated gradient boosting'}
SOURCES={
 'development':{'path':'2017','prefix':'P_','weight':'WTMECPRP','files':['DEMO','LUX','BMX','BIOPRO','CBC','GHB','DIQ']},
 'validation':{'path':'2021','suffix':'_L','weight':'WTPH2YR','files':['DEMO','LUX','BMX','BIOPRO','CBC','GHB','DIQ']}
}
COLUMNS={
 'DEMO':['SEQN','RIDAGEYR','RIAGENDR','RIDRETH3','INDFMPIR','SDMVSTRA','SDMVPSU','WTMECPRP','WTMEC2YR'],
 'LUX':['SEQN','LUAXSTAT','LUANMVGP','LUXSMED','LUXSIQRM','LUXCAPM'],
 'BMX':['SEQN','BMXBMI'],
 'BIOPRO':['SEQN','LBXSASSI','LBXSATSI','LBXSGTSI','WTPH2YR'],
 'CBC':['SEQN','LBXPLTSI','WTPH2YR'],
 'GHB':['SEQN','LBXGH','WTPH2YR'],
 'DIQ':['SEQN','DIQ010'],
}

def src_name(period,kind):
 conf=SOURCES[period]
 return (conf.get('prefix','')+kind+conf.get('suffix',''))

def download_if_needed(root,period,kind,download):
 name=src_name(period,kind)
 for ext in ['xpt','XPT','csv','CSV']:
  p=root/(name+'.'+ext)
  if p.exists(): return p
 if not download:
  raise FileNotFoundError(f'Missing {name}.xpt. Download from https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{SOURCES[period]["path"]}/DataFiles/{name}.xpt and save in {root}')
 import requests
 url=f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{SOURCES[period]["path"]}/DataFiles/{name}.xpt'
 print('Downloading',url,flush=True)
 try:
  r=requests.get(url,timeout=45); r.raise_for_status()
  if len(r.content)<500: raise RuntimeError(f'Unexpected short file from {url}')
  p=root/(name+'.xpt'); p.write_bytes(r.content); return p
 except Exception as exc:
  raise RuntimeError(f'Could not download official NHANES data from {url}. Download this XPT manually and save it to {root}. Original error: {exc}') from exc

def get_table(root,period,kind,download):
 p=download_if_needed(root,period,kind,download)
 d=pd.read_sas(p,format='xport') if p.suffix.lower()=='.xpt' else pd.read_csv(p)
 d.columns=[str(x).upper() for x in d.columns]
 if 'SEQN' not in d: raise RuntimeError(f'{p}: missing SEQN')
 available=[c for c in COLUMNS[kind] if c in d]
 if period=='development' and kind=='DEMO' and 'WTMECPRP' not in d: raise RuntimeError('WTMECPRP not found in development DEMO')
 if kind=='LUX' and ('LUXSMED' not in d or 'LUXSIQRM' not in d): raise RuntimeError(f'{p}: missing elastography variables')
 if kind=='CBC' and 'LBXPLTSI' not in d: raise RuntimeError(f'{p}: platelet count not found')
 if kind=='BIOPRO' and ('LBXSASSI' not in d or 'LBXSATSI' not in d): raise RuntimeError(f'{p}: transaminases not found')
 d=d[available].drop_duplicates(subset=['SEQN'])
 for c in d:
  d[c]=pd.to_numeric(d[c],errors='coerce')
 if kind in ['BIOPRO','CBC','GHB']:
  d=d.drop(columns=['WTPH2YR'],errors='ignore') if kind!='GHB' else d
 return d

def make_cohort(root,period,download):
 tables={k:get_table(root,period,k,download) for k in SOURCES[period]['files']}
 d=tables['DEMO']
 for kind in SOURCES[period]['files'][1:]:
  d=d.merge(tables[kind],on='SEQN',how='left',validate='one_to_one',suffixes=('','_'+kind))
 ren={'RIDAGEYR':'age','RIAGENDR':'sex','BMXBMI':'bmi','LBXGH':'hba1c','LBXSASSI':'ast','LBXSATSI':'alt','LBXSPLTSI':'platelets','LBXPLTSI':'platelets','LBXSGTSI':'ggt','LUXSMED':'lsm','LUXCAPM':'cap','RIDRETH3':'race','INDFMPIR':'pir','SDMVSTRA':'stratum','SDMVPSU':'psu'}
 d=d.rename(columns=ren)
 if 'WTPH2YR' not in d and period=='validation':
  raise RuntimeError('WTPH2YR missing in NHANES 2021-2023 lab file. Required for blood-analyte analyses.')
 d['weight']=d[SOURCES[period]['weight']]
 # Historical 2017-2020 lab specimen nonresponse may persist; MEC weight recommended by source.
 d['diabetes']=np.where(d['DIQ010'].isin([1]),1,np.where(d['DIQ010'].isin([2,3]),0,np.nan))
 d['at_risk']=((d['bmi']>=25)|(d['diabetes']==1)|(d['hba1c']>=5.7))
 d['vcte_valid']=(d['LUAXSTAT']==1)&(d['lsm']>0)&(d['LUANMVGP']>=10)&(d['LUXSIQRM']<30)
 d['fib4']=(d['age']*d['ast'])/(d['platelets']*np.sqrt(d['alt']))
 d['eligible']=((d['age']>=20)&d['at_risk']&d['vcte_valid']&(d['ast']>0)&(d['alt']>0)&(d['platelets']>0)&(d['weight']>0))
 d['outcome12']=(d['lsm']>=12).astype(int)
 d['outcome8']=(d['lsm']>=8).astype(int)
 d['cap_steatosis']=(d['cap']>=248)
 d['period']=period
 d['age_group']=pd.cut(d['age'],[-1,44,64,200],labels=['20-44','45-64','65+']).astype('str')
 d['sex_group']=d['sex'].map({1:'Male',2:'Female'}).fillna('Other/unknown')
 races={1:'Mexican American',2:'Other Hispanic',3:'NH White',4:'NH Black',6:'NH Asian',7:'Other/multiracial'}
 d['race_group']=d['race'].map(races).fillna('Other/unknown') if 'race' in d else 'Unknown'
 raw={'examined_20plus':int((d['age']>=20).sum()),'at_risk_20plus':int(((d['age']>=20)&d['at_risk']).sum()),'valid_vcte_at_risk':int(((d['age']>=20)&d['at_risk']&d['vcte_valid']).sum()),'eligible':int(d['eligible'].sum())}
 kept=d[d['eligible']].copy().reset_index(drop=True)
 if kept.empty: raise RuntimeError(f'No eligible adults in {period}; verify codebook/data values.')
 if kept['outcome12'].nunique()<2: raise RuntimeError(f'No case/non-case outcome events for {period}.')
 return kept,raw

def norm_weights(d):
 w=d['weight'].to_numpy(dtype=float)
 return w/(np.mean(w) or 1.0)

def metric(y,s,w,probability=True):
 if len(np.unique(y))<2: return {'auroc':None,'auprc':None,'brier':None}
 out={'auroc':float(roc_auc_score(y,s,sample_weight=w)),'auprc':float(average_precision_score(y,s,sample_weight=w))}
 out['brier']=float(np.average((y-s)**2,weights=w)) if probability else None
 if probability:
  try:
   q=np.clip(np.asarray(s),1e-5,1-1e-5); recal=LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000)
   recal.fit(logit(q).reshape(-1,1),y,sample_weight=w)
   out['calibration_slope']=float(recal.coef_[0][0]);out['calibration_intercept']=float(recal.intercept_[0])
  except Exception:out['calibration_slope']=None;out['calibration_intercept']=None
 else:out['calibration_slope']=None;out['calibration_intercept']=None
 return out

def select_fraction(scores,weights,k):
 """Weighted top-k with fractional boundary to enforce exact total weighted capacity."""
 scores=np.asarray(scores,dtype=float); weights=np.asarray(weights,dtype=float)
 order=np.argsort(-scores,kind='mergesort')
 target=k*weights.sum(); starts=np.r_[0,weights[order].cumsum()[:-1]]
 z=np.maximum(0,np.minimum(1,(target-starts)/np.maximum(weights[order],1e-9)))
 frac=np.zeros(len(scores),dtype=float); frac[order]=z
 return frac

def policy(y,s,w,k,groups=None):
 y=np.asarray(y,dtype=int); s=np.asarray(s,dtype=float); w=np.asarray(w,dtype=float)
 if groups is None:
  f=select_fraction(s,w,k)
 else:
  g=np.asarray(groups).astype(str); f=np.zeros(len(y))
  for x in np.unique(g):
   ix=np.where(g==x)[0]; f[ix]=select_fraction(s[ix],w[ix],k)
 case_tot=float((y*w).sum()); mass=float(w.sum()); referred=float((f*w).sum())
 detected=float((f*w*y).sum())
 return {'referral_fraction':float(referred/mass),'case_capture':float(detected/case_tot) if case_tot else None,
         'yield':float(detected/referred) if referred else None,
         'missed_per_1000':float(1000*(case_tot-detected)/mass),
         'selected_fraction':f}

def bootstrap_groups(d, rng):
 """Approximate stratified PSU bootstrap; predictions held fixed (validation uncertainty)."""
 strata=d['stratum'].to_numpy() if 'stratum' in d else np.zeros(len(d))
 psu=d['psu'].to_numpy() if 'psu' in d else np.arange(len(d))
 m=np.zeros(len(d),dtype=float)
 for st in np.unique(strata):
  where=np.where(strata==st)[0]; clusters=np.unique(psu[where]); picks=rng.choice(clusters,size=len(clusters),replace=True)
  for cl in picks: m[where[psu[where]==cl]]+=1
 return m

def ci(values):
 v=np.asarray([x for x in values if x is not None and np.isfinite(x)],dtype=float)
 return [float(np.quantile(v,.025)),float(np.quantile(v,.975))] if len(v)>=40 else [None,None]

def reliability(y,s,w,n_bins=8):
 result=[]; cuts=np.linspace(0,1,n_bins+1)
 for i,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
  ix=((s>=a)&(s<b)) if i<n_bins-1 else ((s>=a)&(s<=b))
  if ix.sum(): result.append({'mean_pred':float(np.average(s[ix],weights=w[ix])),'obs':float(np.average(y[ix],weights=w[ix])),'n':int(ix.sum())})
 return result

def do_models(train,val,outcome):
 ytr=train[outcome].to_numpy(); yv=val[outcome].to_numpy(); wv=norm_weights(val)
 if min(int(ytr.sum()),int((1-ytr).sum()))<20: raise RuntimeError(f'Insufficient events for {outcome} development model.')
 models={
 'Logistic':Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=False)),('scale',StandardScaler()),('model',LogisticRegression(C=1.0,max_iter=1500,random_state=SEED))]),
 'Gradient boosting':Pipeline([('imputer',SimpleImputer(strategy='median')),('model',HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=9,learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=30,random_state=SEED))])
 }
 Xtr=train[FEATURES]; Xv=val[FEATURES]
 results={}; fits={}; preds={'FIB-4':val['fib4'].to_numpy(dtype=float)}
 for name,model in models.items():
  cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   cv_aucs=cross_val_score(model,Xtr,ytr,cv=cv,scoring='roc_auc',n_jobs=1)
  if name=='Gradient boosting':
   # Prespecified sigmoid calibration using only development cross-validation.
   model=CalibratedClassifierCV(estimator=model,method='sigmoid',cv=3)
  model.fit(Xtr,ytr)
  scores=model.predict_proba(Xv)[:,1]
  preds[name]=np.clip(scores,1e-6,1-1e-6); fits[name]=model
  results[name]={'development_cv_auroc_mean':float(np.mean(cv_aucs)),'development_cv_auroc_sd':float(np.std(cv_aucs))}
 return fits,preds,results

def save_plots(val,preds,primary_policy,model_stats,output,importance):
 output.mkdir(parents=True,exist_ok=True)
 y=val['outcome12'].to_numpy(); w=norm_weights(val)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
 fig,ax=plt.subplots(figsize=(6.5,5))
 for name,s in preds.items():
  f,t,_=roc_curve(y,s,sample_weight=w)
  ax.plot(f,t,label=f'{name} (AUC {model_stats[name]["auroc"]:.3f})',linewidth=2)
 ax.plot([0,1],[0,1],ls='--',color='.65');ax.set(xlabel='1 - specificity',ylabel='Sensitivity',title='Temporal validation: elevated liver stiffness');ax.legend(fontsize=9)
 fig.tight_layout();fig.savefig(output/'figure2_roc.png',dpi=240);plt.close(fig)
 fig,ax=plt.subplots(figsize=(6.5,5))
 for name in ['Logistic','Gradient boosting']:
  r=reliability(y,preds[name],w)
  ax.plot([v['mean_pred'] for v in r],[v['obs'] for v in r],'o-',label=name)
 ax.plot([0,1],[0,1],'--',color='.65'); ax.set(xlim=(0,max(.15,max(preds['Logistic'].max(),preds['Gradient boosting'].max())*.8)),ylim=(0,.6),xlabel='Mean predicted probability',ylabel='Observed fraction with LSM >= 12 kPa',title='Weighted calibration bins')
 ax.legend();fig.tight_layout();fig.savefig(output/'figure3_calibration.png',dpi=240);plt.close(fig)
 fig,ax=plt.subplots(figsize=(6.8,4.7))
 for name in preds:
  arr=[primary_policy[name][str(v)]['case_capture'] for v in [10,20,30]]
  ax.plot([10,20,30],[100*a for a in arr],marker='o',linewidth=2,label=name)
 ax.set(xlabel='Testing capacity (% of eligible adults)',ylabel='Elevated-LSM cases captured (%)',title='Capacity-constrained diagnostic prioritization',xticks=[10,20,30],ylim=(0,100));ax.legend()
 fig.tight_layout();fig.savefig(output/'figure4_capacity.png',dpi=240);plt.close(fig)
 # Flow summary is generated from the samples after eligibility restrictions.
 fig,ax=plt.subplots(figsize=(7.2,2.7)); ax.axis('off')
 boxes=[(0.17,f'Development\nN = {len(val)}' if False else '2017-Mar 2020\nDevelopment'),(0.5,'Pre-test variables\n+ quality-controlled VCTE'),(0.83,'2021-Aug 2023\nTemporal validation')]
 for x,t in boxes:
  ax.text(x,.55,t,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.65',facecolor='#edf3f8',edgecolor='#7f9ba9'))
 ax.annotate('',xy=(.365,.55),xytext=(.31,.55),arrowprops=dict(arrowstyle='->',lw=1.6))
 ax.annotate('',xy=(.68,.55),xytext=(.65,.55),arrowprops=dict(arrowstyle='->',lw=1.6))
 fig.tight_layout();fig.savefig(output/'figure1_design.png',dpi=240);plt.close(fig)
 if importance:
  fig,ax=plt.subplots(figsize=(7,4.8))
  ordered=sorted(importance.items(),key=lambda x:x[1])
  ax.barh([x[0] for x in ordered],[x[1] for x in ordered]);ax.set(xlabel='Mean AUROC decrease on permutation',title='Permutation importance: gradient boosting, temporal validation')
  fig.tight_layout();fig.savefig(output/'figure5_importance.png',dpi=240);plt.close(fig)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--output',default='results');ap.add_argument('--download',action='store_true');ap.add_argument('--bootstrap',type=int,default=200)
 args=ap.parse_args();root=Path(args.data);root.mkdir(parents=True,exist_ok=True);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 dev,raw_dev=make_cohort(root,'development',args.download);val,raw_val=make_cohort(root,'validation',args.download)
 print('Input cohorts loaded:',len(dev),len(val),flush=True)
 fits,preds,model_cv=do_models(dev,val,'outcome12')
 y=val['outcome12'].to_numpy();wv=norm_weights(val)
 model_stats={name:metric(y,s,wv,name!='FIB-4') for name,s in preds.items()}
 for n in model_cv:model_stats[n].update(model_cv[n])
 policy_stats={};group_stats={};parity_stats={}
 for name,sc in preds.items():
  policy_stats[name]={};parity_stats[name]={}
  for cap in [10,20,30]:
   z=policy(y,sc,wv,cap/100); f=z.pop('selected_fraction');policy_stats[name][str(cap)]=z
   if cap==20:
    for group in ['sex_group','age_group','race_group']:
     group_stats.setdefault(name,{})[group]={}
     for level in sorted(val[group].unique()):
      ix=(val[group]==level).to_numpy(); n=int(ix.sum());events=int(y[ix].sum())
      if n<50 or events<5: continue
      wt=wv[ix]; pos=wt*y[ix]
      group_stats[name][group][level]={'n':n,'events':events,'referral_rate':float(np.sum(wv[ix]*f[ix])/wt.sum()),'case_capture':float(np.sum(pos*f[ix])/pos.sum()) if pos.sum()>0 else None}
    # Equal referral opportunity across sex strata; a policy benchmark, not a fairness optimum.
    zz=policy(y,sc,wv,.2,groups=val['sex_group'].to_numpy());zz.pop('selected_fraction');parity_stats[name]=zz
 names=[src_name(p,k) for p in SOURCES for k in SOURCES[p]['files']]
 all_xpt=all(any((root/(name+'.'+e)).exists() for e in ['xpt','XPT']) for name in names)
 evidence='Observed NHANES XPT data' if all_xpt else 'Local CSV data; source not verified, NOT publication-ready'
 results={'analysis_type':evidence,'data_sources':{k:{'base':f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{x["path"]}/DataFiles/','files':[src_name(k,v)+'.xpt' for v in x['files']]} for k,x in SOURCES.items()},'inclusion':{'development':raw_dev,'validation':raw_val},'development':{'n':len(dev),'events12':int(dev['outcome12'].sum()),'weighted_prevalence12':float(np.average(dev['outcome12'],weights=dev['weight'])),'median_age':float(dev['age'].median()),'mean_age_weighted':float(np.average(dev['age'],weights=dev['weight'])),'mean_bmi_weighted':float(np.average(dev['bmi'].dropna(),weights=dev.loc[dev['bmi'].notna(),'weight'])),'female_percent':float(100*np.average(dev['sex']==2,weights=dev['weight'])),'diabetes_percent':float(100*np.average(dev['diabetes'].fillna(0)==1,weights=dev['weight'])),'cap_steatosis_n':int(dev['cap_steatosis'].sum())},'validation':{'n':len(val),'events12':int(val['outcome12'].sum()),'weighted_prevalence12':float(np.average(val['outcome12'],weights=val['weight'])),'median_age':float(val['age'].median()),'mean_age_weighted':float(np.average(val['age'],weights=val['weight'])),'mean_bmi_weighted':float(np.average(val['bmi'].dropna(),weights=val.loc[val['bmi'].notna(),'weight'])),'female_percent':float(100*np.average(val['sex']==2,weights=val['weight'])),'diabetes_percent':float(100*np.average(val['diabetes'].fillna(0)==1,weights=val['weight'])),'cap_steatosis_n':int(val['cap_steatosis'].sum())},'model_stats':model_stats,'policies':policy_stats,'equal_opportunity_sex':parity_stats,'subgroups':group_stats,'features':FEATURES,'seed':SEED,'bootstrap_reps':args.bootstrap,'data_verified_as_xpt':bool(all_xpt)}
 # Approximate design-based cluster bootstrap for AUROC and referral case capture.
 rng=np.random.default_rng(SEED); reps={n:{'auroc':[],'capture20':[],'delta_capture20':[]} for n in preds}
 for b in range(args.bootstrap):
  multiplicity=bootstrap_groups(val,rng); ww=wv*multiplicity
  if ww[y==1].sum()==0 or ww[y==0].sum()==0:continue
  captures={}
  for n,sc in preds.items():
   auc=float(roc_auc_score(y,sc,sample_weight=ww)); c=policy(y,sc,ww,.2)['case_capture']
   reps[n]['auroc'].append(auc);reps[n]['capture20'].append(c);captures[n]=c
  for n in preds:
   reps[n]['delta_capture20'].append(captures[n]-captures['FIB-4'])
   reps[n].setdefault('delta_auroc',[]).append(roc_auc_score(y,preds[n],sample_weight=ww)-roc_auc_score(y,preds['FIB-4'],sample_weight=ww))
 for n in preds:
  model_stats[n]['auroc_ci95']=ci(reps[n]['auroc']);policy_stats[n]['20']['case_capture_ci95']=ci(reps[n]['capture20']);policy_stats[n]['20']['difference_vs_fib4_ci95']=ci(reps[n]['delta_capture20']);model_stats[n]['difference_auroc_vs_fib4_ci95']=ci(reps[n]['delta_auroc'])
 # secondary analysis: LSM >=8 with FIB-4 and re-fitted models, no outcome leakage.
 fits8,preds8,_=do_models(dev,val,'outcome8')
 y8=val['outcome8'].to_numpy();results['sensitivity_lsm8']={n:metric(y8,sc,wv,n!='FIB-4') for n,sc in preds8.items()}
 # separate CAP-defined subset; retains entire-cohort model and is NOT independent pre-test selection.
 cap=val['cap_steatosis'].to_numpy();results['sensitivity_cap_subset']={'n':int(cap.sum()),'events12':int(y[cap].sum()),'auroc':{n:float(roc_auc_score(y[cap],sc[cap],sample_weight=wv[cap])) if len(np.unique(y[cap]))==2 else None for n,sc in preds.items()}}
 # Permutation importance is descriptive on temporal validation; not used for model selection.
 importance={}
 try:
  imp=permutation_importance(fits['Gradient boosting'],val[FEATURES].iloc[:min(1500,len(val))],y[:min(1500,len(val))],scoring='roc_auc',n_repeats=4,random_state=SEED,n_jobs=1)
  importance={k:float(v) for k,v in zip(FEATURES,imp.importances_mean)}
 except Exception as exc:print('Importance skipped:',exc)
 results['permutation_importance']=importance
 with open(out/'results.json','w') as f:json.dump(results,f,indent=2,allow_nan=False)
 for n,sc in preds.items():val['pred_'+n.lower().replace(' ','_').replace('-','_')]=sc
 val[['SEQN','period','age','sex_group','race_group','outcome12','outcome8','weight','fib4','pred_logistic','pred_gradient_boosting']].to_csv(out/'validation_predictions.csv',index=False)
 save_plots(val,preds,policy_stats,model_stats,out,importance)
 print(json.dumps({'sample_sizes':[len(dev),len(val)],'events':[int(dev.outcome12.sum()),int(val.outcome12.sum())],'AUC':{n:round(v['auroc'],3) for n,v in model_stats.items()},'20_percent_capture':{n:round(v['20']['case_capture'],3) for n,v in policy_stats.items()}},indent=2))
 print('Output:',out.absolute())

if __name__=='__main__':main()
