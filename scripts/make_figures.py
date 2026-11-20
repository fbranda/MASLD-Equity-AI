#!/usr/bin/env python3
"""Regenerates only empirical figure panels; never creates artificial participant data."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import run_analysis as A

C={'FIB-4':'#7f8c99','Logistic':'#3978a6','Gradient boosting':'#0b7665'}

plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def save(fig,out,name):
 fig.savefig(out/name,dpi=260,bbox_inches='tight');plt.close(fig)

def quantile_bins(s,y,w,bins=10):
 ix=np.argsort(s);ss=s[ix];ww=w[ix];cy=np.cumsum(ww)/ww.sum()
 edges=np.interp(np.linspace(0,1,bins+1),cy,ss)
 edges[0]=min(0.0,ss.min()-1e-6);edges[-1]=ss.max()+1e-6
 groups=np.digitize(s,edges[1:-1]);lst=[]
 for j in range(bins):
  mask=groups==j
  if not mask.any():continue
  lst.append([np.average(s[mask],weights=w[mask]),np.average(y[mask],weights=w[mask]),int(mask.sum())])
 return np.array(lst)

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--results',required=True)
 a=p.parse_args();out=Path(a.results);res=json.loads((out/'results.json').read_text())
 train,_=A.make_cohort(Path(a.data),'development',False)
 val,_=A.make_cohort(Path(a.data),'validation',False)
 rows=pd.read_csv(out/'validation_predictions.csv')
 assert np.allclose(rows.SEQN,val.SEQN)
 ss={'FIB-4':rows.fib4.to_numpy(),'Logistic':rows.pred_logistic.to_numpy(),'Gradient boosting':rows.pred_gradient_boosting.to_numpy()}
 w=A.norm_weights(val);y=val.outcome12.to_numpy()
 # Figure 1: observed flow across independently collected survey periods.
 flow=res['inclusion']
 fig,ax=plt.subplots(figsize=(7.1,3.55));ax.set_xlim(0,7);ax.set_ylim(0,3.4);ax.axis('off')
 labels=['Adults >=20 y','With metabolic risk','Complete valid VCTE','FIB-4 labs and weights','LSM >=12 kPa']
 xpositions=[.75,2.13,3.55,4.98,6.3]
 for yi,period in [(2.40,'development'),(1.10,'validation')]:
  vals=[flow[period]['examined_20plus'],flow[period]['at_risk_20plus'],flow[period]['valid_vcte_at_risk'],flow[period]['eligible'],res[period]['events12']]
  for i,(xx,nn) in enumerate(zip(xpositions,vals)):
   ax.text(xx,yi,f'{nn:,}',ha='center',va='center',fontsize=10.3,weight='bold',bbox=dict(boxstyle='round,pad=.6',facecolor='#eff5f7',edgecolor='#c1d1d7'))
   if i<4:ax.annotate('',xy=(xpositions[i+1]-.46,yi),xytext=(xx+.44,yi),arrowprops=dict(arrowstyle='->',color='#5b6d78',lw=1.25))
 ax.text(.08,3.05,'NHANES 2017–March 2020',fontsize=10,weight='bold')
 ax.text(.08,1.75,'NHANES August 2021–August 2023',fontsize=10,weight='bold')
 for xx,ll in zip(xpositions,labels):ax.text(xx,.36,ll,ha='center',va='top',fontsize=8.2)
 fig.tight_layout();save(fig,out,'fig1_cohort_flow.png')
 # Figure 2 ROC.
 fig,ax=plt.subplots(figsize=(6.35,4.8))
 for k,s in ss.items():
  fpr,tpr,_=roc_curve(y,s,sample_weight=w)
  ax.plot(fpr,tpr,label=f'{k}: AUROC {res["model_stats"][k]["auroc"]:.3f}',c=C[k],lw=2.1)
 ax.plot([0,1],[0,1],linestyle='--',color='#a9adae',lw=1)
 ax.set(xlabel='1 – specificity',ylabel='Sensitivity',xlim=(0,1),ylim=(0,1),title='Temporal-validation discrimination')
 ax.legend(loc='lower right',frameon=False,fontsize=9);ax.grid(alpha=.12)
 fig.tight_layout();save(fig,out,'fig2_roc.png')
 # Figure 3 weighted reliability bins.
 fig,ax=plt.subplots(figsize=(6.35,4.8))
 for k in ['Logistic','Gradient boosting']:
  b=quantile_bins(ss[k],y,w,10)
  ax.plot(b[:,0],b[:,1],marker='o',label=k,c=C[k],lw=1.9,ms=4.5)
 ax.plot([0,.55],[0,.55],ls='--',lw=1,c='#a0a4a7',label='Perfect calibration')
 ax.set(xlim=(0,.55),ylim=(0,.55),xlabel='Mean model-predicted probability',ylabel='Survey-weighted observed fraction',title='Temporal-validation calibration')
 ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.12)
 fig.tight_layout();save(fig,out,'fig3_calibration.png')
 # Figure 4 capacity case capture.
 fig,ax=plt.subplots(figsize=(6.45,4.8))
 for k in ['FIB-4','Logistic','Gradient boosting']:
  yy=[100*res['policies'][k][str(c)]['case_capture'] for c in [10,20,30]]
  ax.plot([10,20,30],yy,marker='o',lw=2.2,ms=6,c=C[k],label=k)
 ax.plot([10,20,30],[10,20,30],ls=':',c='#8a8f95',label='Random selection (expected)')
 ax.set(xlabel='Hypothetical VCTE capacity (% of eligible adults)',ylabel='Elevated-LSM cases captured (%)',xlim=(7.5,32.5),ylim=(0,90),xticks=[10,20,30],title='Case capture under fixed testing capacity')
 ax.grid(alpha=.14);ax.legend(loc='upper left',frameon=False,fontsize=9)
 fig.tight_layout();save(fig,out,'fig4_capacity.png')
 # Figure 5: subgroup opportunities and case detection.
 g=res['detailed_equity'];fig,ax=plt.subplots(figsize=(7.2,4.85))
 labs=['Female','Male','Age 20–44','Age 45–64','Age 65+']
 vals=[('sex','Female'),('sex','Male'),('age','20-44'),('age','45-64'),('age','65+')]
 ypos=np.arange(len(labs));barw=.33
 for jj,model in enumerate(['FIB-4','Gradient boosting']):
  x=[100*g[model][dim][name]['fraction_referred'] for dim,name in vals]
  ax.barh(ypos+(jj-.5)*barw,x,height=barw,color=C[model],label=model)
 ax.set_yticks(ypos,labs);ax.invert_yaxis();ax.set(xlabel='Fraction referred at 20% overall testing capacity (%)',title='Referral opportunity by demographic group')
 ax.axvline(20,ls='--',lw=.9,c='#888',label='Population-level quota');ax.legend(frameon=False,fontsize=9);ax.grid(axis='x',alpha=.12)
 fig.tight_layout();save(fig,out,'fig5_equity_referral.png')
 # Figure 6: weighted permutation importance, full temporal validation, weighted AUROC scoring.
 models,_,_=A.do_models(train,val,'outcome12')
 model=models['Gradient boosting'];base=roc_auc_score(y,model.predict_proba(val[A.FEATURES])[:,1],sample_weight=w)
 rng=np.random.default_rng(A.SEED+17);importance={}
 X=val[A.FEATURES].reset_index(drop=True)
 for col in A.FEATURES:
  drops=[]
  for b in range(9):
   xp=X.copy();xp[col]=rng.permutation(xp[col].to_numpy())
   scr=model.predict_proba(xp)[:,1]
   drops.append(float(base-roc_auc_score(y,scr,sample_weight=w)))
  importance[col]={'mean_auroc_drop':float(np.mean(drops)),'sd_auroc_drop':float(np.std(drops,ddof=1))}
 res['weighted_permutation_importance']=importance
 (out/'results.json').write_text(json.dumps(res,indent=2,allow_nan=False))
 order=sorted(importance,key=lambda c:importance[c]['mean_auroc_drop'])
 fig,ax=plt.subplots(figsize=(6.55,4.65));vbar=np.array([importance[c]['mean_auroc_drop'] for c in order]);err=[importance[c]['sd_auroc_drop'] for c in order]
 ax.barh(order,vbar,xerr=err,color='#347c8c',ecolor='#334d59',capsize=2)
 ax.axvline(0,color='black',lw=.6);ax.set(xlabel='Decrease in weighted AUROC after permutation',title='Model-agnostic feature importance')
 ax.grid(axis='x',alpha=.12);fig.tight_layout();save(fig,out,'fig6_importance.png')
 print('Saved 6 empirical figures; weighted importance:',importance)
if __name__=='__main__':main()
