#!/usr/bin/env python3
"""Additional prespecified sensitivity, missingness, and equity audit for NHANES.
Requires true CDC XPT files and run_analysis.py outputs.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import run_analysis as A

def pct(x): return 100*float(x)
def group_ranking(y, s, w, groups, k=.2):
    chosen=A.select_fraction(s,w,k); out={}
    for level in sorted(np.unique(groups)):
        ix=groups==level
        nw=int(ix.sum()); ne=int(y[ix].sum()); sw=float(w[ix].sum())
        out[str(level)]={'n':nw,'events':ne,
                         'fraction_referred':float(np.sum(w[ix]*chosen[ix])/sw) if sw>0 else None,
                         'case_capture':float(np.sum(w[ix]*chosen[ix]*y[ix])/np.sum(w[ix]*y[ix])) if np.sum(w[ix]*y[ix])>0 else None,
                         'weighted_prevalence':float(np.sum(w[ix]*y[ix])/sw) if sw>0 else None}
    return out

def imbalance(groupmetrics, a, b, feature):
    try:return groupmetrics[b][feature]-groupmetrics[a][feature]
    except (TypeError,KeyError):return None

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--results',required=True);p.add_argument('--bootstrap',type=int,default=450)
    ar=p.parse_args();root=Path(ar.data); out=Path(ar.results)
    d,tr_flow=A.make_cohort(root,'development',False);v,v_flow=A.make_cohort(root,'validation',False)
    res=json.loads((out/'results.json').read_text()); yy=v.outcome12.to_numpy(); w=A.norm_weights(v)
    missing={};extra_descriptive={}
    for label,tab in [('development',d),('validation',v)]:
        missing[label]={f:{'n_missing':int(tab[f].isna().sum()),'percent_missing':float(100*tab[f].isna().mean())} for f in A.FEATURES}
        extra_descriptive[label]={
          'outcome8_n':int(tab.outcome8.sum()),'outcome8_weighted_prevalence':float(np.average(tab.outcome8,weights=tab.weight)),
          'weight_sum':float(tab.weight.sum()), 'age_unweighted_median':float(tab.age.median()),
          'bmi_unweighted_median':float(tab.bmi.median()),
          'age_by_group':{n:int((tab.age_group==n).sum()) for n in ['20-44','45-64','65+']},
          'n_vcte12_ratio':int((tab['lsm']>=12).sum()),
          'liver_stiffness_median':float(tab.lsm.median()),
          'liver_stiffness_p95':float(tab.lsm.quantile(.95)),
          'laboratory_extremes':{c:{'min':float(tab[c].min()),'max':float(tab[c].max())} for c in ['ast','alt','platelets','ggt']},
          'female_n':int((tab.sex==2).sum()),
          'lsm_ge50_n':int((tab.lsm>=50).sum()),
        }
    res['missingness']=missing;res['further_descriptives']=extra_descriptive
    # Sensitivity removes potentially less ubiquitous biochemical markers GGT and HbA1c.
    all_feats=A.FEATURES[:];reduced=[x for x in all_feats if x not in ('ggt','hba1c')]
    A.FEATURES[:]=reduced
    fitted,red_pred,_=A.do_models(d,v,'outcome12')
    A.FEATURES[:]=all_feats
    res['reduced_predictors']={'features':reduced,'models':{n:{'auroc':float(roc_auc_score(yy,sc,sample_weight=w)),'case_capture_at_20':A.policy(yy,sc,w,.20)['case_capture']} for n,sc in red_pred.items()}}
    # Alternative fixed conventional age-aware FIB-4 thresholds, separate from quota matched ranking.
    low=np.where(v.age>=65,2.,1.3)
    flag=(v.fib4>=low).astype(float).to_numpy()
    res['fib4_clinical_threshold']={'cutoff_if_age_below65':1.3,'cutoff_if_age65plus':2.0,
        'fraction_referred':float(np.average(flag,weights=w)),
        'case_capture':float(np.sum(w*yy*flag)/np.sum(w*yy)),
        'ppv':float(np.sum(w*yy*flag)/np.sum(w*flag))}
    # Full-sample main predictors drawn from published individual predictions, which are not edited.
    pred=pd.read_csv(out/'validation_predictions.csv');assert len(pred)==len(v) and np.allclose(pred.SEQN,v.SEQN)
    scores={'FIB-4':v.fib4.to_numpy(),'Logistic':pred.pred_logistic.to_numpy(),'Gradient boosting':pred.pred_gradient_boosting.to_numpy()}
    sex=v.sex_group.to_numpy();age=v.age_group.to_numpy();race=v.race_group.to_numpy()
    main_group={}
    for name,sc in scores.items():
        main_group[name]={g:group_ranking(yy,sc,w,vals) for g,vals in [('sex',sex),('age',age),('race',race)]}
    res['detailed_equity']=main_group
    # Paired cluster reweighting and reapplication of quota: percentile CIs are approximate.
    rng=np.random.default_rng(A.SEED+7);reps={}
    metrics=['male_minus_female_referral','male_minus_female_capture','old_minus_young_referral','old_minus_young_capture', 'equal_sex_minus_global_capture','equal_sex_minus_global_referral_gap']
    for name in scores: reps[name]={k:[] for k in metrics}
    for ib in range(ar.bootstrap):
        ww=w*A.bootstrap_groups(v,rng)
        if not np.any(ww):continue
        for name,sc in scores.items():
            groups_sex=group_ranking(yy,sc,ww,sex)
            groups_age=group_ranking(yy,sc,ww,age)
            z=A.policy(yy,sc,ww,.2)
            zz=A.policy(yy,sc,ww,.2,groups=sex)
            eg=zz.pop('selected_fraction');global_fraction=z.pop('selected_fraction')
            sex_gap_global=abs(groups_sex['Male']['fraction_referred']-groups_sex['Female']['fraction_referred'])
            # zero referee difference within sex in equalized allocation by definition
            for key,val in [('male_minus_female_referral',imbalance(groups_sex,'Female','Male','fraction_referred')),
                            ('male_minus_female_capture',imbalance(groups_sex,'Female','Male','case_capture')),
                            ('old_minus_young_referral',imbalance(groups_age,'20-44','65+','fraction_referred')),
                            ('old_minus_young_capture',imbalance(groups_age,'20-44','65+','case_capture')),
                            ('equal_sex_minus_global_capture',zz['case_capture']-z['case_capture']),
                            ('equal_sex_minus_global_referral_gap',-sex_gap_global)]:
                if val is not None and np.isfinite(val):reps[name][key].append(float(val))
    equity_intervals={}
    for name in scores:equity_intervals[name]={k:{'point_estimate':None,'ci95':A.ci(vv)} for k,vv in reps[name].items()}
    for name in scores:
        g=main_group[name]
        glob=A.policy(yy,scores[name],w,.2)
        eq=A.policy(yy,scores[name],w,.2,groups=sex)
        point={
           'male_minus_female_referral':imbalance(g['sex'],'Female','Male','fraction_referred'),
           'male_minus_female_capture':imbalance(g['sex'],'Female','Male','case_capture'),
           'old_minus_young_referral':imbalance(g['age'],'20-44','65+','fraction_referred'),
           'old_minus_young_capture':imbalance(g['age'],'20-44','65+','case_capture'),
           'equal_sex_minus_global_capture':eq['case_capture']-glob['case_capture'],
           'equal_sex_minus_global_referral_gap':-abs(g['sex']['Male']['fraction_referred']-g['sex']['Female']['fraction_referred']),
        }
        for k in metrics:equity_intervals[name][k]['point_estimate']=point[k]
    res['equity_bootstrap_intervals']=equity_intervals
    res['analysis_addendum']='Valid XPT input, threshold and data were fixed by local scripts; all results observational.'
    (out/'results.json').write_text(json.dumps(res,indent=2,allow_nan=False))
    print(json.dumps({'n':len(v),'missingness':missing,'reduced_predictors':res['reduced_predictors'],'fib4_threshold':res['fib4_clinical_threshold'],'equity':equity_intervals['Gradient boosting']},indent=2))
if __name__=='__main__':main()
