#!/usr/bin/env python3
"""Recompute directional Accuracy and raw summaries; no Overall score."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from io_atomic import save_csv

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'tables'
OUT.mkdir(exist_ok=True)
methods = pd.read_csv(PROJECT/'data/methods.csv').sort_values('method_order').method.tolist()
assert len(methods) == 28 and len(set(methods)) == 28
excluded = {'Region Velocity', 'Region-Velocity', 'Region velocity', 'PhyloVelo'}
assert not excluded.intersection(methods)
primary = ['Angle consistency', 'CBDir', 'Transition V1', 'Ground-truth correlation']
shared = ['Angle consistency', 'CBDir', 'Transition V1', 'Consistency score', 'ICCoh']
all_metrics = primary + ['Consistency score', 'ICCoh', 'Performance under downsampling']
raw = pd.read_csv(PROJECT/'data/primary_clean_long.csv')
assert set(raw.method).issubset(methods)
assert set(all_metrics).issubset(set(raw.metric))
assert not raw.duplicated(['method','metric','data_type','dataset_id']).any()
raw['value'] = pd.to_numeric(raw.clean_value, errors='coerce').where(lambda x: np.isfinite(x))
task_keys = ['metric','data_type','dataset_id']
task = raw.groupby(task_keys).value.agg(['count','mean','std']).rename(columns={'count':'task_n','mean':'task_mean','std':'task_sample_sd'})
raw = raw.join(task, on=task_keys)
valid = raw.task_n.ge(2) & raw.task_sample_sd.gt(0)
raw['z'] = ((raw.value-raw.task_mean)/raw.task_sample_sd).where(valid)
raw['z_missing_reason'] = np.select([raw.value.isna(),raw.task_n.lt(2),raw.task_sample_sd.eq(0)],['raw_missing','fewer_than_2_valid_methods','zero_task_sample_sd'],default='')
save_csv(raw,OUT/'task_z.csv')
coverage=[]
for m in methods:
    for metric in all_metrics:
        for typ in ['real','simulation']:
            q=raw[raw.method.eq(m)&raw.metric.eq(metric)&raw.data_type.eq(typ)]
            applicable=not (typ=='real' and metric in ['Ground-truth correlation','Performance under downsampling'])
            expected=raw[raw.metric.eq(metric)&raw.data_type.eq(typ)].dataset_id.nunique() if applicable else 0
            coverage.append({'method':m,'metric':metric,'data_type':typ,'applicable':applicable,'n_expected':expected,
                'n_source_rows':len(q),'n_raw_valid':int(q.value.notna().sum()),'n_z_valid':int(q.z.notna().sum()),
                'zero_count':int(q.value.eq(0).sum()),'negative_count':int(q.value.lt(0).sum()),
                'n_raw_valid_but_unstandardizable':int((q.value.notna()&q.z.isna()).sum())})
save_csv(pd.DataFrame(coverage),OUT/'coverage.csv')

def stats(values):
    v=values.dropna();n=len(v)
    return {'raw_mean':v.mean() if n else np.nan, 'raw_sd':v.std(ddof=1) if n>1 else np.nan,
            'raw_se':v.std(ddof=1)/np.sqrt(n) if n>1 else np.nan, 'n_raw_valid':n}

rows=[]
for m in methods:
    for metric in all_metrics:
        q=raw[raw.method.eq(m)&raw.metric.eq(metric)]
        parts={}
        for typ in ['real','simulation']:
            x=q[q.data_type.eq(typ)]
            parts[typ]={**stats(x.value),'score':x.z.mean(),'n_z_valid':int(x.z.notna().sum())}
        row={'method':m,'metric':metric}
        for typ,data in parts.items():row.update({typ+'_'+k:v for k,v in data.items()})
        if metric in shared:
            ok=np.isfinite(parts['real']['score']) and np.isfinite(parts['simulation']['score'])
            row['score']=.5*parts['real']['score']+.5*parts['simulation']['score'] if ok else np.nan
            row['combined_raw_mean']=.5*parts['real']['raw_mean']+.5*parts['simulation']['raw_mean']
            row['combined_raw_se']=np.sqrt(.25*parts['real']['raw_se']**2+.25*parts['simulation']['raw_se']**2)
            row['type_weighting']='0.5_real_plus_0.5_simulation'
        else:
            row['score']=parts['simulation']['score']
            row['combined_raw_mean']=parts['simulation']['raw_mean']
            row['combined_raw_se']=parts['simulation']['raw_se']
            row['type_weighting']='simulation_only'
        rows.append(row)
summary=pd.DataFrame(rows)
save_csv(summary,OUT/'metric_summary_5050.csv')
wide=summary.pivot(index='method',columns='metric',values='score').reindex(methods)
rank=pd.DataFrame({'method':methods,'fixed_method_order':np.arange(1,29)})
rank['Accuracy']=wide[primary].mean(axis=1).to_numpy()
rank['Accuracy_n_components']=wide[primary].notna().sum(axis=1).to_numpy()
rank['Accuracy_expected_components']=4
for component in primary:
    slug=component.replace(' ','_').replace('-','_')
    vals=wide[component].to_numpy()
    rank['component_'+slug]=vals
    rank['effective_weight_'+slug]=np.where(np.isfinite(vals),1/rank.Accuracy_n_components,np.nan)
rank['Accuracy_missing_components']=[';'.join(k for k in primary if not np.isfinite(wide.loc[m,k])) for m in methods]
rank['Accuracy_rank']=rank.Accuracy.rank(ascending=False,method='average')
rank['Accuracy_reversed_rank']=29-rank.Accuracy_rank
rank=rank.sort_values(['Accuracy_rank','fixed_method_order'],na_position='last',kind='stable')
rank['display_order']=np.arange(1,29)
save_csv(rank,OUT/'accuracy_scores_ranks.csv')
order=dict(zip(rank.method,rank.display_order))
summary['display_order']=summary.method.map(order)
save_csv(summary.sort_values(['display_order','metric']),OUT/'metric_plot_data_accuracy_order.csv')

# Batch SD stays in its original metric units; no average across six metrics.
sd=pd.read_csv(PROJECT/'data/batch_sd_clean_long.csv')
assert set(sd.method).issubset(methods)
assert not sd.duplicated(['method','metric','data_type','dataset_id']).any()
sd['valid_sd']=pd.to_numeric(sd.clean_value,errors='coerce')
sd['valid_sd']=sd.valid_sd.where(np.isfinite(sd.valid_sd)&sd.valid_sd.ge(0))
sdrows=[]
for m in methods:
    for metric in primary+['Consistency score','ICCoh']:
        q=sd[sd.method.eq(m)&sd.metric.eq(metric)]
        vr=q[q.data_type.eq('real')].valid_sd
        vs=q[q.data_type.eq('simulation')].valid_sd
        mr=vr.mean();ms=vs.mean()
        val=.5*mr+.5*ms if metric in shared else ms
        sdrows.append({'method':m,'metric':metric,'real_mean_sd':mr,'simulation_mean_sd':ms,'combined_mean_sd':val,
          'real_n_valid':int(vr.notna().sum()),'simulation_n_valid':int(vs.notna().sum()),
          'real_zero_count':int(vr.eq(0).sum()),'simulation_zero_count':int(vs.eq(0).sum()),
          'display_order':order[m],'label_when_missing':'blank','data_status':'measured' if np.isfinite(val) else 'not_available'})
sds=pd.DataFrame(sdrows).sort_values(['display_order','metric'])
save_csv(sds,OUT/'batch_sd_numeric_accuracy_order.csv')
save_csv(sds[sds.combined_mean_sd.notna()],OUT/'batch_sd_numeric_measured_only.csv')

# Independent numerical verification is performed by independent_verify.py.
checks={'status':'PASS','methods':len(methods),'excluded_before_z':True,
 'Accuracy_components':primary,'real_simulation_shared_weights':[.5,.5],
 'raw_numeric_zero_count':int(raw.value.eq(0).sum()),
 'raw_negative_count':int(raw.value.lt(0).sum()),
 'batch_sd_measured_methods':sds[sds.combined_mean_sd.notna()].method.unique().tolist(),
 'downsampling_valid_methods':int(summary[summary.metric.eq('Performance under downsampling')].score.notna().sum()),
 'palette_selection':'A','prior_results_used_as_inputs':False}
(PROJECT/'validation').mkdir(exist_ok=True)
(PROJECT/'validation/accuracy_checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks))
