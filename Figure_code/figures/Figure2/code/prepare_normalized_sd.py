#!/usr/bin/env python3
"""Unitless cross-metric SD composites: low = less relative batch variation.

These are descriptive relative indices over eight measured methods, not raw SD,
not within-run z values, and not an input to Accuracy.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from io_atomic import save_csv
P=Path(__file__).resolve().parents[1];T=P/'tables'
rank=pd.read_csv(T/'accuracy_scores_ranks.csv');methods=rank.method.tolist();order=dict(zip(methods,rank.display_order))
q=pd.read_csv(P/'data/batch_sd_clean_long.csv')
q['raw_sd']=pd.to_numeric(q.clean_value,errors='raise')
assert not q.raw_sd.dropna().lt(0).any()
keys=['metric','data_type','dataset_id']
g=q.groupby(keys).raw_sd.agg(['count','mean','std']).rename(columns={'count':'n_valid_methods','mean':'mean_SD_across_methods','std':'sample_SD_of_SD_across_methods'})
q=q.join(g,on=keys)
ok=q.n_valid_methods.ge(2)&q.sample_SD_of_SD_across_methods.gt(0)
q['normalized_SD_z']=((q.raw_sd-q.mean_SD_across_methods)/q.sample_SD_of_SD_across_methods).where(ok)
q['SD_fractional_rank']=((q.groupby(keys).raw_sd.rank(ascending=True,method='average')-1)/(q.n_valid_methods-1)).where(q.n_valid_methods.ge(2))
q['normalization_status']=np.select([q.raw_sd.isna(),q.n_valid_methods.lt(2),q.sample_SD_of_SD_across_methods.eq(0)],['raw_SD_missing','fewer_than_2_valid_methods','all_valid_SD_equal'],default='valid')
q['display_order']=q.method.map(order)
save_csv(q,T/'batch_SD_task_normalized.csv')
catalog=q.groupby(['data_type','dataset_id']).metric.agg(lambda x:sorted(set(x))).to_dict()
rows=[]
for (m,t,d),f in q.groupby(['method','data_type','dataset_id'],sort=False):
    applicable=catalog[t,d];valid=f[f.normalized_SD_z.notna()];missing=f[f.normalized_SD_z.isna()]
    rows.append({'method':m,'data_type':t,'dataset_id':d,'normalized_variability':valid.normalized_SD_z.mean(),
        'fractional_rank_variability':f.SD_fractional_rank.mean(),'n_valid_components':len(valid),'n_expected_components':len(applicable),
        'applicable_metrics':';'.join(applicable),'valid_metrics':';'.join(sorted(valid.metric)),
        'missing_components':';'.join(sorted(missing.metric)),
        'missing_reasons':';'.join(r.metric+':'+r.normalization_status for r in missing.itertuples()),
        'n_valid_raw_SD':int(f.raw_sd.notna().sum()),'n_true_raw_SD_zero':int(f.raw_sd.eq(0).sum()),
        'complete_components':len(valid)==len(applicable),'display_order':order[m],
        'source_run_count':'unknown','direction':'lower_is_less_relative_variability'})
dataset=pd.DataFrame(rows).sort_values(['display_order','data_type','dataset_id'])
save_csv(dataset,T/'batch_SD_one_value_per_dataset.csv')

def summarise(ds,value='normalized_variability',complete=False):
    if complete:ds=ds[ds.complete_components]
    rows=[]
    for m in methods:
        row={'method':m,'display_order':order[m]}
        for typ in ['real','simulation']:
            f=ds[ds.method.eq(m)&ds.data_type.eq(typ)];v=f[value].dropna()
            row[typ+'_score']=v.mean();row[typ+'_n_datasets']=len(v)
            row[typ+'_n_complete_datasets']=int((f.complete_components&f[value].notna()).sum())
            row[typ+'_n_expected_datasets']=sum(t==typ for t,d in catalog)
        row['combined_score']=.5*row['real_score']+.5*row['simulation_score']
        row['direction']='lower_is_less_relative_variability';row['available']=np.isfinite(row['combined_score'])
        row['source_run_count']='unknown';rows.append(row)
    f=pd.DataFrame(rows);f['combined_variability_rank']=f.combined_score.rank(ascending=True,method='average')
    return f
main=summarise(dataset);save_csv(main,T/'batch_SD_normalized_method_summary.csv')
save_csv(summarise(dataset,'fractional_rank_variability'),T/'batch_SD_fractional_rank_sensitivity.csv')
save_csv(summarise(dataset,complete=True),T/'batch_SD_complete_dataset_sensitivity.csv')
metric=q.groupby(['method','metric','data_type']).normalized_SD_z.mean().unstack('data_type')
metric['combined_score']=np.where(metric.index.get_level_values('metric')=='Ground-truth correlation',metric.simulation,.5*metric.real+.5*metric.simulation)
save_csv(metric,T/'batch_SD_normalized_metric_summary.csv',index=True)
save_csv(metric.groupby('method').combined_score.mean().reindex(methods).rename('metric_first_combined_score'),T/'batch_SD_metric_first_sensitivity.csv',index=True)
result={'status':'PASS','measured_methods':int(main.available.sum()),'task_SD_rows':len(q),'valid_task_SD_z':int(q.normalized_SD_z.notna().sum()),
 'dataset_composites':len(dataset),'valid_dataset_composites':int(dataset.normalized_variability.notna().sum()),
 'true_raw_SD_zeros_preserved':int(q.raw_sd.eq(0).sum()),'Accuracy_unchanged':True,'direction':'lower_is_less_relative_variability',
 'normalization':'positive SD z within metric x data_type x dataset across measured methods; ddof=1',
 'method_aggregation':'available components equal within dataset; datasets equal within type; real/simulation 50:50',
 'n_runs':'unknown; source SD summaries only, no individual batch-run values supplied'}
(P/'validation/normalized_SD_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result))
