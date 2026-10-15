#!/usr/bin/env python3
"""Export independently sorted metric ranks from the Fig2/Fig3 result tables.

Scores are read from computed results, never inferred from a figure. Fig2
metric ranks use the standardized metric score, not the raw plotted mean.
Batch-run variability is ranked in ascending order (lower is better).
No Overall rank is created. Missing scores and ranks remain NA.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd
import numpy as np


def read(path):
    return pd.read_csv(path,float_precision='round_trip')


def export(repo, fig3, fig2, destination):
    palette=read(repo/'Figure_code/figures/Figure3/config/method_colors.csv').sort_values('method_order')
    methods=palette.method.tolist();fixed=dict(zip(methods,range(1,len(methods)+1)))
    destination.mkdir(parents=True,exist_ok=True)
    frames=[];index=[]

    def add(name, data, score_column, source, *, direction='higher_is_better', rank_column=None,
            n_valid=None,n_expected=None,definition=''):
        assert not data.method.duplicated().any()
        assert set(data.method)<=set(methods)
        data=data.set_index('method').reindex(methods)
        result=pd.DataFrame({'method':methods,'score':pd.to_numeric(data[score_column],errors='raise').to_numpy()})
        result['rank']=(pd.to_numeric(data[rank_column],errors='raise').to_numpy() if rank_column
                        else result.score.rank(ascending=direction=='lower_is_better',method='average'))
        assert result.loc[result.score.isna(),'rank'].isna().all()
        assert result.loc[result.score.notna(),'rank'].gt(0).all()
        result['fixed_method_order']=result.method.map(fixed)
        result=result.sort_values(['rank','fixed_method_order'],kind='stable',na_position='last')
        result.insert(3,'display_order',range(1,len(result)+1))
        result['metric']=name;result['direction']=direction
        result['score_definition']=definition;result['source_table']=source.name
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        result['source_sha256']=digest
        for label,column in [('n_valid',n_valid),('n_expected',n_expected)]:
            result[label]=result.method.map(data[column]) if column else np.nan
        result['status']=np.where(result.score.notna(),'available','missing')
        result.to_csv(destination/(name+'_rank.csv'),index=False,na_rep='NA')
        frames.append(result)
        index.append({'metric':name,'file':name+'_rank.csv','direction':direction,
                      'score_definition':definition,'source_table':source.name,'source_sha256':digest,
                      'n_methods':len(result),'n_ranked':int(result.score.notna().sum())})

    sc_path=fig3/'tables/scalability_independent_rank.csv';sc=read(sc_path)
    add('scalability',sc,'Sc',sc_path,rank_column='Sc_rank',n_valid='n_valid_tasks_total',
        n_expected='n_expected_tasks_total',definition='mean of runtime and CPU-RAM reverse task-z components')
    add('runtime_efficiency',sc,'T',sc_path,rank_column='T_rank',n_valid='T_n_valid',n_expected='T_n_expected',
        definition='mean reverse sample-z of runtime seconds across 68 top-scRNA tasks')
    add('ram_efficiency',sc,'R',sc_path,rank_column='R_rank',n_valid='R_n_valid',n_expected='R_n_expected',
        definition='mean reverse sample-z of CPU RAM across 68 top-scRNA tasks')
    u_path=fig3/'tables/usability_independent_rank.csv';u=read(u_path)
    add('usability',u,'U',u_path,rank_column='U_rank',n_valid='n_valid_categories',n_expected='n_expected_categories',
        definition='weighted total of five category scores recomputed from 24 items')
    categories=fig3/'tables/usability_categories_recomputed.csv';c=read(categories)
    for category in ['Installability','Code quality','Usage guidance','Maintenance','Publication quality']:
        add('usability_'+category.lower().replace(' ','_'),c[c.category.eq(category)],'score',categories,
            rank_column='category_rank',n_valid='valid_subscores',n_expected='expected_items',
            definition='S7-weighted usability item sum within '+category)

    fig2_available=all((fig2/'tables'/name).is_file() for name in [
        'metric_summary_5050.csv','accuracy_scores_ranks.csv','batch_SD_normalized_method_summary.csv','coverage.csv'])
    if fig2_available:
        path=fig2/'tables/accuracy_scores_ranks.csv';d=read(path)
        add('directional_accuracy',d,'Accuracy',path,rank_column='Accuracy_rank',
            n_valid='Accuracy_n_components',n_expected='Accuracy_expected_components',
            definition='equal available components: Angle, CBDir, Transition V1 and simulation-only GT')
        path=fig2/'tables/metric_summary_5050.csv';d=read(path)
        coverage=read(fig2/'tables/coverage.csv')
        expected=coverage.groupby(['method','metric']).n_expected.sum()
        for metric,slug in [('Angle consistency','angle_consistency'),('CBDir','cbdir'),
                            ('Transition V1','transition_v1'),('Ground-truth correlation','ground_truth_correlation'),
                            ('Consistency score','consistency_score'),('ICCoh','iccoh'),
                            ('Performance under downsampling','downsampling')]:
            q=d[d.metric.eq(metric)].copy()
            q['n_tasks_valid']=q.real_n_z_valid+q.simulation_n_z_valid
            q['n_tasks_expected']=[expected.loc[m,metric] for m in q.method]
            add(slug,q,'score',path,n_valid='n_tasks_valid',n_expected='n_tasks_expected',
                definition='task-z metric score; shared metrics real/simulation 50:50; GT/DS simulation only')
        path=fig2/'tables/batch_SD_normalized_method_summary.csv';d=read(path)
        d['n_datasets_valid']=d.real_n_datasets+d.simulation_n_datasets
        d['n_datasets_expected']=d.real_n_expected_datasets+d.simulation_n_expected_datasets
        add('batch_run_variability',d,'combined_score',path,direction='lower_is_better',
            rank_column='combined_variability_rank',n_valid='n_datasets_valid',n_expected='n_datasets_expected',
            definition='positive SD task-z; metrics within dataset, then datasets within type, then 50:50')
    else:
        print('Fig2 results were not found; exported the nine Fig3 ranks only. '
              'Run Fig2, then rerun this exporter to refresh all 18 metric ranks.')
    combined=pd.concat(frames,ignore_index=True)
    combined.to_csv(destination/'all_metric_ranks.csv',index=False,na_rep='NA')
    combined.pivot(index='method',columns='metric',values='rank').reindex(methods).to_csv(
        destination/'rank_matrix.csv',na_rep='NA')
    pd.DataFrame(index).to_csv(destination/'rank_index.csv',index=False,na_rep='NA')
    summary={'status':'PASS','rank_tables':len(frames),'methods_per_table':28,
             'Fig2_results_included':fig2_available,'Overall_rank_created':False}
    print(json.dumps(summary))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--fig2-results',type=Path)
    parser.add_argument('--fig3-results',type=Path)
    parser.add_argument('--rank-dir',type=Path)
    args=parser.parse_args();repo=args.repo_root.expanduser().resolve()
    export(repo,(args.fig3_results or repo/'Figure_code/figures/Figure3/results').resolve(),
           (args.fig2_results or repo/'Figure_code/figures/Figure2/results').resolve(),
           (args.rank_dir or repo/'rank/scRNA').resolve())
