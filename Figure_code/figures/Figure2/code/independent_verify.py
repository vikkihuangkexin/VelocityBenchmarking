#!/usr/bin/env python3
"""Independent raw-source + numerical verification, using Python stdlib math.

Does not import any production parser/scoring function; never uses prior ranks
as calculation inputs. Re-reads every raw CSV and the locked dataset ID dictionary.
"""
from pathlib import Path
from collections import defaultdict
import csv,io,zipfile,math,re,json

P=Path(__file__).resolve().parents[1];NAN=float('nan');checks=0;max_diff=0.
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))
methods=[r['method'] for r in sorted(read(P/'data/methods.csv'),key=lambda r:int(r['method_order']))]
def num(x):
    s=str(x).strip()
    if s.casefold() in {'','na','nan','n/a','none','null','error'}:return NAN
    v=float(s);assert math.isfinite(v);return v
def mean(xs):
    xs=[v for v in xs if math.isfinite(v)]
    return math.fsum(xs)/len(xs) if xs else NAN
def sd(xs):
    xs=[v for v in xs if math.isfinite(v)]
    if len(xs)<2:return NAN
    mu=mean(xs);return math.sqrt(math.fsum((v-mu)**2 for v in xs)/(len(xs)-1))
def assert_number(a,b,label):
    global checks,max_diff
    a=num(a);b=num(b);checks+=1
    if math.isnan(a) or math.isnan(b):assert math.isnan(a) and math.isnan(b),label
    else:
        max_diff=max(max_diff,abs(a-b))
        assert math.isclose(a,b,rel_tol=1e-10,abs_tol=3e-12),(label,a,b)
def canonical(s):
    s=str(s).strip().replace('κ','k').replace('–','-').replace('—','-')
    s=re.sub(r'\s+',' ',s).casefold()
    lookup={m.casefold():m for m in methods+['Region Velocity','PhyloVelo']}
    lookup.update({'region-velocity':'Region Velocity','tsvelo':'TSvelo','scvelo stochastic':'scVelo-stochastic','scvelo dynamical':'scVelo-dynamical','kappa-velo':'k-velo','pyro velocity':'Pyro-Velocity'})
    assert s in lookup,s
    return lookup[s]
dictionary={r['dataset_number']:r['dataset_id'] for r in read(P/'data/real_dataset_ids.csv')}
names={'AngleConsistency':'Angle consistency','CBDir':'CBDir','ConsistencyScore':'Consistency score',
 'GroundTruthCorrelation':'Ground-truth correlation','ICCoh':'ICCoh',
 'TransitionScoreV1':'Transition V1','TransitionScore':'Transition V1'}
original={'primary':{},'std':{}};original_tokens={'primary':{},'std':{}}
for filename in sorted((P/'data/raw').glob('*/*.csv')):
    short=filename.relative_to(P/'data/raw').as_posix()
    name=filename.name.split('_scRNA_')[0]
    match=re.fullmatch(r'\w+_scRNA_(real|simulation)_(02_summary|03_batchrun_std)\.csv',filename.name)
    assert match,filename
    typ,version=match.groups();kind='std' if version=='03_batchrun_std' else 'primary'
    metric='Performance under downsampling' if filename.parent.name=='downsampling' else names[name]
    reader=csv.DictReader(io.StringIO(filename.read_text(encoding='utf-8-sig')))
    fields=reader.fieldnames;assert fields[0].strip().casefold() in {'','method'};columns=fields[1:]
    records={}
    for record in reader:
        if not any(str(v or '').strip() for v in record.values()):continue
        m=canonical(record[fields[0]])
        if m in {'Region Velocity','PhyloVelo'}:continue
        assert m not in records;records[m]=record
    selected=methods if kind=='primary' else [m for m in methods if m in records]
    for m in selected:
        for column in columns:
            dataset=dictionary[column.strip()] if typ=='real' else column.strip()
            k=(m,metric,typ,dataset);assert k not in original[kind]
            token=records[m][column] if m in records else ''
            original[kind][k]=num(token);original_tokens[kind][k]=token
            assert kind!='std' or not math.isfinite(original[kind][k]) or original[kind][k]>=0
for kind,filename in [('primary','primary_clean_long.csv'),('std','batch_sd_clean_long.csv')]:
    observed=read(P/'data'/filename);keys={(r['method'],r['metric'],r['data_type'],r['dataset_id']) for r in observed}
    assert len(keys)==len(observed) and keys==set(original[kind])
    for r in observed:
        k=(r['method'],r['metric'],r['data_type'],r['dataset_id'])
        assert r['raw_token']==original_tokens[kind][k],k
        assert_number(r['clean_value'],original[kind][k],k)

def task_z(values):
    groups=defaultdict(list)
    for (m,k,t,d),v in values.items():
        if math.isfinite(v):groups[k,t,d].append(v)
    result={};stats={}
    for task,vals in groups.items():stats[task]=(len(vals),mean(vals),sd(vals))
    for (m,k,t,d),v in values.items():
        n,mu,s=stats.get((k,t,d),(0,NAN,NAN))
        result[m,k,t,d]=(v-mu)/s if n>=2 and math.isfinite(s) and s>0 and math.isfinite(v) else NAN
    return result,stats
z,taskstats=task_z(original['primary'])
task_rows=read(P/'tables/task_z.csv')
assert len(task_rows)==len(z) and {(r['method'],r['metric'],r['data_type'],r['dataset_id']) for r in task_rows}==set(z)
for r in task_rows:
    k=(r['method'],r['metric'],r['data_type'],r['dataset_id']);n,mu,s=taskstats.get(k[1:],(0,NAN,NAN))
    for field,expected in [('z',z[k]),('task_n',n),('task_mean',mu),('task_sample_sd',s)]:assert_number(r[field],expected,(k,field))
shared={'Angle consistency','CBDir','Transition V1','Consistency score','ICCoh'}
primary=['Angle consistency','CBDir','Transition V1','Ground-truth correlation']
rawgroups=defaultdict(list);zgroups=defaultdict(list)
for (m,k,t,d),v in original['primary'].items():rawgroups[m,k,t].append(v);zgroups[m,k,t].append(z[m,k,t,d])
scores={}
metric_rows=read(P/'tables/metric_summary_5050.csv')
assert len(metric_rows)==28*7 and len({(r['method'],r['metric']) for r in metric_rows})==28*7
for r in metric_rows:
    m,k=r['method'],r['metric'];parts={}
    for typ in ['real','simulation']:
        values=[v for v in rawgroups[m,k,typ] if math.isfinite(v)];zs=[v for v in zgroups[m,k,typ] if math.isfinite(v)]
        parts[typ]={'raw_mean':mean(values),'raw_sd':sd(values),'raw_se':sd(values)/math.sqrt(len(values)) if len(values)>1 else NAN,'n_raw_valid':len(values),'score':mean(zs),'n_z_valid':len(zs)}
        for field,v in parts[typ].items():assert_number(r[typ+'_'+field],v,(m,k,typ,field))
    score=.5*parts['real']['score']+.5*parts['simulation']['score'] if k in shared else parts['simulation']['score']
    rm=.5*parts['real']['raw_mean']+.5*parts['simulation']['raw_mean'] if k in shared else parts['simulation']['raw_mean']
    se=math.sqrt(.25*parts['real']['raw_se']**2+.25*parts['simulation']['raw_se']**2) if k in shared else parts['simulation']['raw_se']
    scores[m,k]=score
    for field,v in [('score',score),('combined_raw_mean',rm),('combined_raw_se',se)]:assert_number(r[field],v,(m,k,field))
accuracy={m:mean(scores[m,k] for k in primary) for m in methods}
def average_rank(vs,v,ascending=False):
    if not math.isfinite(v):return NAN
    valid=[x for x in vs if math.isfinite(x)]
    return 1+sum((x<v if ascending else x>v) for x in valid)+(sum(x==v for x in valid)-1)/2
rank_rows=read(P/'tables/accuracy_scores_ranks.csv');assert len(rank_rows)==28 and {r['method'] for r in rank_rows}==set(methods)
for r in rank_rows:
    m=r['method'];ar=average_rank(accuracy.values(),accuracy[m])
    for field,v in [('Accuracy',accuracy[m]),('Accuracy_rank',ar),('Accuracy_reversed_rank',29-ar),('Accuracy_n_components',sum(math.isfinite(scores[m,k]) for k in primary))]:assert_number(r[field],v,(m,field))

stdz,stdstats=task_z(original['std']);fraction={}
raw_std_groups=defaultdict(list)
for (m,k,t,d),v in original['std'].items():raw_std_groups[m,k,t].append(v)
for r in read(P/'tables/batch_sd_numeric_accuracy_order.csv'):
    m,k=r['method'],r['metric'];a=mean(raw_std_groups[m,k,'real']);b=mean(raw_std_groups[m,k,'simulation'])
    for field,v in [('real_mean_sd',a),('simulation_mean_sd',b),('combined_mean_sd',b if k=='Ground-truth correlation' else .5*a+.5*b)]:assert_number(r[field],v,('raw SD',m,k,field))
for (m,k,t,d),v in original['std'].items():
    vals=[x for (_,kk,tt,dd),x in original['std'].items() if (kk,tt,dd)==(k,t,d) and math.isfinite(x)]
    n=len(vals);fraction[m,k,t,d]=(average_rank(vals,v,True)-1)/(n-1) if n>=2 and math.isfinite(v) else NAN
sd_task_rows=read(P/'tables/batch_SD_task_normalized.csv')
assert len(sd_task_rows)==len(stdz) and {(r['method'],r['metric'],r['data_type'],r['dataset_id']) for r in sd_task_rows}==set(stdz)
for r in sd_task_rows:
    k=(r['method'],r['metric'],r['data_type'],r['dataset_id']);n,mu,s=stdstats.get(k[1:],(0,NAN,NAN))
    for field,v in [('normalized_SD_z',stdz[k]),('SD_fractional_rank',fraction[k]),('n_valid_methods',n),('mean_SD_across_methods',mu),('sample_SD_of_SD_across_methods',s)]:assert_number(r[field],v,(k,field))
ds_groups=defaultdict(list);catalog=defaultdict(set)
for (m,k,t,d),v in stdz.items():ds_groups[m,t,d].append((k,v,fraction[m,k,t,d]));catalog[t,d].add(k)
ds={};ds_fraction={};complete={}
for key,vals in ds_groups.items():
    ds[key]=mean(v for _,v,_ in vals);ds_fraction[key]=mean(f for _,_,f in vals)
    complete[key]=sum(math.isfinite(v) for _,v,_ in vals)==len(catalog[key[1:]])
dataset_rows=read(P/'tables/batch_SD_one_value_per_dataset.csv')
assert len(dataset_rows)==len(ds) and {(r['method'],r['data_type'],r['dataset_id']) for r in dataset_rows}==set(ds)
for r in dataset_rows:
    key=(r['method'],r['data_type'],r['dataset_id']);vals=ds_groups[key]
    for field,v in [('normalized_variability',ds[key]),('fractional_rank_variability',ds_fraction[key]),('n_valid_components',sum(math.isfinite(v) for _,v,_ in vals)),('n_expected_components',len(catalog[key[1:]]))]:assert_number(r[field],v,(key,field))
def summary_check(filename,values,complete_only=False):
    calculated={}
    for m in methods:
        parts={}
        for typ in ['real','simulation']:
            vals=[v for (mm,t,d),v in values.items() if mm==m and t==typ and (not complete_only or complete[mm,t,d]) and math.isfinite(v)]
            parts[typ+'_score']=mean(vals);parts[typ+'_n_datasets']=len(vals)
        parts['combined_score']=.5*parts['real_score']+.5*parts['simulation_score'];calculated[m]=parts
    summary_rows=read(P/'tables'/filename);assert len(summary_rows)==28 and {r['method'] for r in summary_rows}==set(methods)
    for r in summary_rows:
        for field,v in calculated[r['method']].items():assert_number(r[field],v,(filename,r['method'],field))
        if 'combined_variability_rank' in r:assert_number(r['combined_variability_rank'],average_rank([s['combined_score'] for s in calculated.values()],calculated[r['method']]['combined_score'],True),(filename,'rank'))
summary_check('batch_SD_normalized_method_summary.csv',ds)
summary_check('batch_SD_fractional_rank_sensitivity.csv',ds_fraction)
summary_check('batch_SD_complete_dataset_sensitivity.csv',ds,True)
std_metric_groups=defaultdict(list)
for (m,k,t,d),v in stdz.items():std_metric_groups[m,k,t].append(v)
metric_scores={}
for r in read(P/'tables/batch_SD_normalized_metric_summary.csv'):
    m,k=r['method'],r['metric'];a=mean(std_metric_groups[m,k,'real']);b=mean(std_metric_groups[m,k,'simulation'])
    c=b if k=='Ground-truth correlation' else .5*a+.5*b;metric_scores[m,k]=c
    for field,v in [('real',a),('simulation',b),('combined_score',c)]:assert_number(r[field],v,('metric SD',m,k,field))
for r in read(P/'tables/batch_SD_metric_first_sensitivity.csv'):
    assert_number(r['metric_first_combined_score'],mean(v for (m,k),v in metric_scores.items() if m==r['method']),('metric first',r['method']))

# Structural invariants and synthetic edge cases independent of supplied data.
assert mean([0.,-2.,2.])==0 and sd([0.,0.])==0 and math.isnan(sd([1.]))
trial,_=task_z({('a','k','t','d'):0.,('b','k','t','d'):0.});assert all(math.isnan(v) for v in trial.values())
assert average_rank([1.,1.,2.],1.,True)==1.5
assert set(methods).isdisjoint({'Region Velocity','PhyloVelo'})
result={'status':'PASS','implementation':'independent CSV DictReader + stdlib math.fsum two-pass sample variance; production scoring functions not imported',
 'original_primary_cells':len(original['primary']),'original_SD_cells':len(original['std']),'numeric_values_checked':checks,
 'maximum_absolute_numeric_difference':max_diff,'acceptance_tolerance':{'rtol':1e-10,'atol':3e-12},
 'all_Accuracy_scores_ranks_raw_statistics_checked':True,'all_task_SD_z_dataset_composites_type_summaries_and_sensitivities_checked':True,
 'zero_negative_NA_and_exact_tie_edge_cases':True,'no_per_run_SD_recreation_claim':True}
(P/'validation/independent_raw_numeric_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
