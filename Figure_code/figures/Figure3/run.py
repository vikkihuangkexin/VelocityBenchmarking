#!/usr/bin/env python3
"""Run independent Fig3 Scalability/Usability analysis and metric-rank exports.

From the VelocityBenchmarking repository root:
    python -m pip install -r Figure_code/Fig3/requirements.txt
    Rscript --vanilla Figure_code/Fig3/code/install_R_packages.R
    python Figure_code/Fig3/run.py

Inputs are standalone CSVs in Source_data/scalability and Source_data/usability.
No supplementary workbook is needed at runtime. The original R calculations,
heatmaps, orange usability bubbles and independent Sc/U orders are preserved.
Fig2 results, when available, are read for the complete per-metric rank export.
Ranks go to rank/scRNA; figures/tables go to Figure_code/Fig3/results.
Run --mode ranks to refresh rankings from existing computed result files.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time


def find_repo(root):
    for parent in root.parents:
        if (parent/'Source_data/scalability').is_dir() and (parent/'Source_data/usability').is_dir():return parent
    raise FileNotFoundError('Place this code in Figure_code/Fig3 with Source_data/scalability and Source_data/usability.')


def check_inputs(repo, code):
    manifest=json.loads((code/'config/input_manifest.json').read_text())
    for entry in manifest['inputs']:
        path=repo/entry['path'];data=path.read_bytes()
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
            raise ValueError(f'Input hash/size mismatch: {path}')
    return manifest


def read_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))


def write_rows(path, rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def stage_inputs(repo, code, stage):
    (stage/'data').mkdir();(stage/'config').mkdir()
    resources=[]
    for name in ['scRNA_runtime.csv','scRNA_cpu_ram.csv','scRNA_gpu_vram.csv']:
        resources.extend(read_rows(repo/'Source_data/scalability'/name))
    write_rows(stage/'data/resource_clean_long.csv',resources)
    items=read_rows(repo/'Source_data/usability/scRNA_usability_items.csv')
    for row in items:row['old_category_score']=row.pop('reported_category_score')
    write_rows(stage/'data/usability_item_contributions.csv',items)
    shutil.copy2(repo/'Source_data/usability/usability_rules.csv',stage/'data/usability_rules_S7.csv')
    shutil.copy2(code/'config/method_colors.csv',stage/'data/method_colors.csv')
    entries=[]
    for path in sorted((stage/'data').glob('*.csv')):
        data=path.read_bytes();entries.append({'relative_path':path.relative_to(stage).as_posix(),
            'source_filename':path.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
            'role':'standalone scoring input','scoring_or_rule_input':'true'})
    write_rows(stage/'config/input_sha256.csv',entries)


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-root',type=Path)
    parser.add_argument('--output',type=Path,default=root/'results')
    parser.add_argument('--fig2-results',type=Path)
    parser.add_argument('--rank-dir',type=Path)
    parser.add_argument('--rscript',default='Rscript')
    parser.add_argument('--mode',choices=['full','ranks'],default='full')
    args=parser.parse_args()
    if not __debug__:raise SystemExit('Run without python -O; verification assertions are required.')
    repo=args.repo_root.expanduser().resolve() if args.repo_root else find_repo(root)
    out=args.output.expanduser().resolve()
    fig2=(args.fig2_results or repo/'Figure_code/Fig2/results').expanduser().resolve()
    ranks=(args.rank_dir or repo/'rank/scRNA').expanduser().resolve()
    protected=[root,root/'code',root/'config',repo/'Source_data',repo/'rank']
    if any(out==p or out in p.parents or (p!=root and p in out.parents) for p in protected):
        raise SystemExit('The output must not overwrite repository code, source data or ranks.')
    if not (out/'tables/scalability_independent_rank.csv').exists() and args.mode=='ranks':
        raise SystemExit('Ranks mode requires completed Fig3 results.')
    rank_command=[sys.executable,str(root/'code/export_ranks.py'),'--repo-root',str(repo),
                  '--fig3-results',str(out),'--fig2-results',str(fig2),'--rank-dir',str(ranks)]
    env=os.environ.copy();env['PYTHONHASHSEED']='0';env['PYTHONDONTWRITEBYTECODE']='1'
    if args.mode=='ranks':
        subprocess.run(rank_command,check=True,env=env);return
    check_inputs(repo,root)
    if shutil.which(args.rscript) is None:raise SystemExit('Install R/dependencies or provide --rscript PATH.')
    if out.exists() and any(out.iterdir()):raise SystemExit('Full mode requires a new or empty output directory.')
    out.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    with tempfile.TemporaryDirectory(prefix='velocity-fig3-') as temporary:
        stage=Path(temporary);stage_inputs(repo,root,stage)
        commands=[[args.rscript,'--vanilla',str(root/'code/install_R_packages.R'),'--check'],
                  [args.rscript,'--vanilla',str(root/'code/plot_fig3.R'),str(stage),str(out)],
                  [sys.executable,str(root/'code/validate_independent.py'),'--project',str(stage),'--result',str(out)],
                  rank_command]
        for index,command in enumerate(commands,1):
            start=time.monotonic()
            # Keep the R output empty until its own guarded entry point starts.
            log=stage/f'fig3-step-{index}.log'
            with log.open('w') as handle:
                result=subprocess.run(command,cwd=repo,env=env,stdout=handle,stderr=subprocess.STDOUT)
            text=log.read_text()
            if result.returncode:
                failure=out.parent/f'{out.name}.failed_step_{index}.log'
                failure.write_text(text)
                raise SystemExit(f'Failed step {index}: {text[-7000:]}\nLog: {failure}')
            records.append({'step':index,'exit_code':0,'seconds':round(time.monotonic()-start,3)})
            if out.exists():
                (out/'logs').mkdir(exist_ok=True)
                shutil.move(log,out/'logs'/f'{index:02d}_execution.log')
                (out/'logs/execution.json').write_text(json.dumps(records,indent=2))
            print(f'Completed {index}/{len(commands)}',flush=True)
        shutil.copy2(root/'config/input_manifest.json',out/'tables/standalone_input_manifest.json')
        # Move the dependency-check log, created before the R output existed.
        initial=stage/'fig3-step-1.log'
        if initial.exists():shutil.move(initial,out/'logs/01_execution.log')
    print(f'Figures and tables: {out}\nMetric ranks: {ranks}')


if __name__=='__main__':main()
