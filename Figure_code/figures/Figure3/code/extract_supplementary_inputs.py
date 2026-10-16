#!/usr/bin/env python3
"""Extract the scRNA S4-S8 inputs as standalone, English-only CSV files.

The workbook is optional after extraction and is not distributed with the code.
Usage: python Figure_code/figures/Figure3/code/extract_supplementary_inputs.py workbook.xlsx
The three resource protocols are top_scRNA, velocity_genes_2000 and
velocity_genes_5000. Only top_scRNA enters the Scalability score.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

CATEGORIES = {'Installability': 6, 'Code quality': 5, 'Usage guidance': 5,
              'Maintenance': 5, 'Publication quality': 3}
RESOURCE_COLUMNS = ['protocol','measure','dataset_id','unit','data_type','dataset_raw',
                    'cells','genes','scale_layer','structure','source_family','source_role',
                    'source_path','method','method_raw','raw_token','clean_value',
                    'value_status','missing_reason','is_zero','is_negative','source_locator',
                    'method_present','dataset_present']
HAN = re.compile('[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]')


def canonical(value):
    return re.sub('[^a-z0-9]+', '', str(value).strip().lower())


def save(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if HAN.search(' '.join(str(v) for v in row.values())):
            raise ValueError(f'Non-English Chinese text found in extracted input: {path.name}')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader();writer.writerows(rows)


def parse_cost(value, present):
    if not present:
        return '', 'NA', 'method_not_provided', 'method row absent from source'
    token = '' if value is None else str(value)
    if token.strip().casefold() in {'','na','nan','n/a','none','null','error'}:
        return token, 'NA', 'missing_marker', 'empty cell' if not token else 'missing marker'
    try:
        number = float(token)
    except ValueError:
        if re.match(r'^\s*[+-]?(?:\d|\.\d)', token):
            raise ValueError(f'Unrecognized possibly numeric resource token: {token!r}')
        return token, 'NA', 'non_numeric_source_text', 'resource value is not numeric; original source text retained'
    if not math.isfinite(number) or number < 0:
        raise ValueError(f'Invalid resource cost: {token!r}')
    return token, number, 'valid_zero' if number == 0 else 'valid_positive', ''


def resource_rows(workbook, sheet_name, measure, unit, methods, source_name):
    rows = list(workbook[sheet_name].values)
    lookup = {canonical(m):m for m in methods}
    blocks = []
    for i, row in enumerate(rows):
        if canonical(row[0]) != 'method':continue
        columns = [(j,str(v).strip()) for j,v in enumerate(row[1:],1)
                   if v is not None and re.fullmatch(r'.+_cell\d+_gene\d+',str(v).strip())]
        if not columns:continue
        if i == 1:protocol = 'top_scRNA'
        else:
            heading = ' '.join(str(r[0] or '') for r in rows[max(0,i-4):i])
            match = re.search(r'(2000|5000)\s+velocity\s+genes',heading,re.I)
            if not match:raise ValueError(f'Unidentified scRNA resource block: {sheet_name}:{i+1}')
            protocol = 'velocity_genes_' + match[1]
        end = next((j for j in range(i+1,len(rows)) if canonical(rows[j][0])=='method'),len(rows))
        seen = {}
        for j in range(i+1,end):
            name = lookup.get(canonical(rows[j][0]))
            if name:
                if name in seen:raise ValueError(f'Duplicate method in {sheet_name}/{protocol}: {name}')
                seen[name] = j,rows[j]
        block = []
        for column,dataset in columns:
            m = re.fullmatch(r'(.+)_cell(\d+)_gene(\d+)',dataset)
            structure,cells,genes = m[1],int(m[2]),int(m[3])
            for method in methods:
                present = method in seen
                row_index,row = seen[method] if present else (None,None)
                raw_value = row[column] if present and column<len(row) else None
                token,value,status,reason = parse_cost(raw_value,present)
                locator = f'{sheet_name}!{get_column_letter(column+1)}{row_index+1}' if present else ''
                block.append(dict(zip(RESOURCE_COLUMNS,[
                    protocol,measure,dataset,unit,'simulation',dataset,cells,genes,
                    f'{cells}x{genes}',structure,'excel','resource_primary',
                    f'{source_name}::{sheet_name}',method,str(row[0]) if present else method,
                    token,value,status,reason,value==0,False,locator,present,True])))
        expected = 68 if protocol == 'top_scRNA' else 7
        if len(columns)!=expected:raise ValueError(f'Unexpected task count: {sheet_name}/{protocol}')
        blocks.append((protocol,block))
    if {p for p,_ in blocks}!={'top_scRNA','velocity_genes_2000','velocity_genes_5000'}:
        raise ValueError(f'Missing resource protocol in {sheet_name}')
    return [row for _,block in blocks for row in block]


def extract(workbook_path, repo_root):
    code = repo_root/'Figure_code/figures/Figure3'
    with (code/'config/method_colors.csv').open(encoding='utf-8-sig') as handle:
        palette = sorted(csv.DictReader(handle),key=lambda row:int(row['method_order']))
    methods = [row['method'] for row in palette]
    assert len(methods)==len(set(methods))==28
    lookup = {canonical(m):m for m in methods}
    wb = load_workbook(workbook_path,read_only=True,data_only=True)
    paths = []
    for sheet,measure,unit,name in [
        ('Table S4','runtime_seconds','seconds','scRNA_runtime.csv'),
        ('Table S5','ram_gb','GB','scRNA_cpu_ram.csv'),
        ('Table S6','vram_gb','GB','scRNA_gpu_vram.csv')]:
        path = repo_root/'Source_data/scalability'/name
        rows = resource_rows(wb,sheet,measure,unit,methods,workbook_path.name)
        save(path,rows,RESOURCE_COLUMNS);paths.append(path)
    rules = []
    for index,row in enumerate(wb['Table S7'].values,1):
        if row[0] not in CATEGORIES or not isinstance(row[2],str):continue
        rules.append({'category':row[0],'category_raw':row[0],'category_weight':row[1],
                      'subcategory':row[2],'subcategory_weight':row[3],
                      'source_locator':f'Table S7!A{index}:J{index}'})
    assert len(rules)==24 and len({(r['category'],r['subcategory']) for r in rules})==24
    for category,count in CATEGORIES.items():
        subset = [r for r in rules if r['category']==category]
        assert len(subset)==count and abs(sum(r['subcategory_weight'] for r in subset)-1)<1e-12
    path = repo_root/'Source_data/usability/usability_rules.csv'
    save(path,rules,list(rules[0]));paths.append(path)
    items = []
    for index,row in enumerate(wb['Table S8'].values,1):
        method = lookup.get(canonical(row[0]))
        if method is None:continue
        category = {'Install-friendly':'Installability','Paper':'Publication quality'}.get(row[1],row[1])
        if category not in CATEGORIES:raise ValueError(f'Unexpected usability category at row {index}')
        assert isinstance(row[5],(int,float)) and 0<=row[5]<=10
        items.append({'method':method,'method_raw':row[0],'category':category,'category_raw':row[1],
                      'category_weight':row[2],'reported_category_score':row[3],
                      'subcategory':row[4],'subscore':row[5],'subcategory_weight':row[6],
                      'contribution':row[5]*row[6],'evidence_note':row[7] or '',
                      'source_url':row[8] or '', 'source_locator':f'Table S8!A{index}:I{index}'})
    assert len(items)==28*24
    assert len({(r['method'],r['category'],r['subcategory']) for r in items})==28*24
    path = repo_root/'Source_data/usability/scRNA_usability_items.csv'
    save(path,items,list(items[0]));paths.append(path)
    wb.close()
    paths.append(code/'config/method_colors.csv')
    manifest = {'source_workbook':workbook_path.name,
                'source_workbook_sha256':hashlib.sha256(workbook_path.read_bytes()).hexdigest(),
                'source_package':'scRNA_1012_fig3.zip',
                'resource_protocols_scored':['top_scRNA'],
                'VRAM_enters_Scalability':False,
                'category_weights':dict(zip(CATEGORIES,[.25,.25,.20,.15,.15])),
                'inputs':[{'path':p.relative_to(repo_root).as_posix(),'bytes':p.stat().st_size,
                           'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    (code/'config/input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workbook',type=Path)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    manifest = extract(args.workbook.expanduser().resolve(),args.repo_root.expanduser().resolve())
    print(f'Extracted five standalone CSV inputs from {manifest["source_workbook"]}.')
