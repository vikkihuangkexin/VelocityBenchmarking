#!/usr/bin/env python3
"""Recompute Figure 2 using the repository's external measurement directory.

Run from the VelocityBenchmarking repository root:
    python -m pip install -r Figure_code/Fig2/requirements.txt
    Rscript --vanilla Figure_code/Fig2/code/install_R_packages.R
    python Figure_code/Fig2/run.py

Default inputs:
    Source_data/measurements_raw/scRNA/methods.csv
    Source_data/measurements_raw/scRNA/real_dataset_ids.csv
    Source_data/measurements_raw/scRNA/measurement/summary/*.csv
    Source_data/measurements_raw/scRNA/measurement/std/*.csv
    Source_data/measurements_raw/scRNA/measurement/downsampling/*.csv

The default output is Figure_code/Fig2/results. It must be new or empty.
Use --data-root to select another scRNA directory and --output for another
output location. The entry point stages a read-only copy in the original
pipeline layout; scoring and R plotting scripts do not need path changes.

Analysis without R:
    python Figure_code/Fig2/run.py --mode analysis
Plot an existing analysis, after editing verified English annotations:
    python Figure_code/Fig2/run.py --mode plots
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time


def resolve_data_root(script_root, supplied):
    if supplied is not None:
        return supplied.expanduser().resolve()
    for parent in script_root.parents:
        candidate = parent / 'Source_data/measurements_raw/scRNA'
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError('Could not find Source_data/measurements_raw/scRNA. '
                            'Place this code in Figure_code/Fig2 or supply --data-root PATH.')


def verify_inputs(root, data_root=None):
    """Check external source files or a canonical, staged pipeline output."""
    manifest = json.loads((root / 'config/input_manifest.json').read_text())
    seen = []
    for entry in manifest['inputs'] + manifest['locked_files']:
        relative = Path(entry['path'])
        if data_root is not None and relative.parts[:2] == ('data', 'raw'):
            path = data_root / 'measurement' / Path(*relative.parts[2:])
        elif data_root is not None and relative.parts[0] == 'data':
            path = data_root / Path(*relative.parts[1:])
        else:
            path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}. Keep summary, std and downsampling '
                                    'as subdirectories of measurement.')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'Input hash mismatch: {path}')
        if 'bytes' in entry and len(data) != entry['bytes']:
            raise ValueError(f'Input size mismatch: {path}')
        seen.append(entry['path'])
    raw = data_root / 'measurement' if data_root is not None else root / 'data/raw'
    actual = {'data/raw/' + p.relative_to(raw).as_posix() for p in raw.rglob('*.csv')}
    expected = {entry['path'] for entry in manifest['inputs']}
    if actual != expected or len(seen) != len(set(seen)):
        raise ValueError('Measurement file set does not match config/input_manifest.json. '
                         f'Missing: {sorted(expected-actual)}; unexpected: {sorted(actual-expected)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parent
    parser.add_argument('--data-root', type=Path, help='Directory containing methods.csv, real_dataset_ids.csv and measurement/')
    parser.add_argument('--output', type=Path, default=root / 'results')
    parser.add_argument('--mode', choices=['full', 'analysis', 'plots'], default='full')
    parser.add_argument('--rscript', default='Rscript')
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    if not __debug__:
        raise SystemExit('Run without python -O; numerical assertions are required.')
    if out == root or out in root.parents or any(out == root / p or root / p in out.parents for p in ['code', 'config']):
        raise SystemExit('The output directory must not overwrite the source or its configuration.')
    if args.mode != 'plots':
        data_root = resolve_data_root(root, args.data_root)
        if out == data_root or data_root in out.parents or out in data_root.parents:
            raise SystemExit('The output directory must not overwrite the measurement source.')
        verify_inputs(root, data_root)
        if out.exists() and any(out.iterdir()):
            raise SystemExit('Analysis/full mode requires a new or empty output directory.')
        if args.mode == 'full' and shutil.which(args.rscript) is None:
            raise SystemExit('Rscript was not found. Install R/dependencies or supply --rscript.')
        out.mkdir(parents=True, exist_ok=True)
        for folder in ['code', 'config']:
            shutil.copytree(root / folder, out / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (out / 'data').mkdir()
        for name in ['methods.csv', 'real_dataset_ids.csv']:
            shutil.copy2(data_root / name, out / 'data' / name)
        shutil.copytree(data_root / 'measurement', out / 'data/raw')
        verify_inputs(out)
    else:
        if not (out / 'tables/accuracy_scores_ranks.csv').is_file():
            raise SystemExit('Plots mode requires a completed analysis output directory.')
        verify_inputs(out)
        shutil.copytree(root / 'code', out / 'code', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(root / 'config/method_annotations.csv', out / 'config/method_annotations.csv')
    for folder in ['tables', 'figures', 'validation', 'logs']:
        (out / folder).mkdir(exist_ok=True)
    commands = []
    if args.mode != 'plots':
        commands += [[sys.executable, str(out / 'code' / name)] for name in [
            'parse_original_inputs.py', 'prepare_accuracy.py', 'prepare_normalized_sd.py', 'independent_verify.py']]
    if args.mode != 'analysis':
        if shutil.which(args.rscript) is None:
            raise SystemExit('Rscript was not found. Install R/dependencies or supply --rscript.')
        commands += [[args.rscript, '--vanilla', str(out / 'code/plot_abc.R'), str(out)],
                     [sys.executable, str(out / 'code/validate_figures.py')]]
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '0'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    records = []
    for index, command in enumerate(commands, 1):
        log = out / 'logs' / f'{args.mode}_{index:02d}_{Path(command[0]).name}.log'
        start = time.monotonic()
        with log.open('w') as handle:
            result = subprocess.run(command, cwd=out, env=env, stdout=handle, stderr=subprocess.STDOUT)
        records.append({'command': [Path(x).name if str(out) in x else x for x in command],
                        'exit_code': result.returncode, 'seconds': round(time.monotonic() - start, 3),
                        'log': log.relative_to(out).as_posix()})
        (out / 'logs' / f'execution_{args.mode}.json').write_text(json.dumps(records, indent=2))
        if result.returncode:
            raise SystemExit(f'Execution failed; see {log}\n{log.read_text()[-4000:]}')
        print(f'Completed {index}/{len(commands)}: {Path(command[-1]).name}', flush=True)
    print(f'Completed {args.mode}: {out}')


if __name__ == '__main__':
    main()
