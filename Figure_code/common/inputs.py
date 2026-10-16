"""Strict Source_data readers: explicit files, explicit aliases, source locators."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


MISSING = {'', 'na', 'nan', 'n/a', 'none', 'null', 'error'}


def key(value):
    text = str(value).replace('κ', 'k').replace('–', '-').replace('—', '-')
    return re.sub(r'[^a-z0-9]', '', text.strip().casefold())


def numeric(token, locator):
    if str(token).strip().casefold() in MISSING:
        return math.nan
    try:
        value = float(token)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Unknown numeric token {token!r} at {locator}') from exc
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite numeric value at {locator}')
    return value


class Inputs:
    def __init__(self, source_root: Path, config: dict):
        self.root, self.cfg = source_root.resolve(), config
        self.manifest, self.exclusions = [], []
        self.missing_inputs = []
        self._cache = {}
        m = self.read(config['methods_csv']).sort_values('method_order')
        if m.method.duplicated().any() or m.method_order.duplicated().any() or len(m) < 2:
            raise ValueError('Invalid methods.csv: unique names/orders and at least two methods required')
        self.methods = m.method.tolist()
        self.names = {key(x): x for x in self.methods}
        self.excluded = {key(x) for x in config.get('excluded_methods', [])}
        if set(self.names) & self.excluded or len(self.names) != len(self.methods):
            raise ValueError('Method roster conflicts with exclusions/aliases')
        for alias, canonical in config.get('method_aliases', {}).items():
            if canonical not in self.methods:
                raise ValueError(f'Alias points outside method roster: {alias} -> {canonical}')
            if key(alias) in self.names and self.names[key(alias)] != canonical:
                raise ValueError(f'Ambiguous method alias: {alias}')
            self.names[key(alias)] = canonical
        mapping_file = config.get('real_id_map_csv')
        self.real_map = {}
        if mapping_file:
            d = self.read(mapping_file, dtype=str)
            if d.dataset_number.duplicated().any() or d.dataset_id.duplicated().any():
                raise ValueError('Duplicate real dataset IDs in dictionary')
            self.real_map = dict(zip(d.dataset_number, d.dataset_id))

    def path(self, relative):
        p = (self.root / relative).resolve()
        if not p.is_relative_to(self.root) or not p.is_file():
            raise FileNotFoundError(f'Missing Source_data input: {relative}')
        return p

    def register(self, relative):
        p = self.path(relative)
        if relative not in self._cache:
            data = p.read_bytes()
            self._cache[relative] = data
            self.manifest.append({'path': relative, 'bytes': len(data),
                                  'sha256': hashlib.sha256(data).hexdigest()})
        return self._cache[relative]

    def read(self, relative, **kwargs):
        return pd.read_csv(io.BytesIO(self.register(relative)), encoding='utf-8-sig',
                           float_precision='round_trip', **kwargs)

    def canonical(self, name, path, row):
        if key(name) in self.excluded:
            self.exclusions.append({'method_raw': name, 'source_file': path,
                                    'row': row, 'reason': 'excluded_before_standardization'})
            return None
        if key(name) not in self.names:
            raise ValueError(f'Unknown method {name!r} in {path}:{row}')
        return self.names[key(name)]

    def matrix(self, entry):
        relative = entry['path']
        table = list(csv.reader(io.StringIO(self.register(relative).decode('utf-8-sig'))))
        if not table or len(table[0]) < 2:
            raise ValueError(f'Empty measurement matrix: {relative}')
        headers = [x.strip() for x in table[0][1:]]
        if entry['data_type'] == 'real' and entry.get('role') != 'downsampling':
            def map_real(h):
                if h in self.real_map:
                    return self.real_map[h]
                if h in self.real_map.values():
                    return h
                if self.real_map:
                    raise ValueError(f'Unknown real ID {h} in {relative}')
                return h
            datasets = [map_real(h) for h in headers]
        else:
            datasets = headers
        if not all(datasets) or len(set(datasets)) != len(datasets):
            raise ValueError(f'Empty/duplicate dataset headers in {relative}')
        seen = {}
        for ri, row in enumerate(table[1:], 2):
            if not row or not any(x.strip() for x in row):
                continue
            if len(row) != len(table[0]):
                raise ValueError(f'Ragged matrix at {relative}:{ri}')
            method = self.canonical(row[0], relative, ri)
            if method is None:
                continue
            if method in seen:
                raise ValueError(f'Duplicate canonical method in {relative}: {method}')
            seen[method] = (ri, row)
        rows = []
        # Primary tasks use the full roster, even when an entire method is absent.
        # SD uses only actually supplied methods; no deterministic/zero assumptions.
        methods = [m for m in self.methods if m in seen] if entry.get('role') == 'std' else self.methods
        for method in methods:
            ri, row = seen.get(method, (None, None))
            for ci, (header, dataset) in enumerate(zip(headers, datasets), 2):
                token = row[ci - 1] if row else ''
                loc = f'{relative}:row={ri}:column={ci}:header={header}'
                value = numeric(token, loc)
                if entry.get('role') == 'std' and np.isfinite(value) and value < 0:
                    raise ValueError(f'Negative SD: {loc}')
                rows.append({'method': method, 'metric': entry['metric'],
                             'data_type': entry['data_type'], 'dataset_id': dataset,
                             'dataset_raw': header, 'raw_token': token, 'value': value,
                             'method_present': row is not None, 'source_file': relative,
                             'source_locator': loc, 'role': entry.get('role', 'summary')})
        return pd.DataFrame(rows)

    def matrices(self, entries):
        parts = []
        for entry in entries:
            if entry.get('optional') and not (self.root / entry['path']).is_file():
                self.missing_inputs.append({'path': entry['path'], 'metric': entry['metric'],
                    'data_type': entry['data_type'], 'role': entry.get('role', 'summary'),
                    'status': 'not_supplied'})
                continue
            parts.append(self.matrix(entry))
        if not parts:
            return pd.DataFrame(columns=['method', 'metric', 'data_type', 'dataset_id', 'value'])
        out = pd.concat(parts, ignore_index=True)
        if out.duplicated(['method', 'metric', 'data_type', 'dataset_id']).any():
            raise ValueError('Duplicate tasks across declared input matrices')
        return out

    def long_resource(self, relative):
        out = self.read(relative)
        out['method'] = [self.canonical(m, relative, i + 2) for i, m in enumerate(out.method)]
        out = out[out.method.notna()].copy()
        # raw_token may contain recorded Docker errors; clean_value was explicitly
        # extracted from S4/S5/S6. Preserve both, never parse error text as zero.
        out['value'] = pd.to_numeric(out.clean_value, errors='raise')
        if (out.value.dropna() < 0).any() or np.isinf(out.value.dropna()).any():
            raise ValueError(f'Invalid resource costs: {relative}')
        if out.duplicated(['method', 'protocol', 'measure', 'dataset_id']).any():
            raise ValueError(f'Duplicate resource observations: {relative}')
        out['source_file'] = relative
        return out

    def save_audit(self, out):
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.manifest).sort_values('path').to_csv(out / 'input_manifest.csv', index=False)
        pd.DataFrame(self.exclusions, columns=['method_raw', 'source_file', 'row', 'reason']).to_csv(
            out / 'excluded_methods.csv', index=False)
        pd.DataFrame(self.missing_inputs, columns=['path', 'metric', 'data_type', 'role', 'status']).drop_duplicates().to_csv(
            out / 'missing_inputs.csv', index=False)
        (out / 'effective_config.json').write_text(json.dumps(self.cfg, ensure_ascii=False, indent=2), encoding='utf-8')
