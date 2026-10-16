#!/usr/bin/env python3
"""Plot current benchmark results; all numerical rankings come from the shared engine."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import io
import json
from pathlib import Path
import re
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import gaussian_kde

CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parent
DIMS = ['accuracy', 'consistency', 'scalability', 'stability', 'usability']
DISPLAY = {'accuracy': 'Directional Accuracy', 'consistency': 'Consistency',
           'scalability': 'Scalability', 'stability': 'Stability', 'usability': 'Usability', 'overall': 'Overall'}
CATEGORY_SHORT = ['Installability', 'Code quality', 'Usage guidance', 'Maintenance', 'Publication quality']


class Plotter:
    def __init__(self, profile, results, source, output):
        self.profile, self.results, self.source, self.output = profile, results, source, output
        self.cfg = json.loads((CODE / 'config' / (profile + '.json')).read_text())
        previous = json.loads((results / 'audit/effective_config.json').read_text())
        if previous != self.cfg:
            raise ValueError('Calculation configuration changed; recompute into a new result directory')
        subprocess.run([sys.executable, str(CODE / 'verify.py'), '--results', str(results),
                        '--source-root', str(source)], check=True, stdout=subprocess.DEVNULL)
        if output.exists() and any(output.iterdir()):
            raise ValueError('Use a new/empty figure output directory')
        output.mkdir(parents=True, exist_ok=True)
        self.methods = pd.read_csv(source / self.cfg['methods_csv']).method.tolist()
        palette = pd.read_csv(CODE / 'figures' / profile / 'method_colors.csv')
        self.colors = dict(zip(palette.method, palette['hex'] if 'hex' in palette else palette.color))
        self.tables = {d: pd.read_csv(results / d / 'scores_ranks.csv') for d in DIMS + ['overall']}
        self.order = self.tables['overall'].sort_values('display_order').method.tolist()
        self.raw = pd.read_csv(results / 'audit/measurement_task_z.csv')
        self.down = pd.read_csv(results / 'stability/downsampling_task_z.csv')
        self.manifest, self.plot_inputs = [], []
        for path in source.rglob('*.csv'):
            if profile in path.parts or path.name.startswith(profile + '_'):
                self.plot_inputs.append({'path': path.relative_to(source).as_posix(),
                                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        if profile == 'ATAC':
            rnacfg = json.loads((CODE / 'config/scRNA.json').read_text())
            rna_paths = [e['path'] for e in rnacfg['measurement_inputs'] if e['data_type'] == 'real']
            rna_paths += [rnacfg['real_id_map_csv']]
            for relative in rna_paths:
                self.plot_inputs.append({'path': relative,
                    'sha256': hashlib.sha256((source / relative).read_bytes()).hexdigest()})
        for candidate in ['/usr/share/fonts/truetype/msttcorefonts/Arial.ttf',
                          '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
                          '/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf']:
            if Path(candidate).is_file():
                font_manager.fontManager.addfont(candidate)
        available = {f.name for f in font_manager.fontManager.ttflist}
        self.font = next(f for f in ['Arial', 'Liberation Sans', 'Nimbus Sans', 'DejaVu Sans'] if f in available)
        plt.rcParams.update({'font.family': self.font, 'font.size': 11,
            'axes.titlesize': 14, 'axes.titleweight': 'bold', 'axes.labelsize': 11,
            'xtick.labelsize': 9, 'ytick.labelsize': 10, 'axes.spines.top': False,
            'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
            'savefig.facecolor': 'white'})

    def csv(self, name, data):
        data.to_csv(self.output / (name + '.csv'), index=False, float_format='%.17g')

    def save(self, name, fig, data=None, square=True):
        fig.canvas.draw()
        for fmt in ['pdf', 'svg', 'png']:
            path = self.output / (name + '.' + fmt)
            buffer = io.BytesIO()
            fig.savefig(buffer, format=fmt, dpi=300)
            path.write_bytes(buffer.getvalue())
            self.manifest.append({'figure': name, 'format': fmt, 'bytes': path.stat().st_size,
                'width_inches': fig.get_figwidth(), 'height_inches': fig.get_figheight(),
                'square_expected': square, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        if data is not None:
            self.csv(name, data)
        plt.close(fig)

    def rank_axis(self, ax, dim):
        table = self.tables[dim].sort_values('display_order')
        y = np.arange(len(table))
        for pos, row in zip(y, table.itertuples()):
            if np.isfinite(row.score):
                ax.barh(pos, row.score, color=self.colors[row.method], edgecolor='white', height=.77)
            else:
                ax.text(0, pos, 'Failed' if getattr(row, 'entire_dimension_failure', False) else 'NA', va='center', color='#777777')
        ax.set_yticks(y, table.method)
        ax.invert_yaxis()
        ax.axvline(0, color='#777777', lw=.6)
        ax.set_title(DISPLAY[dim])
        ax.set_xlabel('Weighted reversed rank' if dim == 'overall' else 'Score (higher is better)')
        ax.grid(axis='x', color='#E7E7E7', lw=.6)
        ax.set_axisbelow(True)

    def ranks(self):
        for dim in ['overall'] + DIMS:
            fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
            self.rank_axis(ax, dim)
            self.save(dim + '_ranking', fig, self.tables[dim])
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout='constrained')
        for ax, dim in zip(axes.flat, ['overall'] + DIMS):
            self.rank_axis(ax, dim)
        self.save('five_dimension_overview', fig, square=False)
        sensitivity = pd.read_csv(self.results / 'sensitivity/weight_sensitivity_long.csv')
        fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
        for i, method in enumerate(self.order):
            values = sensitivity.loc[sensitivity.method.eq(method), 'rank'].dropna()
            if len(values):
                box = ax.boxplot([values], positions=[i], widths=.6, patch_artist=True, showfliers=False)
                box['boxes'][0].set(facecolor=self.colors[method], alpha=.6)
        ax.set_xticks(range(len(self.order)), self.order, rotation=55, ha='right')
        ax.set_ylabel('Overall rank across declared weight settings')
        ax.invert_yaxis()
        ax.grid(axis='y', color='#EEEEEE')
        self.save('overall_weight_sensitivity', fig, sensitivity)

    def boxes(self, name, frame, methods=None, colors=None, title='', separator=None, ylabel='Raw metric value'):
        methods = methods or self.order
        colors = colors or self.colors
        fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
        xcol = 'value'
        if 'point_key' not in frame:
            frame = frame.assign(point_key=frame.dataset_id.astype(str))
        keys = sorted(frame.point_key.dropna().unique())
        offsets = dict(zip(keys, np.linspace(-.19, .19, len(keys))))
        for i, method in enumerate(methods):
            f = frame[frame.method.eq(method) & frame[xcol].notna()]
            if len(f) >= 3:
                box = ax.boxplot([f[xcol]], positions=[i], widths=.58, patch_artist=True,
                    showfliers=False, medianprops={'color': '#111111', 'lw': 1.2})
                box['boxes'][0].set(facecolor=colors[method], alpha=.4)
            elif len(f):
                ax.plot([i - .2, i + .2], [f[xcol].median()] * 2, color='#111111', lw=1.3)
            if len(f):
                for k, g in f.groupby('point_key'):
                    marker = '^' if separator is not None and i >= separator else 'o'
                    ax.scatter(np.full(len(g), i + offsets[k]), g[xcol], c=colors[method],
                               marker=marker, s=31, edgecolors='#222222', lw=.4, zorder=3, alpha=.9)
            else:
                ax.text(i, .02, 'NA', transform=ax.get_xaxis_transform(), ha='center', color='#888888')
        ax.set_xticks(range(len(methods)), [f'{m}\n(n={int((frame.method.eq(m)&frame[xcol].notna()).sum())})' for m in methods],
                      rotation=55, ha='right')
        if separator is not None:
            ax.axvline(separator - .5, color='#888888', ls=':', lw=1)
        ax.set_xlim(-.6, len(methods) - .4)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis='y', color='#E7E7E7', lw=.6)
        ax.set_axisbelow(True)
        self.save(name, fig, frame)

    def metrics(self):
        for metric, frame in self.raw.groupby('metric', sort=False):
            self.boxes('metric_' + slug(metric), frame, title=metric)
        if self.profile == 'Spatial':
            fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True, layout='constrained')
            data = []
            for ax, dim in zip(axes, ['accuracy', 'consistency']):
                f = self.raw[self.raw.metric.isin(self.cfg[dim + '_metrics'])].copy()
                for i, method in enumerate(self.order):
                    vals = f.loc[f.method.eq(method), 'z'].dropna().to_numpy()
                    if len(vals) > 1 and np.std(vals) > 0:
                        grid = np.linspace(vals.min() - .2, vals.max() + .2, 240)
                        density = gaussian_kde(vals)(grid)
                        ax.fill_between(grid, i, i + .8 * density / density.max(), color=self.colors[method], alpha=.6)
                    if len(vals):
                        ax.scatter(np.mean(vals), i + .1, marker='D', c='black', s=25, zorder=4)
                ax.set_yticks(np.arange(len(self.order)), self.order)
                ax.set_title(DISPLAY[dim])
                ax.set_xlabel('Task z-score; diamond = task mean')
                ax.axvline(0, color='#888888', ls=':', lw=.7)
                f['dimension'] = dim
                data.append(f)
            axes[0].invert_yaxis()
            self.save('directionality_consistency_ridges', fig, pd.concat(data), square=False)
        d = self.tables['accuracy'].set_index('method')
        c = self.tables['consistency'].set_index('method')
        u = self.tables['usability'].set_index('method')
        fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
        coords = []
        for i, method in enumerate(self.methods):
            x, y = d.loc[method, 'score'], c.loc[method, 'score']
            size = 90 + 20 * u.loc[method, 'score']
            ax.scatter(x, y, s=size, color=self.colors[method], edgecolor='white', lw=.7)
            coords.append({'method': method, 'accuracy': x, 'consistency': y, 'usability': u.loc[method, 'score']})
        coord = pd.DataFrame(coords)
        lo, hi = np.nanmin(coord[['accuracy', 'consistency']]), np.nanmax(coord[['accuracy', 'consistency']])
        pad = max((hi - lo) * .23, .1)
        ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad), xlabel='Directional Accuracy', ylabel='Consistency')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(color='#EEEEEE')
        place_labels(fig, ax, coord, self.colors)
        self.save('accuracy_consistency_bubble', fig, coord)

    def resources(self):
        raw = pd.read_csv(self.results / 'scalability/task_z.csv')
        cmap = LinearSegmentedColormap.from_list('resource', ['#F4F6F8', '#D6E2EC', '#A2BFD3', '#608DAA', '#284F6B'])
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), layout='constrained')
        exports = []
        for ax, measure, title, divisor in zip(axes, ['runtime_seconds', 'ram_gb'], ['Runtime (min)', 'CPU RAM (GB)'], [60, 1]):
            f = raw[raw.measure.eq(measure)].copy()
            f['display_value'] = f.value / divisor
            if self.profile == 'ATAC':
                ids = sorted(f.dataset_id.unique(), key=lambda d: int(d.split('_', 1)[0]))
                labels = [d.split('_')[0] for d in ids]
            else:
                dims = f[['dataset_id', 'cells', 'genes']].drop_duplicates().sort_values(['genes', 'cells'])
                ids = dims.dataset_id.tolist()
                labels = [f'{int(r.cells):,} x {int(r.genes):,}' for r in dims.itertuples()]
            matrix = f.pivot(index='method', columns='dataset_id', values='display_value').reindex(index=self.order, columns=ids)
            v = matrix.to_numpy()
            positive = v[np.isfinite(v) & (v > 0)]
            use_log = len(positive) and positive.min() < positive.max() and np.all(v[np.isfinite(v)] > 0)
            norm = LogNorm(positive.min(), positive.max()) if use_log else Normalize()
            image = ax.imshow(np.ma.masked_invalid(v), cmap=cmap, norm=norm, aspect='auto')
            ax.set_xticks(range(len(ids)), labels, rotation=55, ha='right')
            ax.set_yticks(range(len(self.order)), self.order)
            for i, j in itertools.product(range(len(self.order)), range(len(ids))):
                if np.isfinite(v[i, j]):
                    ax.text(j, i, f'{v[i,j]:.1f}', ha='center', va='center', fontsize=7,
                            color='white' if norm(v[i,j]) > .6 else '#222222')
                else:
                    ax.text(j, i, 'NA', ha='center', va='center', color='#888888', fontsize=7)
            ax.set_title(title)
            fig.colorbar(image, ax=ax, fraction=.025, pad=.015)
            exports.append(f)
        self.save('resource_heatmaps', fig, pd.concat(exports))
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout='constrained')
        records = []
        for row, measure in enumerate(['runtime_seconds', 'ram_gb']):
            for col, (axis, fixed, target) in enumerate([('cells', 'genes', 1000 if self.profile == 'ATAC' else 9836),
                                                        ('genes', 'cells', 1000 if self.profile == 'ATAC' else 1967)]):
                ax = axes[row, col]
                f = raw[raw.measure.eq(measure) & raw[fixed].eq(target)].copy()
                f['display_value'] = f.value / (60 if measure == 'runtime_seconds' else 1)
                group = f.groupby(['method', axis]).display_value.agg(['mean', 'count']).reset_index()
                for method, g in group.groupby('method'):
                    g = g.sort_values(axis)
                    ax.plot(g[axis], g['mean'], 'o-', color=self.colors[method], label=method, ms=4, lw=1)
                group['scaling_axis'], group['measure'] = axis, measure
                records.append(group)
                ax.set_xlabel(axis.capitalize())
                ax.set_ylabel('Runtime (min)' if row == 0 else 'CPU RAM (GB)')
                ax.set_xscale('log')
                ax.grid(color='#EEEEEE')
        axes[0, 0].legend(fontsize=7, ncol=2)
        self.save('resource_scaling', fig, pd.concat(records, ignore_index=True))

    def stability(self):
        for metric, f in self.down.groupby('metric', sort=False):
            self.boxes('downsampling_' + slug(metric), f, title='Downsampling: ' + metric)
        dataset = pd.read_csv(self.results / 'stability/downsampling_one_value_per_dataset.csv')
        self.boxes('downsampling_standardized', dataset.rename(columns={'downsampling_z': 'value'}),
                   title='Downsampling: mean task z-score', ylabel='Mean task z-score')
        if self.profile == 'ATAC':
            f = self.down.copy()
            f['retention_percent'] = f.dataset_id.str.extract(r'_downsample(\d+)_').astype(int)
            summary = f.groupby(['method', 'retention_percent']).value.agg(['mean', 'count']).reset_index()
            fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
            for method, g in summary.groupby('method'):
                ax.plot(g.retention_percent, g['mean'], 'o-', color=self.colors[method], label=method)
            ax.set(xlabel='Cell retention (%)', ylabel='Mean ground-truth correlation', xticks=[60, 70, 80, 90, 95])
            ax.legend(fontsize=9, ncol=2)
            ax.grid(color='#EEEEEE')
            self.save('downsampling_retention_curve', fig, summary)

    def usability(self):
        cats = pd.read_csv(self.results / 'usability/category_scores.csv')
        ncols = 3 if len(self.methods) == 9 else 2
        nrows = int(np.ceil(len(self.methods) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 11), subplot_kw={'projection': 'polar'}, layout='constrained')
        theta = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        for ax, method in zip(axes.flat, self.order):
            f = cats[cats.method.eq(method)].set_index('category').reindex(CATEGORY_SHORT)
            t, v = np.r_[theta, theta[0]], np.r_[f.score, f.score.iloc[0]]
            ax.plot(t, v, color=self.colors[method], lw=1.5)
            ax.fill(t, v, color=self.colors[method], alpha=.25)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(theta, ['Install', 'Code', 'Guidance', 'Maintenance', 'Publication'], fontsize=8)
            ax.set_ylim(0, 10)
            ax.set_yticks([2, 4, 6, 8, 10], labels=['2', '4', '6', '8', '10'], fontsize=6)
            ax.set_title(method, pad=15, fontsize=11)
        for ax in list(axes.flat)[len(self.methods):]:
            ax.set_visible(False)
        self.save('usability_radars', fig, cats)

    def comparisons(self):
        if self.profile == 'ATAC':
            sparse = self.raw[self.raw.data_type.eq('simulation') & self.raw.dataset_id.str.contains('sparse')]
            if sparse.dataset_id.nunique() != 4:
                raise ValueError('Expected four sparse tasks')
            for metric, f in sparse.groupby('metric', sort=False):
                self.boxes('sparse_' + slug(metric), f, title='Sparse tasks: ' + metric)
            pairs = pd.read_csv(self.source / 'measurements_raw/ATAC/dataset_pairs.csv')
            rnacfg = json.loads((CODE / 'config/scRNA.json').read_text())
            selected = ['TSvelo', 'scVelo-dynamical', 'scVelo-stochastic']
            methods = self.methods + selected
            palette = pd.read_csv(CODE / 'figures/Figure2/config/method_colors.csv')
            colors = {**self.colors, **dict(zip(palette.method, palette['hex']))}
            for metric in [m for m in self.cfg['accuracy_metrics'] + self.cfg['consistency_metrics'] if m in self.raw[self.raw.data_type.eq('real')].metric.unique()]:
                atac = self.raw[self.raw.metric.eq(metric) & self.raw.data_type.eq('real')].merge(
                    pairs[['pair_id', 'ATAC_dataset']], left_on='dataset_id', right_on='ATAC_dataset')
                entry = next(e for e in rnacfg['measurement_inputs'] if e['metric'] == metric and e['data_type'] == 'real')
                rna = read_matrix(self.source / entry['path'], 'real', metric,
                                  self.source / rnacfg['real_id_map_csv'])
                rna = rna[rna.method.isin(selected)].merge(pairs[['pair_id', 'scRNA_dataset']], left_on='dataset_id', right_on='scRNA_dataset')
                frame = pd.concat([atac, rna], ignore_index=True)
                frame['point_key'] = frame.pair_id.astype(str)
                self.boxes('paired_rna_' + slug(metric), frame, methods, colors, 'Matched real datasets: ' + metric, len(self.methods))
                self.paired_differences('paired_rna_differences_' + slug(metric), frame, 'point_key', self.methods, selected)
        else:
            styles = json.loads((CODE / 'figures/Spatial/comparison_styles.json').read_text())
            methods, colors = styles['methods'], styles['styles']['colors']
            entries = self.cfg['measurement_inputs'] + self.cfg['downsampling_inputs']
            for e in entries:
                typ = 'downsampling' if e['role'] == 'downsampling' else 'real'
                f = read_matrix(self.source / e['path'], typ, e['metric'],
                                self.source / self.cfg['real_id_map_csv'])
                self.boxes('rna_comparison_' + typ + '_' + slug(e['metric']), f, methods, colors,
                           f'{typ.capitalize()}: {e["metric"]}', 7)
                self.paired_differences('rna_differences_' + typ + '_' + slug(e['metric']), f, 'dataset_id', methods[:7], methods[7:])
            for metric in ['esvc_annotation_matched', 'esvc_all_cell']:
                for typ in ['real', 'downsampling']:
                    path = self.source / f'measurements_raw/Spatial/measurement/auxiliary/{metric}_Spatial_{typ}.csv'
                    if path.is_file():
                        f = read_matrix(path, typ, metric, None)
                        self.boxes('diagnostic_' + metric + '_' + typ, f, methods, colors,
                                   f'Auxiliary {metric.replace("_", " ")}: {typ}', 7)

    def paired_differences(self, name, frame, key, a_methods, b_methods):
        wide = frame.pivot(index=key, columns='method', values='value')
        rows = []
        for a, b in itertools.product(a_methods, b_methods):
            diff = (wide[a] - wide[b]).dropna()
            rows.append({'method_a': a, 'method_b': b, 'n_common_datasets': len(diff),
                         'mean_a_minus_b': diff.mean(), 'median_a_minus_b': diff.median()})
        self.csv(name, pd.DataFrame(rows))

    def run(self):
        for action in [self.ranks, self.metrics, self.resources, self.stability, self.usability, self.comparisons]:
            action()
            print(f'{self.profile}: completed {action.__name__}', flush=True)
        manifest = pd.DataFrame(self.manifest)
        self.csv('figure_manifest', manifest)
        self.csv('plot_input_manifest', pd.DataFrame(self.plot_inputs))
        for record in self.manifest:
            path = self.output / (record['figure'] + '.' + record['format'])
            if record['format'] == 'png':
                with Image.open(path) as im:
                    if record['square_expected'] and im.width != im.height:
                        raise ValueError('Expected a square figure: ' + path.name)
                    if abs(im.info.get('dpi', [0])[0] - 300) > 1:
                        raise ValueError('PNG resolution check failed')
            elif record['format'] == 'pdf' and not path.read_bytes().startswith(b'%PDF'):
                raise ValueError('Invalid PDF')
            elif record['format'] == 'svg' and '<svg' not in path.read_text():
                raise ValueError('Invalid SVG')
        (self.output / 'plot_validation.json').write_text(json.dumps({'status': 'PASS',
            'profile': self.profile, 'figures': len(self.manifest) // 3,
            'formats': ['PDF', 'SVG', 'PNG'], 'png_dpi': 300,
            'font_family': self.font,
            'ranking_source': 'independently verified current calculations'}, indent=2) + '\n')


def slug(text):
    return re.sub('[^a-z0-9]+', '_', text.lower()).strip('_')


def place_labels(fig, ax, coords, colors):
    """Place every method label in display coordinates without label overlap."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = []
    bounds = ax.get_window_extent(renderer)
    point_pixels = ax.transData.transform(coords[['accuracy', 'consistency']].to_numpy())
    for row, anchor in zip(coords.itertuples(), point_pixels):
        text = ax.text(row.accuracy, row.consistency, row.method, fontsize=11, va='center',
                       bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .8, 'pad': .5}, zorder=5)
        choices = []
        for radius in [18, 30, 45, 65, 85, 110]:
            for angle in [0, 180, 45, -45, 135, -135, 90, -90]:
                dx, dy = radius * np.cos(np.deg2rad(angle)), radius * np.sin(np.deg2rad(angle))
                xy = ax.transData.inverted().transform(anchor + [dx, dy])
                text.set_position(xy)
                text.set_ha('left' if dx >= 0 else 'right')
                box = text.get_window_extent(renderer).expanded(1.05, 1.15)
                penalty = 0.0
                for other in occupied:
                    penalty += max(0, min(box.x1, other.x1) - max(box.x0, other.x0)) * max(0, min(box.y1, other.y1) - max(box.y0, other.y0))
                if box.x0 < bounds.x0 or box.x1 > bounds.x1 or box.y0 < bounds.y0 or box.y1 > bounds.y1:
                    penalty += 1e6
                for point in point_pixels:
                    if box.x0 - 4 <= point[0] <= box.x1 + 4 and box.y0 - 4 <= point[1] <= box.y1 + 4:
                        penalty += 1e4
                choices.append((penalty, radius, dx, dy, xy, box))
                if penalty == 0:
                    break
            if choices[-1][0] == 0:
                break
        chosen = min(choices, key=lambda c: (c[0], c[1]))
        _, _, dx, _, xy, box = chosen
        text.set_position(xy)
        text.set_ha('left' if dx >= 0 else 'right')
        occupied.append(box)
        ax.plot([row.accuracy, xy[0]], [row.consistency, xy[1]], color=colors[row.method], lw=.6, alpha=.6, zorder=2)


def read_matrix(path, typ, metric, real_map):
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.map(lambda x: 'TSvelo' if x == 'Tsvelo' else x)
    if typ == 'real' and real_map is not None:
        ids = pd.read_csv(real_map, dtype=str)
        mapping = dict(zip(ids.dataset_number, ids.dataset_id))
        matrix.columns = [mapping.get(str(c), str(c)) for c in matrix.columns]
    matrix = matrix.replace({'error': np.nan}).apply(pd.to_numeric, errors='raise')
    frame = matrix.rename_axis('method').reset_index().melt(id_vars='method', var_name='dataset_id', value_name='value')
    frame['metric'], frame['data_type'] = metric, typ
    return frame


def main(profile):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=REPO / 'Results' / profile)
    parser.add_argument('--source-root', type=Path, default=REPO / 'Source_data')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    results = args.results.resolve()
    if not (results / 'overall/scores_ranks.csv').is_file():
        subprocess.run([sys.executable, str(CODE / 'run.py'), '--profile', profile,
                        '--source-root', str(args.source_root), '--output', str(results)], check=True)
    output = (args.output or results / 'figures').resolve()
    Plotter(profile, results, args.source_root.resolve(), output).run()
