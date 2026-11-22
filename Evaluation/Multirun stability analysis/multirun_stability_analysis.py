"""Compare saved RNA velocity vectors across repeated runs."""

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
from anndata.io import read_elem
from scipy.sparse import issparse


MANIFEST_COLUMNS = ("method", "dataset_id", "run_id", "path", "vkey")
OUTPUT_COLUMNS = ("tool_name", "dataset_name", "group", "group_cosine", "average_cosine")


def read_manifest(path):
    groups = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing manifest columns: {', '.join(sorted(missing))}")
        for row in reader:
            values = [(row.get(column) or "").strip() for column in MANIFEST_COLUMNS]
            if not all(values) or None in row:
                raise ValueError(f"Invalid manifest row {reader.line_num} in {path}")
            method, dataset, run_id, h5ad_path, vkey = values
            runs = groups.setdefault((method, dataset), {})
            if run_id in runs:
                raise ValueError(f"Duplicate run ID {run_id!r} for {method} / {dataset}")
            runs[run_id] = (path.parent / Path(h5ad_path).expanduser(), vkey)
    return groups


def load_velocity(path, vkey):
    with h5py.File(path, "r") as h5ad:
        layer = f"layers/{vkey}"
        if layer not in h5ad:
            raise ValueError(f"Missing {layer} in {path}")
        obs, var = h5ad["obs"], h5ad["var"]
        cells = read_elem(obs[obs.attrs["_index"]])
        features = read_elem(var[var.attrs["_index"]])
        velocity = read_elem(h5ad[layer])
    if issparse(velocity):
        velocity = velocity.toarray()
    if len(set(cells)) != len(cells) or len(set(features)) != len(features):
        raise ValueError(f"Duplicate cell or feature identifiers in {path}")
    if velocity.shape != (len(cells), len(features)):
        raise ValueError(f"Invalid velocity shape in {layer} of {path}")
    if velocity.dtype.kind not in "fiu" or not np.isfinite(velocity).all():
        raise ValueError(f"Expected finite real velocities in {layer} of {path}")
    return velocity, {cell: index for index, cell in enumerate(cells)}, features


def rescaled_cosine(a, b):
    a, b = a.astype(np.float64, copy=False), b.astype(np.float64, copy=False)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine = np.dot(a, b) / (norm_a * norm_b)
    return (np.clip(cosine, -1.0, 1.0) + 1.0) / 2.0


def score_group(method, dataset, runs):
    if len(runs) == 1:
        print(
            f"Skipping {method} / {dataset}: only one run ({next(iter(runs))})",
            file=sys.stderr,
        )
        return []

    reference_features = None
    loaded = {}
    for run_id, (path, vkey) in runs.items():
        velocity, cells, features = load_velocity(path, vkey)
        if reference_features is None:
            reference_features = features
        elif not np.array_equal(features, reference_features):
            feature_indices = {feature: index for index, feature in enumerate(features)}
            if feature_indices.keys() != set(reference_features):
                raise ValueError(f"Feature set differs from the first run in {path}")
            velocity = velocity[:, [feature_indices[feature] for feature in reference_features]]
        loaded[run_id] = velocity, cells

    pairs = []
    for a, b in combinations(loaded, 2):
        velocity_a, cells_a = loaded[a]
        velocity_b, cells_b = loaded[b]
        shared_cells = sorted(cells_a.keys() & cells_b.keys())
        if not shared_cells:
            raise ValueError(f"No shared cells for {method} / {dataset}, runs {a!r} and {b!r}")
        similarities = [
            rescaled_cosine(velocity_a[cells_a[cell]], velocity_b[cells_b[cell]])
            for cell in shared_cells
        ]
        pairs.append((f"{a}_vs_{b}", np.mean(similarities)))

    average_cosine = np.mean([mean for _, mean in pairs])
    return [
        (method, dataset, group, f"{mean:.6f}", f"{average_cosine:.6f}")
        for group, mean in pairs
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV of result H5AD files")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file")
    args = parser.parse_args()

    try:
        groups = read_manifest(args.manifest)
        input_paths = {path.resolve() for runs in groups.values() for path, _ in runs.values()}
        if args.output.resolve() in input_paths | {args.manifest.resolve()}:
            raise ValueError("The output file must differ from the input files")

        rows = []
        for (method, dataset), runs in groups.items():
            rows.extend(score_group(method, dataset, runs))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(OUTPUT_COLUMNS)
            writer.writerows(rows)
    except (OSError, KeyError, TypeError, ValueError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Wrote {len(rows)} run-pair scores to {args.output}")


if __name__ == "__main__":
    main()
