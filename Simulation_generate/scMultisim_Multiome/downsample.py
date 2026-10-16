#!/usr/bin/env python3
"""Create paired cell-retention replicates from the supplied source plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import anndata as ad
import numpy as np

from common import write_json
from run import select_ids
from validate_pair import validate_pair

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_file(root: Path, name: str, modality: str) -> Path:
    filename = f"{name}_{modality}.h5ad"
    for path in (root / "h5ad" / filename, root / name / filename):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing source pair member: {filename}")


def subset_to_memory(obj, indices):
    # Explicitly retain backed X; some AnnData versions omit it in to_memory().
    matrix = obj.X[indices, :].copy()
    subset = obj[indices, :].to_memory()
    subset.X = matrix
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Completed simulation output root")
    parser.add_argument("--outdir", type=Path, help="Default: same output root")
    parser.add_argument("--ids", default="31-40")
    args = parser.parse_args()
    root = args.root.resolve()
    destination = (args.outdir or root).resolve()
    with (HERE / "datasets_downsampling_31_40.tsv").open() as handle:
        plan = {int(row["dataset_id"]): row for row in csv.DictReader(handle, delimiter="\t")}
    selected = select_ids(args.ids, set(plan))
    source = plan[selected[0]]["source_dataset"]
    rna_path, atac_path = [source_file(root, source, modality) for modality in ("rna", "atac")]
    if not (root / source / "_SUCCESS").is_file():
        raise ValueError("Source completion marker is missing; finish source validation first")
    source_signature = {"rna_sha256": sha256(rna_path), "atac_sha256": sha256(atac_path)}
    source_bed = root / source / "sckinetics_differential_peaks.bed"
    if not source_bed.is_file():
        raise FileNotFoundError(source_bed)
    rna, atac = ad.read_h5ad(rna_path, backed="r"), ad.read_h5ad(atac_path, backed="r")
    try:
        if not rna.obs_names.equals(atac.obs_names):
            raise ValueError("Source RNA/ATAC cell order differs")
        export = destination / "h5ad"
        export.mkdir(parents=True, exist_ok=True)
        summary = []
        for idx in selected:
            row = plan[idx]
            name = row["dataset_name"]
            fraction, seed = float(row["fraction"]), int(row["seed"])
            count = int(round(rna.n_obs * fraction))
            indices = np.sort(np.random.default_rng(seed).choice(rna.n_obs, size=count, replace=False))
            outdir = destination / name
            rna_out, atac_out = [outdir / f"{name}_{modality}.h5ad" for modality in ("rna", "atac")]
            manifest_path = outdir / "downsampling_manifest.json"
            metadata = {
                "source_dataset": source, "fraction": fraction, "replicate": row["replicate"],
                "seed": seed, "source_n_cells": rna.n_obs, "sampled_n_cells": count,
                "sampling_axis": "cells", "source_signature": source_signature,
                "processing": "subset inherited features, graphs, topic factors, and differential peak tables",
                "source_bed_sha256": sha256(source_bed),
                "selected_cell_names": rna.obs_names[indices].tolist(),
            }
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != metadata:
                raise ValueError(f"Changed downsampling source or plan for {name}; choose a new --outdir")
            completed = (outdir / "_SUCCESS").exists()
            if (completed and manifest_path.exists() and rna_out.is_file() and atac_out.is_file()
                and (outdir / source_bed.name).is_file()):
                rna_sub, atac_sub = ad.read_h5ad(rna_out), ad.read_h5ad(atac_out)
                if rna_sub.obs_names.tolist() != metadata["selected_cell_names"]:
                    raise ValueError(f"Stored cell selection differs for {name}")
                validate_pair(rna_sub, atac_sub)
                if sha256(outdir / source_bed.name) != metadata["source_bed_sha256"]:
                    raise ValueError(f"Inherited BED differs for {name}")
                print(f"SKIP validated downsampling pair: {name}")
            else:
                if outdir.exists() and any(outdir.iterdir()) and not manifest_path.exists():
                    raise FileExistsError(f"Untracked output exists: {outdir}")
                outdir.mkdir(parents=True, exist_ok=True)
                (outdir / "_SUCCESS").unlink(missing_ok=True)
                write_json(manifest_path, metadata)
                rna_sub, atac_sub = subset_to_memory(rna, indices), subset_to_memory(atac, indices)
                for obj, modality, paired in [(rna_sub, "RNA", atac_out), (atac_sub, "ATAC", rna_out)]:
                    obj.obs["downsample_source"] = source
                    obj.obs["downsample_fraction"] = fraction
                    obj.obs["downsample_seed"] = seed
                    obj.uns["downsampling"] = {k: v for k, v in metadata.items() if k != "selected_cell_names"}
                    obj.uns["paired_atac_h5ad" if modality == "RNA" else "paired_rna_h5ad"] = paired.name
                    # Keep source simulation metadata; the derivative is identified here.
                    obj.uns["dataset_name"] = name
                rna_sub.write_h5ad(rna_out, compression="gzip")
                atac_sub.write_h5ad(atac_out, compression="gzip")
                result = validate_pair(ad.read_h5ad(rna_out), ad.read_h5ad(atac_out))
                write_json(outdir / "pair_validation.json", result)
                shutil.copyfile(source_bed, outdir / source_bed.name)
                (outdir / "_SUCCESS").write_text("validated\n")
            for path in (rna_out, atac_out):
                link = export / path.name
                if link.is_symlink() and link.resolve() == path.resolve():
                    continue
                if link.exists() or link.is_symlink():
                    raise FileExistsError(f"Unrelated H5AD export: {link}")
                link.symlink_to(Path("..") / name / path.name)
            summary.append({"dataset": name, "source": source, "fraction": fraction,
                "replicate": row["replicate"], "seed": seed, "cells": count,
                "genes": rna_sub.n_vars, "peaks": atac_sub.n_vars,
                "rna_h5ad": str(rna_out), "atac_h5ad": str(atac_out)})
            del rna_sub, atac_sub
        summary_path = destination / ("downsampling_31_40_summary.tsv" if len(selected) == 10
                                      else "downsampling_selected_summary.tsv")
        with summary_path.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(summary)
        print(f"Validated {len(summary)} paired cell-downsampling tasks")
    finally:
        rna.file.close()
        atac.file.close()


if __name__ == "__main__":
    main()
