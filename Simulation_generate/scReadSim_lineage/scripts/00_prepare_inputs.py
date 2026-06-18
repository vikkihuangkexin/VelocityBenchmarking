#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def normalize_barcode(x: object) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    # remove common wrappers
    s = re.sub(r"^[^:]*:", "", s)    # sample prefix before colon
    s = re.sub(r"^[^_]*_", "", s)    # sample prefix before underscore
    s = re.sub(r"-1$", "", s)        # 10x suffix
    s = re.sub(r"x$", "", s)         # velocyto trailing x
    m = re.search(r"([ACGTN]{14,})", s.upper())
    if m:
        return m.group(1)
    dna = re.sub(r"[^ACGTN]", "", s.upper())
    return dna if len(dna) >= 14 else None


def read_fai_chroms(path: Path):
    chroms = []
    with open(path, "rt") as fh:
        for line in fh:
            if not line.strip():
                continue
            chroms.append(line.split("\t", 1)[0])
    return chroms


def read_first_n_gtf_chroms(path: Path, n: int = 5000):
    chroms = []
    with open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            chroms.append(line.split("\t", 1)[0])
            if len(chroms) >= n:
                break
    return chroms


def detect_style(chroms):
    if not chroms:
        return "unknown"
    frac_chr = sum(c.startswith("chr") for c in chroms) / max(1, len(chroms))
    return "chr" if frac_chr > 0.8 else "nochr"


def copy_or_link(src: Path, dst: Path, symlink: bool = False):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def prepare_reference(genome_fa: Path, genome_fai: Path, genes_gtf: Path, chrom_sizes: Path | None, outdir: Path, symlink: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    out_fa = outdir / "genome.fa"
    out_fai = outdir / "genome.fa.fai"
    out_gtf = outdir / "genes.gtf"
    out_sizes = outdir / "chrom.sizes"

    copy_or_link(genome_fa, out_fa, symlink)
    copy_or_link(genome_fai, out_fai, symlink)

    fai_chroms = read_fai_chroms(genome_fai)
    gtf_chroms = read_first_n_gtf_chroms(genes_gtf)
    target_style = detect_style(fai_chroms)

    def convert_chrom(name: str) -> str:
        name = name.strip()
        if target_style == "nochr":
            return name[3:] if name.startswith("chr") else name
        return name if name.startswith("chr") else f"chr{name}"

    with open_text(genes_gtf) as fin, open(out_gtf, "wt") as fout:
        for line in fin:
            if line.startswith("#") or not line.strip():
                fout.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            fields[0] = convert_chrom(fields[0])
            fout.write("\t".join(fields) + "\n")

    if chrom_sizes is not None and chrom_sizes.exists():
        copy_or_link(chrom_sizes, out_sizes, symlink)
    else:
        with open(genome_fai, "rt") as fin, open(out_sizes, "wt") as fout:
            for line in fin:
                if not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                fout.write(f"{fields[0]}\t{fields[1]}\n")

    meta = {
        "genome_fa": str(out_fa),
        "genome_fai": str(out_fai),
        "genes_gtf": str(out_gtf),
        "chrom_sizes": str(out_sizes),
        "fai_style": target_style,
        "fai_example": fai_chroms[:10],
        "gtf_example_before": gtf_chroms[:10],
    }
    (outdir / "reference_prep.json").write_text(json.dumps(meta, indent=2))
    return meta


def pick_barcode_column(df: pd.DataFrame, loom_set: set[str]) -> tuple[str, pd.Series, dict]:
    report = {}
    best = None
    for col in df.columns:
        s = df[col].astype(str)
        norm = s.map(normalize_barcode)
        overlap_rows = int(norm.dropna().isin(loom_set).sum())
        overlap_unique = len(set(norm.dropna()) & loom_set)
        report[col] = {
            "n_nonnull_norm": int(norm.notna().sum()),
            "overlap_rows": overlap_rows,
            "overlap_unique": overlap_unique,
        }
        score = (overlap_unique, overlap_rows)
        if best is None or score > best[0]:
            best = (score, col, norm)
    if best is None or best[0][0] == 0:
        raise SystemExit("[ERROR] Could not find a barcode-like column in the CSV that overlaps the loom CellID barcodes.")
    return best[1], best[2], report


def pick_cluster_column(df: pd.DataFrame) -> str:
    candidates = [
        "cluster", "Cluster", "Clusters", "seurat_clusters", "seurat_cluster",
        "leiden", "Leiden", "louvain", "Louvain",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: small-cardinality numeric or string column
    best = None
    for col in df.columns:
        nunique = int(df[col].astype(str).nunique(dropna=True))
        if 2 <= nunique <= 100:
            score = -abs(nunique - 10)
            if best is None or score > best[0]:
                best = (score, col)
    if best is None:
        raise SystemExit("[ERROR] Could not identify a cluster column in the CSV.")
    return best[1]


def parse_args():
    p = argparse.ArgumentParser(description="Prepare selected-cell inputs from a loom file and a metadata CSV.")
    p.add_argument("--loom", required=True, help="Loom file containing CellID and, optionally, embedding or cluster column attributes.")
    p.add_argument("--csv", required=True, help="CSV containing the selected cells and their cluster labels.")
    p.add_argument("--project-root", required=True, help="Project output root. The pipeline writes prepared/, rawdata/, screadsim/, and lineage/ under this directory.")
    p.add_argument("--sample", default="sample")
    p.add_argument("--mode-name", default="selected_cells")
    p.add_argument("--raw-bam", required=True, help="Input BAM containing reads with CB tags. This path is recorded in manifest.json for the BAM-subsetting step.")
    p.add_argument("--genome-fa", required=True)
    p.add_argument("--genome-fai", required=True)
    p.add_argument("--genes-gtf", required=True)
    p.add_argument("--chrom-sizes", default=None)
    p.add_argument("--symlink-reference", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import loompy
    except Exception as e:
        raise SystemExit(f"Failed to import loompy: {e}")

    project_root = Path(args.project_root)
    prepared_root = project_root / "prepared" / args.mode_name
    barcode_dir = prepared_root / "barcode"
    meta_dir = prepared_root / "metadata"
    ref_dir = prepared_root / "reference"
    qc_dir = prepared_root / "qc"

    if prepared_root.exists() and args.force:
        shutil.rmtree(prepared_root)
    barcode_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    with loompy.connect(args.loom, mode="r") as ds:
        if "CellID" not in ds.ca:
            raise SystemExit("[ERROR] Loom does not contain CellID in column attributes.")
        loom_df = pd.DataFrame({
            "loom_order": np.arange(ds.shape[1]),
            "loom_cellid": ds.ca["CellID"].astype(str),
        })
        if "Clusters" in ds.ca:
            loom_df["loom_clusters"] = ds.ca["Clusters"]
        if "_X" in ds.ca:
            loom_df["loom_x"] = ds.ca["_X"]
        if "_Y" in ds.ca:
            loom_df["loom_y"] = ds.ca["_Y"]

    loom_df["barcode_norm"] = loom_df["loom_cellid"].map(normalize_barcode)
    loom_set = set(loom_df["barcode_norm"].dropna())

    csv_df = pd.read_csv(args.csv)
    barcode_col, csv_bc_norm, bc_report = pick_barcode_column(csv_df, loom_set)
    cluster_col = pick_cluster_column(csv_df)

    csv_df = csv_df.copy()
    csv_df["barcode_norm"] = csv_bc_norm
    csv_df = csv_df.dropna(subset=["barcode_norm"]).copy()
    csv_df["cluster_raw"] = csv_df[cluster_col].astype(str).str.strip()
    csv_df["celllabel"] = csv_df["cluster_raw"].map(lambda x: f"cluster_{x}")

    # Keep first occurrence per barcode in the metadata CSV
    csv_first = csv_df.drop_duplicates("barcode_norm", keep="first").copy()

    selected = loom_df.merge(
        csv_first[["barcode_norm", "cluster_raw", "celllabel"]],
        on="barcode_norm",
        how="inner",
    ).sort_values("loom_order").reset_index(drop=True)

    if selected.empty:
        raise SystemExit("[ERROR] No overlapping cells between loom and CSV subset.")
    if selected["barcode_norm"].nunique() != len(selected):
        raise SystemExit("[ERROR] Duplicate normalized barcodes remain after matching; please inspect the CSV.")

    subset_barcodes = barcode_dir / "barcodes.tsv"
    subset_labels = barcode_dir / f"{args.sample}.celllabels.txt"
    mapping_tsv = meta_dir / "barcode_mapping.tsv"

    with open(subset_barcodes, "wt") as f:
        for bc in selected["barcode_norm"]:
            f.write(bc + "-1\n")

    with open(subset_labels, "wt") as f:
        for lab in selected["celllabel"]:
            f.write(lab + "\n")

    selected.to_csv(mapping_tsv, sep="\t", index=False)

    ref_info = prepare_reference(
        genome_fa=Path(args.genome_fa),
        genome_fai=Path(args.genome_fai),
        genes_gtf=Path(args.genes_gtf),
        chrom_sizes=Path(args.chrom_sizes) if args.chrom_sizes else None,
        outdir=ref_dir,
        symlink=args.symlink_reference,
    )

    raw_bam_dir = project_root / "rawdata" / "bam"
    bam_cells = raw_bam_dir / f"{args.sample}.cells_only.bam"
    bam_cb = raw_bam_dir / f"{args.sample}.cells_only.CBinReadName.bam"

    manifest = {
        "sample": args.sample,
        "mode_name": args.mode_name,
        "project_root": str(project_root),
        "old_root": str(project_root),
        "new_root": str(project_root),
        "bam": {
            "raw_bam": str(Path(args.raw_bam)),
            "cells_only": str(bam_cells),
            "cb_in_read_name": str(bam_cb),
        },
        "prepared_paths": {
            "prepared_root": str(prepared_root),
            "barcode_dir": str(barcode_dir),
            "metadata_dir": str(meta_dir),
            "reference_dir": str(ref_dir),
            "subset_barcodes_tsv": str(subset_barcodes),
            "subset_celllabels_txt": str(subset_labels),
            "barcode_mapping_tsv": str(mapping_tsv),
            "genome_fa": ref_info["genome_fa"],
            "genome_fai": ref_info["genome_fai"],
            "chrom_sizes": ref_info["chrom_sizes"],
            "genes_gtf": ref_info["genes_gtf"],
        },
        "selection": {
            "loom_n_cells": int(len(loom_df)),
            "csv_n_rows": int(len(csv_df)),
            "selected_n_cells": int(len(selected)),
            "barcode_column_in_csv": barcode_col,
            "cluster_column_in_csv": cluster_col,
            "barcode_source_overlap_report": bc_report,
        },
        "label_counts": selected["celllabel"].value_counts().sort_index().to_dict(),
        "notes": [
            "subset barcodes are written as normalized 10x-style barcodes with -1 suffix",
            "labels come directly from the selected blood-development CSV cluster column",
            "cells retain loom order to preserve the original analyzed subset ordering",
        ],
    }

    (prepared_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (qc_dir / "prepare_from_csv.summary.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps({
        "status": "ok",
        "prepared_root": str(prepared_root),
        "selected_cells": int(len(selected)),
        "barcode_column_in_csv": barcode_col,
        "cluster_column_in_csv": cluster_col,
        "subset_barcodes_tsv": str(subset_barcodes),
        "subset_celllabels_txt": str(subset_labels),
        "manifest_json": str(prepared_root / "manifest.json"),
        "label_counts": selected["celllabel"].value_counts().sort_index().to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
