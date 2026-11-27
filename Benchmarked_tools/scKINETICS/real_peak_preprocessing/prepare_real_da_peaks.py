#!/usr/bin/env python3
"""Select cluster-enriched ATAC peaks and export a scKINETICS BED."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.stats import hypergeom


FORMATS = ("10x", "share-seq")
ID_KINDS = ("rna", "atac")
PATH_FIELDS = ("input_h5ad", "matrix_dir", "atac_counts", "atac_barcodes", "atac_peaks", "barcode_map")
DATASET_FIELDS = ("sample_name", "input_format", "genome", "cluster_key", "rna_id_kind", "matrix_id_kind", *PATH_FIELDS)
PEAK_PATTERN = re.compile(r"([^:\s]+):(\d+)-(\d+)")
TENX_PATTERN = re.compile(r"(?:^|:)([ACGTN]{16})(?:-([1-9]\d*)|(x))?$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-file", type=Path, help="CSV with one dataset per row.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the BED and TSV/JSON reports.")
    parser.add_argument("--sample-name", help="Prefix for the output filenames.")
    parser.add_argument("--input-format", choices=FORMATS)
    parser.add_argument("--input-h5ad", help="RNA H5AD with barcode index and cell labels in obs.")
    parser.add_argument("--genome", help="Assembly label for the supplied ATAC coordinates.")
    parser.add_argument("--cluster-key", help="RNA obs label column (default: celltype).")
    parser.add_argument("--matrix-dir", help="10x MEX directory containing features, barcodes and the count matrix.")
    parser.add_argument("--atac-counts", help="SHARE-seq feature-by-cell Matrix Market counts.")
    parser.add_argument("--atac-barcodes", help="SHARE-seq matrix-column barcode file.")
    parser.add_argument("--atac-peaks", help="SHARE-seq matrix-row peak coordinates in BED format.")
    parser.add_argument("--barcode-map", help="SHARE-seq TSV with atac.bc and rna.bc columns.")
    parser.add_argument("--rna-id-kind", choices=ID_KINDS, help="Modality of H5AD obs_names for SHARE-seq (default: rna).")
    parser.add_argument("--matrix-id-kind", choices=ID_KINDS, help="Modality of matrix barcodes for SHARE-seq (default: rna).")
    parser.add_argument("--allow-partial-overlap", action="store_true",
                        help="Use the matched intersection and report unmatched RNA cells.")
    parser.add_argument("--padj-cutoff", type=float, default=0.05)
    parser.add_argument("--log2fc-cutoff", type=float, default=1.0)
    parser.add_argument("--top-n-per-cluster", type=int, default=1000)
    parser.add_argument("--max-peak-width", type=int, default=2000, help="Exclusive width limit in bp.")
    parser.add_argument("--pseudocount", type=float, default=0.5, help="Used for effect-size estimates only.")
    args = parser.parse_args()
    if args.metadata_file and any(getattr(args, key) is not None for key in DATASET_FIELDS):
        parser.error("--metadata-file cannot be combined with per-dataset arguments")
    if not 0 <= args.padj_cutoff <= 1:
        parser.error("--padj-cutoff must be between 0 and 1")
    if not np.isfinite(args.log2fc_cutoff) or args.log2fc_cutoff < 0:
        parser.error("--log2fc-cutoff must be finite and nonnegative")
    if args.top_n_per_cluster < 1 or args.max_peak_width < 2:
        parser.error("--top-n-per-cluster must be positive and --max-peak-width must exceed 1")
    if not np.isfinite(args.pseudocount) or args.pseudocount <= 0:
        parser.error("--pseudocount must be finite and positive")
    return args


def dataset_options(row, base):
    options = {key: (row.get(key) or "").strip() for key in DATASET_FIELDS}
    sample = options["sample_name"]
    input_format = options["input_format"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sample):
        raise ValueError("sample_name must start with a letter or digit and contain only letters, digits, . _ or -")
    if input_format not in FORMATS:
        raise ValueError(f"{sample}: input_format must be one of {', '.join(FORMATS)}")
    if not options["genome"]:
        raise ValueError(f"{sample}: genome must name the assembly of the ATAC coordinates")
    options["cluster_key"] = options["cluster_key"] or "celltype"
    share_paths = ("atac_counts", "atac_barcodes", "atac_peaks", "barcode_map")
    share_fields = (*share_paths, "rna_id_kind", "matrix_id_kind")
    if input_format == "10x":
        required = ("input_h5ad", "matrix_dir")
        disallowed = share_fields
    else:
        required = ("input_h5ad", *share_paths)
        disallowed = ("matrix_dir",)
        for field in ("rna_id_kind", "matrix_id_kind"):
            options[field] = options[field] or "rna"
            if options[field] not in ID_KINDS:
                raise ValueError(f"{sample}: {field} must be rna or atac")
    if missing := [name for name in required if not options[name]]:
        raise ValueError(f"{sample}: missing required inputs: {', '.join(missing)}")
    if unused := [name for name in disallowed if options[name]]:
        raise ValueError(f"{sample}: fields not used with {input_format}: {', '.join(unused)}")
    for field in required:
        path = (base / options[field]).resolve()
        valid = path.is_dir() if field == "matrix_dir" else path.is_file()
        if not valid:
            raise FileNotFoundError(f"{sample}: {field} not found: {path}")
        options[field] = path
    return options


def read_datasets(args):
    if args.metadata_file is None:
        return [dataset_options(vars(args), Path.cwd())]
    path = args.metadata_file.resolve()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(set(columns)) != len(columns) or set(columns) - set(DATASET_FIELDS):
            raise ValueError("Metadata CSV contains duplicated or unrecognized column names")
        datasets = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{reader.line_num}: wrong number of CSV fields")
            try:
                datasets.append(dataset_options(row, path.parent))
            except ValueError as error:
                raise ValueError(f"{path}:{reader.line_num}: {error}") from error
    if not datasets:
        raise ValueError(f"{path}: no dataset rows")
    names = [d["sample_name"] for d in datasets]
    if len(set(names)) != len(names):
        raise ValueError("sample_name must be unique in the metadata CSV")
    return datasets


def read_rna_labels(path, key):
    obj = anndata.read_h5ad(path, backed="r")
    try:
        if key not in obj.obs:
            raise ValueError(f"{path}: obs has no {key!r} column")
        labels = obj.obs[key].copy()
    finally:
        obj.file.close()
    labels.index = unique_barcodes(labels.index, "RNA obs_names")
    if labels.isna().any() or (labels.astype(str).str.strip() == "").any():
        raise ValueError(f"{path}: obs[{key!r}] must label every RNA cell")
    return labels.astype(str)


def unique_barcodes(values, description):
    values = pd.Index(values, dtype=str)
    if len(values) == 0 or values.isna().any() or (values.str.strip() == "").any():
        raise ValueError(f"{description} contains no identifiers or contains empty identifiers")
    if not values.is_unique:
        duplicate = values[values.duplicated()][0]
        raise ValueError(f"{description} contains duplicate barcode {duplicate!r}")
    return values


def read_barcode_file(path):
    barcodes = pd.read_csv(path, sep="\t", header=None, usecols=[0], dtype=str, keep_default_na=False)[0]
    return unique_barcodes(barcodes, str(path))


def mex_file(directory, name):
    for filename in (name + ".gz", name):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {name} or {name}.gz in {directory}")


def parse_coordinates(value):
    match = PEAK_PATTERN.fullmatch(value)
    if match is None:
        return None
    return match[1], int(match[2]), int(match[3])


def read_tenx_peaks(path):
    features = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    if features.shape[1] not in (2, 3, 6):
        raise ValueError(f"{path}: expected 2, 3 or 6 feature columns")
    rows = []
    peak_features = features if features.shape[1] == 2 else features.loc[features[2].eq("Peaks")]
    for row in peak_features.itertuples(index=True, name=None):
        row_index, feature_id, feature_name, *_ = row
        coordinates = parse_coordinates(feature_name) or parse_coordinates(feature_id)
        if len(row) == 7 and any(row[-3:]):
            try:
                peak_fields = row[-3], int(row[-2]), int(row[-1])
            except ValueError as error:
                raise ValueError(f"{path}: invalid peak coordinates in feature row {row_index + 1}") from error
            if coordinates is not None and coordinates != peak_fields:
                raise ValueError(f"{path}: conflicting peak coordinates in feature row {row_index + 1}")
            coordinates = peak_fields
        if coordinates is None:
            raise ValueError(f"{path}: no chr:start-end coordinates in peak feature row {row_index + 1}")
        rows.append((row_index, *coordinates))
    return pd.DataFrame(rows, columns=["index", "chrom", "start", "end"]), len(features)


def read_share_peaks(path):
    coordinates = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], keep_default_na=False,
                              dtype={0: str, 1: np.int64, 2: np.int64})
    coordinates.columns = ["chrom", "start", "end"]
    coordinates.insert(0, "index", np.arange(len(coordinates)))
    return coordinates, len(coordinates)


def eligible_peaks(peaks, width_max):
    if peaks.empty:
        raise ValueError("No ATAC peak features were found")
    if not peaks["chrom"].str.match(r"^[^:\s]+$").all():
        raise ValueError("Peak chromosomes must be nonempty names without spaces or colons")
    if (peaks["start"] < 0).any() or (peaks["end"] <= peaks["start"]).any():
        raise ValueError("Peak coordinates must satisfy 0 <= start < end")
    if peaks.duplicated(["chrom", "start", "end"]).any():
        raise ValueError("Duplicate ATAC peak coordinates")
    peaks = peaks.assign(width=peaks["end"] - peaks["start"])
    peaks = peaks.loc[peaks["width"] < width_max].reset_index(drop=True)
    if peaks.empty:
        raise ValueError(f"No peaks satisfy the width limit of < {width_max} bp")
    names = peaks["chrom"] + ":" + peaks["start"].astype(str) + "-" + peaks["end"].astype(str)
    return peaks.assign(feature_id=names)


def tenx_identity(value):
    match = TENX_PATTERN.search(value)
    if match is None:
        return None
    gem = int(match[2]) if match[2] else 1 if match[3] else None
    return match[1].upper(), gem


def align_tenx(rna, atac):
    by_identity = defaultdict(list)
    by_sequence = defaultdict(list)
    for index, barcode in enumerate(atac):
        if (identity := tenx_identity(barcode)) is not None:
            by_identity[identity].append(index)
            by_sequence[identity[0]].append(index)

    matched = atac.get_indexer(rna)
    for index in np.flatnonzero(matched < 0):
        identity = tenx_identity(rna[index])
        if identity is None:
            continue
        seq, gem = identity
        same_sequence = by_sequence[seq]
        candidates = same_sequence if gem is None else by_identity[identity]
        if gem is not None and not candidates and len(same_sequence) == 1:
            candidates = by_identity[(seq, None)]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous 10x barcode {rna[index]!r}; supply unambiguous GEM suffixes")
        if candidates:
            matched[index] = candidates[0]
    return matched


def normalize_share_barcodes(barcodes):
    return pd.Index(barcodes).str.replace(r"\.(?=R[23]\.|P1\.)", ",", regex=True)


def align_share(rna, atac, map_path, rna_kind, matrix_kind):
    crosswalk = pd.read_csv(map_path, sep="\t", dtype=str, keep_default_na=False)
    for name in ("rna.bc", "atac.bc"):
        if name not in crosswalk:
            raise ValueError(f"{map_path}: required column {name!r} is missing")
        normalized = normalize_share_barcodes(crosswalk[name])
        crosswalk[name] = unique_barcodes(normalized, f"{map_path}: {name}")
    rna = unique_barcodes(normalize_share_barcodes(rna), "normalized SHARE-seq RNA obs_names")
    atac = unique_barcodes(normalize_share_barcodes(atac), "normalized SHARE-seq matrix barcodes")
    source_key, target_key = f"{rna_kind}.bc", f"{matrix_kind}.bc"
    translated = crosswalk.set_index(source_key, drop=False)[target_key].reindex(rna)
    return atac.get_indexer(translated)


def open_matrix(path):
    return gzip.open(path, "rt", encoding="ascii") if path.suffix == ".gz" else path.open(encoding="ascii")


def read_accessibility(path, n_features, n_barcodes, peak_indexes, cell_indexes):
    """Stream the feature-by-cell MEX, keeping positive counts in the selected axes."""
    row_lookup = np.full(n_features, -1, dtype=np.int32)
    col_lookup = np.full(n_barcodes, -1, dtype=np.int32)
    row_lookup[peak_indexes] = np.arange(len(peak_indexes), dtype=np.int32)
    col_lookup[cell_indexes] = np.arange(len(cell_indexes), dtype=np.int32)
    rows_kept, columns_kept = [], []
    read_entries = 0
    with open_matrix(path) as handle:
        header = handle.readline().strip().lower()
        if header != "%%matrixmarket matrix coordinate integer general":
            raise ValueError(f"{path}: expected a general coordinate Matrix Market file with integer counts")
        dimensions = next((line for line in handle if line.strip() and not line.lstrip().startswith("%")), "")
        dims = dimensions.split()
        if len(dims) != 3:
            raise ValueError(f"{path}: missing Matrix Market dimensions")
        rows, columns, entries = map(int, dims)
        if (rows, columns) != (n_features, n_barcodes) or entries < 0:
            raise ValueError(f"{path}: matrix dimensions must be {n_features} features × {n_barcodes} barcodes")
        if entries:
            chunks = pd.read_csv(handle, sep=r"\s+", comment="%", header=None,
                                 dtype=np.int64, chunksize=2_000_000, na_filter=False)
            for chunk in chunks:
                if chunk.shape[1] != 3:
                    raise ValueError(f"{path}: expected row, column and count in each matrix entry")
                row, column, count = chunk.to_numpy().T
                if (row < 1).any() or (row > rows).any() or (column < 1).any() or (column > columns).any():
                    raise ValueError(f"{path}: matrix index is out of bounds")
                if (count < 0).any():
                    raise ValueError(f"{path}: raw ATAC counts must be nonnegative")
                read_entries += len(count)
                peak, cell = row_lookup[row - 1], col_lookup[column - 1]
                keep = (peak >= 0) & (cell >= 0) & (count > 0)
                rows_kept.append(cell[keep])
                columns_kept.append(peak[keep])
        elif any(line.strip() and not line.lstrip().startswith("%") for line in handle):
            raise ValueError(f"{path}: matrix entries present despite a zero entry count")
    if read_entries != entries:
        raise ValueError(f"{path}: expected {entries} entries, read {read_entries}")
    row = np.concatenate(rows_kept) if rows_kept else np.empty(0, dtype=np.int32)
    column = np.concatenate(columns_kept) if columns_kept else np.empty(0, dtype=np.int32)
    shape = len(cell_indexes), len(peak_indexes)
    return coo_matrix((np.ones(len(row), dtype=bool), (row, column)), shape=shape).tocsr()


def bh_adjust(pvalues):
    order = np.argsort(pvalues, kind="mergesort")
    adjusted = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1, dtype=np.float64)
    reordered = np.empty_like(adjusted)
    reordered[order] = np.minimum(np.minimum.accumulate(adjusted[::-1])[::-1], 1.0)
    return reordered


def select_peaks(matrix, peaks, labels, args):
    labels = labels.to_numpy()
    groups = sorted(set(labels))
    if len(groups) < 2:
        raise ValueError("At least two cell types must remain after barcode alignment")
    total = matrix.getnnz(axis=0).astype(np.int64)
    tests = []
    summary = []
    for group in groups:
        in_group = labels == group
        n_group = int(in_group.sum())
        n_rest = len(labels) - n_group
        group_open = matrix[in_group, :].getnnz(axis=0).astype(np.int64)
        rest_open = total - group_open
        p = args.pseudocount
        log2fc = np.log2(((group_open + p) / (n_group + 2 * p)) / ((rest_open + p) / (n_rest + 2 * p)))
        odds = ((group_open + p) * (n_rest - rest_open + p)) / ((n_group - group_open + p) * (rest_open + p))
        pvalues = hypergeom.sf(group_open - 1, len(labels), total, n_group)
        pvalues = np.nan_to_num(pvalues, nan=1.0, posinf=1.0, neginf=1.0)
        padj = bh_adjust(pvalues)
        candidates = np.flatnonzero((padj <= args.padj_cutoff) & (log2fc >= args.log2fc_cutoff) & (group_open > 0))
        scores = -np.log10(np.maximum(padj[candidates], 1e-300)) * log2fc[candidates]
        order = np.lexsort((-log2fc[candidates], padj[candidates], -scores))[:args.top_n_per_cluster]
        kept = candidates[order]
        summary.append({"cluster": group, "n_cells": n_group, "candidate_peak_count": len(candidates),
                        "selected_peak_count": len(kept)})
        if len(kept):
            coords = peaks.iloc[kept][["feature_id", "chrom", "start", "end", "width"]].copy()
            coords.insert(0, "cluster", group)
            tests.append(coords.assign(group_open_cells=group_open[kept], rest_open_cells=rest_open[kept],
                                       group_open_fraction=group_open[kept] / n_group,
                                       rest_open_fraction=rest_open[kept] / n_rest,
                                       log2fc=log2fc[kept], odds_ratio=odds[kept], pvalue=pvalues[kept],
                                       padj=padj[kept], score=scores[order]))
        print(f"  {group}: {n_group:,} cells, {len(candidates):,} candidates, {len(kept):,} selected", flush=True)
    if not tests:
        raise ValueError("No peaks passed the differential accessibility thresholds")
    return pd.concat(tests, ignore_index=True), pd.DataFrame(summary)


def chrom_rank(chrom):
    name = re.sub(r"^chr", "", chrom, flags=re.IGNORECASE)
    if name.isdigit():
        return 0, int(name), chrom
    return 1, {"X": 1000, "Y": 1001, "M": 1002, "MT": 1002}.get(name.upper(), 2000), chrom


def write_outputs(output, prefix, selected, summary, alignment, audit):
    by_cluster = selected[["chrom", "start", "end", "cluster"]].drop_duplicates().copy()
    by_cluster["peak_name"] = by_cluster["chrom"] + ":" + by_cluster["start"].astype(str) + "-" + by_cluster["end"].astype(str)
    by_cluster = by_cluster.rename(columns={"start": "chromStart", "end": "chromEnd"})
    bed = selected[["chrom", "start", "end"]].drop_duplicates().copy()
    bed["_rank"] = bed["chrom"].map(chrom_rank)
    bed = bed.sort_values(["_rank", "start", "end"]).drop(columns="_rank")
    audit["selected_unique_peaks"] = len(bed)
    tables = {"peaks.bed": bed, "peaks_by_cluster.tsv": by_cluster, "peak_table.tsv": selected,
              "peak_summary.tsv": summary, "cell_alignment.tsv": alignment}
    input_paths = {Path(path).resolve() for path in audit["sources"].values()}
    for suffix in (*tables, "input_audit.json"):
        if (output / (prefix + suffix)).resolve() in input_paths:
            raise ValueError(f"Output {prefix + suffix} would replace an input file")
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temporary:
        staging = Path(temporary)
        for suffix, table in tables.items():
            table.to_csv(staging / (prefix + suffix), sep="\t", index=False, header=not suffix.endswith(".bed"))
        audit_path = staging / (prefix + "input_audit.json")
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for suffix in tables:
            path = staging / (prefix + suffix)
            path.replace(output / path.name)
        audit_path.replace(output / audit_path.name)
    print(f"  Wrote {len(bed):,} unique peaks to {output / (prefix + 'peaks.bed')}", flush=True)


def run_dataset(dataset, args):
    sample = dataset["sample_name"]
    print(sample, flush=True)
    labels = read_rna_labels(dataset["input_h5ad"], dataset["cluster_key"])
    if dataset["input_format"] == "10x":
        directory = dataset["matrix_dir"]
        matrix_path = mex_file(directory, "matrix.mtx")
        barcode_path = mex_file(directory, "barcodes.tsv")
        peak_path = mex_file(directory, "features.tsv")
        atac = read_barcode_file(barcode_path)
        matched = align_tenx(labels.index, atac)
        peaks, n_features = read_tenx_peaks(peak_path)
    else:
        matrix_path = dataset["atac_counts"]
        barcode_path = dataset["atac_barcodes"]
        peak_path = dataset["atac_peaks"]
        atac = read_barcode_file(barcode_path)
        matched = align_share(labels.index, atac, dataset["barcode_map"],
                              dataset["rna_id_kind"], dataset["matrix_id_kind"])
        peaks, n_features = read_share_peaks(peak_path)

    keep = matched >= 0
    n_matched = int(keep.sum())
    if len(set(matched[keep])) != n_matched:
        raise ValueError("Multiple RNA cells resolve to the same ATAC column")
    if not n_matched:
        raise ValueError("No RNA cells matched ATAC barcodes")
    if not args.allow_partial_overlap and not keep.all():
        preview = ", ".join(labels.index[~keep][:5])
        raise ValueError(f"{len(labels) - n_matched} RNA cells did not match ATAC, e.g. {preview}; "
                         "check barcode identifiers or use --allow-partial-overlap")
    if labels.iloc[keep].nunique() < 2:
        raise ValueError("At least two cell types must remain after barcode alignment")
    print(f"  Matched {n_matched:,} of {len(labels):,} RNA cells to {len(atac):,} ATAC columns", flush=True)
    alignment = pd.DataFrame({"rna_barcode": labels.index,
                              "atac_matrix_barcode": [atac[pos] if pos >= 0 else "" for pos in matched],
                              "cluster": labels.to_numpy(), "status": np.where(keep, "matched", "unmatched")})
    n_source_peaks = len(peaks)
    peaks = eligible_peaks(peaks, args.max_peak_width)
    matrix = read_accessibility(matrix_path, n_features, len(atac), peaks["index"].to_numpy(), matched[keep])
    selected, summary = select_peaks(matrix, peaks, labels.iloc[keep], args)
    audit = {
        "sample_name": sample,
        "input_format": dataset["input_format"],
        "genome": dataset["genome"],
        "cluster_key": dataset["cluster_key"],
        "sources": {"input_h5ad": str(dataset["input_h5ad"]), "atac_counts": str(matrix_path),
                    "atac_barcodes": str(barcode_path), "atac_features_or_peaks": str(peak_path)},
        "rna_cells": len(labels),
        "atac_cells": len(atac),
        "matched_cells": n_matched,
        "unmatched_rna_cells": len(labels) - n_matched,
        "unused_atac_cells": len(atac) - n_matched,
        "allow_partial_overlap": args.allow_partial_overlap,
        "source_peak_count": n_source_peaks,
        "tested_peak_count": len(peaks),
        "method": "one-sided hypergeometric enrichment of nonzero accessibility, BH within each cell type",
        "parameters": {key: getattr(args, key) for key in ("padj_cutoff", "log2fc_cutoff", "top_n_per_cluster",
                                                           "max_peak_width", "pseudocount")},
    }
    if dataset["input_format"] == "share-seq":
        audit["sources"]["barcode_map"] = str(dataset["barcode_map"])
        audit["rna_id_kind"] = dataset["rna_id_kind"]
        audit["matrix_id_kind"] = dataset["matrix_id_kind"]
    write_outputs(args.output_dir, sample + "_sckinetics_da_", selected, summary, alignment, audit)


def main():
    args = parse_args()
    try:
        for dataset in read_datasets(args):
            try:
                run_dataset(dataset, args)
            except (ValueError, OSError) as error:
                raise ValueError(f"{dataset['sample_name']}: {error}") from error
    except (ValueError, OSError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
