#!/usr/bin/env python3
"""
Run scReadSim fast mode from prepared selected-cell inputs.

Key behavior change
-------------------
If the intergene count matrix is empty for the selected cells, this script:
- skips intergene synthetic count generation
- skips intergene BED coordinate generation
- builds the combined BED from gene-only BED files

This is useful for lineage-focused subsets where scReadSim may produce an empty intergenic count matrix and fail in the synthetic-count step.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

try:
    import scReadSim.Utility as Utility
    import scReadSim.GenerateSyntheticCount as GenerateSyntheticCount
    import scReadSim.scRNA_GenerateBAM as scRNA_GenerateBAM
except Exception as e:
    Utility = None
    GenerateSyntheticCount = None
    scRNA_GenerateBAM = None
    IMPORT_ERROR = e
else:
    IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run scReadSim fast mode from prepared inputs with a gene-only fallback.")
    p.add_argument("--manifest", required=True, help="manifest.json from prepare step")
    p.add_argument("--outdir", default=None, help="Output directory. Default: <new_root>/screadsim/<mode_name>")
    p.add_argument("--sample", default=None, help="Override sample prefix in output filenames. Default: manifest sample.")
    p.add_argument("--n-cores", type=int, default=8, help="Number of CPU cores.")
    p.add_argument("--read-len", type=int, default=98, help="Synthetic read length.")
    p.add_argument("--jitter-size", type=int, default=5, help="Jitter size for BAM coordinate generation.")
    p.add_argument("--n-cell-new", type=int, default=None, help="Optional scReadSim n_cell_new.")
    p.add_argument("--total-count-new", type=int, default=None, help="Optional scReadSim total_count_new.")
    p.add_argument("--force-regenerate-features", action="store_true", help="Regenerate feature BEDs even if they already exist.")
    p.add_argument("--force-regenerate-countmat", action="store_true", help="Regenerate real count matrices even if they already exist.")
    p.add_argument("--force-regenerate-synthetic", action="store_true", help="Regenerate synthetic count matrices even if they already exist.")
    p.add_argument("--skip-fastq", action="store_true", help="Stop after combined BED and do not create FASTQ.")
    p.add_argument("--fgbio-jar", default="", help="Optional path to fgbio jar for sequencing error modeling.")
    return p.parse_args()


def require_tools() -> dict:
    samtools_bin = shutil.which("samtools")
    bedtools_bin = shutil.which("bedtools")
    seqtk_bin = shutil.which("seqtk")
    missing = []
    if samtools_bin is None:
        missing.append("samtools")
    if bedtools_bin is None:
        missing.append("bedtools")
    if seqtk_bin is None:
        missing.append("seqtk")
    if missing:
        sys.exit("Missing required executables in PATH: " + ", ".join(missing))
    return {
        "samtools_bin": samtools_bin,
        "bedtools_bin": bedtools_bin,
        "seqtk_bin": seqtk_bin,
        "samtools_dir": os.path.dirname(samtools_bin),
        "bedtools_dir": os.path.dirname(bedtools_bin),
        "seqtk_path": seqtk_bin,
    }


def check_required_files(paths: list[str]) -> None:
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit("Missing required input files:\n" + "\n".join(missing))


def file_has_nonblank_line(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rt") as fh:
        for line in fh:
            if line.strip():
                return True
    return False


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_gene_beds_as_combined(outdir: Path, gene_prefix: str, combined_prefix: str) -> None:
    copied = []
    for suffix in [".read1.bed", ".read2.bed"]:
        src = outdir / f"{gene_prefix}{suffix}"
        dst = outdir / f"{combined_prefix}{suffix}"
        if src.exists():
            safe_copy(src, dst)
            copied.append(str(dst))
    if not copied:
        raise SystemExit(
            f"Gene BED files were not found for gene-only combine fallback. "
            f"Expected at least {outdir / (gene_prefix + '.read1.bed')}"
        )
    print("[Step 6] Gene-only fallback: copied gene BED files to combined BED:")
    for x in copied:
        print("  ", x)


def main() -> None:
    args = parse_args()
    if IMPORT_ERROR is not None:
        sys.exit(f"Failed to import scReadSim: {IMPORT_ERROR}")

    manifest = json.loads(Path(args.manifest).read_text())
    sample = args.sample or manifest["sample"]
    mode_name = manifest.get("mode_name", "fast")
    new_root = Path(manifest["new_root"])

    outdir = Path(args.outdir) if args.outdir else new_root / "screadsim" / mode_name
    outdir.mkdir(parents=True, exist_ok=True)

    tools = require_tools()

    INPUT_BAM = manifest["bam"]["cb_in_read_name"]
    INPUT_BARCODES = manifest["prepared_paths"]["subset_barcodes_tsv"]
    CELLLABEL_FILE = manifest["prepared_paths"]["subset_celllabels_txt"]
    REFERENCE_FASTA = manifest["prepared_paths"]["genome_fa"]
    GENOME_ANNOTATION = manifest["prepared_paths"]["genes_gtf"]
    GENOME_SIZE_FILE = manifest["prepared_paths"]["chrom_sizes"]

    check_required_files([
        INPUT_BAM,
        INPUT_BARCODES,
        CELLLABEL_FILE,
        REFERENCE_FASTA,
        GENOME_ANNOTATION,
        GENOME_SIZE_FILE,
    ])

    READ_LEN = args.read_len
    JITTER_SIZE = args.jitter_size
    N_CORES = args.n_cores
    N_CELL_NEW = args.n_cell_new
    TOTAL_COUNT_NEW = args.total_count_new

    FGBIO_JAR = args.fgbio_jar.strip()
    USE_ERROR_MODEL = bool(FGBIO_JAR) and os.path.exists(FGBIO_JAR)

    UMI_gene_count_mat_filename = f"{sample}.gene.countmatrix"
    UMI_intergene_count_mat_filename = f"{sample}.intergene.countmatrix"

    gene_bed = outdir / "scReadSim.Gene.bed"
    intergene_bed = outdir / "scReadSim.InterGene.bed"

    gene_count_txt = outdir / f"{UMI_gene_count_mat_filename}.txt"
    intergene_count_txt = outdir / f"{UMI_intergene_count_mat_filename}.txt"

    synthetic_countmat_gene = outdir / f"{UMI_gene_count_mat_filename}.scDesign2Simulated.txt"
    synthetic_countmat_intergene = outdir / f"{UMI_intergene_count_mat_filename}.scDesign2Simulated.txt"

    gene_read_bedfile_prename = f"{sample}.gene"
    intergene_read_bedfile_prename = f"{sample}.intergene"
    combined_pre = f"{sample}.combined"
    cell_barcode_out = f"{sample}.synthetic_cell_barcode.txt"
    fastq_pre = f"{sample}.{mode_name}"

    combined_R1 = outdir / f"{fastq_pre}.R1.fastq"
    combined_R2 = outdir / f"{fastq_pre}.R2.fastq"

    run_meta = {
        "manifest": str(Path(args.manifest).resolve()),
        "outdir": str(outdir),
        "sample": sample,
        "INPUT_BAM": INPUT_BAM,
        "INPUT_BARCODES": INPUT_BARCODES,
        "CELLLABEL_FILE": CELLLABEL_FILE,
        "REFERENCE_FASTA": REFERENCE_FASTA,
        "GENOME_ANNOTATION": GENOME_ANNOTATION,
        "GENOME_SIZE_FILE": GENOME_SIZE_FILE,
        "READ_LEN": READ_LEN,
        "JITTER_SIZE": JITTER_SIZE,
        "N_CORES": N_CORES,
        "N_CELL_NEW": N_CELL_NEW,
        "TOTAL_COUNT_NEW": TOTAL_COUNT_NEW,
        "USE_ERROR_MODEL": USE_ERROR_MODEL,
        "tools": tools,
    }
    (outdir / "run_config.json").write_text(json.dumps(run_meta, indent=2))

    print("===== Clean scReadSim fast run (patched intergene-safe) =====")
    print(json.dumps(run_meta, indent=2))

    # Step 1-2: feature sets
    if args.force_regenerate_features or not (gene_bed.exists() and intergene_bed.exists()):
        print("\n[Step 1-2] Create feature sets")
        Utility.scRNA_CreateFeatureSets(
            INPUT_bamfile=INPUT_BAM,
            samtools_directory=tools["samtools_dir"],
            bedtools_directory=tools["bedtools_dir"],
            outdirectory=str(outdir),
            genome_annotation=GENOME_ANNOTATION,
            genome_size_file=GENOME_SIZE_FILE,
        )
    else:
        print("\n[Step 1-2] Reuse feature sets")

    if not gene_bed.exists():
        sys.exit(f"Missing gene BED after Step 1-2: {gene_bed}")
    if not intergene_bed.exists():
        print(f"[WARN] Missing intergene BED after Step 1-2: {intergene_bed}")

    # Step 3: real count matrices
    def ensure_count_matrix(count_name: str, bed_file: Path, out_txt: Path) -> bool:
        if out_txt.exists() and not args.force_regenerate_countmat:
            print(f"[Step 3] Reuse count matrix: {out_txt}")
            return file_has_nonblank_line(out_txt)
        print(f"[Step 3] Generate count matrix: {count_name}")
        Utility.scRNA_bam2countmat_paral(
            cells_barcode_file=INPUT_BARCODES,
            bed_file=str(bed_file),
            INPUT_bamfile=INPUT_BAM,
            outdirectory=str(outdir),
            count_mat_filename=count_name,
            UMI_modeling=True,
            UMI_tag="UB:Z",
            n_cores=N_CORES,
        )
        if not out_txt.exists():
            sys.exit(f"Failed to generate count matrix: {out_txt}")
        return file_has_nonblank_line(out_txt)

    print("\n[Step 3] Prepare real count matrices")
    gene_has_data = ensure_count_matrix(UMI_gene_count_mat_filename, gene_bed, gene_count_txt)
    intergene_has_data = False
    if intergene_bed.exists():
        intergene_has_data = ensure_count_matrix(UMI_intergene_count_mat_filename, intergene_bed, intergene_count_txt)
    else:
        print("[Step 3] Skip intergene count matrix because intergene BED is missing")

    if not gene_has_data:
        sys.exit(f"Gene count matrix is empty: {gene_count_txt}")
    if not intergene_has_data:
        print(f"[WARN] Intergene count matrix is empty or missing: {intergene_count_txt}")
        print("[WARN] Proceeding in gene-only mode for this subset.")

    # Step 4: synthetic count matrices
    def ensure_synthetic_count(count_name: str, out_txt: Path) -> None:
        if out_txt.exists() and not args.force_regenerate_synthetic:
            print(f"[Step 4] Reuse synthetic count matrix: {out_txt}")
            return
        print(f"[Step 4] Generate synthetic count matrix: {count_name}")
        GenerateSyntheticCount.scRNA_GenerateSyntheticCount(
            count_mat_filename=count_name,
            directory=str(outdir),
            outdirectory=str(outdir),
            celllabel_file=CELLLABEL_FILE,
            n_cell_new=N_CELL_NEW,
            total_count_new=TOTAL_COUNT_NEW,
            n_cores=N_CORES,
        )
        if not out_txt.exists():
            sys.exit(f"Failed to generate synthetic count matrix: {out_txt}")

    print("\n[Step 4] Prepare synthetic count matrices")
    ensure_synthetic_count(UMI_gene_count_mat_filename, synthetic_countmat_gene)
    if intergene_has_data:
        ensure_synthetic_count(UMI_intergene_count_mat_filename, synthetic_countmat_intergene)
    else:
        print("[Step 4] Skip intergene synthetic count generation (gene-only mode)")

    # Step 5: BED coordinates
    gene_read1 = outdir / f"{gene_read_bedfile_prename}.read1.bed"
    intergene_read1 = outdir / f"{intergene_read_bedfile_prename}.read1.bed"

    if not gene_read1.exists():
        print("\n[Step 5A] Generate gene BED coordinates")
        scRNA_GenerateBAM.scRNA_GenerateBAMCoord(
            bed_file=str(gene_bed),
            UMI_count_mat_file=str(synthetic_countmat_gene),
            synthetic_cell_label_file=CELLLABEL_FILE,
            read_bedfile_prename=gene_read_bedfile_prename,
            INPUT_bamfile=INPUT_BAM,
            outdirectory=str(outdir),
            OUTPUT_cells_barcode_file=cell_barcode_out,
            jitter_size=JITTER_SIZE,
            read_len=READ_LEN,
        )
    else:
        print("\n[Step 5A] Reuse gene BED coordinates")

    if intergene_has_data:
        if not intergene_read1.exists():
            print("\n[Step 5B] Generate intergene BED coordinates")
            scRNA_GenerateBAM.scRNA_GenerateBAMCoord(
                bed_file=str(intergene_bed),
                UMI_count_mat_file=str(synthetic_countmat_intergene),
                synthetic_cell_label_file=CELLLABEL_FILE,
                read_bedfile_prename=intergene_read_bedfile_prename,
                INPUT_bamfile=INPUT_BAM,
                outdirectory=str(outdir),
                OUTPUT_cells_barcode_file=cell_barcode_out,
                jitter_size=JITTER_SIZE,
                read_len=READ_LEN,
            )
        else:
            print("\n[Step 5B] Reuse intergene BED coordinates")
    else:
        print("\n[Step 5B] Skip intergene BED coordinates (gene-only mode)")

    # Step 6: combine BED
    combined_read1 = outdir / f"{combined_pre}.read1.bed"
    if not combined_read1.exists():
        print("\n[Step 6] Build combined BED")
        if intergene_has_data and intergene_read1.exists():
            scRNA_GenerateBAM.scRNA_CombineBED(
                outdirectory=str(outdir),
                gene_read_bedfile_prename=gene_read_bedfile_prename,
                intergene_read_bedfile_prename=intergene_read_bedfile_prename,
                BED_filename_combined_pre=combined_pre,
            )
        else:
            copy_gene_beds_as_combined(outdir, gene_read_bedfile_prename, combined_pre)
    else:
        print("\n[Step 6] Reuse combined BED")

    # Step 7: BED -> FASTQ
    if args.skip_fastq:
        print("\n[Step 7] Skip FASTQ by request (--skip-fastq)")
        return

    if not (combined_R1.exists() and combined_R2.exists()):
        print("\n[Step 7] BED -> FASTQ")
        scRNA_GenerateBAM.scRNA_BED2FASTQ(
            bedtools_directory=tools["bedtools_dir"],
            seqtk_directory=tools["seqtk_path"],
            referenceGenome_file=REFERENCE_FASTA,
            outdirectory=str(outdir),
            BED_filename_combined=combined_pre,
            synthetic_fastq_prename=fastq_pre,
        )
    else:
        print("\n[Step 7] Reuse FASTQ")

    # Step 8: sequencing errors (optional)
    if USE_ERROR_MODEL:
        print("\n[Step 8] Add sequencing errors with fgbio")
        scRNA_GenerateBAM.scRNA_ErrorBase(
            fgbio_jarfile=FGBIO_JAR,
            INPUT_bamfile=INPUT_BAM,
            referenceGenome_file=REFERENCE_FASTA,
            outdirectory=str(outdir),
            synthetic_fastq_prename=fastq_pre,
        )
        print("[Step 8] Error-included FASTQ generated.")
    else:
        print("\n[Step 8] Skip scRNA_ErrorBase because no usable --fgbio-jar was provided.")

    summary = {
        "sample": sample,
        "outdir": str(outdir),
        "gene_has_data": gene_has_data,
        "intergene_has_data": intergene_has_data,
        "mode": "gene_only" if not intergene_has_data else "gene_plus_intergene",
        "outputs": {
            "synthetic_barcodes": str(outdir / cell_barcode_out),
            "gene_synthetic_count": str(synthetic_countmat_gene),
            "intergene_synthetic_count": str(synthetic_countmat_intergene),
            "fastq_r1": str(combined_R1),
            "fastq_r2": str(combined_R2),
        }
    }
    (outdir / "run_summary.patched.json").write_text(json.dumps(summary, indent=2))
    print("\nPatched scReadSim fast pipeline finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
