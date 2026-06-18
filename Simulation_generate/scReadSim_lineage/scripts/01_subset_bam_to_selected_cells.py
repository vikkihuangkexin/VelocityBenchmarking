#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Subset raw embryo BAM to selected barcodes and rewrite QNAME to contain CB prefix.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--raw-bam", default=None, help="Override raw BAM path. Default: manifest bam.raw_bam.")
    p.add_argument("--sample", default=None)
    p.add_argument("--threads", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    import json
    import pysam

    manifest = json.loads(Path(args.manifest).read_text())
    sample = args.sample or manifest["sample"]
    raw_bam = args.raw_bam or manifest.get("bam", {}).get("raw_bam")
    if not raw_bam:
        raise SystemExit("Raw BAM path is missing. Provide --raw-bam or include bam.raw_bam in manifest.json.")
    if not Path(raw_bam).exists():
        raise SystemExit(f"Raw BAM does not exist: {raw_bam}")
    keep_barcodes = manifest["prepared_paths"]["subset_barcodes_tsv"]
    out_cells_only = Path(manifest["bam"]["cells_only"])
    out_cb_qname = Path(manifest["bam"]["cb_in_read_name"])
    out_cells_only.parent.mkdir(parents=True, exist_ok=True)

    with open(keep_barcodes, "rt") as handle:
        keep = {x.strip() for x in handle if x.strip()}
    if not keep:
        raise SystemExit("No barcodes loaded from subset_barcodes.tsv")

    n_total = n_keep = n_missing_cb = 0
    with pysam.AlignmentFile(raw_bam, "rb") as fin, pysam.AlignmentFile(out_cells_only, "wb", template=fin) as fout:
        for read in fin:
            n_total += 1
            try:
                cb = read.get_tag("CB")
            except KeyError:
                n_missing_cb += 1
                continue
            if cb in keep:
                fout.write(read)
                n_keep += 1

    pysam.index(str(out_cells_only), "-@", str(args.threads))

    # rewrite qname: <CB>:<old_qname>
    n_rewritten = 0
    with pysam.AlignmentFile(out_cells_only, "rb") as fin, pysam.AlignmentFile(out_cb_qname, "wb", template=fin) as fout:
        for read in fin:
            try:
                cb = read.get_tag("CB")
            except KeyError:
                continue
            read.query_name = f"{cb}:{read.query_name}"
            fout.write(read)
            n_rewritten += 1
    pysam.index(str(out_cb_qname), "-@", str(args.threads))

    print(json.dumps({
        "sample": sample,
        "raw_bam": raw_bam,
        "n_total_reads": n_total,
        "n_reads_kept": n_keep,
        "n_reads_missing_CB": n_missing_cb,
        "cells_only_bam": str(out_cells_only),
        "cb_in_read_name_bam": str(out_cb_qname),
        "n_reads_rewritten": n_rewritten,
    }, indent=2))


if __name__ == "__main__":
    main()
