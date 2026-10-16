#!/usr/bin/env python3
"""Run the paired multiome simulation and preprocessing stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MANIFESTS = ("datasets_01_21.tsv", "datasets_sparse_25_28.tsv")


def select_ids(value: str, available: set[int]) -> list[int]:
    selected = set()
    for token in value.split(","):
        bounds = token.strip().split("-")
        if len(bounds) == 1:
            selected.add(int(bounds[0]))
        elif len(bounds) == 2:
            low, high = map(int, bounds)
            if low > high:
                raise ValueError(f"Reversed ID range: {token}")
            selected.update(range(low, high + 1))
        else:
            raise ValueError(f"Invalid ID selection: {token}")
    unknown = selected - available
    if unknown:
        raise ValueError(f"IDs are not in the retained manifests: {sorted(unknown)}")
    return sorted(selected)


def load_datasets() -> dict[int, dict]:
    rows = {}
    for filename in MANIFESTS:
        with (HERE / filename).open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                idx = int(row["dataset_id"])
                if idx in rows:
                    raise ValueError(f"Duplicate dataset ID {idx}")
                rows[idx] = row
    return rows


def prefix(environment: str, program: str) -> list[str]:
    if environment == "current":
        return [sys.executable if program == "python" else program]
    return ["conda", "run", "--no-capture-output", "-n", environment, program]


def execute(command: list[str], env: dict, log: Path | None = None) -> None:
    print("+ " + shlex.join(command), flush=True)
    if log is None:
        subprocess.run(command, env=env, check=True)
        return
    with log.open("w") as handle:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def settings(row: dict, args) -> dict[str, str]:
    sparse_task = "atac_p_zero" in row
    defaults = {
        "ATAC_P_ZERO": row.get("atac_p_zero", "0.80"),
        "SCMULTISIM_NUM_CIFS": "40" if sparse_task else
            ("100" if int(row["n_genes"]) >= 5000 else "50"),
        "RUN_CHROMVAR": "0" if sparse_task else "1",
        "SAVE_SIM_RDS": "0" if sparse_task else "1",
        "SAVE_CISTOPIC_OBJECT": "0",
        "N_TOPICS": "20",
        "CISTOPIC_ITERATIONS": "60" if sparse_task else "150",
        "CISTOPIC_CPUS": "8" if sparse_task else "1",
    }
    if args.preset == "queue":
        defaults.update({"SCMULTISIM_NUM_CIFS": "40", "CISTOPIC_ITERATIONS": "60",
                         "CISTOPIC_CPUS": "4", "RUN_CHROMVAR": "0", "SAVE_SIM_RDS": "0"})
    resolved = {key: os.environ.get(key, value) for key, value in defaults.items()}
    # Sparse probabilities are part of the supplied task definition.
    if sparse_task:
        resolved["ATAC_P_ZERO"] = row["atac_p_zero"]
    for key, value in [("N_TOPICS", args.topics),
                       ("CISTOPIC_ITERATIONS", args.iterations),
                       ("CISTOPIC_CPUS", args.cpus)]:
        if value is not None:
            resolved[key] = str(value)
    for key in ("SCMULTISIM_NUM_CIFS", "N_TOPICS", "CISTOPIC_ITERATIONS", "CISTOPIC_CPUS"):
        if int(resolved[key]) < (5 if key == "SCMULTISIM_NUM_CIFS" else 1):
            raise ValueError(f"Invalid positive integer setting: {key}")
    if not 0 <= float(resolved["ATAC_P_ZERO"]) < 1:
        raise ValueError("ATAC_P_ZERO must be in [0, 1)")
    for key in ("RUN_CHROMVAR", "SAVE_SIM_RDS", "SAVE_CISTOPIC_OBJECT"):
        if resolved[key] not in {"0", "1"}:
            raise ValueError(f"{key} must be 0 or 1")
    return resolved


def preflight(args) -> None:
    if (args.r_env != "current" or args.python_env != "current") and not shutil.which("conda"):
        raise RuntimeError("conda is required for named environments")
    if not shutil.which("gzip"):
        raise RuntimeError("gzip is required by the R matrix export")
    r_code = (
        'p <- c("scMultiSim", "Matrix"); '
        'stopifnot(all(vapply(p, requireNamespace, logical(1), quietly=TRUE))); '
        'cat(R.version.string, "\\nscMultiSim:", as.character(packageVersion("scMultiSim")), "\\n"); '
        'p <- c("chromVAR", "SummarizedExperiment", "GenomicRanges", "IRanges"); '
        'print(setNames(vapply(p, requireNamespace, logical(1), quietly=TRUE), p))'
    )
    py_code = (
        'import anndata, matplotlib, numpy, pandas, scipy, sklearn, pycisTopic; '
        'from pycisTopic.cistopic_class import create_cistopic_object; '
        'from pycisTopic.lda_models import run_cgs_models; '
        'from packaging.version import Version as V; '
        'assert V(numpy.__version__) < V("2"), "Install the supplied NumPy bounds"; '
        'assert V("1.10") <= V(scipy.__version__) < V("1.15"), "Install the supplied SciPy bounds"; '
        'assert V("0.10") <= V(anndata.__version__) < V("0.12"), "Install the supplied AnnData bounds"; '
        'print("Python dependency and CGS API checks passed")'
    )
    execute(prefix(args.r_env, "Rscript") + ["-e", r_code], os.environ.copy())
    execute(prefix(args.python_env, "python") + ["-c", py_code], os.environ.copy())


def run_one(row: dict, args) -> None:
    name = row["dataset_name"]
    outdir = args.outdir.resolve() / name
    resolved = settings(row, args)
    env = dict(os.environ, **resolved, MPLBACKEND="Agg")
    r = prefix(args.r_env, "Rscript")
    py = prefix(args.python_env, "python")
    stages = [
        ("simulation", r + [str(HERE / "01_simulate.R"), "--dataset-name", name,
            "--outdir", str(outdir), "--topology", row["topology"], "--cells", row["n_cells"],
            "--genes", row["n_genes"], "--seed", row["seed"]],
         ["intermediate/simulation_manifest.tsv.gz", "intermediate/chromvar_status.tsv.gz",
          "intermediate/spliced_genes_cells.mtx.gz", "intermediate/unspliced_genes_cells.mtx.gz",
          "intermediate/atac_counts_regions_cells.mtx.gz", "intermediate/atac_true_regions_cells.mtx.gz",
          "intermediate/region_to_gene_regions_genes.mtx.gz", "intermediate/region_to_tf_regions_tfs.mtx.gz",
          "intermediate/chromatin_genes_cells.mtx.gz", "intermediate/true_velocity_genes_cells.f32",
          "intermediate/gene_names.tsv", "intermediate/cell_names.tsv", "intermediate/region_names.tsv",
          "intermediate/tf_names.tsv", "intermediate/gene_metadata.tsv.gz",
          "intermediate/peak_metadata.tsv.gz", "intermediate/cell_metadata.tsv.gz"]),
        ("diff_peaks", py + [str(HERE / "02_compute_diff_peaks.py"), "--outdir", str(outdir)],
         ["sckinetics_differential_peaks.bed", "intermediate/sckinetics_diff_peaks.tsv.gz",
          "intermediate/true_differential_peaks.tsv.gz"]),
        ("cistopic", py + [str(HERE / "03_run_cistopic.py"), "--outdir", str(outdir),
            "--topics", resolved["N_TOPICS"], "--iterations", resolved["CISTOPIC_ITERATIONS"],
            "--cpus", resolved["CISTOPIC_CPUS"], "--seed", row["seed"]],
         ["intermediate/cistopic_factors.npz", "intermediate/cistopic_audit.json"]),
        ("build_pair", py + [str(HERE / "04_build_pair.py"), "--outdir", str(outdir),
            "--dataset-name", name, "--topology", row["topology"], "--seed", row["seed"]],
         [f"{name}_rna.h5ad", f"{name}_atac.h5ad", f"{name}_validation_audit.json"]),
    ]
    if args.tf_activity_proxy:
        stages.append(("tf_activity", py + [str(HERE / "05_add_tf_activity.py"),
                      "--atac", str(outdir / f"{name}_atac.h5ad")], [f"{name}_atac.h5ad"]))
    validation = py + [str(HERE / "validate_pair.py"),
        "--rna", str(outdir / f"{name}_rna.h5ad"),
        "--atac", str(outdir / f"{name}_atac.h5ad"),
        "--output", str(outdir / "pair_validation.json")]
    if args.require_chromvar:
        validation.append("--require-chromvar")
    print(f"Dataset {name}: {json.dumps(resolved, sort_keys=True)}", flush=True)
    if args.dry_run:
        for _, command, _ in stages:
            print(shlex.join(command))
        print(shlex.join(validation))
        return

    config = {
        "dataset": row, "settings": resolved, "r_env": args.r_env, "python_env": args.python_env,
        "preset": args.preset, "tf_activity_proxy": args.tf_activity_proxy,
        "require_chromvar": args.require_chromvar,
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(HERE.iterdir()) if p.suffix in {".R", ".py"}},
    }
    config_path = outdir / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError(f"Configuration/code changed for {name}; choose a new --outdir")
    elif outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Untracked existing data in {outdir}; choose a new --outdir")
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    # A completion marker is published only after the final pair validation.
    (outdir / "_SUCCESS").unlink(missing_ok=True)
    upstream_changed = False
    for label, command, products in stages:
        marker = outdir / f".stage_{label}_success"
        complete = marker.exists() and all((outdir / x).is_file() and (outdir / x).stat().st_size for x in products)
        if complete and not upstream_changed:
            print(f"SKIP completed stage: {label}", flush=True)
            continue
        upstream_changed = True
        marker.unlink(missing_ok=True)
        execute(command, env, outdir / "logs" / f"{label}.log")
        if not all((outdir / x).is_file() and (outdir / x).stat().st_size for x in products):
            raise RuntimeError(f"Stage {label} did not produce all required files")
        marker.write_text("completed\n")
    (outdir / "_SUCCESS").unlink(missing_ok=True)
    execute(validation, env, outdir / "logs" / "validate_pair.log")
    export = args.outdir.resolve() / "h5ad"
    export.mkdir(exist_ok=True)
    for modality in ("rna", "atac"):
        source = outdir / f"{name}_{modality}.h5ad"
        link = export / source.name
        if link.is_symlink() and link.resolve() == source.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Refusing to replace an unrelated export: {link}")
        link.symlink_to(Path("..") / name / source.name)
    (outdir / "_SUCCESS").write_text("validated\n")
    print(f"DATASET COMPLETE: {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("output"))
    parser.add_argument("--ids", default="1", help="IDs/ranges, e.g. 1-12,25-28")
    parser.add_argument("--r-env", default=os.environ.get("SCMULTISIM_ENV", "scmulti_velo"))
    parser.add_argument("--python-env", default=os.environ.get("CISTOPIC_ENV", "crak_cistopic"))
    parser.add_argument("--topics", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--preset", choices=("source", "queue"), default="source",
                        help="Original phase/sparse driver defaults, or original queue-launch settings")
    parser.add_argument("--tf-activity-proxy", action="store_true", help="Fill absent TF activity with the labeled source approximation")
    parser.add_argument("--require-chromvar", action="store_true", help="Require actual chromVAR deviations, excluding proxy values")
    parser.add_argument("--preflight", action="store_true", help="Check dependencies and exit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", help="List retained simulation tasks and exit")
    args = parser.parse_args()
    datasets = load_datasets()
    if args.list:
        for idx, row in datasets.items():
            print(f"{idx:02d}\t{row['dataset_name']}")
        return
    ids = select_ids(args.ids, set(datasets))
    if not args.dry_run:
        preflight(args)
        if args.preflight:
            return
    for idx in ids:
        run_one(datasets[idx], args)


if __name__ == "__main__":
    main()
