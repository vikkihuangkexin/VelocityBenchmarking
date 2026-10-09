#!/usr/bin/env python3
"""
DynaVelo velocity analysis pipeline for VelocityBenchmarking.

DynaVelo is a latent neural ODE model that learns multiomic (RNA expression + TF
motif accessibility) velocities of single cells, infers dynamic cell-state-specific
gene regulatory networks, and performs in-silico perturbations.

Real and simulated datasets are processed by the same script and switched with the
--simulate flag:

* real data: reads the H5AD, runs the standard scVelo pipeline to obtain the RNA
  velocity prior that DynaVelo consumes, then trains DynaVelo.
* simulated data: labels the cells with the `milestone` cluster (the input is used
  as-is; no embedding is remapped and no extra filtering is applied).

DynaVelo needs two aligned modalities; this wrapper accepts either:

* an ATAC H5AD via --atac-input whose `obsm['chromVAR']` holds the TF motif
  accessibility z-scores (the upstream layout), or
* the same `obsm['chromVAR']` stored inside the RNA H5AD itself.

Installation:
    pip install dynamo-release scanpy scvelo pyarrow simplejson GPUtil pynvml
    git clone https://github.com/karbalayghareh/DynaVelo.git
    # The upstream model itself (dynavelo.models) is installed from that clone.

Usage:
    python DynaVelo.py --rna-input rna.h5ad --atac-input atac.h5ad --output-dir ./output --cluster-key celltype
    python DynaVelo.py --rna-input sim.h5ad --atac-input sim_atac.h5ad --output-dir ./output --cluster-key milestone --simulate
    python DynaVelo.py --metadata-file datasets.csv --output-dir ./output
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
from pathlib import Path
from typing import Optional

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvelo as scv

def seed_everything(seed: int) -> None:
    """Seed the random number generators used by DynaVelo."""
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_resources() -> None:
    """Release memory held by the last dataset before moving on."""
    gc.collect()
    plt.close("all")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def derive_output_stem(input_path: Path) -> str:
    """Strip the benchmark `_dataset` suffix from an input file stem."""
    stem = input_path.stem
    if stem.endswith("_dataset"):
        return stem[:-8]
    return stem


def parse_bool(value) -> bool:
    """Convert common textual/boolean values into a real bool."""
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False

    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def detect_separator(path: Path) -> str:
    """Infer the column separator of a metadata table from its extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    if suffix in {".tsv", ".txt"}:
        return "\t"

    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    return "\t" if "\t" in first_line else ","


def load_metadata_file(metadata_path: Path) -> pd.DataFrame:
    """Read and validate the batch metadata table."""
    df = pd.read_csv(metadata_path, sep=detect_separator(metadata_path))

    if "rna_path" not in df.columns and "file_path" not in df.columns:
        raise ValueError("The metadata table needs an 'rna_path' or 'file_path' column.")
    if "rna_path" not in df.columns:
        df["rna_path"] = df["file_path"]
    if "atac_path" not in df.columns:
        df["atac_path"] = ""
    if "dataset_name" not in df.columns:
        df["dataset_name"] = df["rna_path"].map(lambda value: Path(str(value)).stem)
    if "cluster_key" not in df.columns:
        df["cluster_key"] = "celltype"
    if "simulate" not in df.columns:
        df["simulate"] = False

    df["dataset_name"] = df["dataset_name"].astype(str)
    df["rna_path"] = df["rna_path"].astype(str)
    df["atac_path"] = df["atac_path"].fillna("").astype(str)
    df["cluster_key"] = df["cluster_key"].astype(str)
    df["simulate"] = df["simulate"].map(parse_bool)

    return df


def resolve_cluster_key(adata, cluster_key: str, simulate: bool) -> str:
    """Return a usable ``obs`` column for the DynaVelo cell-type weights."""
    if cluster_key in adata.obs.columns:
        return cluster_key

    for fallback in ("celltype", "cell_type", "milestone", "clusters"):
        if fallback in adata.obs.columns:
            print(f"  Column '{cluster_key}' not found; using '{fallback}' instead.")
            return fallback

    label = "milestone" if simulate else "cluster"
    print(f"  No cluster column found; writing a constant '{label}' label.")
    adata.obs[label] = label
    return label


def read_chromvar(rna_path: Path, atac_path: Optional[Path]):
    """Load the TF motif accessibility matrix required by DynaVelo."""
    if atac_path is not None and str(atac_path) and Path(atac_path).exists():
        adata_atac = ad.read_h5ad(atac_path)
        if "chromVAR" not in adata_atac.obsm:
            raise ValueError(f"ATAC file '{atac_path}' has no obsm['chromVAR'].")
        return adata_atac.obsm["chromVAR"].copy(), adata_atac.obs.copy()

    adata_rna = ad.read_h5ad(rna_path)
    if "chromVAR" in adata_rna.obsm:
        return adata_rna.obsm["chromVAR"].copy(), adata_rna.obs.copy()

    raise ValueError(
        f"No TF motif accessibility found: neither '{atac_path}' nor obsm['chromVAR'] in '{rna_path}'."
    )


def determine_preprocessing_params(adata) -> tuple[int, int]:
    """Choose safe PCA / neighbourhood sizes for small datasets."""
    n_pcs = max(1, min(30, adata.n_obs - 1, adata.n_vars - 1))
    n_neighbors = max(1, min(30, adata.n_obs - 1))
    return n_pcs, n_neighbors


def _noop_makedirs(*args, **kwargs):
    return None


class _suppress_makedirs:
    """Temporarily make ``os.makedirs`` a no-op.

    ``DynaVelo.__init__`` calls ``os.makedirs('../checkpoints/<dataset>/<sample>/')``
    relative to the current working directory, which would create directories outside
    the per-dataset output folder. Model construction is wrapped in this context
    manager and ``model.train_dir`` / ``model.ckpt_path`` are then set explicitly to
    keep every artefact inside the output directory.
    """

    def __enter__(self):
        self._original = os.makedirs
        os.makedirs = _noop_makedirs
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        os.makedirs = self._original
        return False


def preprocess_adata(
    adata,
    cluster_key: str,
    simulate: bool,
    n_top_genes: int = 2000,
    min_shared_counts: int = 20,
    preprocess_input: bool = False,
):
    """Prepare the RNA object and its cell-label column.

    Mirrors the per-dataset setup of ``DynaVelo_simdata_multigpu.py``: only unique
    names and the cluster labels are handled here, and ``obsm`` embeddings are never
    remapped. The input must already carry the ``Ms``/``Mu`` moment layers; the scVelo
    prior is computed afterwards in :func:`run_dynavelo_analysis`.

    When ``preprocess_input`` is set the wrapper additionally runs the
    ``filter_and_normalize`` + ``pca`` + ``moments`` sequence, which must stay disabled
    for the prepared multiome input of the original driver.
    """
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    if simulate:
        if cluster_key and cluster_key in adata.obs:
            adata.obs[cluster_key] = "milestone"
        adata.obs["celltype"] = "milestone"

    if preprocess_input:
        scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts, n_top_genes=n_top_genes)
        n_pcs, n_neighbors = determine_preprocessing_params(adata)
        sc.tl.pca(adata, n_comps=n_pcs)
        sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
        scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)

    active_cluster_key = resolve_cluster_key(adata, cluster_key, simulate)
    if active_cluster_key != "celltype":
        adata.obs["celltype"] = adata.obs[active_cluster_key].astype(str)

    return adata, active_cluster_key


def build_dataloaders(adata_rna, adata_chromvar, batch_size: int, test_fraction: float = 0.1):
    """Build the train / test / full PyTorch dataloaders used by DynaVelo."""
    import torch
    from torch.utils.data import DataLoader

    from dynavelo.models import MultiomeDataset

    dataset = MultiomeDataset(adata_rna, adata_chromvar)

    n_test = int(test_fraction * len(dataset))
    idx_random = np.random.permutation(len(dataset))
    idx_test = idx_random[:n_test]
    idx_train = idx_random[n_test:]

    dataset_train = MultiomeDataset(adata_rna[idx_train], adata_chromvar[idx_train])
    dataset_test = MultiomeDataset(adata_rna[idx_test], adata_chromvar[idx_test])

    pin_memory = torch.cuda.is_available()
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=pin_memory
    )
    dataloader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=pin_memory
    )

    return dataloader_train, dataloader_test, dataloader


def write_velocity_layer(adata_rna_pred):
    """Expose the DynaVelo RNA velocity as ``layers['velocity']``.

    DynaVelo writes its predicted RNA velocity to ``layers['vx_pred_mean']`` (the
    posterior mean over ``n_samples`` draws). That is the quantity the benchmark
    compares, so it is copied to the conventional ``layers['velocity']`` while the
    scVelo velocity that DynaVelo consumed as input is preserved separately.
    """
    if "vx_pred_mean" not in adata_rna_pred.layers:
        raise RuntimeError(
            "DynaVelo did not produce layers['vx_pred_mean'], so layers['velocity'] cannot be exported."
        )

    if "velocity" in adata_rna_pred.layers and "velocity_scvelo_input" not in adata_rna_pred.layers:
        adata_rna_pred.layers["velocity_scvelo_input"] = adata_rna_pred.layers["velocity"].copy()

    adata_rna_pred.layers["velocity"] = np.asarray(adata_rna_pred.layers["vx_pred_mean"])
    return adata_rna_pred


def run_dynavelo_analysis(
    rna_input: str | Path,
    output_dir: str | Path,
    cluster_key: str = "celltype",
    atac_input: Optional[str | Path] = None,
    dataset_name: Optional[str] = None,
    batch_size: int = 32,
    max_epoch: int = 200,
    n_samples: int = 20,
    learning_rate: float = 1e-3,
    test_fraction: float = 0.1,
    simulate: bool = False,
    n_top_genes: int = 2000,
    min_shared_counts: int = 20,
    preprocess_input: bool = False,
    device: Optional[str] = None,
    overwrite: bool = False,
    seed: int = 2024,
) -> Path:
    """Train DynaVelo on one dataset and export the predicted velocity."""
    import torch
    import torch.optim as optim

    from dynavelo.models import DynaVelo

    rna_input = Path(rna_input)
    output_dir = Path(output_dir)
    if not rna_input.exists():
        raise FileNotFoundError(f"Input file not found: {rna_input}")

    if dataset_name is None:
        dataset_name = derive_output_stem(rna_input)

    dataset_output_dir = output_dir / str(dataset_name)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Native DynaVelo file names, preserved from code/DynaVelo_simdata_multigpu.py.
    output_h5ad = dataset_output_dir / f"{dataset_name}_adata_rna_pred_DynaVelo.h5ad"

    if output_h5ad.exists() and not overwrite:
        print(f"Skipping existing output: {output_h5ad}")
        return output_h5ad

    seed_everything(seed)

    adata_rna = None
    adata_atac = None
    adata_rna_pred = None

    try:
        print(f"\nProcessing: {rna_input.name}")

        print("  Loading modalities...")
        chromvar_X, chromvar_obs = read_chromvar(rna_input, atac_input)

        adata_rna = sc.read(rna_input)

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device)
        print(f"  Device: {device_obj}")

        print("  Preparing RNA object...")
        adata_rna, active_cluster_key = preprocess_adata(
            adata_rna,
            cluster_key=cluster_key,
            simulate=simulate,
            n_top_genes=n_top_genes,
            min_shared_counts=min_shared_counts,
            preprocess_input=preprocess_input,
        )

        print("  Computing the scVelo prior...")
        scv.tl.recover_dynamics(adata_rna)
        scv.pp.neighbors(adata_rna, n_neighbors=30, n_pcs=30)
        scv.tl.velocity(adata_rna, mode="dynamical")
        scv.tl.velocity_graph(adata_rna)
        scv.tl.latent_time(adata_rna)

        if not np.array_equal(np.asarray(adata_rna.obs_names), np.asarray(chromvar_obs.index)):
            raise ValueError("The RNA and chromVAR objects do not share the same cell order.")

        adata_chromvar = ad.AnnData(X=np.asarray(chromvar_X), obs=chromvar_obs)
        if "celltype" not in adata_chromvar.obs and "cell_type" in adata_chromvar.obs:
            adata_chromvar.obs["celltype"] = adata_chromvar.obs["cell_type"].astype(str)
        if "celltype" not in adata_chromvar.obs:
            adata_chromvar.obs["celltype"] = adata_rna.obs[active_cluster_key].astype(str).values

        print("  Building dataloaders...")
        dataloader_train, dataloader_test, dataloader = build_dataloaders(
            adata_rna, adata_chromvar, batch_size=batch_size, test_fraction=test_fraction
        )

        print("  Training DynaVelo...")
        # DynaVelo.__init__ calls os.makedirs('../checkpoints/<dataset>/<sample>/') relative to
        # the current working directory; suppress it so nothing is written outside the
        # per-dataset output dir, then point train_dir / ckpt_path inside it below.
        with _suppress_makedirs():
            model = DynaVelo(
                x_dim=adata_rna.shape[1], y_dim=adata_chromvar.shape[1], device=device_obj
            ).to(device_obj)

        # DynaVelo writes checkpoints and training logs through paths relative to the
        # working directory; redirect both into the per-dataset output folder.
        model.train_dir = str(dataset_output_dir / "checkpoints") + os.sep
        os.makedirs(model.train_dir, exist_ok=True)
        model.ckpt_path = model.train_dir + model.model_suffix + ".pth"

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.fit(dataloader_train, dataloader_test, optimizer, max_epoch=max_epoch)

        print("  Predicting velocities and latent times...")
        model.mode = "evaluation-sample"
        adata_rna_pred, adata_atac_pred = model.evaluate(
            adata_rna, adata_chromvar, dataloader, n_samples=n_samples
        )

        adata_rna_pred = write_velocity_layer(adata_rna_pred)
        adata_rna_pred.uns["dynavelo_run"] = {
            "dataset_name": str(dataset_name),
            "rna_input": str(rna_input.resolve()),
            "atac_input": str(Path(atac_input).resolve()) if atac_input else "",
            "cluster_key": active_cluster_key,
            "simulate": bool(simulate),
            "batch_size": int(batch_size),
            "max_epoch": int(max_epoch),
            "n_samples": int(n_samples),
            "seed": int(seed),
            "output_path": str(output_h5ad.resolve()),
        }

        adata_rna_pred.write_h5ad(output_h5ad, compression="lzf")
        adata_atac_pred.write_h5ad(
            dataset_output_dir / f"{dataset_name}_adata_atac_pred_DynaVelo.h5ad", compression="lzf"
        )
        shutil.copyfile(output_h5ad, dataset_output_dir / "rc.h5ad")
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    finally:
        del adata_rna
        del adata_atac
        del adata_rna_pred
        cleanup_resources()


def run_batch_dynavelo(
    metadata_file: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    seed: int = 2024,
) -> list[Path]:
    """Run DynaVelo over every entry of a metadata table."""
    metadata_file = Path(metadata_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata_file(metadata_file)
    outputs: list[Path] = []
    print(f"Batch mode: {len(metadata_df)} datasets")

    for _, row in metadata_df.iterrows():
        rna_path = Path(row["rna_path"])
        if not rna_path.exists():
            print(f"Skipping missing file: {rna_path}")
            continue

        try:
            output_path = run_dynavelo_analysis(
                rna_input=rna_path,
                output_dir=output_dir,
                cluster_key=row["cluster_key"],
                atac_input=row["atac_path"] or None,
                dataset_name=row["dataset_name"],
                simulate=bool(row["simulate"]),
                overwrite=overwrite,
                seed=seed,
            )
            outputs.append(output_path)
        except Exception as exc:
            print(f"Failed: {row['dataset_name']}: {exc}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DynaVelo multi-omic velocity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--rna-input", "--input", dest="rna_input", help="RNA AnnData file")
    input_group.add_argument("--metadata-file", help="Metadata CSV/TSV file for batch processing")

    parser.add_argument("--atac-input", default=None, help="ATAC AnnData file carrying obsm['chromVAR']")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset-name", default=None, help="Dataset folder name in single-file mode")
    parser.add_argument(
        "--cluster-key",
        default="celltype",
        help="Column name in adata.obs used for the cell-type weights; use milestone for simulated data",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--max-epoch", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--n-samples", type=int, default=20, help="Posterior samples drawn at evaluation")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Fraction of cells held out for testing")
    parser.add_argument(
        "--preprocess-input",
        action="store_true",
        default=False,
        help=(
            "Opt-in: run filter_and_normalize + PCA + moments before the scVelo prior. "
            "Disabled by default because the original input already carries Ms/Mu."
        ),
    )
    parser.add_argument(
        "--n-top-genes", type=int, default=2000,
        help="Only used with --preprocess-input: number of highly variable genes",
    )
    parser.add_argument(
        "--min-shared-counts", type=int, default=20,
        help="Only used with --preprocess-input: minimum shared counts filter",
    )
    parser.add_argument("--device", default=None, help="Torch device, for example cuda:0 or cpu")
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Simulated-data branch: labels the cells milestone (no obsm remap and no extra filtering)",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_dynavelo(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            seed=args.seed,
        )

    return run_dynavelo_analysis(
        rna_input=args.rna_input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        atac_input=args.atac_input,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        n_samples=args.n_samples,
        learning_rate=args.learning_rate,
        test_fraction=args.test_fraction,
        simulate=args.simulate,
        n_top_genes=args.n_top_genes,
        min_shared_counts=args.min_shared_counts,
        preprocess_input=args.preprocess_input,
        device=args.device,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
