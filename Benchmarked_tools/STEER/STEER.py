#!/usr/bin/env python3
"""
STEER spatial RNA velocity analysis pipeline for VelocityBenchmarking.

STEER combines a spatial graph, an expression-expert prior and a graph neural
network to estimate a spatial RNA velocity. The workflow runs in two stages:
the first trains the model and infers the expert-based velocity direction, the
second re-trains on the selected velocity genes and builds the final velocity
graph. It needs the spliced/unspliced layers and the spatial coordinates.

Installation:
    See the repository Dockerfile, which installs the upstream STEER source plus
    the pinned PyTorch (cu124), PyTorch Geometric and R/mclust stack:
        git clone https://github.com/lzygenomics/STEER.git /opt/STEER
        pip install -r requirements.txt && pip install --no-deps .

Usage:
    python STEER.py --input spatial.h5ad --output-dir ./output --cluster-key celltype
    python STEER.py --metadata-file datasets.csv --output-dir ./output
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional

# numba needs a writable cache directory; the path stays configurable instead of
# relying on the upstream fixed /tmp/numba_cache.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import torch
import torch.backends.cudnn as cudnn
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

DEFAULT_SPLICED_LAYER = "spliced"
DEFAULT_UNSPLICED_LAYER = "unspliced"

# STEER keeps the predicted spliced velocity in 'pred_vs_norm' (produced by
# steer.utils.normalize_l2_anndata and used as the vkey of steer.velocity_graph).
NATIVE_VELOCITY_KEY = "pred_vs_norm"


def import_steer():
    """Import the installed ``steer`` package (lazy, so --help works without it).

    On case-insensitive file systems the sibling ``STEER.py`` can be picked up as
    the ``steer`` module, so the script directory is temporarily removed from
    ``sys.path`` while the real package is imported.
    """
    script_dir = str(Path(__file__).resolve().parent)
    removed_path = False
    if script_dir in sys.path:
        sys.path.remove(script_dir)
        removed_path = True

    local_module = None
    restore_local_module = __name__ == "STEER" and "STEER" in sys.modules
    if restore_local_module:
        local_module = sys.modules.pop("STEER")

    try:
        import steer  # noqa: PLC0415 - deliberately lazy
        from steer.prior.prior import PriorInferenceManager  # noqa: PLC0415
    finally:
        if restore_local_module and local_module is not None:
            sys.modules["STEER"] = local_module
        if removed_path:
            sys.path.insert(0, script_dir)

    return steer, PriorInferenceManager


def seed_everything(seed: int) -> None:
    """Seed the random number generators used by STEER."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def cleanup_resources() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    required_columns = ["dataset_name", "file_path"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "cluster_key" not in df.columns:
        df["cluster_key"] = "celltype"
    if "spatial_key" not in df.columns:
        df["spatial_key"] = "spatial"

    df["dataset_name"] = df["dataset_name"].astype(str)
    df["file_path"] = df["file_path"].astype(str)
    df["cluster_key"] = df["cluster_key"].fillna("celltype").astype(str)
    df["spatial_key"] = df["spatial_key"].fillna("spatial").astype(str)

    return df


# ---------------------------------------------------------------------------
# Graph / dataset helpers (unchanged upstream logic)
# ---------------------------------------------------------------------------
def us_moments(adata, npc=50, n_neighbor=100, neighbor_metric="cosine"):
    """Smooth the spliced/unspliced counts with a joint PCA neighbourhood graph."""
    spliced = adata.layers["spliced"].toarray() if hasattr(adata.layers["spliced"], "toarray") else adata.layers["spliced"]
    unspliced = adata.layers["unspliced"].toarray() if hasattr(adata.layers["unspliced"], "toarray") else adata.layers["unspliced"]

    # Combine the two layers along the feature axis before the joint PCA.
    combined = np.concatenate([spliced, unspliced], axis=1)

    combined_adata = sc.AnnData(X=combined)
    sc.tl.pca(combined_adata, n_comps=npc)

    adata.obsm["X_pca_combined"] = combined_adata.obsm["X_pca"]
    sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=n_neighbor, use_rep="X_pca_combined", metric=neighbor_metric)
    scv.pp.moments(adata, n_pcs=npc, n_neighbors=n_neighbor, use_rep="X_pca_combined")
    return adata


def convert_anndata(data):
    """Flatten the AnnData moments into the long table consumed by the datasets."""
    n_cells, n_genes = data.n_obs, data.n_vars

    cell_ids = data.obs_names.to_numpy()
    gene_names = data.var_names.to_numpy()

    Mu = data.layers["Mu"]
    Ms = data.layers["Ms"]
    orig_u = data.layers["unspliced"]
    orig_s = data.layers["spliced"]

    if issparse(Mu):
        Mu = Mu.toarray()
    if issparse(Ms):
        Ms = Ms.toarray()
    if issparse(orig_u):
        orig_u = orig_u.toarray()
    if issparse(orig_s):
        orig_s = orig_s.toarray()

    combined_df = pd.DataFrame(
        {
            "cellID": np.tile(cell_ids, n_genes),
            "gene_name": np.repeat(gene_names, n_cells),
            "unsplice": np.asarray(Mu).T.reshape(-1),
            "splice": np.asarray(Ms).T.reshape(-1),
            "orig_unsplice": np.asarray(orig_u).T.reshape(-1),
            "orig_splice": np.asarray(orig_s).T.reshape(-1),
        }
    )

    combined_df["Mu"] = combined_df["unsplice"]
    combined_df["Ms"] = combined_df["splice"]

    return combined_df


def construct_latent_adj(adata, npc=50, NUM_AD_NEIGH=30, graph_adj="connectivities"):
    """Binary adjacency built from the latent embedding graph."""
    sc.pp.neighbors(adata, n_neighbors=NUM_AD_NEIGH, use_rep="X_pre_embed")

    binary_connectivities = adata.obsp[graph_adj].copy()
    binary_connectivities.data = np.ones_like(binary_connectivities.data)
    binary_connectivities.setdiag(1)
    binary_connectivities = binary_connectivities.astype(int)

    adjacency_matrix = binary_connectivities.toarray()
    return adata, adjacency_matrix


def construct_momented_adj(adata, npc=50, NUM_AD_NEIGH=30, neighbor_metric="cosine", graph_adj="connectivities", use_us=True):
    """Binary adjacency built from the moments (or Ms) graph."""
    if use_us:
        Mu = adata.layers["Mu"]
        Ms = adata.layers["Ms"]
        concatenated_data = np.concatenate((Mu, Ms), axis=1)

        combined_adata = sc.AnnData(X=concatenated_data)
        sc.tl.pca(combined_adata, n_comps=npc)
        adata.obsm["X_pca_moments"] = combined_adata.obsm["X_pca"]
        sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=NUM_AD_NEIGH, use_rep="X_pca_moments", metric=neighbor_metric)
    else:
        sc.tl.pca(adata, n_comps=npc, layer="Ms")
        sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=NUM_AD_NEIGH, use_rep="X_pca", metric=neighbor_metric)

    binary_connectivities = adata.obsp[graph_adj].copy()
    binary_connectivities.data = np.ones_like(binary_connectivities.data)
    binary_connectivities.setdiag(1)
    binary_connectivities = binary_connectivities.astype(int)

    adjacency_matrix = binary_connectivities.toarray()
    return adata, adjacency_matrix


def preprocess_anndata_spatial(
    adata,
    npc=30,
    NUM_AD_NEIGH=30,
    SMOOTH_NEIGH=100,
    use_us=True,
    neighbor_metric="cosine",
    graph_adj="connectivities",
    latent_graph=False,
    moments_adj=True,
    smooth=True,
    use_spatial=True,
    spatial_neighbors=8,
    spatial_key="spatial",
    combine_mode="union",
    spatial_first=True,
):
    """Smooth the counts, build the expression graph and merge it with the spatial graph."""
    adata.obs.index.name = "cellID"
    adata.var.index.name = "gene_name"

    if smooth:
        if use_us:
            adata = us_moments(adata=adata, npc=npc, n_neighbor=SMOOTH_NEIGH, neighbor_metric=neighbor_metric)
        else:
            sc.pp.pca(adata, n_comps=npc, use_highly_variable=False)
            sc.pp.neighbors(adata, n_pcs=npc, n_neighbors=SMOOTH_NEIGH, metric=neighbor_metric)
            scv.pp.moments(adata, n_pcs=npc, n_neighbors=SMOOTH_NEIGH)

    df = convert_anndata(adata)

    if latent_graph:
        adata, original_adjacency = construct_latent_adj(
            adata, npc=npc, NUM_AD_NEIGH=NUM_AD_NEIGH, graph_adj=graph_adj
        )

    elif moments_adj:
        adata, original_adjacency = construct_momented_adj(
            adata,
            npc=npc,
            NUM_AD_NEIGH=NUM_AD_NEIGH,
            neighbor_metric=neighbor_metric,
            graph_adj=graph_adj,
            use_us=use_us,
        )

        if not issparse(original_adjacency):
            original_adjacency = csr_matrix(original_adjacency)
        else:
            original_adjacency = original_adjacency.tocsr()

        original_adjacency.data[:] = 1
        original_adjacency.setdiag(1)
        original_adjacency.eliminate_zeros()

    else:
        if use_us:
            sc.pp.neighbors(
                adata, n_pcs=npc, n_neighbors=NUM_AD_NEIGH, use_rep="X_pca_combined",
                metric=neighbor_metric, key_added="adj",
            )
        else:
            sc.pp.neighbors(
                adata, n_pcs=npc, n_neighbors=NUM_AD_NEIGH, use_rep="X_pca",
                metric=neighbor_metric, key_added="adj",
            )

        binary_connectivities = adata.obsp["adj_" + graph_adj].copy().tocsr()
        binary_connectivities.data = np.ones_like(binary_connectivities.data)
        binary_connectivities.setdiag(1)
        binary_connectivities.eliminate_zeros()
        original_adjacency = binary_connectivities

    if use_spatial and spatial_key in adata.obsm:
        spatial_coords = adata.obsm[spatial_key]

        knn = NearestNeighbors(n_neighbors=spatial_neighbors + 1)
        knn.fit(spatial_coords)
        spatial_adj_matrix = knn.kneighbors_graph(spatial_coords)

        spatial_adj_matrix = spatial_adj_matrix.tocsr()
        spatial_adj_matrix.data = np.ones_like(spatial_adj_matrix.data)
        spatial_adj_matrix.setdiag(1)
        spatial_adj_matrix.eliminate_zeros()
        spatial_adjacency = spatial_adj_matrix

        if combine_mode == "union":
            adjacency_matrix = original_adjacency.maximum(spatial_adjacency)
            adjacency_matrix.data[:] = 1

        elif combine_mode == "overlap":
            adjacency_matrix = original_adjacency.multiply(spatial_adjacency)
            adjacency_matrix.data[:] = 1

            no_overlap_cells = np.where(np.asarray(adjacency_matrix.sum(axis=1)).ravel() == 1)[0]
            print(f"Number of cells with no overlap: {len(no_overlap_cells)}", flush=True)

            adjacency_matrix = adjacency_matrix.tolil()
            source_adjacency = spatial_adjacency if spatial_first else original_adjacency
            for cell in no_overlap_cells:
                adjacency_matrix[cell, :] = source_adjacency.getrow(cell)
            adjacency_matrix = adjacency_matrix.tocsr()

        else:
            raise ValueError("combine_mode must be 'union' or 'overlap'")

    else:
        adjacency_matrix = original_adjacency

    # Normalize every gene to its own maximum, as the upstream graph model expects.
    df["unsplice"] = df.groupby("gene_name")["unsplice"].transform(lambda x: x / x.max() if x.max() != 0 else x)
    df["splice"] = df.groupby("gene_name")["splice"].transform(lambda x: x / x.max() if x.max() != 0 else x)

    return df, adjacency_matrix, adata


def normalize_adj(adj):
    """Symmetric normalization ``D^-1/2 A D^-1/2``."""
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    d_mat_inv_sqrt = torch.diag(deg_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def create_pyg_data(dataset, adjacency_matrix, normalize=False):
    """Wrap a dataset and its adjacency matrix into a PyG ``Data`` object."""
    from torch_geometric.data import Data
    from torch_geometric.utils import dense_to_sparse, to_dense_adj

    if issparse(adjacency_matrix):
        adj_coo = adjacency_matrix.tocoo()
        edge_index = torch.tensor(np.vstack((adj_coo.row, adj_coo.col)), dtype=torch.long)
    else:
        edge_index, _ = dense_to_sparse(torch.tensor(adjacency_matrix, dtype=torch.float))

    PyG_data = Data(
        x=dataset.features,
        edge_index=edge_index,
        type_features=dataset.type_features,
        orig_features=dataset.orig_features,
        cell_ids=dataset.cell_ids,
    )

    PyG_data.adj = to_dense_adj(PyG_data.edge_index)[0]
    if normalize:
        PyG_data.adj = normalize_adj(PyG_data.adj)

    PyG_data.x = PyG_data.x.detach()
    return PyG_data


class GeneExpressionDatasetAllGenes(Dataset):
    """Dataset variant that pivots without re-indexing to ``adata.obs``.

    Kept from the upstream module for completeness; the two-stage workflow uses
    :class:`GeneExpressionDatasetAllGenes_adata`.
    """

    def __init__(self, df, gene_list):
        type_mapping = {"Up": 1, "Auto": 0, "Down": -1}
        df["pred_cell_type"] = df["pred_cell_type"].map(type_mapping)

        unsplice_matrix = df.pivot_table(index="cellID", columns="gene_name", values="unsplice", aggfunc="first").fillna(0)
        splice_matrix = df.pivot_table(index="cellID", columns="gene_name", values="splice", aggfunc="first").fillna(0)
        type_matrix = df.pivot_table(index="cellID", columns="gene_name", values="pred_cell_type", aggfunc="first").fillna(0)
        orig_unsplice_matrix = df.pivot_table(index="cellID", columns="gene_name", values="orig_unsplice", aggfunc="first").fillna(0)
        orig_splice_matrix = df.pivot_table(index="cellID", columns="gene_name", values="orig_splice", aggfunc="first").fillna(0)

        unsplice_matrix = unsplice_matrix[gene_list]
        splice_matrix = splice_matrix[gene_list]
        type_matrix = type_matrix[gene_list]
        orig_unsplice_matrix = orig_unsplice_matrix[gene_list]
        orig_splice_matrix = orig_splice_matrix[gene_list]

        features_matrix = pd.concat([unsplice_matrix, splice_matrix], axis=1)
        orig_features_matrix = pd.concat([orig_unsplice_matrix, orig_splice_matrix], axis=1)

        self.features = torch.tensor(features_matrix.values, dtype=torch.float32)
        self.orig_features = torch.tensor(orig_features_matrix.values, dtype=torch.float32)
        self.type_features = torch.tensor(type_matrix.values, dtype=torch.float32)
        self.cell_ids = torch.tensor(features_matrix.index.values, dtype=torch.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.type_features[idx], self.cell_ids[idx], self.orig_features[idx]


class GeneExpressionDatasetAllGenes_adata(Dataset):
    """Dataset used by the stage-1 (pretrain) model."""

    def __init__(self, df, gene_list, adata):
        type_mapping = {"Up": 1, "Auto": 0, "Down": -1}
        df["pred_cell_type"] = df["pred_cell_type"].map(type_mapping)

        def pivot_and_reindex(df, value, index, columns, fillna_value, gene_list, adata):
            matrix = df.pivot_table(index=index, columns=columns, values=value, aggfunc="first").fillna(fillna_value)
            matrix = matrix.reindex(adata.obs.index).fillna(0)
            matrix = matrix[gene_list].astype(np.float32)
            return matrix

        unsplice_matrix = pivot_and_reindex(df, "unsplice", "cellID", "gene_name", 0, gene_list, adata)
        splice_matrix = pivot_and_reindex(df, "splice", "cellID", "gene_name", 0, gene_list, adata)
        type_matrix = pivot_and_reindex(df, "pred_cell_type", "cellID", "gene_name", 0, gene_list, adata).astype(np.float32)
        orig_unsplice_matrix = pivot_and_reindex(df, "orig_unsplice", "cellID", "gene_name", 0, gene_list, adata)
        orig_splice_matrix = pivot_and_reindex(df, "orig_splice", "cellID", "gene_name", 0, gene_list, adata)

        features_matrix = np.concatenate([unsplice_matrix.values, splice_matrix.values], axis=1)
        orig_features_matrix = np.concatenate([orig_unsplice_matrix.values, orig_splice_matrix.values], axis=1)

        self.features = torch.tensor(features_matrix, dtype=torch.float32)
        self.orig_features = torch.tensor(orig_features_matrix, dtype=torch.float32)
        self.type_features = torch.tensor(type_matrix.values, dtype=torch.float32)
        self.cell_ids = unsplice_matrix.index.values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.type_features[idx], self.cell_ids[idx], self.orig_features[idx]


class GeneExpressionDataset_adata_velo(Dataset):
    """Dataset used by the stage-2 (whole-model) model."""

    def __init__(self, adata):
        type_mapping = {"Up": 1, "Auto": 0, "Down": -1}
        map_func = np.vectorize(type_mapping.get)
        adata.layers["pred_cell_type"] = map_func(adata.layers["pred_cell_type"])

        unsplice_matrix = adata.layers["scale_Mu"]
        splice_matrix = adata.layers["scale_Ms"]
        type_matrix = adata.layers["pred_cell_type"]
        orig_unsplice_matrix = adata.layers["unspliced"]
        orig_splice_matrix = adata.layers["spliced"]

        features_matrix = np.concatenate([unsplice_matrix, splice_matrix], axis=1)
        orig_unsplice_matrix = orig_unsplice_matrix.toarray() if hasattr(orig_unsplice_matrix, "toarray") else orig_unsplice_matrix
        orig_splice_matrix = orig_splice_matrix.toarray() if hasattr(orig_splice_matrix, "toarray") else orig_splice_matrix
        orig_features_matrix = np.concatenate([orig_unsplice_matrix, orig_splice_matrix], axis=1)

        self.features = torch.tensor(features_matrix, dtype=torch.float32)
        self.orig_features = torch.tensor(orig_features_matrix, dtype=torch.float32)
        self.type_features = torch.tensor(type_matrix, dtype=torch.float32)
        self.cell_ids = adata.obs.index.values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.type_features[idx], self.cell_ids[idx], self.orig_features[idx]


def preload_datasets_all_genes_anndata(df=None, MODEL_MODE="pretrain", adata=None):
    """Build the stage-1 or stage-2 dataset from the long table / AnnData."""
    if adata is None:
        raise ValueError("The 'adata' parameter must be provided.")

    if MODEL_MODE == "pretrain":
        if df is None:
            raise ValueError("DataFrame 'df' must be provided in pretrain mode")

        df["pred_cell_type"] = "Auto"
        relevant_df = df[
            ["unsplice", "splice", "cellID", "gene_name", "pred_cell_type", "orig_unsplice", "orig_splice"]
        ].copy()

        genes = adata.var.index.tolist()
        dataset = GeneExpressionDatasetAllGenes_adata(relevant_df, genes, adata)

    else:
        if "pred_cell_type" not in adata.layers:
            adata.layers["pred_cell_type"] = np.zeros((adata.shape[0], adata.shape[1]), dtype=np.float32)

        Mu = np.asarray(adata.layers["Mu"], dtype=np.float32)
        Ms = np.asarray(adata.layers["Ms"], dtype=np.float32)

        max_Mu = np.max(Mu, axis=0, keepdims=True)
        max_Ms = np.max(Ms, axis=0, keepdims=True)

        adata.layers["scale_Mu"] = np.divide(Mu, max_Mu, out=np.zeros_like(Mu, dtype=np.float32), where=max_Mu > 0)
        adata.layers["scale_Ms"] = np.divide(Ms, max_Ms, out=np.zeros_like(Ms, dtype=np.float32), where=max_Ms > 0)

        dataset = GeneExpressionDataset_adata_velo(adata)

    return dataset


# ---------------------------------------------------------------------------
# Training-mode presets (unchanged)
# ---------------------------------------------------------------------------
def get_stage1_training_overrides(mode: str) -> dict:
    if mode == "full":
        return {}
    if mode == "fast":
        return {
            "expert_mode": "slim",
            "pretrain_epochs": 500,
            "cluster_epochs": 200,
        }
    raise ValueError(f"Unsupported training_mode: {mode}")


def get_stage2_training_overrides(mode: str, finetune_epochs=None) -> dict:
    if mode == "full":
        return {}
    if mode == "fast":
        if finetune_epochs is None:
            raise ValueError("finetune_epochs must be set when training_mode='fast'.")
        return {
            "expert_mode": "slim",
            "pretrain_epochs": 500,
            "cluster_epochs": 200,
            "velo_batch_size": 512,
            "MIN_IMPRO": 0.01,
            "PATIENCE": 100,
            "num_epochs": finetune_epochs,
        }
    raise ValueError(f"Unsupported training_mode: {mode}")


def run_steer_analysis(
    input_path,
    output_dir,
    cluster_key: str = "celltype",
    spatial_key: str = "spatial",
    spliced_layer: str = DEFAULT_SPLICED_LAYER,
    unspliced_layer: str = DEFAULT_UNSPLICED_LAYER,
    dataset_name: Optional[str] = None,
    device: str = "cuda:0",
    expert: int = 10,
    smooth_neigh: int = 50,
    spatial_neighbors: int = 8,
    n_top_genes: int = 2000,
    training_mode: str = "full",
    finetune_epochs: int = 2000,
    graph: str = "union",
    corr_mode: str = "u",
    neighbor_metric: str = "cosine",
    use_us: bool = True,
    use_filter: bool = True,
    fine_method: str = "none",
    target_size: int = 300,
    direction_base: str = "expert",
    overwrite: bool = False,
    seed: int = 618,
) -> Path:
    """Run the full two-stage STEER pipeline for a single spatial dataset.

    ``root``, ``folder`` and ``out`` are the ones the original ``run_all`` loop
    builds around ``run_one``::

        name = path.stem
        folder = root / name
        out = folder / f"{name}.h5ad"
    """
    steer, PriorInferenceManager = import_steer()

    path = Path(input_path)
    root = Path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    name = path.stem
    if dataset_name is None:
        dataset_name = name
    folder = root / str(dataset_name)
    out = folder / f"{name}.h5ad"

    if out.exists() and not overwrite:
        print(f"Skipping existing output: {out}")
        return out
    folder.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    device = torch.device(device)

    adata = None
    try:
        print(f"Processing {path.name}...")
        adata = sc.read_h5ad(path)
        if cluster_key is None or cluster_key not in adata.obs:
            raise KeyError(f"Missing cell type column: {cluster_key}")
        if spatial_key not in adata.obsm:
            raise KeyError(f"Missing spatial coordinates: {spatial_key}")
        if spatial_key != "spatial":
            adata.obsm["spatial"] = adata.obsm[spatial_key].copy()
        for source, target in ((spliced_layer, "spliced"), (unspliced_layer, "unspliced")):
            if source not in adata.layers:
                raise KeyError(f"Missing layer: {source}")
            if source != target:
                adata.layers[target] = adata.layers[source].copy()

        adata = steer.prepare_spatial_adata(
            adata,
            obs_copy_map={"celltype": cluster_key},
            celltype_key="celltype",
            remove_celltypes=[],
            spatial_key="spatial",
        )

        missing_layers = [layer for layer in ("spliced", "unspliced") if layer not in adata.layers]
        if missing_layers:
            raise ValueError(f"Missing required layers: {missing_layers}")
        if "spatial" not in adata.obsm:
            raise ValueError("Spatial coordinates were not standardized into adata.obsm['spatial'].")

        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=n_top_genes)

        df, adjacency_matrix, adata = preprocess_anndata_spatial(
            adata,
            npc=30,
            NUM_AD_NEIGH=30,
            spatial_neighbors=spatial_neighbors,
            SMOOTH_NEIGH=smooth_neigh,
            moments_adj=True,
            combine_mode=graph,
            neighbor_metric=neighbor_metric,
            spatial_key="spatial",
            use_us=use_us,
        )

        dataset = preload_datasets_all_genes_anndata(df=df, MODEL_MODE="pretrain", adata=adata)
        pyg_data = create_pyg_data(dataset, adjacency_matrix, normalize=True)

        stage1_kwargs = dict(
            device=device,
            device2=device,
            pyg_data=pyg_data,
            MODEL_MODE="pretrain",
            adata=adata,
            NUM_LOSS_NEIGH=30,
            corr_mode=corr_mode,
            max_n_cluster=expert,
            path=str(folder),
        )
        stage1_kwargs.update(get_stage1_training_overrides(training_mode))

        result_adata = steer.model_training_share_neighbor_adata(**stage1_kwargs)

        prior_manager = PriorInferenceManager(result_adata, df, str(folder), seed=seed)
        prior_manager.task1_define_fine_clusters(method=fine_method, target_size=target_size)
        prior_manager.task2_filter_genes(based_on="expert", keep_ngene=1000, use_filter=use_filter)
        prior_manager.task3_calc_convexity(based_on=direction_base)

        result_adata, df_updated, fine_clus_vec_np = prior_manager.finalize_for_training()
        prior_adata = result_adata.copy()

        raw_adata = sc.read_h5ad(path)
        if spatial_key != "spatial":
            raw_adata.obsm["spatial"] = raw_adata.obsm[spatial_key].copy()
        for source, target in ((spliced_layer, "spliced"), (unspliced_layer, "unspliced")):
            if source != target:
                raw_adata.layers[target] = raw_adata.layers[source].copy()
        raw_adata = steer.prepare_spatial_adata(
            raw_adata,
            obs_copy_map={"celltype": cluster_key},
            celltype_key="celltype",
            remove_celltypes=[],
            spatial_key="spatial",
        )

        raw_adata = raw_adata[:, raw_adata.var_names.isin(prior_adata.var_names)].copy()
        if not raw_adata.var_names.is_unique or not prior_adata.var_names.is_unique:
            raise ValueError("Non-unique gene names prevent safe stage-2 alignment")
        raw_adata = raw_adata[:, prior_adata.var_names].copy()

        raw_adata.layers["pred_cell_type"] = prior_adata.layers["pred_cell_type"]
        raw_adata.obsm["X_pre_embed"] = prior_adata.obsm["X_pre_embed"]
        raw_adata.obs["pred_cluster"] = prior_adata.obs["pred_cluster"].astype(int)

        velocity_genes = prior_adata.var_names[prior_adata.var["is_velocity_gene"].astype(bool)]
        velo_adata = raw_adata[:, velocity_genes].copy()

        df_fine, adjacency_matrix_fine, velo_adata = preprocess_anndata_spatial(
            velo_adata,
            npc=30,
            NUM_AD_NEIGH=30,
            spatial_neighbors=spatial_neighbors,
            SMOOTH_NEIGH=smooth_neigh,
            moments_adj=True,
            neighbor_metric=neighbor_metric,
            combine_mode=graph,
            spatial_key="X_spatial" if "X_spatial" in velo_adata.obsm else "spatial",
            use_us=use_us,
        )

        dataset_fine = preload_datasets_all_genes_anndata(df=df_fine, MODEL_MODE="whole", adata=velo_adata)
        pyg_data_fine = create_pyg_data(dataset_fine, adjacency_matrix_fine, normalize=True)
        pyg_data_fine.fine_clus_vec = torch.tensor(fine_clus_vec_np, dtype=torch.long, device=device)

        stage2_kwargs = dict(
            device=device,
            device2=device,
            pyg_data=pyg_data_fine,
            MODEL_MODE="whole",
            adata=velo_adata,
            NUM_LOSS_NEIGH=30,
            max_n_cluster=expert,
            corr_mode=corr_mode,
            path=str(folder),
            num_epochs=finetune_epochs,
        )
        stage2_kwargs.update(get_stage2_training_overrides(training_mode, finetune_epochs))

        velo_adata = steer.model_training_share_neighbor_adata(**stage2_kwargs)
        result_adata = velo_adata.copy()

        sc.pp.neighbors(result_adata, use_rep="X_refine_embed_t", n_neighbors=30)
        steer.velocity_graph(result_adata, vkey="pred_vs_norm", xkey="model_Ms")

        # Benchmark contract: STEER keeps the predicted spliced velocity in
        # layers['pred_vs_norm'] (the key its own driver passes to
        # steer.velocity_graph), so that native layer is aliased to layers['velocity']
        # and preserved under its own name.
        if NATIVE_VELOCITY_KEY not in result_adata.layers:
            raise RuntimeError(f"STEER did not produce layers['{NATIVE_VELOCITY_KEY}'].")
        result_adata.layers["velocity"] = np.asarray(result_adata.layers[NATIVE_VELOCITY_KEY])

        result_adata.uns["steer_run"] = {
            "dataset_name": str(dataset_name),
            "input_path": str(path.resolve()),
            "spatial_key": spatial_key,
            "cluster_key": cluster_key,
            "spliced_layer": spliced_layer,
            "unspliced_layer": unspliced_layer,
            "device": str(device),
            "expert": expert,
            "smooth_neigh": smooth_neigh,
            "spatial_neighbors": spatial_neighbors,
            "n_top_genes": n_top_genes,
            "training_mode": training_mode,
            "finetune_epochs": finetune_epochs,
            "graph": graph,
            "corr_mode": corr_mode,
            "neighbor_metric": neighbor_metric,
            "use_us": bool(use_us),
            "use_filter": bool(use_filter),
            "fine_method": fine_method,
            "target_size": target_size,
            "direction_base": direction_base,
            "native_velocity_key": NATIVE_VELOCITY_KEY,
            "output_path": str(out.resolve()),
        }

        print(f"Saving results to {out}")
        result_adata.write_h5ad(str(out))
        return out
    finally:
        del adata
        cleanup_resources()


def run_batch_steer(
    metadata_file,
    output_dir,
    spatial_key: str = "spatial",
    spliced_layer: str = DEFAULT_SPLICED_LAYER,
    unspliced_layer: str = DEFAULT_UNSPLICED_LAYER,
    device: str = "cuda:0",
    expert: int = 10,
    smooth_neigh: int = 50,
    spatial_neighbors: int = 8,
    n_top_genes: int = 2000,
    training_mode: str = "full",
    finetune_epochs: int = 2000,
    graph: str = "union",
    corr_mode: str = "u",
    neighbor_metric: str = "cosine",
    use_us: bool = True,
    use_filter: bool = True,
    fine_method: str = "none",
    target_size: int = 300,
    direction_base: str = "expert",
    overwrite: bool = False,
    seed: int = 618,
) -> list[Path]:
    """Run STEER for every dataset listed in a metadata table."""
    metadata_file = Path(metadata_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata_file(metadata_file)
    outputs: list[Path] = []
    print(f"Batch mode: {len(metadata_df)} datasets")

    for _, row in metadata_df.iterrows():
        file_path = Path(str(row["file_path"]))
        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue

        try:
            outputs.append(
                run_steer_analysis(
                    input_path=file_path,
                    output_dir=output_dir,
                    cluster_key=str(row["cluster_key"]),
                    spatial_key=str(row["spatial_key"]) or spatial_key,
                    spliced_layer=spliced_layer,
                    unspliced_layer=unspliced_layer,
                    dataset_name=str(row["dataset_name"]),
                    device=device,
                    expert=expert,
                    smooth_neigh=smooth_neigh,
                    spatial_neighbors=spatial_neighbors,
                    n_top_genes=n_top_genes,
                    training_mode=training_mode,
                    finetune_epochs=finetune_epochs,
                    graph=graph,
                    corr_mode=corr_mode,
                    neighbor_metric=neighbor_metric,
                    use_us=use_us,
                    use_filter=use_filter,
                    fine_method=fine_method,
                    target_size=target_size,
                    direction_base=direction_base,
                    overwrite=overwrite,
                    seed=seed,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep batch runs going
            print(f"Failed: {row['dataset_name']}: {exc}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="STEER spatial RNA velocity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Input H5AD file (single-file mode)")
    input_group.add_argument("--metadata-file", help="Metadata CSV/TSV file for batch processing")

    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset-name", dest="dataset_name", default=None, help="Dataset folder name")
    parser.add_argument(
        "--cluster-key",
        dest="cluster_key",
        default="celltype",
        help="Column name in adata.obs used for the cell-type labels (required by STEER)",
    )
    parser.add_argument(
        "--spatial-key",
        dest="spatial_key",
        default="spatial",
        help="Spatial coordinates key in adata.obsm",
    )
    parser.add_argument("--spliced-layer", dest="spliced_layer", default=DEFAULT_SPLICED_LAYER,
                        help="Spliced counts layer")
    parser.add_argument("--unspliced-layer", dest="unspliced_layer", default=DEFAULT_UNSPLICED_LAYER,
                        help="Unspliced counts layer")
    parser.add_argument("--device", default="cuda:0", help="Torch device, for example cuda:0 or cpu")
    parser.add_argument("--expert", type=int, default=10, help="Maximum number of expert clusters")
    parser.add_argument("--smooth-neigh", dest="smooth_neigh", type=int, default=50,
                        help="Neighbours used when smoothing the counts")
    parser.add_argument("--spatial-neighbors", dest="spatial_neighbors", type=int, default=8,
                        help="Neighbours used for the spatial graph")
    parser.add_argument("--n-top-genes", dest="n_top_genes", type=int, default=2000,
                        help="Number of highly variable genes retained")
    parser.add_argument("--training-mode", dest="training_mode", choices=["full", "fast"], default="full",
                        help="Training preset; 'fast' uses fewer epochs")
    parser.add_argument("--finetune-epochs", dest="finetune_epochs", type=int, default=2000,
                        help="Stage-2 fine-tuning epochs")
    parser.add_argument("--graph", choices=["union", "overlap"], default="union",
                        help="How the expression graph is combined with the spatial graph")
    parser.add_argument("--corr-mode", dest="corr_mode", choices=["u", "s"], default="u",
                        help="Correlation direction used by the expert model")
    parser.add_argument("--neighbor-metric", dest="neighbor_metric", default="cosine",
                        help="Distance metric for the neighbour graphs")
    parser.add_argument("--use-us", dest="use_us", action="store_true", default=True,
                        help="Smooth and build the graph from the joint unspliced/spliced moments")
    parser.add_argument("--no-use-us", dest="use_us", action="store_false",
                        help="Use Ms only instead of the joint moments")
    parser.add_argument("--use-filter", dest="use_filter", action="store_true", default=True,
                        help="Enable the prior gene filter")
    parser.add_argument("--no-use-filter", dest="use_filter", action="store_false",
                        help="Disable the prior gene filter")
    parser.add_argument("--fine-method", dest="fine_method", default="none",
                        help="Fine-cluster method used by the prior inference manager")
    parser.add_argument("--target-size", dest="target_size", type=int, default=300,
                        help="Target fine-cluster size")
    parser.add_argument("--direction-base", dest="direction_base", default="expert",
                        help="Basis used for the convexity direction")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=618, help="Random seed (upstream default)")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_steer(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            spatial_key=args.spatial_key,
            spliced_layer=args.spliced_layer,
            unspliced_layer=args.unspliced_layer,
            device=args.device,
            expert=args.expert,
            smooth_neigh=args.smooth_neigh,
            spatial_neighbors=args.spatial_neighbors,
            n_top_genes=args.n_top_genes,
            training_mode=args.training_mode,
            finetune_epochs=args.finetune_epochs,
            graph=args.graph,
            corr_mode=args.corr_mode,
            neighbor_metric=args.neighbor_metric,
            use_us=args.use_us,
            use_filter=args.use_filter,
            fine_method=args.fine_method,
            target_size=args.target_size,
            direction_base=args.direction_base,
            overwrite=args.overwrite,
            seed=args.seed,
        )

    return run_steer_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        spatial_key=args.spatial_key,
        spliced_layer=args.spliced_layer,
        unspliced_layer=args.unspliced_layer,
        dataset_name=args.dataset_name,
        device=args.device,
        expert=args.expert,
        smooth_neigh=args.smooth_neigh,
        spatial_neighbors=args.spatial_neighbors,
        n_top_genes=args.n_top_genes,
        training_mode=args.training_mode,
        finetune_epochs=args.finetune_epochs,
        graph=args.graph,
        corr_mode=args.corr_mode,
        neighbor_metric=args.neighbor_metric,
        use_us=args.use_us,
        use_filter=args.use_filter,
        fine_method=args.fine_method,
        target_size=args.target_size,
        direction_base=args.direction_base,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
