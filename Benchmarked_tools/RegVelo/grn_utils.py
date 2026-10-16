"""
GRN helpers for RegVelo on simulated (scmultisim / dyngen) data.

RegVelo needs a prior gene regulatory network (GRN). For real data the prior is
inferred by pySCENIC (see ``run_pyscenic_grn_regvelo.py``); for simulated data the
GRN is supplied a-priori through the helpers in this module:

* scmultisim datasets carry the generating network as a long-format csv, which is
  converted into the adjacency matrix RegVelo expects.
* dyngen datasets have no usable prior file, so the GRN is always rebuilt from
  Spearman correlation between the known TFs and every gene.

Representation
--------------
RegVelo stores the prior as a **square 0/1 adjacency matrix** that
``regvelo.pp.set_prior_grn`` consumes; the caller transposes it so that rows are
targets and columns are regulators. This differs from TSvelo, which uses a
long-format ``adata.varm['TFs*']`` representation, so only the RegVelo builders
are kept here.

Data files
----------
All GRN files live next to this module. Paths are resolved relative to this file
(not the current working directory) so the script can be launched from anywhere.

``GRN_params_100.csv`` / ``GRN_params_1139.csv``
    scmultisim long-format priors with columns
    ``regulated.gene, regulator.gene, regulator.effect``. Gene ids are integers
    that map to h5ad gene names via ``str(id)``. The 100-parameter file covers
    the small networks (~100 regulators, e.g. genes500/800/1000) and the
    1139-parameter file the large networks (~1100+ regulators, e.g.
    genes1500/2000/2500).

``dyngen_bifurcating_cell10000_gene500_feature_network.csv``
    dyngen ``model$feature_network`` edge list with columns
    ``regulator, target, effect, strength``, dumped from the dyngen model of the
    ``bifurcating_cell10000_gene500`` dataset. It is **reference data**: the
    default dyngen path rebuilds the GRN by Spearman correlation instead, so this
    file is currently not read. See the commented block below for the code that
    consumes it.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# All GRN files live in the RegVelo folder next to this module.
GRN_DIR = os.path.dirname(os.path.abspath(__file__))

GRN_100_CSV = os.path.join(GRN_DIR, "GRN_params_100.csv")
GRN_1139_CSV = os.path.join(GRN_DIR, "GRN_params_1139.csv")

# ---------------------------------------------------------------------------
# Optional dyngen GRN prior (reference only, disabled by default)
# ---------------------------------------------------------------------------
# dyngen datasets currently always rebuild their GRN from Spearman correlation
# (see ``build_regvelo_grn_from_correlation``). If a reliable simulated GRN for a
# dyngen dataset is available -- for example the dataset's dyngen model
# ``model$feature_network`` edge list dumped to csv, as shipped in
# ``dyngen_bifurcating_cell10000_gene500_feature_network.csv`` -- uncomment the
# definitions below and enable the matching branch in
# ``RegVelo_sim.load_prior_grn()`` to use those true edges instead.
#
# DYNGEN_FEATURE_NETWORK_CSV = os.path.join(
#     GRN_DIR, "dyngen_bifurcating_cell10000_gene500_feature_network.csv"
# )
#
# # Map a dyngen dataset key to its pre-extracted ``model$feature_network`` csv.
# # The key is the dataset name with the benchmark ``_dataset`` suffix stripped,
# # matching RegVelo_sim.derive_output_stem(). Add an entry per dataset that has a
# # reliable dumped GRN; datasets absent from this map keep the Spearman fallback.
# DYNGEN_MODEL_GRN = {
#     "bifurcating_cell10000_gene500": DYNGEN_FEATURE_NETWORK_CSV,
# }
#
#
# def load_dyngen_feature_network(
#     csv_path: str = DYNGEN_FEATURE_NETWORK_CSV,
# ) -> pd.DataFrame:
#     """Load a dyngen ``model$feature_network`` edge list.
#
#     The dyngen model object stores the true GRN in ``model$feature_network``
#     with columns ``from / to / effect / strength / hill``; the shipped dump
#     renames ``from``/``to`` to ``regulator``/``target``. ``effect`` is +1
#     (activation) or -1 (repression) and ``strength`` is the interaction
#     strength. Feed the result to ``build_regvelo_grn``.
#     """
#     return pd.read_csv(csv_path)


def detect_sim_type(adata) -> tuple:
    """Detect the simulation engine from the gene names.

    scmultisim (Bursting-tree) names its regulatory genes with pure numbers
    ("1", "2", ..., "N"), whereas dyngen uses names such as "M1_TF1"/"Burn1_TF1".

    Returns
    -------
    (sim_type, n_regulatory_genes)
        ``sim_type`` is ``'scmultisim'`` or ``'dyngen'``; ``n_regulatory_genes``
        is the number of pure-numeric gene names (0 for dyngen).
    """
    pure = [g for g in adata.var_names if re.fullmatch(r"\d+", str(g))]
    if pure:
        return "scmultisim", len(pure)
    return "dyngen", 0


def choose_grn_csv(
    n_regulatory_genes: int,
    grn_100_csv: str = GRN_100_CSV,
    grn_1139_csv: str = GRN_1139_CSV,
) -> str:
    """Pick the scmultisim GRN csv matching the number of regulatory genes.

    The scmultisim regulators are numbered ``1..N`` but the h5ad only keeps the
    genes that actually take part in the GRN, so the count of pure-numeric names
    is not always equal to the largest gene id and varies slightly between
    datasets. The two prior files differ by an order of magnitude (~100 vs
    ~1100 regulators), so a threshold of 500 separates them without depending on
    an exact value.
    """
    if n_regulatory_genes <= 500:
        return grn_100_csv
    return grn_1139_csv


def load_grn(csv_path: str) -> pd.DataFrame:
    """Load a long-format GRN csv into a DataFrame."""
    return pd.read_csv(csv_path)


def grn_regulatory_genes(grn_df: pd.DataFrame) -> List[str]:
    """Return the sorted gene names (as str) that appear anywhere in the GRN."""
    ids = set()
    for column in ("regulated.gene", "regulator.gene", "target", "regulator"):
        if column in grn_df.columns:
            ids.update(grn_df[column].astype(int))
    return sorted([str(i) for i in ids])


def detect_tf_from_names(adata) -> List[str]:
    """Infer TF gene names from dyngen-style names (``*_TF1`` suffix).

    Used for dyngen datasets, whose TFs are named ``<module>_TF1`` while their
    targets are named ``Target<id>``, so the ``_TF1`` suffix reliably marks TFs.
    The resulting list feeds the Spearman-correlation GRN rebuild.
    """
    return sorted([str(g) for g in adata.var_names if re.search(r"_TF\d*$", str(g))])


def build_regvelo_grn(
    edges_df: pd.DataFrame, genes: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Build a square 0/1 regulator-target matrix for ``set_prior_grn``.

    ``set_prior_grn`` expects a ``gt_net`` DataFrame with rows = targets and
    columns = regulators; the caller transposes this matrix before passing it
    (``grn.T``), matching how the real-data pipeline stores
    ``regulon_mat_processed_all_regulons.parquet`` (a square 0/1 matrix with
    index == columns).

    Parameters
    ----------
    edges_df
        Edge list with regulator/target columns named either
        ``regulator``/``target`` or ``from``/``to``. An explicit dyngen
        ``feature_network`` dump is such an edge list (see the commented block
        above); the scmultisim priors are converted into one by
        :func:`build_regvelo_grn_from_scmultisim`.
    genes
        If provided, the returned matrix is square over exactly these genes;
        otherwise it is square over the union of regulators and targets.

    Returns
    -------
    pandas.DataFrame
        Square 0/1 matrix (int), ``index == columns`` = genes, with
        ``mat.loc[reg, tgt] = 1`` for every regulatory edge. Gene names keep
        their original case (RegVelo does not uppercase names).
    """
    reg_col = "regulator" if "regulator" in edges_df.columns else "from"
    tgt_col = "target" if "target" in edges_df.columns else "to"

    regulators = sorted({str(r) for r in edges_df[reg_col]})
    targets = sorted({str(t) for t in edges_df[tgt_col]})
    if genes is None:
        genes = sorted(set(regulators) | set(targets))

    mat = pd.DataFrame(0, index=genes, columns=genes, dtype=int)
    for _, row in edges_df.iterrows():
        regulator = str(row[reg_col])
        target = str(row[tgt_col])
        if regulator in mat.index and target in mat.columns:
            mat.loc[regulator, target] = 1

    n_edges = int((mat.values != 0).sum())
    print(
        f"[build_regvelo_grn] {len(regulators)} regulators, {len(targets)} targets, "
        f"{n_edges} edges, matrix {mat.shape}"
    )
    return mat


def build_regvelo_grn_from_scmultisim(grn_df: pd.DataFrame) -> pd.DataFrame:
    """Build a RegVelo GRN matrix from a scmultisim long-format GRN.

    scmultisim stores edges as ``regulated.gene / regulator.gene`` with integer
    gene ids; the h5ad names them by ``str(id)``. This helper converts the ids to
    gene-name strings and builds the 0/1 matrix.
    """
    columns = list(grn_df.columns)
    if "regulated.gene" in columns and "regulator.gene" in columns:
        tgt_col, reg_col = "regulated.gene", "regulator.gene"
    else:
        tgt_col, reg_col = "target", "regulator"

    edges = pd.DataFrame(
        {
            "regulator": [str(int(r)) for r in grn_df[reg_col]],
            "target": [str(int(t)) for t in grn_df[tgt_col]],
        }
    )
    return build_regvelo_grn(edges)


def build_regvelo_grn_from_correlation(
    adata, tf_list: Sequence[str], top_k: int = 10, layer: str = "Ms"
) -> pd.DataFrame:
    """Rebuild a RegVelo 0/1 GRN from Spearman correlation (fallback).

    Used for dyngen datasets, which have no usable prior file, so the true
    ``feature_network`` edges are unavailable. For every gene the ``top_k`` most
    correlated TFs (by ``|spearman|``) are kept as its regulators, using the same
    expression/mask convention as the real-data TF lookup.

    ``adata`` must already have ``layers[layer]`` (e.g. ``Ms`` after
    ``scv.pp.moments``).
    """
    from scipy import stats

    gene_names = list(adata.var_names)
    gene_index = {g: i for i, g in enumerate(gene_names)}

    tf_list = [str(t).upper() for t in tf_list]
    tf_list = sorted(
        {t for t in tf_list if t in gene_index}, key=lambda x: gene_index[x]
    )
    if not tf_list:
        raise ValueError("No TF in tf_list is present in adata.var_names.")

    expr = np.asarray(adata.layers[layer], dtype=float)
    k = min(top_k, len(tf_list))

    mat = pd.DataFrame(0, index=gene_names, columns=gene_names, dtype=int)
    n_edges = 0
    for gene_id, gene in enumerate(gene_names):
        target_expr = expr[:, gene_id]
        scored = []
        for tf in tf_list:
            if tf == gene:
                continue
            tf_id = gene_index[tf]
            tf_expr = expr[:, tf_id]
            flag = (tf_expr > 0.1) & (target_expr > 0.1)
            if flag.sum() < 2:
                corr = 0.0
            else:
                corr, _ = stats.spearmanr(target_expr[flag], tf_expr[flag])
                if np.isnan(corr):
                    corr = 0.0
            scored.append((tf, corr))
        scored.sort(key=lambda x: -abs(x[1]))
        for tf, _ in scored[:k]:
            mat.loc[tf, gene] = 1
            n_edges += 1

    print(
        f"[build_regvelo_grn_from_correlation] {len(tf_list)} TFs, {n_edges} edges "
        f"(top_k={k}, layer={layer})"
    )
    return mat
