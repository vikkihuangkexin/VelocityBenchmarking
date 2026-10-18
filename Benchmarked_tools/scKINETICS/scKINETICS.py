#!/usr/bin/env python
"""
scKINETICS benchmark pipeline using the maintained local fork and updated environment.
"""

import argparse
import inspect
import pickle
import subprocess
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from tqdm.auto import tqdm

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

PALETTE = [
    "#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4",
    "#66c2a5", "#3288bd", "#abdda4", "#e6f598", "#fee08b",
    "#f46d43", "#e7298a", "#a6cee3", "#1f78b4", "#b2df8a",
    "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
    "#cab2d6", "#6a3d9a", "#ffff99", "#b15928", "#8dd3c7",
    "#bc80bd", "#ccebc5", "#ffed6f", "#999999",
    "#8B0000", "#006400", "#FF69B4", "#00CED1", "#FFD700",
]


# Paired simulation input validation and TF-target annotations.

def resolve_annotations(
    dataset_name, simulated, cluster_key=None, embedding_basis=None
):
    if cluster_key is None:
        cluster_key = (
            "pseudo_celltype"
            if simulated and "linear" in dataset_name.lower()
            else "celltype"
        )
    return cluster_key, embedding_basis or ("X_tsne" if simulated else "X_umap")


def align_modalities(rna, atac):
    key = "simulation_cell_id"
    if (key in rna.obs) != (key in atac.obs):
        raise ValueError(
            "simulation_cell_id must be present in both modalities or neither."
        )
    r = pd.Index(rna.obs[key] if key in rna.obs else rna.obs_names)
    a = pd.Index(atac.obs[key] if key in atac.obs else atac.obs_names)
    if r.hasnans or a.hasnans:
        raise ValueError("Cell identifiers must not contain missing values.")
    r, a = r.astype(str), a.astype(str)
    if not r.is_unique or not a.is_unique:
        raise ValueError("Cell identifiers must be unique within each modality.")
    if len(r) != len(a) or not r.isin(a).all():
        raise ValueError(
            "RNA/ATAC cell sets differ; explicitly subset them before running."
        )
    rna, atac = rna.copy(), atac[a.get_indexer(r)].copy()
    if not rna.obs_names.is_unique:
        raise ValueError("RNA obs_names must be unique.")
    atac.obs_names = rna.obs_names.copy()
    if not rna.var_names.is_unique or not rna.var_names.str.upper().is_unique:
        raise ValueError(
            "RNA gene identifiers must be unique, including after uppercasing."
        )
    if "spliced" not in rna.layers:
        raise KeyError(
            "scMultiSim RNA requires layers['spliced']; no implicit expression fallback."
        )
    rna.X = rna.layers["spliced"].copy()
    return rna, atac


def boolean_mask(values, size):
    values = np.asarray(values)
    if values.shape != (size,) or pd.isna(values).any():
        raise ValueError(
            "Peak selection must be a non-missing vector with one entry per peak."
        )
    if values.dtype.kind == "b":
        return values.copy()
    if values.dtype.kind in "iuf" and np.isin(values, [0, 1]).all():
        return values.astype(bool)
    strings = np.char.lower(values.astype(str))
    if not np.isin(strings, ["true", "false", "0", "1"]).all():
        raise ValueError("Peak selection accepts only booleans, true/false, or 0/1.")
    return np.isin(strings, ["true", "1"])


def select_peaks(atac, diff_peak_key, width_max, all_peaks=False):
    required = {"chrom", "chromStart", "chromEnd"}
    if not required.issubset(atac.var):
        raise KeyError(f"ATAC var must include {sorted(required)}.")
    start = pd.to_numeric(atac.var["chromStart"], errors="raise").to_numpy(dtype=float)
    end = pd.to_numeric(atac.var["chromEnd"], errors="raise").to_numpy(dtype=float)
    if (
        not np.isfinite(start).all()
        or not np.isfinite(end).all()
        or np.any(start != np.floor(start))
        or np.any(end != np.floor(end))
        or np.any(start < 0)
        or np.any(end <= start)
        or atac.var["chrom"].isna().any()
    ):
        raise ValueError("Invalid ATAC peak coordinates.")
    if all_peaks:
        mask = np.ones(atac.n_vars, dtype=bool)
    elif diff_peak_key in atac.var:
        mask = boolean_mask(atac.var[diff_peak_key], atac.n_vars)
    elif "selected" in atac.uns.get("sckinetics_diff_peaks", {}):
        mask = boolean_mask(atac.uns["sckinetics_diff_peaks"]["selected"], atac.n_vars)
    else:
        raise KeyError(
            f"Missing ATAC differential peak mask {diff_peak_key!r}; use --all-peaks explicitly."
        )
    mask &= end - start < width_max
    if not mask.any():
        raise ValueError("No peaks remain after selection and peak-width filtering.")
    if not atac.var_names.is_unique:
        raise ValueError("ATAC peak identifiers must be unique.")
    return mask


def read_tf_map(path):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"tf", "gene"}.issubset(table.columns):
        raise ValueError("TF mapping CSV requires columns tf,gene.")
    if (
        table.empty
        or table[["tf", "gene"]].apply(lambda col: col.str.strip().eq("").any()).any()
    ):
        raise ValueError("TF mapping CSV must contain nonempty identifiers.")
    table = table.assign(
        tf=table.tf.str.strip().str.upper(), gene=table.gene.str.strip().str.upper()
    )
    if table.tf.duplicated().any():
        raise ValueError("TF mapping CSV must contain exactly one gene per TF.")
    return dict(zip(table.tf, table.gene))


def annotation_matrix(atac, key, names_key):
    if key not in atac.varm or names_key not in atac.uns:
        raise KeyError(f"ATAC requires varm[{key!r}] and uns[{names_key!r}].")
    names = np.asarray(atac.uns[names_key]).astype(str)
    matrix = sparse.csr_matrix(atac.varm[key], dtype=float)
    if names.ndim != 1 or matrix.shape != (atac.n_vars, len(names)):
        raise ValueError(f"{key} dimensions do not match regions and annotation names.")
    if pd.Index(names).str.upper().duplicated().any():
        raise ValueError(f"{names_key} must contain unique identifiers.")
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError(f"{key} must contain finite nonnegative link weights.")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix, np.char.upper(names)


def reference_tf_mapping(atac, tf_matrix, tf_names, gene_matrix, gene_names):
    """Recover the team's template identities only after full annotation agreement."""
    if atac.uns.get("simulation", {}).get("simulator") != "scMultiSim":
        raise ValueError(
            "Simulator TF labels have no RNA identity; supply an explicit mapping."
        )
    reference = pd.read_csv(Path(__file__).with_name("scmultisim_grn_params_100.csv"))
    regulators = sorted(reference.regulator.unique())
    expected_labels = np.asarray([f"TF_{i + 1:03d}" for i in range(len(regulators))])
    if not np.array_equal(tf_names, expected_labels):
        raise ValueError(
            "TF labels have no RNA identity: TF labels/order do not match the supported GRN_params_100 template."
        )
    gene_lookup = {name: i for i, name in enumerate(gene_names)}
    template_genes = {f"GENE_{i:05d}" for i in set(reference.target) | set(regulators)}
    if not template_genes.issubset(gene_lookup):
        raise ValueError(
            "TF labels have no RNA identity: region_to_gene gene names do not match GRN_params_100."
        )
    positive = reference[reference.effect > 0]
    reg_lookup = {reg: i for i, reg in enumerate(regulators)}
    effects = sparse.csr_matrix(
        (
            np.ones(len(positive)),
            (
                [gene_lookup[f"GENE_{i:05d}"] for i in positive.target],
                [reg_lookup[i] for i in positive.regulator],
            ),
        ),
        shape=(len(gene_names), len(regulators)),
    )
    predicted = (gene_matrix @ effects > 0).tocsr()
    actual = (tf_matrix > 0).tocsr()
    if (predicted != actual).nnz or np.any(np.asarray(actual.sum(axis=0)).ravel() == 0):
        raise ValueError(
            "TF labels have no RNA identity: full region_to_tf reconstruction disagrees with GRN_params_100; supply a verified mapping."
        )
    return {tf: f"GENE_{reg:05d}" for tf, reg in zip(tf_names, regulators)}


def build_tf_target_pairs(
    atac, mask, rna_genes, tf_map=None, original_rna_genes=None
):
    tf_matrix, tf_names = annotation_matrix(atac, "region_to_tf", "region_to_tf_names")
    gene_matrix, gene_names = annotation_matrix(
        atac, "region_to_gene", "region_to_gene_genes"
    )
    genes = set(pd.Index(rna_genes).str.upper())
    original_genes = (
        genes
        if original_rna_genes is None
        else set(pd.Index(original_rna_genes).str.upper())
    )
    mapping_source = (
        "csv"
        if tf_map is not None
        else "atac_uns"
        if "sckinetics_tf_to_gene" in atac.uns
        else "gene_names"
    )
    if tf_map is None:
        tf_map = atac.uns.get("sckinetics_tf_to_gene", {})
    if not isinstance(tf_map, dict):
        raise ValueError(
            "uns['sckinetics_tf_to_gene'] must be a TF-to-RNA-gene dictionary."
        )
    tf_map = {str(k).upper(): str(v).upper() for k, v in tf_map.items()}
    template_map = {}
    if (
        not tf_map
        and any(tf not in original_genes for tf in tf_names)
    ):
        template_map = reference_tf_mapping(
            atac, tf_matrix, tf_names, gene_matrix, gene_names
        )
        mapping_source = "GRN_params_100_full_annotation_match"
    if set(tf_map.values()) - original_genes:
        raise ValueError(
            "TF mapping references genes absent from the original RNA input."
        )
    unknown = set(tf_map) - set(tf_names)
    if unknown:
        raise ValueError(
            f"TF map contains labels absent from region_to_tf_names: {sorted(unknown)}"
        )
    mapping = {}
    for tf in tf_names:
        if tf in tf_map:
            mapping[tf] = tf_map[tf]
        elif tf in original_genes:
            mapping[tf] = tf
        elif tf in template_map:
            mapping[tf] = template_map[tf]
        else:
            raise ValueError(
                f"TF {tf!r} has no RNA identity. Supply ATAC uns['sckinetics_tf_to_gene'] or --tf-map; numbered labels are not gene IDs."
            )
    records = []
    for i in np.flatnonzero(mask):
        tfs = {
            mapping[tf_names[j]] for j in tf_matrix[i].indices if tf_names[j] in mapping
        }
        targets = set(gene_names[gene_matrix[i].indices]) & genes
        for tf in sorted(tfs & genes):
            for target in sorted(targets):
                records.append((str(atac.var_names[i]), tf, target))
    if not records:
        raise ValueError(
            "Selected, width-filtered peaks produced no RNA-compatible TF-target pairs. No all-peak retry was performed."
        )
    pairs = pd.DataFrame(records, columns=["peak_name", "TF", "target"])
    provenance = {
        "grn_source": "simulator_region_to_tf_region_to_gene",
        "tf_mapping_mode": "validated_reference_template"
        if template_map
        else "explicit_or_gene_names",
        "tf_mapping_source": mapping_source,
        "tf_mapping": mapping,
        "selected_peaks": int(np.sum(mask)),
        "paired_peaks": int(pairs.peak_name.nunique()),
        "tf_target_peak_pairs": len(pairs),
    }
    return pairs.drop_duplicates().set_index("peak_name"), provenance


def set_cluster_colors(adata, cluster_key, palette):
    if adata.obs[cluster_key].isna().any():
        raise ValueError(f"obs[{cluster_key!r}] contains missing labels.")
    adata.obs[cluster_key] = pd.Categorical(adata.obs[cluster_key].astype(str))
    adata.uns[f"{cluster_key}_colors"] = [
        palette[i % len(palette)]
        for i in range(len(adata.obs[cluster_key].cat.categories))
    ]


def drop_transient_data(adata, cluster_key):
    """Drop training DataFrames on the export copy; retain the runtime object."""
    adata.obsm.pop("X_transformed", None)
    for cluster in adata.obs[cluster_key].astype(str).unique():
        adata.uns.pop(f"X_{cluster}", None)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run scKINETICS on real or paired scMultiSim datasets."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-h5ad", help="Input AnnData file for single-dataset mode.")
    input_group.add_argument("--metadata-file", help="Metadata table for batch mode.")

    parser.add_argument("--peaks-bed", help="ATAC peaks BED file for single-dataset mode.")
    parser.add_argument("--atac-h5ad", help="Paired scMultiSim ATAC file; enables simulation mode.")
    parser.add_argument("--tf-map", help="CSV with explicit simulator tf,gene identities.")
    parser.add_argument("--diff-peak-key", default="sckinetics_diff_peak")
    parser.add_argument("--all-peaks", action="store_true", help="Explicitly use all width-filtered simulation peaks.")
    parser.add_argument("--sklearn-n-jobs", type=int, default=1, help="Jobs for fork kNN operations.")
    parser.add_argument("--output-dir", required=True, help="Output directory for exported h5ad files.")
    parser.add_argument(
        "--fig-dir",
        default=None,
        help="Optional root directory for the fixed benchmark figures.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name in single-dataset mode. Defaults to input file stem.",
    )
    parser.add_argument(
        "--cluster-key",
        default=None,
        help="Default: pseudo_celltype for simulated IDs containing linear; otherwise celltype.",
    )
    parser.add_argument(
        "--embedding-basis",
        default=None,
        help="Default: X_tsne for simulation, X_umap for real data.",
    )
    parser.add_argument(
        "--genome",
        default="mm10",
        choices=["mm10", "mm39", "hg38"],
        help="Genome build for peak annotation.",
    )
    parser.add_argument("--peak-width-max", type=int, default=2000, help="Maximum allowed peak width.")
    parser.add_argument("--min-genes", type=int, default=200, help="Basic cell filter threshold.")
    parser.add_argument("--min-cells", type=int, default=3, help="Basic gene filter threshold.")
    parser.add_argument("--target-sum", type=float, default=1e4, help="Target sum for normalize_total.")
    parser.add_argument("--skip-normalize", action="store_true", help="Skip normalize_total.")
    parser.add_argument("--skip-log1p", action="store_true", help="Skip log1p.")
    parser.add_argument("--pca-n-comps", type=int, default=50, help="PCA dimensions if X_pca is missing.")
    parser.add_argument("--motif-pvalue", type=float, default=1e-10, help="Motif calling p-value.")
    parser.add_argument("--threads", type=int, default=1, help="Thread count for EM fitting.")
    parser.add_argument("--maxiter", type=int, default=20, help="Maximum EM iterations.")
    parser.add_argument("--tol", type=float, default=0.005, help="EM tolerance.")
    parser.add_argument("--model-knn", type=int, default=50, help="kNN used in the EM model.")
    parser.add_argument("--graph-knn", type=int, default=30, help="kNN used by VelocityGraph.")
    parser.add_argument("--sigma", type=float, default=5.0, help="EM sigma.")
    parser.add_argument("--sigma-prior", type=float, default=1.0, help="EM sigma prior.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for scVelo graph-related plotting calculations.",
    )
    return parser.parse_args()


def import_sckinetics():
    from sckinetics import EM, VelocityGraph, tf_targets

    if any("sklearn_n_jobs" not in inspect.signature(cls).parameters
           for cls in (EM.ExpectationMaximization, VelocityGraph)):
        raise RuntimeError("Use the fork revision pinned in README.md; the imported source lacks sklearn_n_jobs support.")
    return EM, VelocityGraph, tf_targets


def ensure_unique_names(adata):
    adata.obs_names_make_unique()
    adata.var_names_make_unique()


def load_bed_three_columns(peak_bed_path):
    first_row = pd.read_csv(peak_bed_path, sep="\t", nrows=1)
    header_names = {str(col) for col in first_row.columns}
    required_names = {"chrom", "chromStart", "chromEnd"}

    if required_names.issubset(header_names):
        peaks = pd.read_csv(peak_bed_path, sep="\t", usecols=["chrom", "chromStart", "chromEnd"])
    else:
        first_line = pd.read_csv(peak_bed_path, sep="\t", nrows=1, header=None)
        has_header = False
        try:
            int(first_line.iloc[0, 1])
            int(first_line.iloc[0, 2])
        except (ValueError, TypeError):
            has_header = True

        if has_header:
            peaks = pd.read_csv(peak_bed_path, sep="\t", header=0).iloc[:, :3].copy()
        else:
            peaks = pd.read_csv(peak_bed_path, sep="\t", header=None).iloc[:, :3].copy()
        peaks.columns = ["chrom", "chromStart", "chromEnd"]

    peaks["chromStart"] = pd.to_numeric(peaks["chromStart"])
    peaks["chromEnd"] = pd.to_numeric(peaks["chromEnd"])
    return peaks


def basic_preprocess(adata, cluster_key, embedding_basis, args):
    ensure_unique_names(adata)
    adata.var["gene_symbol_original"] = adata.var_names.astype(str)
    adata.var["gene_symbol_upper"] = adata.var_names.astype(str).str.upper()

    if cluster_key not in adata.obs:
        raise KeyError(f"{cluster_key!r} not found in adata.obs.")
    set_cluster_colors(adata, cluster_key, PALETTE)

    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_genes(adata, min_cells=args.min_cells)

    if sparse.issparse(adata.X):
        nonzero_columns = np.where(np.asarray(adata.X.sum(axis=0)).ravel() > 0)[0]
    else:
        nonzero_columns = np.where(np.sum(adata.X, axis=0) > 0)[0]
    adata = adata[:, nonzero_columns].copy()
    if min(adata.shape) <= 50:
        raise ValueError("The fork's prepare_data requires more than 50 cells and genes after filtering.")

    if not args.skip_normalize:
        sc.pp.normalize_total(adata, target_sum=args.target_sum)

    adata.layers["norm_counts"] = adata.X.copy()

    if not args.skip_log1p:
        sc.pp.log1p(adata)

    if "X_pca" not in adata.obsm:
        if "X_PCA" in adata.obsm:
            adata.obsm["X_pca"] = adata.obsm["X_PCA"].copy()
        else:
            n_comps = min(args.pca_n_comps, max(2, min(adata.n_obs - 1, adata.n_vars - 1)))
            sc.pp.pca(adata, n_comps=n_comps)

    if embedding_basis not in adata.obsm:
        if embedding_basis == "X_umap":
            sc.pp.neighbors(adata, use_rep="X_pca")
            sc.tl.umap(adata)
        else:
            raise KeyError(f"{embedding_basis!r} not found in adata.obsm.")

    return adata


def export_result(adata, model, vg, velocity_embedding, cluster_key, embedding_basis, genome):
    velocity_genes_upper = model.velocities_.columns.tolist()
    available_genes = [gene for gene in velocity_genes_upper if gene in adata.var_names]
    adata_export = adata[:, available_genes].copy()
    adata_export = adata_export[:, model.velocities_.columns].copy()

    velocity_df = model.velocities_.loc[adata_export.obs_names, adata_export.var_names]
    adata_export.layers["velocity"] = velocity_df.to_numpy(dtype=np.float32)
    adata_export.var["sckinetics_modeled"] = np.ones(adata_export.n_vars, dtype=bool)
    adata_export.uns["sckinetics_velocity_genes_upper"] = np.asarray(adata_export.var_names.astype(str))

    if not model.alpha_all.empty:
        alpha_df = model.alpha_all.loc[adata_export.obs_names, adata_export.var_names]
        adata_export.layers["sckinetics_alpha"] = alpha_df.to_numpy(dtype=np.float32)

    if not model.beta_all.empty:
        beta_df = model.beta_all.loc[adata_export.obs_names, adata_export.var_names]
        adata_export.layers["sckinetics_beta"] = beta_df.to_numpy(dtype=np.float32)

    basis_name = embedding_basis.replace("X_", "")
    adata_export.obsm[f"velocity_{basis_name}"] = velocity_embedding.astype(np.float32)

    if vg.T is not None:
        adata_export.obsp["sckinetics_T"] = vg.T.tocsr()

    if vg.backwards_T is not None:
        adata_export.obsp["sckinetics_T_backward"] = vg.backwards_T.tocsr()

    if model.kNN_graph is not None:
        adata_export.obsp["sckinetics_knn_graph"] = sparse.csr_matrix(model.kNN_graph)

    if "gene_symbol_original" in adata_export.var:
        original_names = adata_export.var["gene_symbol_original"].astype(str).to_numpy()
        adata_export.uns["sckinetics_velocity_genes"] = original_names
        adata_export.var_names = original_names
        adata_export.var_names_make_unique()
    else:
        adata_export.uns["sckinetics_velocity_genes"] = np.asarray(adata_export.var_names.astype(str))

    adata_export.uns["sckinetics_params"] = {
        "tool": "scKINETICS",
        "cluster_key": cluster_key,
        "embedding_basis": embedding_basis,
        "velocity_key": "velocity",
        "genome": genome,
        "model_knn": int(model.knn),
        "graph_knn": int(vg.knn),
        "maxiter": int(model.maxiter),
        "tol": float(model.tol),
        "sigma": float(model.sigma),
        "sigma_prior": float(model.sigma_prior),
        "threads": int(model.threads),
        "n_obs": int(adata_export.n_obs),
        "n_vars": int(adata_export.n_vars),
        "has_alpha_layer": "sckinetics_alpha" in adata_export.layers,
        "has_beta_layer": "sckinetics_beta" in adata_export.layers,
    }
    adata_export.uns["sckinetics_celltypes"] = np.asarray(
        [str(x) for x in getattr(model, "celltypes_list", [])],
        dtype=str,
    )

    set_cluster_colors(adata_export, cluster_key, PALETTE)
    drop_transient_data(adata_export, cluster_key)
    # h5py does not support the longdouble transition weights used internally.
    for key in ("sckinetics_T", "sckinetics_T_backward"):
        if key in adata_export.obsp:
            adata_export.obsp[key] = adata_export.obsp[key].astype(np.float64)
    return adata_export


def convert_svg_to_pdf(svg_path: Path, pdf_path: Path) -> bool:
    try:
        import cairosvg

        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        return True
    except ImportError:
        pass
    except Exception:
        pass

    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF

        drawing = svg2rlg(str(svg_path))
        renderPDF.drawToFile(drawing, str(pdf_path))
        return True
    except ImportError:
        pass
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["inkscape", "--export-filename", str(pdf_path), str(svg_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass

    return False


def save_stream_plot_with_fallback(scv, plt, adata, save_path: Path, **plot_kwargs):
    # Save the figure directly: scVelo prepends its own filename prefix and may
    # swallow PDF errors, which prevents checking/converting the requested file.
    scv.pl.velocity_embedding_stream(adata, save=None, show=False, **plot_kwargs)
    figure = plt.gcf()
    try:
        try:
            figure.savefig(save_path, dpi=plot_kwargs.get("dpi", 400), bbox_inches="tight")
        except ValueError as exc:
            if save_path.suffix != ".pdf" or "finite" not in str(exc).lower():
                raise
            warnings.warn(f"PDF stream rendering failed; converting SVG instead: {exc}")
            svg_path = save_path.with_suffix(".svg")
            figure.savefig(svg_path, bbox_inches="tight")
            if not convert_svg_to_pdf(svg_path, save_path):
                if save_path.exists():
                    save_path.unlink()
                raise RuntimeError(f"Could not convert {svg_path} to PDF") from exc
            svg_path.unlink()
    finally:
        plt.close(figure)


def ensure_scvelo_graph(adata, basis_name, n_jobs):
    import scvelo as scv

    if "neighbors" not in adata.uns:
        if "X_pca" in adata.obsm:
            sc.pp.neighbors(adata, use_rep="X_pca")
        else:
            sc.pp.neighbors(adata, use_rep=f"X_{basis_name}")

    if "velocity_graph" not in adata.uns and "velocity_graph" not in adata.obsp:
        scv.tl.velocity_graph(adata, vkey="velocity", n_jobs=n_jobs)


def add_normalized_velocity_pseudotime(adata):
    import scvelo as scv

    scv.tl.velocity_pseudotime(adata, vkey="velocity")
    pseudotime = np.asarray(adata.obs["velocity_pseudotime"], dtype=float)
    if np.nanmax(pseudotime) != np.nanmin(pseudotime):
        pseudotime = (pseudotime - np.nanmin(pseudotime)) / (np.nanmax(pseudotime) - np.nanmin(pseudotime))
    else:
        pseudotime = np.zeros_like(pseudotime)
    adata.obs["velocity_pseudotime_normalized"] = pseudotime


def generate_standard_figures(adata_export, fig_dir, cluster_key, embedding_basis, n_jobs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scvelo as scv

    basis_name = embedding_basis.replace("X_", "")
    velocity_embed_key = f"velocity_{basis_name}"
    if velocity_embed_key not in adata_export.obsm:
        warnings.warn(
            f"Skip plotting because `{velocity_embed_key}` is missing in adata.obsm.",
            UserWarning,
        )
        return

    fig_dir = Path(fig_dir)
    png_dir = fig_dir / "png"
    pdf_dir = fig_dir / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    set_cluster_colors(adata_export, cluster_key, PALETTE)
    ensure_scvelo_graph(adata_export, basis_name, n_jobs)

    try:
        add_normalized_velocity_pseudotime(adata_export)
        has_pseudotime = True
    except Exception as exc:
        warnings.warn(f"Velocity pseudotime plotting skipped: {exc}", UserWarning)
        has_pseudotime = False

    output_basename = "scKINETICS"

    for fmt in ["png", "pdf"]:
        save_dir = png_dir if fmt == "png" else pdf_dir
        save_name = f"{output_basename}_{basis_name}.{fmt}"
        scv.settings.figdir = str(save_dir)
        scv.pl.scatter(
            adata_export,
            basis=basis_name,
            vkey="velocity",
            color=cluster_key,
            palette=PALETTE,
            size=100,
            alpha=0.6,
            legend_loc="right margin",
            legend_fontsize=9,
            fontsize=None,
            title="scKINETICS",
            dpi=400,
            show=False,
            save=save_name,
        )
        plt.close("all")

    for fmt in ["png", "pdf"]:
        save_dir = png_dir if fmt == "png" else pdf_dir
        save_path = save_dir / f"{output_basename}_{basis_name}_stream.{fmt}"
        save_stream_plot_with_fallback(
            scv,
            plt,
            adata_export,
            save_path,
            size=100,
            alpha=0.6,
            vkey="velocity",
            V=adata_export.obsm[velocity_embed_key],
            basis=basis_name,
            color=cluster_key,
            legend_fontsize=9,
            legend_loc="right margin",
            fontsize=None,
            density=2,
            dpi=400,
            arrow_size=1,
            linewidth=1,
            palette=PALETTE,
            title="scKINETICS",
        )

    for fmt in ["png", "pdf"]:
        save_dir = png_dir if fmt == "png" else pdf_dir
        save_name = f"{output_basename}_{basis_name}_grid.{fmt}"
        scv.settings.figdir = str(save_dir)
        scv.pl.velocity_embedding_grid(
            adata_export,
            vkey="velocity",
            size=100,
            alpha=0.6,
            V=adata_export.obsm[velocity_embed_key],
            basis=basis_name,
            color=cluster_key,
            legend_fontsize=9,
            legend_loc="right margin",
            fontsize=None,
            density=0.8,
            dpi=400,
            arrow_size=1,
            linewidth=0.3,
            palette=PALETTE,
            title="scKINETICS",
            save=save_name,
            show=False,
        )
        plt.close("all")

    if has_pseudotime:
        for fmt in ["png", "pdf"]:
            save_dir = png_dir if fmt == "png" else pdf_dir
            save_name = f"{output_basename}_{basis_name}_pseudotime.{fmt}"
            scv.settings.figdir = str(save_dir)
            scv.pl.scatter(
                adata_export,
                basis=basis_name,
                color="velocity_pseudotime_normalized",
                cmap="gnuplot",
                size=100,
                dpi=400,
                figsize=(8, 6),
                colorbar=True,
                title="scKINETICS",
                save=save_name,
                show=False,
            )
            plt.close("all")


def load_metadata_file(metadata_path: Path) -> pd.DataFrame:
    suffix = metadata_path.suffix.lower()
    if suffix == ".csv":
        sep = ","
    elif suffix in [".tsv", ".txt"]:
        sep = "\t"
    else:
        with open(metadata_path) as handle:
            first_line = handle.readline()
        sep = "\t" if "\t" in first_line else ","

    df = pd.read_csv(metadata_path, sep=sep)
    required_cols = ["dataset_name", "file_path"]
    missing_cols = [column for column in required_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata file: {missing_cols}")
    if df[required_cols].isna().any().any():
        raise ValueError("dataset_name and file_path must not be missing.")
    if df.dataset_name.duplicated().any():
        raise ValueError("dataset_name must be unique in the metadata file.")
    if "dimred_key" in df.columns and "embedding_basis" in df.columns:
        raise ValueError("Use either 'dimred_key' or 'embedding_basis', not both")
    if "embedding_basis" in df.columns:
        df = df.rename(columns={"embedding_basis": "dimred_key"})

    optional_defaults = {
        "cluster_key": None,
        "dimred_key": None,
        "genome": "mm10",
        "peaks_bed": None,
        "atac_h5ad": None,
        "tf_map": None,
    }
    for column, default_value in optional_defaults.items():
        if column not in df.columns:
            df[column] = default_value
        elif default_value is not None:
            df[column] = df[column].fillna(default_value)

    for _, row in df.iterrows():
        if pd.notna(row.peaks_bed) == pd.notna(row.atac_h5ad):
            raise ValueError(f"{row.dataset_name}: supply exactly one of peaks_bed or atac_h5ad.")

    return df


def run_sckinetics_analysis(
    dataset_name,
    input_h5ad,
    peaks_bed,
    output_dir,
    fig_dir,
    cluster_key,
    embedding_basis,
    genome,
    args,
    atac_h5ad=None,
    tf_map_path=None,
):
    EM, VelocityGraph, tf_targets = import_sckinetics()

    input_h5ad = Path(input_h5ad).resolve()
    simulated = atac_h5ad is not None
    cluster_key, embedding_basis = resolve_annotations(
        dataset_name, simulated, cluster_key, embedding_basis,
    )
    if simulated and peaks_bed is not None:
        raise ValueError("Supply either peaks_bed or atac_h5ad, not both.")
    if not simulated and peaks_bed is None:
        raise ValueError("Real data requires --peaks-bed.")
    peaks_bed = Path(peaks_bed).resolve() if peaks_bed else None
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_h5ad = output_dir / f"scKINETICS_{dataset_name}.h5ad"

    print(f"\nProcessing dataset: {dataset_name}")
    print("  Input H5AD:", input_h5ad)
    print("  Peaks BED:", peaks_bed)
    print("  Output H5AD:", output_h5ad)
    print("  Cluster key:", cluster_key)
    print("  Embedding basis:", embedding_basis)
    print("  Threads:", args.threads)

    print("\n[1/6] Load peak inputs")
    if not simulated:
        peaks = load_bed_three_columns(peaks_bed)
        width = peaks["chromEnd"] - peaks["chromStart"]
        peaks = peaks.loc[(width > 0) & (width < args.peak_width_max)].reset_index(drop=True)
        if peaks.empty:
            raise ValueError("No valid peaks remain after width filtering.")
        print("  peaks_lt_width", peaks.shape)

    print("\n[2/6] Load and preprocess AnnData")
    adata = sc.read_h5ad(input_h5ad)
    original_rna_genes = adata.var_names.copy()
    if simulated:
        atac = sc.read_h5ad(atac_h5ad)
        adata, atac = align_modalities(adata, atac)
        selected = select_peaks(atac, args.diff_peak_key, args.peak_width_max, args.all_peaks)
        tf_mapping = read_tf_map(tf_map_path) if tf_map_path else None
        if tf_mapping:
            unknown_genes = set(tf_mapping.values()) - set(adata.var_names.str.upper())
            if unknown_genes:
                raise ValueError(f"TF map references genes absent from input RNA: {sorted(unknown_genes)}")
    print("  loaded", adata)
    adata = basic_preprocess(adata, cluster_key, embedding_basis, args)
    print("  preprocessed", adata)

    print("\n[3/6] Build TF-target peak annotations")
    peak_annotation = tf_targets.PeakAnnotation(adata=adata, genome=None if simulated else genome)
    if simulated:
        peak_annotation.pairs, provenance = build_tf_target_pairs(
            atac, selected, adata.var_names, tf_mapping,
            original_rna_genes=original_rna_genes,
        )
        provenance["peak_selection"] = "all" if args.all_peaks else args.diff_peak_key
        provenance["tf_map_file"] = str(Path(tf_map_path).resolve()) if tf_map_path else ""
        adata.uns["sckinetics_simulation"] = provenance
        print("  simulation GRN", provenance)
    else:
        peak_annotation.call_motifs(peaks, pvalue=args.motif_pvalue)
        print("  targets", peak_annotation.targets.shape)
        print("  motifs", peak_annotation.motifs.shape)
    print("  pairs", peak_annotation.pairs.shape)

    print("\n[4/6] Prepare target annotations")
    cluster_values = pd.Series(adata.obs[cluster_key]).dropna().unique().tolist()
    G_clusters = {}
    for cluster in tqdm(cluster_values, desc="clusters", unit="cluster", dynamic_ncols=True):
        G_clusters[cluster] = peak_annotation.prepare_target_annotations(
            cluster_key=cluster_key,
            cluster=cluster,
        )
        if not G_clusters[cluster]:
            raise ValueError(f"No targets remain for cluster {cluster!r}.")
    print("  n_clusters", len(G_clusters))

    print("\n[5/6] EM fitting and velocity projection")
    model = EM.ExpectationMaximization(
        maxiter=args.maxiter,
        tol=args.tol,
        knn=args.model_knn,
        sigma=args.sigma,
        sigma_prior=args.sigma_prior,
        threads=args.threads,
        sklearn_n_jobs=args.sklearn_n_jobs,
    )
    adata_runtime = peak_annotation.adata
    model.fit(adata_runtime, G_clusters, celltype_basis=cluster_key)

    embedding = adata_runtime.obsm[embedding_basis]
    vg = VelocityGraph(model, adata_runtime, knn=args.graph_knn, sklearn_n_jobs=args.sklearn_n_jobs)
    vg.create_velocity_graph()
    vg.compute_transitions()
    velocity_embedding = vg.embed_graph(embedding)
    print("  velocity_embedding", velocity_embedding.shape)

    print("\n[6/6] Export final h5ad")
    adata_export = export_result(
        adata_runtime,
        model,
        vg,
        velocity_embedding,
        cluster_key=cluster_key,
        embedding_basis=embedding_basis,
        genome="synthetic" if simulated else genome,
    )
    adata_export.write(output_h5ad, compression="gzip")
    print("  saved", output_h5ad)
    print("  export_adata", adata_export)

    with open(output_dir / "sckinetics_model.pickle", "wb") as handle:
        pickle.dump(model, handle)
    with open(output_dir / "sckinetics_runtime_adata.pickle", "wb") as handle:
        pickle.dump(adata_runtime, handle)
    np.save(output_dir / f"velocity_{embedding_basis.replace('X_', '')}.npy", velocity_embedding)

    if fig_dir is not None:
        print("\nGenerating fixed figures...")
        generate_standard_figures(
            adata_export,
            fig_dir=fig_dir,
            cluster_key=cluster_key,
            embedding_basis=embedding_basis,
            n_jobs=args.n_jobs,
        )


def main(args):
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fig_root = Path(args.fig_dir).resolve() if args.fig_dir else None
    if fig_root is not None:
        fig_root.mkdir(parents=True, exist_ok=True)

    if args.metadata_file:
        metadata_df = load_metadata_file(Path(args.metadata_file).resolve())
        print(f"Batch mode: {len(metadata_df)} datasets")

        for _, row in metadata_df.iterrows():
            dataset_name = str(row["dataset_name"])
            dataset_output_dir = output_root / dataset_name
            dataset_fig_dir = fig_root / dataset_name if fig_root is not None else None
            run_sckinetics_analysis(
                dataset_name=dataset_name,
                input_h5ad=row["file_path"],
                peaks_bed=None if pd.isna(row["peaks_bed"]) else row["peaks_bed"],
                output_dir=dataset_output_dir,
                fig_dir=dataset_fig_dir,
                cluster_key=args.cluster_key if pd.isna(row["cluster_key"]) else str(row["cluster_key"]),
                embedding_basis=args.embedding_basis if pd.isna(row["dimred_key"]) else str(row["dimred_key"]),
                genome=str(row["genome"]),
                args=args,
                atac_h5ad=None if pd.isna(row["atac_h5ad"]) else row["atac_h5ad"],
                tf_map_path=args.tf_map if pd.isna(row["tf_map"]) else row["tf_map"],
            )
    else:
        if bool(args.peaks_bed) == bool(args.atac_h5ad):
            raise ValueError("Supply exactly one of --peaks-bed or --atac-h5ad.")

        dataset_name = args.dataset_name or Path(args.input_h5ad).stem
        dataset_fig_dir = fig_root / dataset_name if fig_root is not None else None
        run_sckinetics_analysis(
            dataset_name=dataset_name,
            input_h5ad=args.input_h5ad,
            peaks_bed=args.peaks_bed,
            output_dir=output_root / dataset_name,
            fig_dir=dataset_fig_dir,
            cluster_key=args.cluster_key,
            embedding_basis=args.embedding_basis,
            genome=args.genome,
            args=args,
            atac_h5ad=args.atac_h5ad,
            tf_map_path=args.tf_map,
        )

    print("\nDone!")


if __name__ == "__main__":
    main(parse_args())
