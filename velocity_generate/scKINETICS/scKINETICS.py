#!/usr/bin/env python
"""
scKINETICS benchmark pipeline using the maintained local fork and updated environment.
"""

import argparse
import io
import os
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run scKINETICS on one or more real datasets for VelocityBenchmarking."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-h5ad", help="Input AnnData file for single-dataset mode.")
    input_group.add_argument("--metadata-file", help="Metadata table for batch mode.")

    parser.add_argument("--peaks-bed", help="ATAC peaks BED file for single-dataset mode.")
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
        default="celltype",
        help="Grouping column in adata.obs used for scKINETICS fitting.",
    )
    parser.add_argument(
        "--embedding-basis",
        default="X_umap",
        help="Low-dimensional embedding key used for velocity projection and plotting.",
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

    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_genes(adata, min_cells=args.min_cells)

    if sparse.issparse(adata.X):
        nonzero_columns = np.where(np.asarray(adata.X.sum(axis=0)).ravel() > 0)[0]
    else:
        nonzero_columns = np.where(np.sum(adata.X, axis=0) > 0)[0]
    adata = adata[:, nonzero_columns].copy()

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
    scv.settings.figdir = str(save_path.parent)
    save_name = save_path.name
    fmt = save_path.suffix[1:]

    if fmt != "pdf":
        scv.pl.velocity_embedding_stream(adata, save=save_name, show=False, **plot_kwargs)
        plt.close("all")
        return

    old_stdout, old_stderr = sys.stdout, sys.stderr
    captured_output = io.StringIO()
    sys.stdout = sys.stderr = captured_output

    try:
        scv.pl.velocity_embedding_stream(adata, save=save_name, show=False, **plot_kwargs)
        plt.close("all")

        sys.stdout, sys.stderr = old_stdout, old_stderr
        output_content = captured_output.getvalue()
        pdf_error = (
            "cannot be saved as pdf" in output_content.lower()
            or "can only output finite numbers in pdf" in output_content.lower()
        )

        if pdf_error:
            if save_path.exists():
                os.remove(save_path)

            png_fallback = save_path.with_suffix(".png")
            if png_fallback.exists():
                os.remove(png_fallback)

            svg_path = save_path.with_suffix(".svg")
            scv.settings.figdir = str(svg_path.parent)
            scv.pl.velocity_embedding_stream(adata, save=svg_path.name, show=False, **plot_kwargs)
            plt.close("all")

            if convert_svg_to_pdf(svg_path, save_path) and svg_path.exists():
                os.remove(svg_path)
    finally:
        if sys.stdout != old_stdout:
            sys.stdout = old_stdout
        if sys.stderr != old_stderr:
            sys.stderr = old_stderr


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

    adata_export.obs[cluster_key] = adata_export.obs[cluster_key].astype(str)
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
    required_cols = ["dataset_name", "file_path", "peaks_bed"]
    missing_cols = [column for column in required_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata file: {missing_cols}")

    optional_defaults = {
        "cluster_key": "celltype",
        "embedding_basis": "X_umap",
        "genome": "mm10",
    }
    for column, default_value in optional_defaults.items():
        if column not in df.columns:
            df[column] = default_value

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
):
    EM, VelocityGraph, tf_targets = import_sckinetics()

    input_h5ad = Path(input_h5ad).resolve()
    peaks_bed = Path(peaks_bed).resolve()
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

    print("\n[1/6] Load peaks")
    peaks = load_bed_three_columns(peaks_bed)
    width = peaks["chromEnd"] - peaks["chromStart"]
    peaks = peaks.loc[width < args.peak_width_max].reset_index(drop=True)
    print("  peaks_lt_width", peaks.shape)

    print("\n[2/6] Load and preprocess AnnData")
    adata = sc.read_h5ad(input_h5ad)
    print("  loaded", adata)
    adata = basic_preprocess(adata, cluster_key, embedding_basis, args)
    print("  preprocessed", adata)

    print("\n[3/6] Peak annotation and motif calling")
    peak_annotation = tf_targets.PeakAnnotation(adata=adata, genome=genome)
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
    print("  n_clusters", len(G_clusters))

    print("\n[5/6] EM fitting and velocity projection")
    model = EM.ExpectationMaximization(
        maxiter=args.maxiter,
        tol=args.tol,
        knn=args.model_knn,
        sigma=args.sigma,
        sigma_prior=args.sigma_prior,
        threads=args.threads,
    )
    adata_runtime = peak_annotation.adata
    model.fit(adata_runtime, G_clusters, celltype_basis=cluster_key)

    embedding = adata_runtime.obsm[embedding_basis]
    vg = VelocityGraph(model, adata_runtime, knn=args.graph_knn)
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
        genome=genome,
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
                peaks_bed=row["peaks_bed"],
                output_dir=dataset_output_dir,
                fig_dir=dataset_fig_dir,
                cluster_key=str(row["cluster_key"]),
                embedding_basis=str(row["embedding_basis"]),
                genome=str(row["genome"]),
                args=args,
            )
    else:
        if not args.peaks_bed:
            raise ValueError("`--peaks-bed` is required in single-dataset mode.")

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
        )

    print("\nDone!")


if __name__ == "__main__":
    main(parse_args())
