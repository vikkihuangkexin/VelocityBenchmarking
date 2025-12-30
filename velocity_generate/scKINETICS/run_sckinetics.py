#!/usr/bin/env python
"""
scKINETICS velocity analysis pipeline

INSTALLATION:
    scKINETICS is not available on PyPI. Please install from GitHub:

    git clone https://github.com/dpeerlab/scKINETICS.git
    cd scKINETICS
    # Add sckinetics directory to your PYTHONPATH or copy it to site-packages

    Or install directly:
    pip install git+https://github.com/dpeerlab/scKINETICS.git
"""

import argparse
import io
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

warnings.filterwarnings('ignore')


# --------------------------------------------------
# helper: convert SVG to PDF
# --------------------------------------------------
def convert_svg_to_pdf(svg_path: Path, pdf_path: Path) -> bool:
    """
    Convert SVG to PDF using available tools (cairosvg, svglib, or Inkscape).
    Returns True if successful, False otherwise.
    """
    # Method 1: Try cairosvg
    try:
        import cairosvg
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        print(f"    ✓ Converted to PDF using cairosvg")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"    × cairosvg conversion failed: {e}")

    # Method 2: Try svglib + reportlab
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
        drawing = svg2rlg(str(svg_path))
        renderPDF.drawToFile(drawing, str(pdf_path))
        print(f"    ✓ Converted to PDF using svglib")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"    × svglib conversion failed: {e}")

    # Method 3: Try Inkscape command line
    try:
        result = subprocess.run(
            ['inkscape', '--export-filename', str(pdf_path), str(svg_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"    ✓ Converted to PDF using Inkscape")
            return True
        else:
            print(f"    × Inkscape conversion failed: {result.stderr}")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"    × Inkscape conversion failed: {e}")

    print(f"    ! Warning: Could not convert SVG to PDF.")
    print(f"    ! Please install one of: cairosvg (pip install cairosvg),")
    print(f"    ! svglib (pip install svglib reportlab), or Inkscape")
    print(f"    ! SVG file kept at: {svg_path}")
    return False


# --------------------------------------------------
# helper: save stream plot with PDF fallback handling
# --------------------------------------------------
def save_stream_plot_with_fallback(adata, save_path: Path, **plot_kwargs):
    """
    Save velocity stream plot with special handling for PDF format issues.
    If PDF save fails with finite number error, converts via SVG.
    """
    import scvelo as scv
    import matplotlib.pyplot as plt

    fmt = save_path.suffix[1:]  # Remove the leading dot

    # For non-PDF formats, just save normally
    if fmt != 'pdf':
        scv.pl.velocity_embedding_stream(
            adata,
            save=str(save_path),
            show=False,
            **plot_kwargs
        )
        plt.close()
        return

    # For PDF: capture both stdout and stderr to detect issues
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    captured_output = io.StringIO()
    sys.stdout = captured_output
    sys.stderr = captured_output

    try:
        scv.pl.velocity_embedding_stream(
            adata,
            save=str(save_path),
            show=False,
            **plot_kwargs
        )
        plt.close()

        # Restore stdout/stderr and check content
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        output_content = captured_output.getvalue()

        # Check for PDF save failure message
        pdf_error_detected = (
            "cannot be saved as pdf" in output_content.lower() or
            "can only output finite numbers in pdf" in output_content.lower()
        )

        if pdf_error_detected:
            print(f"  ! Detected PDF compatibility issue for stream plot")
            print(f"  → Converting via SVG format...")

            # Remove corrupted PDF if it exists
            if save_path.exists():
                os.remove(save_path)
                print(f"    × Removed corrupted PDF: {save_path.name}")

            # Remove auto-generated PNG fallback if it exists
            png_fallback = save_path.with_suffix('.png')
            if png_fallback.exists():
                os.remove(png_fallback)
                print(f"    × Removed fallback PNG: {png_fallback.name}")

            # Save as SVG
            svg_path = save_path.with_suffix('.svg')
            scv.pl.velocity_embedding_stream(
                adata,
                save=str(svg_path),
                show=False,
                **plot_kwargs
            )
            plt.close()
            print(f"    ✓ Saved as SVG: {svg_path.name}")

            # Convert SVG to PDF
            success = convert_svg_to_pdf(svg_path, save_path)

            # Remove temporary SVG file only if conversion succeeded
            if success and svg_path.exists():
                os.remove(svg_path)
                print(f"    ✓ Removed temporary SVG: {svg_path.name}")
                print(f"    ✓ Final PDF saved: {save_path.name}")

    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        raise e
    finally:
        # Ensure stdout/stderr are always restored
        if sys.stdout != old_stdout:
            sys.stdout = old_stdout
        if sys.stderr != old_stderr:
            sys.stderr = old_stderr


def main(args):
    input_h5ad = Path(args.input_h5ad)
    peaks_bed = Path(args.peaks_bed)
    output_h5ad = Path(args.output_h5ad)

    print(f"Input H5AD: {input_h5ad}")
    print(f"Peaks BED: {peaks_bed}")
    print(f"Output H5AD: {output_h5ad}")

    # --------------------------------------------------
    # 1. Load peaks data
    # --------------------------------------------------
    print("\n[1/5] Loading peaks data...")

    # Auto-detect header by checking if first row contains numeric values
    first_line = pd.read_csv(peaks_bed, sep='\t', nrows=1, header=None)
    has_header = False
    try:
        # Try to convert second and third columns to int
        int(first_line.iloc[0, 1])
        int(first_line.iloc[0, 2])
    except (ValueError, TypeError):
        # If conversion fails, first row is a header
        has_header = True

    # Read BED file with or without header
    if has_header:
        normal_peaks = pd.read_csv(peaks_bed, sep='\t', header=0, usecols=[0, 1, 2])
    else:
        normal_peaks = pd.read_csv(peaks_bed, sep='\t', header=None, usecols=[0, 1, 2])

    normal_peaks.columns = ['chrom', 'chromStart', 'chromEnd']

    # Ensure numeric types for coordinates
    normal_peaks['chromStart'] = pd.to_numeric(normal_peaks['chromStart'])
    normal_peaks['chromEnd'] = pd.to_numeric(normal_peaks['chromEnd'])

    # Filter peaks by width
    width = normal_peaks['chromEnd'] - normal_peaks['chromStart']
    normal_peaks = normal_peaks.iloc[np.where(width < 2000)]
    normal_peaks = normal_peaks.reset_index(drop=True)
    print(f"  Loaded {len(normal_peaks)} peaks (width < 2000bp)")

    # --------------------------------------------------
    # 2. Load scRNA-seq data
    # --------------------------------------------------
    print("\n[2/5] Loading scRNA-seq data...")
    X = sc.read_h5ad(input_h5ad)
    print(f"  Loaded {X.n_obs} cells × {X.n_vars} genes")

    # Ensure 'cluster' column exists as numeric values
    if 'cluster' not in X.obs.columns:
        if 'celltype' in X.obs.columns:
            # Convert celltype to numeric cluster codes
            X.obs['celltype'] = X.obs['celltype'].astype('category')
            X.obs['cluster'] = X.obs['celltype'].cat.codes
            print(f"  Created numeric 'cluster' column from 'celltype' ({X.obs['cluster'].nunique()} clusters)")
        else:
            raise ValueError("Input data must have either 'cluster' or 'celltype' column in obs")

    # Remove zero-only columns
    nonzero_columns = np.where(~np.all(X.X.toarray() == 0, axis=0))[0]
    X = X[:, nonzero_columns]
    X.layers['norm_counts'] = X.X.copy()
    print(f"  After filtering: {X.n_obs} cells × {X.n_vars} genes")

    # Preprocessing - only compute UMAP if not present
    sc.pp.log1p(X)

    if 'X_umap' not in X.obsm:
        print("  Computing neighbors and UMAP...")
        sc.pp.neighbors(X, n_neighbors=args.n_neighbors, n_pcs=args.n_pcs)
        sc.tl.umap(X)
    else:
        print("  Using existing UMAP coordinates")
        # Still need neighbors for velocity graph later
        sc.pp.neighbors(X, n_neighbors=args.n_neighbors, n_pcs=args.n_pcs)

    # --------------------------------------------------
    # 3. Peak annotation and motif calling
    # --------------------------------------------------
    print("\n[3/5] Annotating peaks and calling motifs...")
    from sckinetics import tf_targets

    peak_annotation = tf_targets.PeakAnnotation(adata=X, genome=args.genome)
    peak_annotation.call_motifs(normal_peaks, pvalue=args.motif_pvalue)
    print(f"  Motif calling complete (p-value threshold: {args.motif_pvalue})")

    # --------------------------------------------------
    # 4. Prepare target annotations and run EM
    # --------------------------------------------------
    print("\n[4/5] Preparing target annotations per cluster...")
    cluster_basis = 'cluster'

    G_clusters = {}
    for this_cluster in tqdm(list(set(X.obs[cluster_basis])), desc="  Processing clusters"):
        G_clusters[this_cluster] = peak_annotation.prepare_target_annotations(
            cluster_key=cluster_basis,
            cluster=this_cluster
        )
    print(f"  Processed {len(G_clusters)} clusters")

    print("\n  Running Expectation-Maximization...")
    from sckinetics import EM

    model = EM.ExpectationMaximization(threads=args.n_threads, maxiter=args.maxiter)
    adata = peak_annotation.adata
    model.fit(adata, G_clusters, celltype_basis=cluster_basis)
    print(f"  EM fitting complete (max iterations: {args.maxiter})")

    # --------------------------------------------------
    # 5. Velocity graph and embedding
    # --------------------------------------------------
    print("\n[5/5] Computing velocity graph and embedding...")
    from sckinetics import VelocityGraph

    embedding = adata.obsm['X_umap']

    vg = VelocityGraph(model, adata, knn=args.knn)
    vg.create_velocity_graph()
    vg.compute_transitions()
    velocity_embedding = vg.embed_graph(embedding)

    # --------------------------------------------------
    # Prepare final output
    # --------------------------------------------------
    print("\nPreparing final output...")

    # Get velocity genes
    if hasattr(model.velocities_, 'columns'):
        velocity_genes = model.velocities_.columns.tolist()
    else:
        raise ValueError("Cannot determine velocity genes from model")

    # Subset adata to velocity genes
    available_genes = [gene for gene in velocity_genes if gene in adata.var.index]
    adata_subset = adata[:, available_genes].copy()

    # Ensure gene order matches
    if hasattr(model.velocities_, 'columns'):
        adata_subset = adata_subset[:, model.velocities_.columns]

    # Add velocity layers
    if adata_subset.n_vars == model.velocities_.shape[1]:
        adata_subset.layers['velocity'] = model.velocities_.values
        adata_subset.obsm['velocity_umap'] = velocity_embedding
        print(f"  Added velocity layer: {adata_subset.layers['velocity'].shape}")
        print(f"  Added velocity_umap: {adata_subset.obsm['velocity_umap'].shape}")
    else:
        raise ValueError(
            f"Dimension mismatch: adata {adata_subset.n_vars} vs model {model.velocities_.shape[1]}"
        )

    # --------------------------------------------------
    # scVelo downstream analysis and visualization
    # --------------------------------------------------
    if args.fig_dir:
        print("\nRunning scVelo downstream analysis...")
        import scvelo as scv
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Compute velocity graph using scVelo
        print("  Computing velocity graph...")
        scv.tl.velocity_graph(adata_subset, vkey='velocity', n_jobs=args.n_jobs)

        # Compute pseudotime
        print("  Computing velocity pseudotime...")
        scv.tl.velocity_pseudotime(adata_subset, vkey='velocity')

        # Fixed color palette for 7 cell types
        fixed_palette = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4", "#66c2a5", "#3288bd"]
        cluster_key = "celltype"
        title = "scKINETICS Velocity"

        # Create subdirectories for PNG and PDF formats
        png_dir = fig_dir / "png"
        pdf_dir = fig_dir / "pdf"
        png_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        print("  Saving velocity embedding plots...")

        # 1. UMAP scatter plot
        for fmt in ['png', 'pdf']:
            save_dir = png_dir if fmt == 'png' else pdf_dir
            save_filename = f"sckinetics_velocity_umap.{fmt}"
            save_path = save_dir / save_filename
            scv.pl.scatter(
                adata_subset,
                basis='umap',
                vkey='velocity',
                color=cluster_key,
                palette=fixed_palette,
                size=100,
                alpha=0.6,
                legend_loc='right margin',
                legend_fontsize=9,
                fontsize=None,
                title=title,
                dpi=400,
                show=False,
                save=str(save_path)
            )
            plt.close()

        # 2. Velocity stream plot (with PDF fallback handling)
        for fmt in ['png', 'pdf']:
            save_dir = png_dir if fmt == 'png' else pdf_dir
            save_filename = f"sckinetics_velocity_stream.{fmt}"
            save_path = save_dir / save_filename

            # Use special handling for stream plots due to potential PDF issues
            save_stream_plot_with_fallback(
                adata_subset,
                save_path,
                vkey='velocity',
                size=100,
                alpha=0.6,
                basis="umap",
                color=cluster_key,
                legend_fontsize=9,
                legend_loc='right margin',
                fontsize=None,
                density=2,
                dpi=400,
                arrow_size=1,
                linewidth=1,
                palette=fixed_palette,
                title=title
            )

        # 3. Velocity grid plot
        for fmt in ['png', 'pdf']:
            save_dir = png_dir if fmt == 'png' else pdf_dir
            save_filename = f"sckinetics_velocity_grid.{fmt}"
            save_path = save_dir / save_filename
            scv.pl.velocity_embedding_grid(
                adata_subset,
                vkey='velocity',
                size=100,
                alpha=0.6,
                basis="umap",
                color=cluster_key,
                legend_fontsize=9,
                legend_loc='right margin',
                fontsize=None,
                density=0.8,
                dpi=400,
                arrow_size=1,
                linewidth=0.3,
                palette=fixed_palette,
                title=title,
                save=str(save_path)
            )
            plt.close()

        # 4. Pseudotime plot
        pseudotime_key = 'velocity_pseudotime'
        if pseudotime_key in adata_subset.obs:
            pseudotime = adata_subset.obs[pseudotime_key].values
            pseudotime_normalized = (pseudotime - np.min(pseudotime)) / (np.max(pseudotime) - np.min(pseudotime))
            adata_subset.obs['velocity_pseudotime_normalized'] = pseudotime_normalized

            for fmt in ['png', 'pdf']:
                save_dir = png_dir if fmt == 'png' else pdf_dir
                save_filename = f"sckinetics_velocity_pseudotime.{fmt}"
                save_path = save_dir / save_filename
                scv.pl.scatter(
                    adata_subset,
                    basis='umap',
                    color='velocity_pseudotime_normalized',
                    cmap='gnuplot',
                    size=100,
                    dpi=400,
                    figsize=(8, 6),
                    colorbar=True,
                    title=title,
                    save=str(save_path)
                )
                plt.close()
        else:
            print(f"  Warning: {pseudotime_key} not found in adata.obs. Skipping pseudotime plot.")

        print(f"  Figures saved to: {fig_dir}")

    # Save final output
    adata_subset.write(output_h5ad, compression="lzf")
    print(f"\nOutput saved: {output_h5ad}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run scKINETICS velocity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-h5ad",
        default="24_mouse_brain.h5ad",
        help="Input scRNA-seq data (.h5ad)"
    )

    parser.add_argument(
        "--peaks-bed",
        default="24_MouseBrain_peaks_for_sckinetics.bed",
        help="Input peaks BED file"
    )

    parser.add_argument(
        "--output-h5ad",
        default="scKINETICS_24_plot.h5ad",
        help="Output H5AD with velocity layers"
    )

    parser.add_argument(
        "--genome",
        default="mm10",
        help="Genome assembly (e.g., mm10, hg38)"
    )

    parser.add_argument(
        "--motif-pvalue",
        type=float,
        default=1e-10,
        help="P-value threshold for motif calling"
    )

    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Number of neighbors for scanpy.pp.neighbors"
    )

    parser.add_argument(
        "--n-pcs",
        type=int,
        default=40,
        help="Number of PCs for scanpy.pp.neighbors"
    )

    parser.add_argument(
        "--n-threads",
        type=int,
        default=15,
        help="Number of threads for EM"
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=20,
        help="Maximum iterations for EM"
    )

    parser.add_argument(
        "--knn",
        type=int,
        default=30,
        help="Number of neighbors for velocity graph"
    )

    parser.add_argument(
        "--fig-dir",
        default=None,
        help="Directory to save velocity visualization figures (optional)"
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for scVelo velocity_graph (-1=all cores)"
    )

    args = parser.parse_args()
    main(args)
