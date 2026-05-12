#!/usr/bin/env python

import argparse
from pathlib import Path
import gc
import io
import sys
import os
import subprocess

import scanpy as sc
import scvelo as scv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# --------------------------------------------------
# helper: load cell x gene CSV into AnnData layer
# --------------------------------------------------
def load_layer(csv_path: Path, adata: sc.AnnData) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df = df.set_index("cell")

    # strict alignment
    df = df.loc[adata.obs_names, adata.var_names]

    arr = df.values.astype("float32")

    # free memory ASAP
    del df
    gc.collect()

    return arr


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
    csv_dir = Path(args.csv_dir)
    output_h5ad = Path(args.output_h5ad)
    fig_dir = Path(args.fig_dir)

    fig_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. read original AnnData
    # --------------------------------------------------
    print("Reading original AnnData...")
    adata = sc.read(input_h5ad)

    # --------------------------------------------------
    # 2. determine RegionVelocity valid genes
    # --------------------------------------------------
    print("Loading RegionVelocity gene list...")
    rv_genes = pd.read_csv(
        csv_dir / "region_EM_parameters.csv"
    )["gene"].tolist()
    rv_genes = set(rv_genes)

    # --------------------------------------------------
    # 3. subset AnnData by genes
    # --------------------------------------------------
    print(f"Subsetting genes: {adata.n_vars} -> {len(rv_genes)}")
    adata = adata[:, [g in rv_genes for g in adata.var_names]].copy()
    gc.collect()

    # --------------------------------------------------
    # 4. load RegionVelocity layers
    # --------------------------------------------------
    print("Loading RegionVelocity layers...")

    adata.layers["region_velocity"] = load_layer(
        csv_dir / "region_velocity.csv", adata
    )
    adata.layers["region_velocity_EM"] = load_layer(
        csv_dir / "region_velocity_EM.csv", adata
    )
    adata.layers["region_projected"] = load_layer(
        csv_dir / "region_projected.csv", adata
    )
    adata.layers["region_projected_EM"] = load_layer(
        csv_dir / "region_projected_EM.csv", adata
    )
    adata.layers["region_Ms"] = load_layer(
        csv_dir / "region_Ms.csv", adata
    )
    adata.layers["region_EM_time"] = load_layer(
        csv_dir / "region_EM_time.csv", adata
    )

    assert adata.layers["region_velocity_EM"].shape == adata.X.shape

    # --------------------------------------------------
    # 5. gene-level metadata
    # --------------------------------------------------
    print("Adding gene-level metadata...")
    for fname in [
        "region_EM_parameters.csv",
        "region_velocity_gamma.csv",
        "used_in_EM_model.csv",
    ]:
        df = pd.read_csv(csv_dir / fname).set_index("gene")
        adata.var = adata.var.join(df)
        del df
        gc.collect()

    # --------------------------------------------------
    # 6. cell-level metadata
    # --------------------------------------------------
    print("Adding cell-level metadata...")
    df = pd.read_csv(
        csv_dir / "region_velocity_cellsize.csv"
    ).set_index("cell")
    adata.obs = adata.obs.join(df)
    del df
    gc.collect()

    # --------------------------------------------------
    # 7. annotate provenance
    # --------------------------------------------------
    adata.uns["region_velocity_params"] = {
        "source": "RegionVelocity (R)",
        "note": (
            "Velocity values imported from R. "
            "Genes without valid RegionVelocity estimates were removed. "
            "scVelo was used only for neighborhood graph, "
            "velocity graph construction, and visualization."
        ),
    }

    # --------------------------------------------------
    # 8. scVelo downstream (EXTERNAL velocity)
    # --------------------------------------------------
    print("Running scVelo downstream analysis...")

    sc.pp.neighbors(adata)

    # CRITICAL: xkey is ONLY used here
    scv.tl.velocity_graph(
        adata,
        vkey="region_velocity",
        xkey="region_Ms",
        n_jobs=args.n_jobs
    )

    scv.tl.velocity_graph(
        adata,
        vkey="region_velocity_EM",
        xkey="region_Ms",
        n_jobs=args.n_jobs
    )

    scv.tl.velocity_pseudotime(
        adata,
        vkey="region_velocity_EM",
    )

    # --------------------------------------------------
    # 9. Plot velocity embedding (NO xkey here!)
    # --------------------------------------------------
    print("Saving velocity embedding plots...")
    
    # Fixed parameters
    cluster_key = "cell_type"
    title = "Region Velocity"
    # Fixed color palette for three cell types
    fixed_palette = ["#d73027", "#fc8d59", "#fee090"]
    
    # Create subdirectories for PNG and PDF formats
    png_dir = fig_dir / "png"
    pdf_dir = fig_dir / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. UMAP scatter plot
    for fmt in ['png', 'pdf']:
        save_dir = png_dir if fmt == 'png' else pdf_dir
        save_filename = f"region_velocity_umap.{fmt}"
        save_path = save_dir / save_filename
        scv.pl.scatter(
            adata,
            basis='umap',
            vkey='region_velocity_EM',
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
            save=str(save_path)  # Convert Path to string
        )
        plt.close()

    # 2. Velocity stream plot (with PDF fallback handling)
    for fmt in ['png', 'pdf']:
        save_dir = png_dir if fmt == 'png' else pdf_dir
        save_filename = f"region_velocity_stream.{fmt}"
        save_path = save_dir / save_filename

        # Use special handling for stream plots due to potential PDF issues
        save_stream_plot_with_fallback(
            adata,
            save_path,
            size=100,
            alpha=0.6,
            vkey='region_velocity_EM',
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
        save_filename = f"region_velocity_grid.{fmt}"
        save_path = save_dir / save_filename
        scv.pl.velocity_embedding_grid(
            adata,
            vkey='region_velocity_EM',
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
            save=str(save_path)  # Convert Path to string
        )
        plt.close()

    # 4. Pseudotime plot
    pseudotime_key = 'region_velocity_EM_pseudotime'
    if pseudotime_key in adata.obs:
        pseudotime = adata.obs[pseudotime_key].values
        pseudotime_normalized = (pseudotime - np.min(pseudotime)) / (np.max(pseudotime) - np.min(pseudotime))
        adata.obs['region_velocity_EM_pseudotime_normalized'] = pseudotime_normalized
        
        for fmt in ['png', 'pdf']:
            save_dir = png_dir if fmt == 'png' else pdf_dir
            save_filename = f"region_velocity_pseudotime.{fmt}"
            save_path = save_dir / save_filename
            scv.pl.scatter(
                adata,
                basis='umap',
                color='region_velocity_EM_pseudotime_normalized',
                cmap='gnuplot',
                size=100,
                dpi=400,
                figsize=(8, 6),
                colorbar=True,
                title=title,
                save=str(save_path)  # Convert Path to string
            )
            plt.close()
    else:
        print(f"Warning: {pseudotime_key} not found in adata.obs. Skipping pseudotime plot.")

    # --------------------------------------------------
    # 10. save AnnData
    # --------------------------------------------------
    print("Saving AnnData...")
    adata.write(output_h5ad, compression='lzf')

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import RegionVelocity results into AnnData for scVelo visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-h5ad", 
        required=True,
        help="Input AnnData (.h5ad) - same as used in run_regionvelocity_1.r"
    )
    
    parser.add_argument(
        "--csv-dir", 
        required=True,
        help="Directory containing RegionVelocity CSV outputs from run_regionvelocity_1.r"
    )
    
    parser.add_argument(
        "--output-h5ad", 
        required=True,
        help="Output AnnData (.h5ad) with RegionVelocity layers"
    )
    
    parser.add_argument(
        "--fig-dir", 
        required=True,
        help="Directory to save velocity visualization figures"
    )
    
    parser.add_argument(
        "--n-jobs", 
        type=int, 
        default=1,
        help="CPU cores for velocity_graph (-1=all)"
    )

    args = parser.parse_args()
    main(args)
