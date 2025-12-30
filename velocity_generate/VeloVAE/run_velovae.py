#!/usr/bin/env python3
"""
VeloVAE velocity analysis pipeline

INSTALLATION:
    git clone https://github.com/welch-lab/VeloVAE.git
    cd VeloVAE && pip install -e .
    If based on Docker, run pip install - e. -- no deps

USAGE:
    python run_velovae.py --input data.h5ad --output-dir ./output --fig-dir ./figures --cluster-key celltype
    python run_velovae.py --metadata-file datasets.csv --output-dir ./output --fig-dir ./figures
"""

import argparse
import gc
import io
import os
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy.sparse import csr_matrix

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

PALETTE = [
    "#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4",
    "#66c2a5", "#3288bd", "#abdda4", "#e6f598", "#fee08b",
    "#f46d43", "#e7298a", "#a6cee3", "#1f78b4", "#b2df8a",
    "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
    "#cab2d6", "#6a3d9a", "#ffff99", "#b15928", "#8dd3c7",
    "#bc80bd", "#ccebc5", "#ffed6f", "#999999",
    "#8B0000", "#006400", "#FF69B4", "#00CED1", "#FFD700",
]


def cleanup_resources(adata=None, vae=None):
    """Clean up memory resources."""
    try:
        if adata is not None:
            del adata
        if vae is not None:
            del vae
        gc.collect()
        plt.close('all')
    except:
        pass


def convert_svg_to_pdf(svg_path: Path, pdf_path: Path) -> bool:
    """Convert SVG to PDF using available tools."""
    # Method 1: cairosvg
    try:
        import cairosvg
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        return True
    except ImportError:
        pass
    except Exception:
        pass

    # Method 2: svglib + reportlab
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

    # Method 3: Inkscape
    try:
        result = subprocess.run(
            ['inkscape', '--export-filename', str(pdf_path), str(svg_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return True
    except:
        pass

    return False


def save_stream_plot_with_fallback(adata, save_path: Path, **plot_kwargs):
    """Save velocity stream plot with PDF fallback handling."""
    fmt = save_path.suffix[1:]

    if fmt != 'pdf':
        scv.pl.velocity_embedding_stream(adata, save=str(save_path), show=False, **plot_kwargs)
        plt.close()
        return

    # For PDF: capture output to detect issues
    old_stdout, old_stderr = sys.stdout, sys.stderr
    captured_output = io.StringIO()
    sys.stdout = sys.stderr = captured_output

    try:
        scv.pl.velocity_embedding_stream(adata, save=str(save_path), show=False, **plot_kwargs)
        plt.close()

        sys.stdout, sys.stderr = old_stdout, old_stderr
        output_content = captured_output.getvalue()

        pdf_error = (
            "cannot be saved as pdf" in output_content.lower() or
            "can only output finite numbers in pdf" in output_content.lower()
        )

        if pdf_error:
            if save_path.exists():
                os.remove(save_path)
            png_fallback = save_path.with_suffix('.png')
            if png_fallback.exists():
                os.remove(png_fallback)

            svg_path = save_path.with_suffix('.svg')
            scv.pl.velocity_embedding_stream(adata, save=str(svg_path), show=False, **plot_kwargs)
            plt.close()

            if convert_svg_to_pdf(svg_path, save_path) and svg_path.exists():
                os.remove(svg_path)

    except Exception as e:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        raise e
    finally:
        if sys.stdout != old_stdout:
            sys.stdout = old_stdout
        if sys.stderr != old_stderr:
            sys.stderr = old_stderr


def determine_preprocessing_params(adata):
    """Determine preprocessing parameters based on data size."""
    n_obs, n_vars = adata.n_obs, adata.n_vars

    if n_vars < 1500:
        n_gene = 500
    elif n_vars < 10000:
        n_gene = 2000
    elif n_vars < 50000:
        n_gene = 4000
    else:
        n_gene = 5000

    if n_obs < 1500:
        npc, batch_size = 30, 64
    elif n_obs < 15000:
        npc, batch_size = 30, 128
    else:
        npc, batch_size = 50, 256

    return n_gene, npc, batch_size


def check_or_compute_dimred(adata, dimred_key: str, npc: int):
    """Check if dimensionality reduction exists, compute UMAP if not."""
    if dimred_key in adata.obsm:
        if dimred_key != 'X_umap':
            adata.obsm['X_umap'] = adata.obsm[dimred_key]
    else:
        print(f"  Computing UMAP ({dimred_key} not found)...")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=npc)
        sc.tl.umap(adata)
        if dimred_key != 'X_umap':
            adata.obsm[dimred_key] = adata.obsm['X_umap'].copy()


def run_velovae_analysis(
    input_path: Path,
    output_dir: Path,
    fig_dir: Path,
    cluster_key: str,
    dimred_key: str = 'X_umap',
    dim_z: int = 5,
    zero_threshold: bool = False,
    device: str = 'cuda:0',
    n_jobs: int = 1
) -> None:
    """Run VeloVAE analysis on a single dataset."""
    import velovae as vv

    adata = None
    vae = None

    try:
        input_basename = input_path.stem.replace('_dataset', '')
        output_basename = f"VeloVAE_{input_basename}_plot"
        output_h5ad = output_dir / f"{output_basename}.h5ad"

        print(f"\nProcessing: {input_path.name}")

        png_dir = fig_dir / "png"
        pdf_dir = fig_dir / "pdf"
        png_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        adata = sc.read(input_path)

        if cluster_key not in adata.obs.columns:
            raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")

        n_gene, npc, batch_size = determine_preprocessing_params(adata)
        check_or_compute_dimred(adata, dimred_key, npc)

        if not isinstance(adata.layers["spliced"], csr_matrix):
            adata.layers["spliced"] = csr_matrix(adata.layers["spliced"])
        if not isinstance(adata.layers["unspliced"], csr_matrix):
            adata.layers["unspliced"] = csr_matrix(adata.layers["unspliced"])

        adata.obs_names_make_unique()
        adata = adata[~adata.to_df().duplicated(), :]

        print("  Preprocessing...")
        if zero_threshold:
            vv.preprocess(adata, n_gene=n_gene, npc=npc, min_shared_counts=0, min_shared_cells=0)
        else:
            vv.preprocess(adata, n_gene=n_gene, npc=npc)

        print("  Training VAE...")
        figure_path = output_dir / "figure"
        figure_path.mkdir(exist_ok=True)

        config = {"batch_size": batch_size}
        vae = vv.VAE(adata, tmax=10, dim_z=dim_z, device=device, config=config)
        vae.train(adata, figure_path=str(figure_path), embed='umap', config=config)
        vae.save_anndata(adata, key='vae', file_path=str(output_dir))

        adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)
        adata.obs['vae_time'] = adata.obs['vae_time'].astype(float)

        print("  Post-analysis...")
        _, _ = vv.post_analysis(
            adata, output_basename,
            methods=['VeloVAE'], keys=['vae'],
            cluster_key=cluster_key, n_jobs=n_jobs
        )

        del vae
        vae = None
        gc.collect()

        print("  Generating plots...")

        # 1. UMAP scatter
        for fmt in ['png', 'pdf']:
            save_dir = png_dir if fmt == 'png' else pdf_dir
            save_path = save_dir / f"{output_basename}_umap.{fmt}"
            scv.pl.scatter(
                adata, basis='umap', vkey='vae_velocity', color=cluster_key,
                palette=PALETTE, size=100, alpha=0.6,
                legend_loc='right margin', legend_fontsize=9, fontsize=None,
                title='VeloVAE', dpi=400, show=False, save=str(save_path)
            )
            plt.close('all')

        # 2. Stream plot (with PDF fallback)
        for fmt in ['png', 'pdf']:
            save_dir = png_dir if fmt == 'png' else pdf_dir
            save_path = save_dir / f"{output_basename}_stream.{fmt}"
            save_stream_plot_with_fallback(
                adata, save_path,
                size=100, alpha=0.6, vkey='vae_velocity',
                V=adata.obsm['vae_velocity_umap'], basis="umap",
                color=cluster_key, legend_fontsize=9, legend_loc='right margin',
                fontsize=None, density=2, dpi=400, arrow_size=1, linewidth=1,
                palette=PALETTE, title='VeloVAE'
            )

        # 3. Grid plot
        for fmt in ['png', 'pdf']:
            save_dir = png_dir if fmt == 'png' else pdf_dir
            save_path = save_dir / f"{output_basename}_grid.{fmt}"
            scv.pl.velocity_embedding_grid(
                adata, vkey='vae_velocity', size=100, alpha=0.6,
                V=adata.obsm['vae_velocity_umap'], basis="umap",
                color=cluster_key, legend_fontsize=9, legend_loc='right margin',
                fontsize=None, density=0.8, dpi=400, arrow_size=1, linewidth=0.3,
                palette=PALETTE, title='VeloVAE', save=str(save_path), show=False
            )
            plt.close('all')

        # 4. Pseudotime
        vae_time = adata.obs['vae_time'].values
        if np.max(vae_time) != np.min(vae_time):
            vae_time_normalized = (vae_time - np.min(vae_time)) / (np.max(vae_time) - np.min(vae_time))
        else:
            vae_time_normalized = np.zeros_like(vae_time)
        adata.obs['vae_time_normalized'] = vae_time_normalized

        for fmt in ['png', 'pdf']:
            save_dir = png_dir if fmt == 'png' else pdf_dir
            save_path = save_dir / f"{output_basename}_pseudotime.{fmt}"
            scv.pl.scatter(
                adata, basis='umap', color='vae_time_normalized',
                cmap='gnuplot', size=100, dpi=400, figsize=(8, 6),
                colorbar=True, title='VeloVAE', save=str(save_path), show=False
            )
            plt.close('all')

        adata.write(output_h5ad, compression='lzf')
        print(f"  Done: {output_h5ad.name}")

    except Exception as e:
        print(f"  Error: {e}")
        raise
    finally:
        cleanup_resources(adata, vae)


def load_metadata_file(metadata_path: Path) -> pd.DataFrame:
    """Load metadata file for batch processing."""
    suffix = metadata_path.suffix.lower()
    if suffix == '.csv':
        sep = ','
    elif suffix in ['.tsv', '.txt']:
        sep = '\t'
    else:
        with open(metadata_path) as f:
            first_line = f.readline()
            sep = '\t' if '\t' in first_line else ','

    df = pd.read_csv(metadata_path, sep=sep)

    required_cols = ['dataset_name', 'file_path', 'cluster_key']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if 'dimred_key' not in df.columns:
        df['dimred_key'] = 'X_umap'
    if 'dim_z' not in df.columns:
        df['dim_z'] = 5
    if 'zero_threshold' not in df.columns:
        df['zero_threshold'] = False

    df['dim_z'] = df['dim_z'].astype(int)
    df['zero_threshold'] = df['zero_threshold'].astype(bool)

    return df


def main(args):
    output_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.metadata_file:
        metadata_df = load_metadata_file(Path(args.metadata_file))
        print(f"Batch mode: {len(metadata_df)} datasets")

        for _, row in metadata_df.iterrows():
            file_path = Path(row['file_path'])
            if not file_path.exists():
                print(f"Skipping (not found): {file_path}")
                continue

            dataset_output_dir = output_dir / row['dataset_name']
            dataset_output_dir.mkdir(exist_ok=True)

            try:
                run_velovae_analysis(
                    input_path=file_path,
                    output_dir=dataset_output_dir,
                    fig_dir=fig_dir,
                    cluster_key=row['cluster_key'],
                    dimred_key=row['dimred_key'],
                    dim_z=int(row['dim_z']),
                    zero_threshold=bool(row['zero_threshold']),
                    device=args.device,
                    n_jobs=args.n_jobs
                )
            except Exception as e:
                print(f"Failed: {row['dataset_name']}: {e}")
    else:
        if not args.input:
            print("Error: --input or --metadata-file required")
            sys.exit(1)
        if not args.cluster_key:
            print("Error: --cluster-key required")
            sys.exit(1)

        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: File not found: {input_path}")
            sys.exit(1)

        run_velovae_analysis(
            input_path=input_path,
            output_dir=output_dir,
            fig_dir=fig_dir,
            cluster_key=args.cluster_key,
            dimred_key=args.dimred_key,
            dim_z=args.dim_z,
            zero_threshold=args.zero_threshold,
            device=args.device,
            n_jobs=args.n_jobs
        )

    print("\nComplete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VeloVAE velocity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input", help="Input H5AD file")
    input_group.add_argument("--metadata-file", help="Metadata file for batch processing")

    parser.add_argument("--output-dir", required=True, help="Output directory for H5AD files")
    parser.add_argument("--fig-dir", required=True, help="Output directory for figures")

    parser.add_argument("--cluster-key", default=None, help="Cell type column name (required)")
    parser.add_argument("--dimred-key", default="X_umap", help="Dimensionality reduction key")
    parser.add_argument("--dim-z", type=int, default=5, help="Latent dimension (5 for real, 4 for simulated)")
    parser.add_argument("--zero-threshold", action="store_true", default=False,
                        help="Use zero thresholds (for simulated data)")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for post-analysis")

    args = parser.parse_args()
    main(args)
