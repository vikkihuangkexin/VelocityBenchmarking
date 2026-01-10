Project summary

This repository contains a utility function to compute and visualize velocity confidence for single‑cell RNA velocity analyses. The function supports two workflows:

AnnData workflow: pass an AnnData object with precomputed velocities and neighbors; the function will call scv.tl.velocity_confidence and produce plots and CSV outputs.

Matrix workflow: pass raw velocity and distances matrices to compute per‑cell confidence scores directly and generate the same visual outputs.

The visualization includes:

Scatter plots on a chosen embedding (default: UMAP) colored by velocity confidence (saved as PDF and PNG).

Kernel density estimate (KDE) of the velocity confidence distribution (saved as PDF and CSV).

Requirements

Python 3.8+

scvelo

scanpy (if using AnnData)

numpy, scipy, pandas

matplotlib, seaborn

Install with pip or conda as appropriate, for example:

conda install -c conda-forge scvelo numpy scipy pandas matplotlib seaborn

Usage examples

Using an AnnData object

from your_module import velocity_confidence_plot

# adata must have velocities and neighbors computed
# e.g., scv.tl.velocity(adata); sc.pp.neighbors(adata)
velocity_confidence_plot(adata=adata, plot_save_path='results/', basis='umap')

Using raw matrices

# velocity: numpy array of shape (n_cells, n_features)
# distances: CSR matrix from sc.pp.neighbors or equivalent
velocity_length, velocity_confidence = velocity_confidence_plot(
    velocity=velocity_matrix,
    distances=distances_csr,
    plot_save_path='results/'
)

Outputs

velocity_confidence_scatter.pdf and .png — embedding scatter colored by confidence

velocity_confidence_KDE.pdf — KDE plot of confidence distribution

velocity_confidence_row.csv — per‑cell confidence values (from AnnData workflow)

KDE_density_1w.csv — KDE density values sampled across [0,1]

Notes and recommendations

When using the AnnData workflow, ensure velocities and neighbors are computed beforehand (scv.tl.velocity, sc.pp.neighbors).

For the matrix workflow, verify that distances is a CSR matrix or convertible to one and that velocity rows correspond to the same cell ordering used to compute distances.

The KDE uses a large grid (10000 points) to produce a smooth density curve; adjust if memory or performance is a concern.

The function saves plots and CSVs to the provided plot_save_path; ensure the directory exists and is writable.
