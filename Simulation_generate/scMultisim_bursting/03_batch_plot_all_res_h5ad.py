# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import anndata as ann
import scvelo as scv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.lines import Line2D

base_dir = "./bursting_benchmark"

scv.settings.set_figure_params(
    "scvelo",
    dpi=120,
    dpi_save=300,
    transparent=False,
    frameon=False
)

def gaussian_weights(dist):
    sigma = np.median(dist)
    if not np.isfinite(sigma) or sigma <= 1e-8:
        sigma = max(np.mean(dist), 1e-6)
    w = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    s = w.sum()
    if not np.isfinite(s) or s <= 1e-12:
        return np.ones_like(dist) / len(dist)
    return w / s

def project_velocity_to_tsne(X_high, X_tsne, V_high, k=30, ridge=1e-3):
    n, d = X_high.shape
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean").fit(X_high)
    dists, inds = nbrs.kneighbors(X_high)

    V_tsne = np.zeros((n, 2), dtype=float)

    for i in range(n):
        idx = inds[i, 1:]   # drop self
        dist_i = dists[i, 1:]

        dX = X_high[idx] - X_high[i]
        dY = X_tsne[idx] - X_tsne[i]

        w = gaussian_weights(dist_i)
        W = np.diag(w)

        XtWX = dX.T @ W @ dX + ridge * np.eye(d)
        XtWY = dX.T @ W @ dY
        B = np.linalg.solve(XtWX, XtWY)

        V_tsne[i] = V_high[i] @ B

    return V_tsne

def smooth_velocity(X_tsne, V_tsne, k=20):
    n = X_tsne.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean").fit(X_tsne)
    dists, inds = nbrs.kneighbors(X_tsne)
    V_sm = np.zeros_like(V_tsne)

    for i in range(n):
        idx = inds[i, 1:]
        dist_i = dists[i, 1:]
        w = gaussian_weights(dist_i)
        V_sm[i] = (V_tsne[idx] * w[:, None]).sum(axis=0)

    return V_sm

def make_pop_legend(ax, palette, pop_levels):
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=palette[x], markersize=8, linestyle="")
        for x in pop_levels
    ]
    ax.legend(handles, pop_levels, title="pop",
              frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

def process_one_h5ad(h5ad_path):
    outdir = os.path.dirname(h5ad_path)
    print(f"Processing: {h5ad_path}")

    adata = ann.read_h5ad(h5ad_path)

    # fix obsm type
    if "tsne" in adata.obsm:
        adata.obsm["tsne"] = np.asarray(adata.obsm["tsne"])
    if "X_tsne" not in adata.obsm and "tsne" in adata.obsm:
        adata.obsm["X_tsne"] = np.asarray(adata.obsm["tsne"])
    if "X_tsne" in adata.obsm:
        adata.obsm["X_tsne"] = np.asarray(adata.obsm["X_tsne"])

    if "X_tsne" not in adata.obsm:
        raise ValueError("No tsne / X_tsne found in obsm")

    if "pop" not in adata.obs.columns:
        raise ValueError("No pop found in obs")

    if "pseudotime" not in adata.obs.columns:
        if "cell_time" in adata.obs.columns:
            adata.obs["pseudotime"] = adata.obs["cell_time"]
        else:
            adata.obs["pseudotime"] = np.arange(adata.n_obs)

    adata.obs["pop"] = adata.obs["pop"].astype(str)
    pop_levels = sorted(pd.unique(adata.obs["pop"]))

    base_colors = scv.pl.palettes.default_20
    palette = {lab: base_colors[i % len(base_colors)] for i, lab in enumerate(pop_levels)}
    point_colors = adata.obs["pop"].map(palette).values

    X_tsne = np.asarray(adata.obsm["X_tsne"], dtype=float)
    X_high = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X, dtype=float)

    if "ground_truth_velocity" in adata.layers:
        V_high = (
            adata.layers["ground_truth_velocity"].toarray()
            if hasattr(adata.layers["ground_truth_velocity"], "toarray")
            else np.asarray(adata.layers["ground_truth_velocity"], dtype=float)
        )
    elif "velocity" in adata.layers:
        V_high = (
            adata.layers["velocity"].toarray()
            if hasattr(adata.layers["velocity"], "toarray")
            else np.asarray(adata.layers["velocity"], dtype=float)
        )
    else:
        raise ValueError("No ground_truth_velocity / velocity in layers")

    V_tsne = project_velocity_to_tsne(X_high, X_tsne, V_high, k=30, ridge=1e-3)
    V_tsne = smooth_velocity(X_tsne, V_tsne, k=20)

    # for per-cell arrows
    vlen = np.sqrt((V_tsne ** 2).sum(axis=1))
    eps = 1e-8
    vlen_safe = np.maximum(vlen, eps)
    V_unit = V_tsne / vlen_safe[:, None]
    arrow_len = 0.8
    V_quiver = V_unit * arrow_len
    keep_arrow = vlen > np.quantile(vlen, 0.10)

    # 1) tSNE by pop
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=point_colors,
        s=28,
        alpha=0.85,
        linewidths=0
    )
    make_pop_legend(ax, palette, pop_levels)
    ax.set_title("tSNE by pop")
    ax.set_xlabel("tSNE1")
    ax.set_ylabel("tSNE2")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scmultisim_tsne_by_pop.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2) tSNE by pseudotime
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=adata.obs["pseudotime"].values,
        s=28,
        alpha=0.85,
        linewidths=0
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("pseudotime")
    ax.set_title("tSNE by pseudotime")
    ax.set_xlabel("tSNE1")
    ax.set_ylabel("tSNE2")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scmultisim_tsne_by_pseudotime.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3) tSNE + velocity stream by pop
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=point_colors,
        s=28,
        alpha=0.8,
        linewidths=0
    )

    scv.pl.velocity_embedding_stream(
        adata,
        basis="tsne",
        X=X_tsne,
        V=V_tsne,
        color=None,
        legend_loc="none",
        density=1.0,
        smooth=0.7,
        linewidth=1.0,
        size=0,
        alpha=0.0,
        show=False,
        ax=ax
    )

    make_pop_legend(ax, palette, pop_levels)
    ax.set_title("tSNE with projected velocity stream (colored by pop)")
    ax.set_xlabel("tSNE1")
    ax.set_ylabel("tSNE2")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scmultisim_tsne_velocity_stream_by_pop.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4) tSNE + per-cell arrows by pop
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=point_colors,
        s=24,
        alpha=0.85,
        linewidths=0
    )

    idx = np.where(keep_arrow)[0]
    ax.quiver(
        X_tsne[idx, 0],
        X_tsne[idx, 1],
        V_quiver[idx, 0],
        V_quiver[idx, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="black",
        width=0.0018,
        headwidth=3.0,
        headlength=4.0,
        headaxislength=3.5,
        alpha=0.9
    )

    make_pop_legend(ax, palette, pop_levels)
    ax.set_title("tSNE with projected per-cell velocity arrows (colored by pop)")
    ax.set_xlabel("tSNE1")
    ax.set_ylabel("tSNE2")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scmultisim_tsne_velocity_arrows_by_pop.png"), dpi=300, bbox_inches="tight")
    plt.close()

def find_h5ad_files(base_dir):
    found = []
    for root, dirs, files in os.walk(base_dir):
        if "res.h5ad" in files:
            found.append(os.path.join(root, "res.h5ad"))
    return sorted(found)

all_h5ad = find_h5ad_files(base_dir)
print(f"Found {len(all_h5ad)} h5ad files.")

for f in all_h5ad:
    try:
        process_one_h5ad(f)
    except Exception as e:
        print(f"[ERROR] {f}")
        print(f"        {e}")

print("Done.")
