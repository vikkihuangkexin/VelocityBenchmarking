import argparse
import os
import datetime
import numpy as np
import torch
import pyro
import scanpy as sc
import matplotlib
matplotlib.use("Agg")

from velocycle import (
    preprocessing,
    utils,
    cycle,
    phases,
    phase_inference_model
)

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ======================
    # Load data
    # ======================
    data = sc.read_h5ad(args.data_dir)
    file_id = os.path.splitext(os.path.basename(args.data_dir))[0]
    print(f"Input data: {data.n_obs} cells, {data.n_vars} genes")

    # ======================
    # Gene filtering (fixed)
    # ======================
    data = data[:, data.layers["unspliced"].toarray().mean(0) > 0.1].copy()
    data = data[:, data.layers["spliced"].toarray().mean(0) > 0.1].copy()

    print(f"Filtered data: {data.n_obs} cells, {data.n_vars} genes")

    # ======================
    # Batch (single batch)
    # ======================
    data.obs["batch"] = "single_batch"
    batch_design_matrix = preprocessing.make_design_matrix(data, ids="batch")

    # ======================
    # Cell cycle score
    # ======================
    if "phase" not in data.obs.columns:
        sc.tl.score_genes_cell_cycle(
            data,
            s_genes=utils.S_genes_human,
            g2m_genes=utils.G2M_genes_human
        )

    preprocessing.normalize_total(data)

    # ======================
    # Cycle prior
    # ======================
    keep_genes = utils.get_cycling_gene_set(size="Medium", species="Human")
    cycle_prior = cycle.Cycle.trivial_prior(
        gene_names=keep_genes,
        harmonics=1
    )

    cycle_prior, data_fit = preprocessing.filter_shared_genes(
        cycle_prior, data, filter_type="intersection"
    )

    # ======================
    # Prior parameters
    # ======================
    S = data_fit.layers["spliced"].toarray()
    nu0 = np.log(S.mean(axis=0) + 1e-6)
    nu0std = np.std(np.log(S + 1), axis=0) / 2

    cycle_prior.set_means(np.vstack((nu0, 0 * nu0, 0 * nu0)))
    cycle_prior.set_stds(np.vstack((nu0std, 0.5 * nu0std, 0.5 * nu0std)))

    # ======================
    # Phase prior
    # ======================
    phase_prior = phases.Phases.from_pca_heuristic(
        data_fit,
        genes_to_use=utils.get_cycling_gene_set(size="Small", species="Human"),
        layer="S_sz",
        concentration=10.0,
        plot=False
    )

    # ======================
    # Model
    # ======================
    pyro.clear_param_store()

    sigma_dnu = torch.ones(
        (batch_design_matrix.shape[1], S.shape[1], 1),
        device=device
    ) * 0.001

    metapar = preprocessing.preprocess_for_phase_estimation(
        anndata=data_fit,
        cycle_obj=cycle_prior,
        phase_obj=phase_prior,
        design_mtx=batch_design_matrix,
        n_harmonics=1,
        σΔν=sigma_dnu,
        device=device
    )

    model = phase_inference_model.PhaseFitModel(metaparams=metapar)

    # ======================
    # Optimizer (only 3 tunable params)
    # ======================
    gamma = args.lr_end / args.lr_start
    lrd = gamma ** (1 / args.num_steps)

    optimizer = pyro.optim.ClippedAdam({
        "lr": args.lr_start,
        "lrd": lrd,
        "betas": (0.8, 0.99)
    })

    print("Training...")
    start = datetime.datetime.now()
    model.fit(optimizer=optimizer, num_steps=args.num_steps)
    print(f"Done. Runtime: {datetime.datetime.now() - start}")

    # ======================
    # Save result
    # ======================
    result = data_fit.copy()
    result.obs["velocycle_phase"] = model.phase_pyro.phis

    os.makedirs(args.save_dir, exist_ok=True)
    out = os.path.join(args.save_dir, f"{file_id}_velocycle.h5ad")
    result.write_h5ad(out)

    print(f"Saved to: {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Velocycle (minimal CLI)")

    parser.add_argument("--data_dir", required=True, help="Input h5ad")
    parser.add_argument("--save_dir", required=True, help="Output directory")

    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--lr_start", type=float, default=0.03)
    parser.add_argument("--lr_end", type=float, default=0.005)

    args = parser.parse_args()
    main(args)
