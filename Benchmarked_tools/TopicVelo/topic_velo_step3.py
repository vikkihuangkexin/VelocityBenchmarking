import numpy as np
import pandas as pd
import os
import argparse

import scanpy as sc
import scvelo as scv
from scipy.sparse import csr_matrix, hstack
from scipy.io import mmread

import tomotopy as tp
from tomotopy.coherence import Coherence
from tomotopy.utils import Corpus

from topicvelo.tm_utils import (
    cells_to_documents, FastTopics_cluster_assign,
    TopicGeneFiltering, remove_U
)
from topicvelo.transition_matrix import Combined_Topics_Transitions



def main(adata_path, step2_dir, input_dir, save_dir):
    # Accept directory or file for adata_path; if directory, pick first .h5ad
    if os.path.isdir(adata_path):
        matches = [f for f in os.listdir(adata_path) if f.endswith('.h5ad')]
        if len(matches) == 0:
            raise FileNotFoundError(f'No .h5ad files found in {adata_path}')
        adata_file = os.path.join(adata_path, matches[0])
    else:
        adata_file = adata_path

    sample_id = os.path.basename(adata_file).replace(".h5ad", "")
    print(f"[INFO] Processing sample: {sample_id}")

    os.makedirs(save_dir, exist_ok=True)

    adata = sc.read_h5ad(adata_file)
    adata.var_names_make_unique()

    genes_S = adata.var_names.to_list()
    genes_U = [g + "_U" for g in genes_S]
    gene_names = np.hstack((genes_S, genes_U))

    S_U = csr_matrix(
        hstack([adata.layers["spliced"], adata.layers["unspliced"]]),
        dtype=np.int32
    )

    scv.tl.velocity(adata, vkey="scvelo_stochastic_velocity")
    scv.tl.velocity_graph(adata, vkey="scvelo_stochastic_velocity")


    ks = [8]
    corpus = Corpus()
    for i in range(adata.n_obs):
        corpus.add_doc(cells_to_documents(S_U[i, :].A[0], gene_names))

    for K in ks:
        lda = tp.LDAModel(k=K, rm_top=0)
        lda.burn_in = 10
        lda.add_corpus(corpus)
        lda.train(iter=10, parallel=64)


    cellWeights = np.genfromtxt(
        os.path.join(step2_dir, f"{sample_id}_scNT_HH_fastTopics_CellWeights_k=8.csv"),
        delimiter=",",
        skip_header=1
    )
    # tolerate different naming conventions (CellWeights or CellWeights with different suffix)
    if cellWeights.size == 0:
        cellWeights = np.genfromtxt(
            os.path.join(step2_dir, f"{sample_id}_scNT_HH_fastTopics_fit_CellWeights_k=8.csv"),
            delimiter=",",
            skip_header=1
        )
    if cellWeights.ndim > 1:
        cellWeights = cellWeights[:, 1:]

    FastTopics_cluster_assign(adata, cellWeights, t_type="fastTopics")

    de_postmean = np.genfromtxt(
        os.path.join(step2_dir, f"{sample_id}_scNT_HH_de_postmean_k=8.csv"),
        delimiter=",",
        skip_header=1
    )
    if de_postmean.ndim > 1:
        de_postmean = de_postmean[:, 1:]

    de_lfsr = np.genfromtxt(
        os.path.join(step2_dir, f"{sample_id}_scNT_HH_de_lfsr_k=8.csv"),
        delimiter=",",
        skip_header=1
    )
    if de_lfsr.ndim > 1:
        de_lfsr = de_lfsr[:, 1:]


    ttgs, _, _ = TopicGeneFiltering(
        de_postmean, de_lfsr,
        lfc_up_th=0.5,
        lfc_down_th=-0.5,
        lfsr_up_th=0.001,
        lfsr_down_th=0.001
    )

    de_genes = pd.read_csv(
        os.path.join(input_dir, f"{sample_id}_scNT_HH_filtered_SU_Genes_names.csv"),
        index_col=0
    )

    SU_counts = mmread(
        os.path.join(input_dir, f"{sample_id}_scNT_HH_filtered_SU_Counts.mtx")
    ).tocsr()

    gene_sums = np.array(SU_counts.sum(axis=0)).ravel()
    mask = gene_sums > 0

    de_genes_filtered = de_genes.loc[mask]
    top_genes = remove_U(ttgs, de_genes_filtered.values.ravel())
    adata.uns["top_genes"] = [str(g) for g in top_genes]

    topics = [0, 1, 3, 4, 7]
    topic_th = [35] * len(topics)

    Combined_Topics_Transitions(
        adata,
        topics=topics,
        velocity_type="burst",
        recompute=True,
        recompute_matrix=True,
        steady_state_perc=95,
        topic_weights_th_percentile=topic_th,
        subset_save_prefix=f"{sample_id}_"
    )


    out_dir = os.path.join(save_dir, sample_id)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{sample_id}_topicvelo.h5ad")
    adata.write_h5ad(out_path)

    print(f"[DONE] Saved to {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata_path", default='./example', help="Path to input .h5ad file or directory (default: ./example)")
    parser.add_argument("--step2_dir", default='./example/output/topic-velo', help="Directory with Step2 outputs (default: ./example/output/topic-velo)")
    parser.add_argument("--input_dir", default='./example/output/topic-velo', help="Directory with Step1 outputs (default: ./example/output/topic-velo)")
    parser.add_argument("--save_dir", default='./example/output/topic-velo', help="Directory to save Step3 outputs (default: ./example/output/topic-velo)")
    args = parser.parse_args()
    main(
        args.adata_path,
        args.step2_dir,
        args.input_dir,
        args.save_dir
    )
