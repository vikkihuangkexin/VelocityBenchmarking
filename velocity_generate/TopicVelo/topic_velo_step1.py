import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.io import mmwrite
from scipy.sparse import csr_matrix, hstack

import tomotopy as tp
from tomotopy.coherence import Coherence
from tomotopy.utils import Corpus

import scvelo as scv
import scanpy as sc
from scvelo.utils import get_transition_matrix

from topicvelo.tm_utils import cells_to_documents, FastTopics_cluster_assign, TopicGeneFiltering, remove_U, TopicGenesQC, aggregate_clusters
from topicvelo.embed_pl import Plot_Topics, pl_TopTopicGenes, Plot_Genes, Plot_Velocity, mfpt_plot, comparision_stacked_bar_plot, comparision_violin_plot_v2, relative_flux_plot
from topicvelo.dist_pl import Experimental_JD_Plot, Burst_Simulation_JD_Plot, OS_Analytical_JD_Plot, ExpJD_Cluster_Focus_HeatMap
from topicvelo.transcription_simulation import GeometricBurstTranscription, JointDistributionAnalysis, JointDistributionAnalysis_exp
from topicvelo.inference_tools import Burst_Inference_Gene, KLdivergence, topic_threshold_heuristic_plot, Burst_Inference
from topicvelo.transition_matrix import Combined_Topics_Transitions, get_cells_indices, velocity_graph
from topicvelo.vel_eval_utils import *

import argparse

def main(data_dir, save_dir):
    # Accept either a file path or a directory containing a single .h5ad
    if os.path.isdir(data_dir):
        matches = [f for f in os.listdir(data_dir) if f.endswith('.h5ad')]
        if len(matches) == 0:
            raise FileNotFoundError(f'No .h5ad files found in {data_dir}')
        data_path = os.path.join(data_dir, matches[0])
    else:
        data_path = data_dir

    adata = sc.read_h5ad(data_path)
    num_id = os.path.basename(data_path).split('.')[0]
    adata.var['gene'] = adata.var.index
    adata.var_names_make_unique()
    adata.var.index = adata.var['gene'].tolist()

    genes_S = adata.var['gene'].to_list()
    genes_U = [g + '_U' for g in genes_S]
    gene_names = np.hstack((genes_S, genes_U))

    S_U = csr_matrix(hstack([csr_matrix(adata.layers['spliced']), csr_matrix(adata.layers['unspliced'])]), dtype=np.int32)
    os.makedirs(save_dir, exist_ok=True)
    mmwrite(os.path.join(save_dir, f"{num_id}_scNT_HH_filtered_SU_Counts.mtx"), S_U)
    pd.DataFrame(gene_names).to_csv(os.path.join(save_dir, f"{num_id}_scNT_HH_filtered_SU_Genes_names.csv"))
    pd.DataFrame(adata.obs_names.to_list()).to_csv(os.path.join(save_dir, f"{num_id}_scNT_HH_filtered_SU_Cells_names.csv"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data path select.")
    parser.add_argument("--save_dir", default='.../example/result/TopicVelo/...')
    parser.add_argument("--data_dir", default='.../example/data/...')
    args = parser.parse_args()  
    main(args.data_dir, args.save_dir)