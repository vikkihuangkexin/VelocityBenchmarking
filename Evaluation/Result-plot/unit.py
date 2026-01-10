import os
from scipy.stats import gaussian_kde
import numpy as np
import scvelo as scv
from scvelo.core import l2_norm, prod_sum
from scvelo.utils import get_indices
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
import re
import anndata
#coherence of the velocity vector field（velocity confidence）
import numpy as np

def cosine_similarity_percentage(velocity_T, velocity_O, threshold=0.7):

    assert velocity_T.shape[0] == velocity_O.shape[0], "Need same shape."
    dot_product = np.dot(velocity_T, velocity_O.T)
    norm_T = np.linalg.norm(velocity_T, axis=1)
    norm_O = np.linalg.norm(velocity_O, axis=1)
    similarity_matrix = dot_product / (norm_T[:, np.newaxis] * norm_O)
    below_threshold = np.sum(similarity_matrix < threshold)
    percentage = below_threshold / similarity_matrix.size
    return similarity_matrix, percentage

import re
from typing import Optional, Tuple
import pandas as pd

def find_cluster_column(adata: anndata.AnnData) -> Optional[str]:
    """
    Find a cluster / annotation column in `adata.obs` by priority.

    The function searches `adata.obs` for common column names that represent
    cell type labels, cluster names, annotations, cell cycle phase, or time.
    It returns the first matching column name according to a predefined
    priority list, or `None` if no match is found.

    Args:
        adata: AnnData object containing `obs` with metadata columns.

    Returns:
        The matching column name (str) if found, otherwise None.
    """
    # Priority groups of candidate column name patterns (regular expressions)
    priority_patterns = [
        # Priority 1: cell type related (celltype, cell_type, CellType, predicted_cell_type)
        [r"^cell[\W_]?type$", r"^Cell[\W_]?Type$", r"^celltype$", r"^predicted_cell_type$"],
        # Priority 2: cluster name related (cluster_name, ClusterName, clustername)
        [r"^cluster[\W_]?name$", r"^Cluster[\W_]?Name$", r"^clustername$"],
        # Priority 3: annotation
        [r"^annotation$"],
        # Priority 4: phase (cell cycle phase)
        [r"^phase$", r"^cell_cycle_phase$"],
        # Priority 5: cluster (cluster, Cluster, clusters)
        [r"^cluster$", r"^Cluster$", r"^clusters$"],
        # Priority 6: time
        [r"^time$"]
    ]

    # Iterate through priority groups and try to match any column name
    for patterns in priority_patterns:
        for col in adata.obs.columns:
            # Normalize column name: remove non-alphanumeric characters and lowercase
            normalized_col = re.sub(r"[\W_]", "", col).lower()
            for pattern in patterns:
                # Normalize pattern similarly and match against normalized column
                std_pattern = re.sub(r"[\W_]", "", pattern).lower().strip("^$")
                if re.match(f"^{std_pattern}$", normalized_col):
                    return col

    # No matching column found
    print("Warning: No cluster/cell type column found in adata.obs!")
    return None
