
from typing import Optional, Tuple
import re
import pandas as pd
import anndata
def find_cluster_column(adata: anndata.AnnData) -> Optional[Tuple[str, pd.Series]]:
    """
    Find a clustering/annotation column in `adata.obs` by priority.

    Args:
        adata: an AnnData object

    Returns:
        tuple: (matched column name, corresponding values) or None if not found
    """
    
    # Define priority groups and possible column name variants (supports regex)
    priority_patterns = [
        # Priority 1: celltype-related (celltype, cell_type, CellType...)
        [r"^cell[\W_]?type$", r"^Cell[\W_]?Type$", r"^celltype$", r"^predicted_cell_type$"],
        # Priority 2: clustername-related (cluster_name, ClusterName...)
        [r"^cluster[\W_]?name$", r"^Cluster[\W_]?Name$", r"^clustername$"],
        # Priority 3: annotation
        [r"^annotation$"],
        # Priority 4: phase (direct match)
        [r"^phase$", r"^cell_cycle_phase$"],
        # Priority 5: cluster (cluster, Cluster...)
        [r"^cluster$", r"^Cluster$", r"^clusters$"],
        # Priority 6: time
        [r"^time$"]
    ]

    # Iterate through priority groups
    for patterns in priority_patterns:
        # Check all possible column names
        for col in adata.obs.columns:
            # Normalize by removing non-alphanumeric/underscore and lowercasing, then match
            normalized_col = re.sub(r"[\W_]", "", col).lower()
            for pattern in patterns:
                # Normalize the regex pattern and match
                std_pattern = re.sub(r"[\W_]", "", pattern).lower().strip("^$")
                if re.match(f"^{std_pattern}$", normalized_col):
                    return col
    # No matching column found
    print("Warning: No cluster/celltype column found in adata.obs!")
    return None