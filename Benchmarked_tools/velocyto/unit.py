
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


def vlm_to_adata(vlm, n_comps=30, basis="umap", trans_mats=None, cells_ixs=None):
    """ Conversion function from the velocyto world to the dynamo world.
		Code original from scSLAM-seq repository

    Parameters
    ----------
		vlm: VelocytoLoom Object
			The VelocytoLoom object that will be converted into adata.
		n_comps: `int` (default: 30)
			The number of pc components that will be stored.
		basis: `str` (default: `umap`)
			The embedding that will be used to store the vlm.ts attribute. Note that velocyto doesn't usually use
			umap as embedding although `umap` as set as default for the convenience of dynamo itself.
		trans_mats: None or dict
			A dict of all relevant transition matrices
		cell_ixs: list of int
			These are the indices of the subsampled cells

    Returns
    -------
		adata: AnnData object
	"""
    from collections import OrderedDict

    # set obs, var
    CA = {k:v for k,v in vlm.ca.items() if len(v.shape)==1}
    RA = {k: v for k, v in vlm.ra.items() if len(v.shape) == 1}
    obs, var = pd.DataFrame(CA), pd.DataFrame(RA)
    if "CellID" in obs.keys():
        obs["obs_names"] = obs.pop("CellID")
    if "Gene" in var.keys():
        var["var_names"] = var.pop("Gene")

    if hasattr(vlm, "q"):
        var["gamma_b"] = vlm.q
    if hasattr(vlm, "gammas"):
        var["gamma"] = vlm.gammas
    if hasattr(vlm, "R2"):
        var["gamma_r2"] = vlm.R2

    # rename clusters to louvain
    try:
        ix = np.where(obs.columns == "Clusters")[0][0]
        obs_names = list(obs.columns)
        obs_names[ix] = "louvain"
        obs.columns = obs_names

        # make louvain a categorical field
        obs["louvain"] = pd.Categorical(obs["louvain"])
    except:
        print("Could not find a filed 'Clusters' in vlm.ca.")

    # set layers basics
    layers = OrderedDict(
        unspliced=csr_matrix(vlm.U.T),
        spliced=csr_matrix(vlm.S.T),
        velocity_S=csr_matrix(vlm.velocity.T),
    )

    # set X_spliced / X_unspliced
    if hasattr(vlm, "S_norm"):
        layers["X_spliced"] = csr_matrix(2**vlm.S_norm - 1).T
    if hasattr(vlm, "U_norm"):
        layers["X_unspliced"] = csr_matrix(2**vlm.U_norm - 1).T
    if hasattr(vlm, "S_sz") and not hasattr(vlm, "S_norm"):
        layers["X_spliced"] = csr_matrix(vlm.S_sz).T
    if hasattr(vlm, "U_sz") and hasattr(vlm, "U_norm"):
        layers["X_unspliced"] = csr_matrix(vlm.U_sz).T

    # set M_s / M_u
    if hasattr(vlm, "Sx"):
        layers["M_s"] = csr_matrix(vlm.Sx).T
    if hasattr(vlm, "Ux"):
        layers["M_u"] = csr_matrix(vlm.Ux).T
    if hasattr(vlm, "Sx_sz") and not hasattr(vlm, "Sx"):
        layers["M_s"] = csr_matrix(vlm.Sx_sz).T
    if hasattr(vlm, "Ux_sz") and hasattr(vlm, "Ux"):
        layers["M_u"] = csr_matrix(vlm.Ux_sz).T

    # set obsm
    obsm = {}
    obsm["X"] = vlm.pcs[:, : min(n_comps, vlm.pcs.shape[1])]
    # set basis and velocity on the basis
    if basis is not None:
        obsm["X_" + basis] = vlm.umap
        if hasattr(vlm, "delta_embedding"):
            obsm["velocity_" + basis] = vlm.delta_embedding

    # set transition matrix:
    uns = {}
    if hasattr(vlm, "corrcoef"):
        uns["transition_matrix"] = vlm.corrcoef
    if hasattr(vlm, "colorandum"):
        uns["louvain_colors"] = list(np.unique(vlm.colorandum))

    # add uns annotations
    if trans_mats is not None:
        for key, value in trans_mats.items():
            uns[key] = trans_mats[key]
    if cells_ixs is not None:
        uns["cell_ixs"] = cells_ixs
    obsp={}
    if hasattr(vlm, "embedding_knn"):

        n_neighbors = np.unique((vlm.embedding_knn > 0).sum(1)).min()
        ind_mat, dist_mat = adj_to_knn(vlm.embedding_knn, n_neighbors)
        uns["neighbors"] = {"indices": ind_mat}
        obsp = {"connectivities": vlm.embedding_knn}

    uns["dynamics"] = {
        "filter_gene_mode": None,
        "t": None,
        "group": None,
        "X_data": None,
        "X_fit_data": None,
        "asspt_mRNA": "ss",
        "experiment_type": "conventional",
        "normalized": True,
        "model": "deterministic",
        "has_splicing": True,
        "has_labeling": False,
        "has_protein": False,
        "use_smoothed": True,
        "NTR_vel": False,
        "log_unnormalized": True,
    }

    # set X
    if hasattr(vlm, "S_norm"):
        X = csr_matrix(vlm.S_norm.T)
    else:
        X = csr_matrix(vlm.S_sz.T) if hasattr(vlm, "S_sz") else csr_matrix(vlm.S.T)

    # create an anndata object with Dynamo characteristics.
    dyn_adata = AnnData(X=X, obs=obs, obsp=obsp, obsm=obsm, var=var, layers=layers, uns=uns)
    return dyn_adata