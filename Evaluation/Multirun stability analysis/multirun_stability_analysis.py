import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from unit import find_h5ad

def load_velocity(file_path, vkey='velocity', use_low_dim=False, dataset_type='real'):
    """Load velocity vectors"""
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    adata = sc.read_h5ad(file_path)
    cell_names = adata.obs_names.tolist()

    if use_low_dim:
        # Low-dimensional vector
        suffix = '_umap'
        vkey_low = f"{vkey}{suffix}"
        if vkey_low not in adata.obsm:
            raise KeyError(f"Missing low-dimensional vector: {vkey_low}")
        velocity = adata.obsm[vkey_low]
    else:
        # High-dimensional vector
        if vkey not in adata.layers:
            raise KeyError(f"Missing velocity vector: {vkey}")
        velocity = adata.layers[vkey]

    # Convert to dense array
    if hasattr(velocity, 'toarray'):
        velocity = velocity.toarray()

    return velocity, cell_names


def cosine_similarity(v1, v2):
    """Calculate cosine similarity of two vectors, normalized to [0,1]"""
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_sim = np.dot(v1, v2) / (norm1 * norm2)
    return (cos_sim + 1) / 2  # Normalize to [0,1]


def calculate_pairwise_similarity(velocity1, velocity2, cells1, cells2):
    """
    Calculate cosine similarity of matched cells between two replicates.
    Returns: (mean, median)
    """
    # Find common cells
    common_cells = set(cells1) & set(cells2)
    if len(common_cells) == 0:
        raise ValueError("The two files have no common cells!")

    # Get indices of common cells
    idx1 = [cells1.index(cell) for cell in common_cells]
    idx2 = [cells2.index(cell) for cell in common_cells]

    # Calculate similarity for each cell
    similarities = [
        cosine_similarity(velocity1[i], velocity2[j])
        for i, j in zip(idx1, idx2)
    ]

    return np.mean(similarities), np.median(similarities)


def analyze_dataset(inputdata_dir, tool_name, ID, file_prefix,
                    vkey='velocity', use_low_dim=False, n_repeats=5):
    """Analyze all replicate combinations for a single dataset"""

    dataset_type = 'real' if ID == '8' else 'simulated'

    print(f"\nProcessing: {ID}")

    # Load all replicates
    velocities = {}
    cell_names_dict = {}

    for r in range(0, n_repeats):
        files = find_h5ad(f'{inputdata_dir}/{r}', ID, 'velo.h5ad')
        if len(files) == 0:
            print(f'{ID}/{r}')
            return
        file_path = Path(files[0])
        try:
            velocity, cell_names = load_velocity(
                file_path, vkey, use_low_dim, dataset_type
            )
            velocities[r] = velocity
            cell_names_dict[r] = cell_names
        except Exception as e:
            print(f"  Warning: Failed to load r{r}: {e}")
            raise

    # Calculate similarities for all pairwise combinations
    results = []
    all_means = []
    all_medians = []

    for r1, r2 in combinations(range(0, n_repeats), 2):
        mean_sim, median_sim = calculate_pairwise_similarity(
            velocities[r1], velocities[r2],
            cell_names_dict[r1], cell_names_dict[r2]
        )

        all_means.append(mean_sim)
        all_medians.append(median_sim)

        results.append({
            'tool_name': tool_name,
            'dataset_name': ID,
            'group': f"r{r1}_vs_r{r2}",
            'group_cosine': mean_sim,
            'group_median': median_sim
        })

    # Calculate overall mean and median
    avg_cosine = np.mean(all_means)
    avg_median = np.mean(all_medians)

    # Add overall statistics to each row
    for row in results:
        row['average_cosine'] = avg_cosine
        row['average_median'] = avg_median

    print(f"  ✓ Completed (average cosine: {avg_cosine:.4f}, average median: {avg_median:.4f})")

    return pd.DataFrame(results)


def calculate_velocity_stability(
        inputdata_dir,
        tool_name,
        ID,
        data_file,
        output_path,
        vkey='velocity',
        use_low_dim=False,
        n_repeats=5
):
    """
    Calculate stability of RNA Velocity tools

    Parameters:
    -----------
    inputdata_dir : str
        Directory path containing tool results (e.g., base_result/VeloVAE)
    tool_name : str
        Tool name (e.g., 'VeloVAE')
    output_path : str
        Output CSV file path
    vkey : str
        Velocity vector key name
    use_low_dim : bool
        Whether to use low-dimensional vectors (False=high-dim, True=low-dim)
    n_repeats : int
        Number of repeats

    Returns:
    --------
    pd.DataFrame : Analysis results
        Columns: tool_name, dataset_name, group, group_cosine, group_median,
            average_cosine, average_median
    """

    inputdata_dir = Path(inputdata_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dataset configuration
    all_results = []

    try:
        df = analyze_dataset(
            inputdata_dir, tool_name,
            ID, data_file,
            vkey, use_low_dim, n_repeats
        )
        all_results.append(df)
    except Exception as e:
        print(f"  ✗ Skipping dataset: {e}\n")
        return

    if not all_results:
        raise FileNotFoundError(
            f"No valid dataset files found!\n"
            f"Please check directory: {inputdata_dir}"
        )

    # Merge and save
    final_df = pd.concat(all_results, ignore_index=True)

    # Ensure column order
    final_df = final_df[[
        'tool_name', 'dataset_name', 'group',
        'group_cosine', 'group_median',
        'average_cosine', 'average_median'
    ]]

    final_df.to_csv(output_path, index=False, float_format='%.6f')

    print(f"\n✓ Analysis completed, results saved: {output_path}\n")

    return final_df


