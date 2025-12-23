#python -m pip install seaborn
import pandas as pd
import anndata as ann
import scvelo as scv
import seaborn as sns
import os
import numpy as np
import scanpy as sc
from pathlib import Path
from unit import find_cluster_column,velocity_confidence_plot,cosine_similarity_percentage
# import math
from pathlib import Path
from itertools import combinations
def sort_categories(categories):
    numeric, non_numeric,nan_list = [], [], []
    for cat in categories:
        try:
            # Attempt to convert to integer
            num = int(cat)
            # Check if the conversion was exact (avoids casting floats like 4.0 to int 4)
            # If you want to include exact floats like 4.0 as numeric, remove the float check
            if isinstance(cat, float):
                non_numeric.append(cat)  # Treat exact floats as non-numeric if not explicitly integers
            else:
                numeric.append(num)
        except (ValueError, TypeError):
            # Handle cases where conversion to int fails (strings, non-integer floats, nan)
            if isinstance(cat, float) and np.isnan(cat):
                nan_list.append(cat)
            else:
                # It's a non-numeric, non-nan value (e.g., string, non-nan float)
                non_numeric.append(cat)
    # 对数字部分按数值排序后转回字符串
    numeric_sorted = sorted(numeric)
    numeric_str = [str(n) for n in numeric_sorted]
    # 非数字部分按字母顺序排序
    non_numeric_sorted = sorted(non_numeric)
    # 合并结果
    return numeric_sorted + non_numeric_sorted

def load_velocity(file_path, vkey='velocity', use_low_dim=False, dataset_type='real'):
    """加载velocity向量"""
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    adata = sc.read_h5ad(file_path)
    cell_names = adata.obs_names.tolist()

    if use_low_dim:
        # 降维向量
        # suffix = '_umap' if dataset_type == 'real' else '_dimred'
        suffix = '_umap'
        vkey_low = f"{vkey}{suffix}"
        if vkey_low not in adata.obsm:
            raise KeyError(f"缺少降维向量: {vkey_low}")
        velocity = adata.obsm[vkey_low]
    else:
        # 高维向量
        if vkey not in adata.layers:
            raise KeyError(f"缺少velocity向量: {vkey}")
        velocity = adata.layers[vkey]

    # 转换为dense array
    if hasattr(velocity, 'toarray'):
        velocity = velocity.toarray()

    return velocity, cell_names


def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度，归一化到[0,1]"""
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_sim = np.dot(v1, v2) / (norm1 * norm2)
    return (cos_sim + 1) / 2  # 归一化到[0,1]


def calculate_pairwise_similarity(velocity1, velocity2, cells1, cells2):
    """
    计算两个重复间匹配细胞的余弦相似度
    返回: (平均值, 中位数)
    """
    # 找到共同细胞
    common_cells = set(cells1) & set(cells2)
    if len(common_cells) == 0:
        raise ValueError("两个文件没有共同的细胞！")

    # 获取共同细胞的索引
    idx1 = [cells1.index(cell) for cell in common_cells]
    idx2 = [cells2.index(cell) for cell in common_cells]

    # 计算每个细胞的相似度
    similarities = [
        cosine_similarity(velocity1[i], velocity2[j])
        for i, j in zip(idx1, idx2)
    ]

    return np.mean(similarities), np.median(similarities)


def analyze_dataset(inputdata_dir, tool_name, ID, file_prefix,
                    vkey='velocity', use_low_dim=False, n_repeats=5):
    """分析单个数据集的所有重复组合"""

    dataset_type = 'real' if ID == '8' else 'simulated'

    print(f"\n处理: {ID}")

    # 加载所有重复
    velocities = {}
    cell_names_dict = {}

    for r in range(0, n_repeats):
        files = find_h5ad(f'{inputdata_dir}/{r}', ID, 'velo.h5ad')
        if len(files) == 0:
            print(f'{ID}/{r}')
            return
        file_path=Path(files[0])
        try:
            velocity, cell_names = load_velocity(
                file_path, vkey, use_low_dim, dataset_type
            )
            velocities[r] = velocity
            cell_names_dict[r] = cell_names
        except Exception as e:
            print(f"  ⚠️  加载 r{r} 失败: {e}")
            raise

    # 计算所有两两组合的相似度
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

    # 计算总体平均值和中位数
    avg_cosine = np.mean(all_means)
    avg_median = np.mean(all_medians)

    # 添加总体统计量到每一行
    for row in results:
        row['average_cosine'] = avg_cosine
        row['average_median'] = avg_median

    print(f"  ✓ 完成 (平均余弦值: {avg_cosine:.4f}, 平均中位数: {avg_median:.4f})")

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
    计算RNA Velocity工具的稳定性

    Parameters:
    -----------
    inputdata_dir : str
        包含工具结果的目录路径 (如: base_result/VeloVAE)
    tool_name : str
        工具名称 (如: 'VeloVAE')
    output_path : str
        输出CSV文件路径
    vkey : str
        velocity向量的key名称
    use_low_dim : bool
        是否使用降维向量 (False=高维, True=低维)
    n_repeats : int
        重复次数

    Returns:
    --------
    pd.DataFrame : 分析结果
        列: tool_name, dataset_name, group, group_cosine, group_median,
            average_cosine, average_median
    """

    inputdata_dir = Path(inputdata_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 数据集配置
    all_results = []

    try:
        df = analyze_dataset(
            inputdata_dir, tool_name,
            ID, data_file,
            vkey, use_low_dim, n_repeats
        )
        all_results.append(df)
    except Exception as e:
        print(f"  ✗ 跳过该数据集: {e}\n")
        return

    if not all_results:
        raise FileNotFoundError(
            f"未找到任何有效的数据集文件！\n"
            f"请检查目录: {inputdata_dir}"
        )

    # 合并并保存
    final_df = pd.concat(all_results, ignore_index=True)

    # 确保列顺序
    final_df = final_df[[
        'tool_name', 'dataset_name', 'group',
        'group_cosine', 'group_median',
        'average_cosine', 'average_median'
    ]]

    final_df.to_csv(output_path, index=False, float_format='%.6f')

    print(f"\n✓ 分析完成，结果已保存: {output_path}\n")

    return final_df


