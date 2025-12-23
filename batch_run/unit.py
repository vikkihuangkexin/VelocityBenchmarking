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
import pandas as pd
import anndata
#coherence of the velocity vector field（velocity confidence）
def velocity_confidence_plot(adata=None,velocity=None,distances=None,plot_save_path=str,basis='umap'):
    """
    定义：将每个cell i 的一致性分数定义为其速度 v 与相邻单元格的速度的平均相关性，即整体一致性，
    可使用velocity_confidence直接计算，返回adata.obs['velocity_confidence']
    需要的数据有计算好的velocity矩阵以及使用sc.pp.neighbors计算出的距离矩阵distances用作邻域细胞信息
    因此有两种方式计算，一是构成anndata数据直接使用API计算，二是获得计算的两个矩阵计算
    第二种方式需要velocity为np矩阵或可转化为np矩阵的格式，distances需要csr_matrix或np.array格式
    这里仅画出scatter图和一致性分布图,这里使用KDE(核概率密度分布)来计算分布密度
    """
    if adata:

        scv.tl.velocity_confidence(adata)
        keys = 'velocity_confidence'#'velocity_length',
        scv.pl.scatter(adata, c=keys, cmap='coolwarm',
            size=100,
            basis=basis,
            alpha = 0.6,
            dpi=400,
            perc=[5, 95],save=os.path.join(plot_save_path,'velocity_confidence_scatter.pdf'))
        scv.pl.scatter(adata, c=keys, cmap='coolwarm',
            size=100,
            basis=basis,
            alpha = 0.6,
            dpi=400,
            perc=[5, 95],
            save=os.path.join(plot_save_path, 'velocity_confidence_scatter.png'))
        velocity_confidence = adata.obs['velocity_confidence']
        sns.kdeplot(velocity_confidence, shade=True)
        plt.xlim(0, 1)
        plt.title('velocity confidence kernel density estimate')
        plt.xlabel('confidence')
        plt.ylabel('density')
        plt.show()
        plt.savefig(os.path.join(plot_save_path,'velocity_confidence_KDE.pdf'))
        plt.close()
        velocity_confidence1 = pd.DataFrame(velocity_confidence)
        velocity_confidence1.to_csv(os.path.join(plot_save_path,'velocity_confidence_row.csv'))
        velocity_confidence.dropna(inplace=True)
        kde = gaussian_kde(velocity_confidence)
        x = np.linspace(0, 1, 10000)
        y = kde.evaluate(x)
        y = pd.DataFrame({'density':y})
        y.to_csv(os.path.join(plot_save_path,'KDE_density_1w.csv'))
        return adata
    elif velocity:
        velocity = np.array(velocity)
        if not distances:
            print('Need neighbors cell info. Run sc.pp.neighbors or others to get distance csr matrix')
        if str(type(distances))!="<class 'scipy.sparse._csr.csr_matrix'>":
            distances = csr_matrix(distances)
        velocity -= velocity.mean(1)[:, None]
        V_norm = l2_norm(V, axis=1)
        R = np.zeros(adata.n_obs)
        indices = get_indices(dist=distances)[0]
        for i in range(adata.n_obs):
            Vi_neighs = velocity[indices[i]]
            Vi_neighs -= Vi_neighs.mean(1)[:, None]
            R[i] = np.mean(
                np.einsum("ij, j", Vi_neighs, V[i])
                / (l2_norm(Vi_neighs, axis=1) * V_norm[i])[None, :]
            )
        velocity_length = V_norm.round(2)
        velocity_confidence = np.clip(R, 0, None)
        sns.kdeplot(velocity_confidence, shade=True)
        plt.xlim(0, 1)
        plt.title('velocity_confidence kernel density estimate')
        plt.xlabel('confidence')
        plt.ylabel('density')
        plt.show()
        plt.savefig(os.path.join(plot_save_path,'velocity_confidence_KDE.pdf'))
        return velocity_length,velocity_confidence

import numpy as np
#获取模拟的真实速率和计算得出的速率
# key = 'velocity'
# velocity_T = np.array(adata.layers['velocity'])
# velocity_O = np.array(adata.layers['velocity_original'])
def cosine_similarity_percentage(velocity_T, velocity_O, threshold=0.7):
    """
    模拟数据中通常带有模拟出来的原始velocity，确保计算出来的velocity矩阵与原始矩阵细胞数量一致
    """
    # 检查矩阵形状是否一致
    assert velocity_T.shape[0] == velocity_O.shape[0], "两个矩阵的行数必须相等"
    dot_product = np.dot(velocity_T, velocity_O.T)
    norm_T = np.linalg.norm(velocity_T, axis=1)
    norm_O = np.linalg.norm(velocity_O, axis=1)
    similarity_matrix = dot_product / (norm_T[:, np.newaxis] * norm_O)
    below_threshold = np.sum(similarity_matrix < threshold)
    percentage = below_threshold / similarity_matrix.size
    return similarity_matrix, percentage

from typing import Optional, Tuple
import re
import pandas as pd
import anndata
def find_cluster_column(adata: anndata.AnnData) -> Optional[Tuple[str, pd.Series]]:
    """
    按优先级查找 adata.obs 中的聚类/注释列

    Args:
        adata: AnnData 对象

    Returns:
        tuple: (匹配的列名, 对应的值) 或 None（若未找到）
    """
    # 定义优先级及可能的列名变体（支持正则表达式）
    priority_patterns = [
        # 第一优先级: celltype 相关 (celltype、cell_type、CellType...)
        [r"^cell[\W_]?type$", r"^Cell[\W_]?Type$", r"^celltype$",r"^predicted_cell_type$"],
        # 第二优先级: clustername 相关 (cluster_name、ClusterName...)
        [r"^cluster[\W_]?name$", r"^Cluster[\W_]?Name$", r"^clustername$"],
        # 第三优先级：r"^annotation$"
        [r"^annotation$"],
        # 第四优先级: phase (直接匹配 phase)
        [r"^phase$",r"^cell_cycle_phase$"],
        # 第五优先级: cluster (cluster、Cluster...)
        [r"^cluster$", r"^Cluster$",r"^clusters$"],
        # 第六优先级: time
        [r"^time$"]
    ]
    # 遍历优先级
    for patterns in priority_patterns:
        # 检查所有可能的列名
        for col in adata.obs.columns:
            # 忽略大小写和下划线，匹配模式
            normalized_col = re.sub(r"[\W_]", "", col).lower()
            for pattern in patterns:
                # 将正则模式标准化后匹配
                std_pattern = re.sub(r"[\W_]", "", pattern).lower().strip("^$")
                if re.match(f"^{std_pattern}$", normalized_col):
                    return col
    # 未找到任何匹配列
    print("Warning: No cluster/celltype column found in adata.obs!")
    return None