
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