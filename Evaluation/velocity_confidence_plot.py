import os
from scipy.stats import gaussian_kde
import numpy as np
import scvelo as scv
from scvelo.core import l2_norm, prod_sum
from scvelo.utils import get_indices
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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