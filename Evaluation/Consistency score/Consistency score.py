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

def velocity_confidence_plot(adata=None, velocity=None, distances=None, plot_save_path=str, basis='umap'):
    """
    Definition:
    The per-cell consistency score is defined as the average correlation between
    a cell's velocity vector and the velocity vectors of its neighboring cells.
    This overall consistency is available via scvelo's API as
    adata.obs['velocity_confidence'] when computed with scv.tl.velocity_confidence.

    Required inputs:
    - Either provide an AnnData object (adata) with precomputed velocities and
      neighbors (sc.pp.neighbors), or provide the raw velocity matrix and a
      distances matrix computed by sc.pp.neighbors.
    - If using the second approach:
        * velocity should be a NumPy array or convertible to one.
        * distances should be a scipy CSR matrix or a NumPy array (will be converted).

    Output and plotting:
    - Produces a scatter plot on the specified embedding (default 'umap') colored
      by velocity confidence and saves PDF/PNG.
    - Plots and saves a kernel density estimate (KDE) of the velocity confidence
      distribution and exports density values to CSV.
    """
    if adata:
        # Compute velocity confidence using scvelo API
        scv.tl.velocity_confidence(adata)
        keys = 'velocity_confidence'

        # Scatter plots (PDF and PNG)
        scv.pl.scatter(
            adata, c=keys, cmap='coolwarm', size=100, basis=basis,
            alpha=0.6, dpi=400, perc=[5, 95],
            save=os.path.join(plot_save_path, 'velocity_confidence_scatter.pdf')
        )
        scv.pl.scatter(
            adata, c=keys, cmap='coolwarm', size=100, basis=basis,
            alpha=0.6, dpi=400, perc=[5, 95],
            save=os.path.join(plot_save_path, 'velocity_confidence_scatter.png')
        )

        # KDE plot of velocity confidence
        velocity_confidence = adata.obs['velocity_confidence']
        sns.kdeplot(velocity_confidence, shade=True)
        plt.xlim(0, 1)
        plt.title('Velocity confidence kernel density estimate')
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.show()
        plt.savefig(os.path.join(plot_save_path, 'velocity_confidence_KDE.pdf'))
        plt.close()

        # Save per-cell confidence values and KDE density values
        velocity_confidence_df = pd.DataFrame(velocity_confidence)
        velocity_confidence_df.to_csv(os.path.join(plot_save_path, 'velocity_confidence_row.csv'))

        velocity_confidence = velocity_confidence.dropna()
        kde = gaussian_kde(velocity_confidence)
        x = np.linspace(0, 1, 10000)
        y = kde.evaluate(x)
        y_df = pd.DataFrame({'density': y})
        y_df.to_csv(os.path.join(plot_save_path, 'KDE_density_1w.csv'))

        return adata

    elif velocity is not None:
        # Compute confidence from raw matrices
        velocity = np.array(velocity)
        if distances is None:
            print('Need neighbor cell information. Run sc.pp.neighbors or similar to obtain a distance CSR matrix.')
            return None

        if str(type(distances)) != "<class 'scipy.sparse._csr.csr_matrix'>":
            distances = csr_matrix(distances)

        # Center velocities
        velocity -= velocity.mean(1)[:, None]

        # Norms and placeholders
        V_norm = l2_norm(velocity, axis=1)
        R = np.zeros(velocity.shape[0])
        indices = get_indices(dist=distances)[0]

        for i in range(velocity.shape[0]):
            Vi_neighs = velocity[indices[i]]
            Vi_neighs -= Vi_neighs.mean(1)[:, None]
            # cosine-like similarity averaged over neighbors
            R[i] = np.mean(
                np.einsum("ij, j", Vi_neighs, velocity[i])
                / (l2_norm(Vi_neighs, axis=1) * V_norm[i])[None, :]
            )

        velocity_length = V_norm.round(2)
        velocity_confidence = np.clip(R, 0, None)

        # KDE plot and save
        sns.kdeplot(velocity_confidence, shade=True)
        plt.xlim(0, 1)
        plt.title('Velocity confidence kernel density estimate')
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.show()
        plt.savefig(os.path.join(plot_save_path, 'velocity_confidence_KDE.pdf'))

        return velocity_length, velocity_confidence
