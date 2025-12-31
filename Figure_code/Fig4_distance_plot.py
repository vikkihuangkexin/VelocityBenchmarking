import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import os

def extract_cell_gene_count(group_name):
    """Extract cell count and gene count from data group name"""
    cell_match = re.search(r"cell(\d+)", group_name)
    gene_match = re.search(r"gene(\d+)", group_name)
    if cell_match and gene_match:
        return int(cell_match.group(1)), int(gene_match.group(1))
    return None, None

def create_heatmap_matrix(df):
    """Create heatmap matrix from dataframe"""
    df_valid = df.dropna(subset=["cell_count", "gene_count"]).copy()
    
    unique_cells = sorted(df_valid["cell_count"].unique())
    unique_genes = sorted(df_valid["gene_count"].unique())
    
    heatmap_matrix = pd.DataFrame(
        index=unique_cells,
        columns=unique_genes,
        dtype=float
    )
    
    for _, row in df_valid.iterrows():
        cell = row["cell_count"]
        gene = row["gene_count"]
        heatmap_matrix.loc[cell, gene] = row["mean_distance"]
    
    return heatmap_matrix, unique_cells, unique_genes

def setup_plot_style():
    """Setup consistent plot styling"""
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "Arial",
        "axes.labelpad": 8,
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
        "pdf.fonttype": 42,  # 确保PDF中的文字可编辑
        "ps.fonttype": 42,   # 确保PostScript中的文字可编辑
        "figure.dpi": 300,   # 统一DPI设置
    })

def plot_heatmap(heatmap_matrix, unique_cells, unique_genes):
    """Create and return heatmap figure and axis"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 处理NaN值，设置为最小值或特定值以确保颜色一致性
    data_for_plot = heatmap_matrix.values.copy()
    
    # 使用固定的颜色范围以确保一致性
    #vmin = np.nanmin(data_for_plot)
    #vmax = np.nanmax(data_for_plot)
    vmin = 0
    vmax = 150
    # 创建热图 - 使用pcolormesh而不是imshow以获得更好的PDF输出
    im = ax.pcolormesh(
        data_for_plot,
        cmap='viridis_r',
        vmin=vmin,
        vmax=vmax,
        edgecolors='none',  # 确保没有边缘线
        shading='auto'
    )
    
    # 设置标签和刻度
    ax.set_xlabel("Number of genes", fontweight="bold")
    ax.set_ylabel("Number of cells", fontweight="bold")
    
    # 设置刻度位置和标签
    ax.set_xticks(np.arange(len(unique_genes)) + 0.5)
    ax.set_xticklabels(unique_genes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(unique_cells)) + 0.5)
    ax.set_yticklabels(unique_cells)
    
    # 添加值标注
    for i in range(len(unique_cells)):
        for j in range(len(unique_genes)):
            value = heatmap_matrix.iloc[i, j]
            if not np.isnan(value):
                text_color = "white" if value < (vmin + vmax) / 2 else "black"
                ax.text(
                    j + 0.5, i + 0.5, f"{value:.2f}",
                    ha="center", va="center",
                    color=text_color,
                    fontsize=9
                )
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Euclidean Distance", fontweight="bold", labelpad=10)
    
    # 设置标题
    ax.set_title(
        "Mean Euclidean Distance of Velocycle Phase",
        fontweight="bold",
        pad=15,
        fontsize=12
    )
    
    # 设置坐标轴范围
    ax.set_xlim(0, len(unique_genes))
    ax.set_ylim(0, len(unique_cells))
    
    plt.tight_layout()
    return fig, ax

def save_plots(fig, base_path):
    """Save plots in both PNG and PDF formats with consistent settings"""
    # 保存PNG
    png_path = f"{base_path}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", 
                facecolor='white', edgecolor='none')
    print(f"PNG heatmap saved to: {png_path}")
    
    # 保存PDF - 使用与PNG相同的设置
    pdf_path = f"{base_path}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", format='pdf',
                facecolor='white', edgecolor='none',
                dpi=300)  # 即使PDF是矢量格式，也设置相同的DPI
    print(f"PDF heatmap saved to: {pdf_path}")

def main():
    # -------------------------- 1. Data Loading and Preprocessing --------------------------
    data_path = "/data_d/ZXL/RNA_Velocity/DeepCycle-main/result/phase_result/otu/euclidean_distance_summary.csv"
    df = pd.read_csv(data_path, usecols=["data_group", "mean_distance"])
    
    # Extract cell and gene counts
    df[["cell_count", "gene_count"]] = df["data_group"].apply(
        lambda x: pd.Series(extract_cell_gene_count(x))
    )
    
    # Create heatmap matrix
    heatmap_matrix, unique_cells, unique_genes = create_heatmap_matrix(df)
    
    # -------------------------- 2. Plotting --------------------------
    setup_plot_style()
    fig, ax = plot_heatmap(heatmap_matrix, unique_cells, unique_genes)
    
    # -------------------------- 3. Save plots --------------------------
    base_save_path = "/data_d/ZXL/RNA_Velocity/DeepCycle-main/result/phase_result/otu/cell_gene_distance_heatmap"
    save_plots(fig, base_save_path)
    
    # Print summary
    print(f"Heatmap structure: {len(unique_cells)} rows (cell counts) * {len(unique_genes)} columns (gene counts)")
    print(f"Data range: {np.nanmin(heatmap_matrix.values):.2f} to {np.nanmax(heatmap_matrix.values):.2f}")
    
    # Close the figure to free memory
    plt.close(fig)

if __name__ == "__main__":
    main()