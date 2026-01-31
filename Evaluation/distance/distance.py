#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import scanpy as sc
import numpy as np
import pandas as pd
import scvelo as scv
from tqdm import tqdm
import argparse

possible_celltype_columns = [
    "cell_type"
]

def process_h5ad_file(input_path, method_name, celltype_column, output_path):
    try:
        adata = sc.read_h5ad(input_path)

        if method_name == "TFvelo":
            print("    This is TFvelo: mapping WX -> Mu, M_total -> Ms")
            adata.layers["Mu"] = adata.layers["WX"]
            adata.layers["Ms"] = adata.layers["M_total"]

        if "Mu" not in adata.layers or "Ms" not in adata.layers:
            print("    Mu/Ms not found, computing moments...")
            scv.pp.moments(adata, n_neighbors=30, n_pcs=30)
        else:
            print("    Mu and Ms already present, skipping moments calculation.")

        adata.obs[celltype_column] = adata.obs[celltype_column].astype('category')
        labels   = adata.obs[celltype_column].cat.codes.to_numpy()
        n_labels = len(adata.obs[celltype_column].cat.categories)

        results = pd.DataFrame(columns=['Gene', 'Intra-class distance', 'Inter-class distance'])

        for gene in tqdm(adata.var_names, desc=f"Calculating for {os.path.basename(input_path)}", leave=False):
            try:
                w  = adata[:, gene].layers['Mu'].flatten()
                mt = adata[:, gene].layers['Ms'].flatten()
                w  = w / (w.std() or 1)
                mt = mt / (mt.std() or 1)
                pts = np.stack([w, mt], axis=1)

                centers = []
                for cid in range(n_labels):
                    sub = pts[labels == cid]
                    if sub.size:
                        centers.append(sub.mean(axis=0))

                intra = []
                for cid, ctr in enumerate(centers):
                    sub = pts[labels == cid]
                    intra.extend(np.linalg.norm(sub - ctr, axis=1))
                intra_mean = np.nan if not intra else np.mean(intra)

                inter = []
                for i in range(len(centers)):
                    for j in range(i+1, len(centers)):
                        inter.append(np.linalg.norm(centers[i] - centers[j]))
                inter_mean = np.nan if not inter else np.mean(inter)

                results.loc[len(results)] = [gene, intra_mean, inter_mean]
            except Exception as e:
                print(f"  Error on gene {gene}: {e}")
                results.loc[len(results)] = [gene, np.nan, np.nan]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results.to_csv(output_path, index=False)
        return True, ""
    except Exception as e:
        return False, str(e)

def main(h5ad_path, method_name, output_dir):
    print(f"\n=== Processing single h5ad file: {os.path.basename(h5ad_path)} ===")
    error_log = []

    # Check if h5ad file exists
    if not os.path.isfile(h5ad_path):
        msg = f"File not found: {h5ad_path}"
        print(f"  Error: {msg}")
        error_log.append({"Error": msg})
        return

    try:
        # Read h5ad and find cell type column
        adata = sc.read_h5ad(h5ad_path)
        celltype_column = None
        for col in possible_celltype_columns:
            if col in adata.obs.columns:
                celltype_column = col
                print(f"  Using '{col}' as cell type column.")
                break

        if celltype_column is None:
            msg = "No suitable cell type column found in adata.obs."
            print(f"  Error: {msg}")
            error_log.append({"Error": msg})
            return

        # Build output paths
        h5ad_basename = os.path.splitext(os.path.basename(h5ad_path))[0]
        output_filename = f"{h5ad_basename}_{method_name}_gene_distance.csv"
        output_path = os.path.join(output_dir, output_filename)
        error_filename = f"{h5ad_basename}_{method_name}_errors.csv"
        error_path = os.path.join(output_dir, error_filename)

        # Skip if output already exists
        if os.path.isfile(output_path):
            print(f"  Output file already exists: {output_path}, skipping processing.")
            return

        # Process the h5ad file
        success, error_msg = process_h5ad_file(h5ad_path, method_name, celltype_column, output_path)
        if success:
            print(f"  Processing success. Results saved to {output_path}")
        else:
            msg = f"Processing failed: {error_msg}"
            print(f"  Error: {msg}")
            error_log.append({"Error": msg})

    except Exception as e:
        msg = f"File loading failed: {str(e)}"
        print(f"  Error: {msg}")
        error_log.append({"Error": msg})

    # Save error log if any errors
    if error_log:
        error_df = pd.DataFrame(error_log)
        os.makedirs(output_dir, exist_ok=True)
        h5ad_basename = os.path.splitext(os.path.basename(h5ad_path))[0]
        error_path = os.path.join(output_dir, f"{h5ad_basename}_{method_name}_errors.csv")
        error_df.to_csv(error_path, index=False)
        print(f"  Error log saved to {error_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single h5ad File Gene Intra/Inter Class Distance Calculation")
    parser.add_argument("--h5ad_path",
                        required=True,
                        help="Path to the input single h5ad file")
    parser.add_argument("--method_name",
                        default="SingleSample",
                        help="Method name (e.g., TFvelo, DeepVelo, default: SingleSample)")
    parser.add_argument("--output_dir",
                        default="./",
                        help="Output directory to save results and error logs (default: current directory)")
    args = parser.parse_args()

    main(args.h5ad_path, args.method_name, args.output_dir)
    print("\nAll processing completed.")