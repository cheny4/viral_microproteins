# Import standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import pickle
from collections import OrderedDict
from adjustText import adjust_text
import gc
from seaborn import clustermap

# Core scverse libraries
import scanpy as sc
import anndata as ad
import scipy.io as sio
import scipy.sparse as sp
import h5py
import pertpy as pt
import networkx as nx
import hdbscan


warnings.filterwarnings('ignore')
sns.set_style('white')
sns.set_palette('deep')


import os
def rescale_kde(g):
    # rescale the curves in the x direction
    max_height = max([np.max(curve.get_data()[1]) for curve in g.lines])
    for curve in g.lines:
        height = np.max(curve.get_data()[1])
        curve.set_ydata(curve.get_data()[1] / height * max_height)


def run_hdbscan(dist_matrix, min_cluster_size=3, eps=0.5):
    clusterer = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=min_cluster_size,
        min_samples=1,
        cluster_selection_method='eom',
        # cluster_selection_epsilon=eps,
        allow_single_cluster=False
    )
    return clusterer.fit_predict(dist_matrix)

def split_large_clusters(
    labels_df,
    dist_df,
    max_size=150,
    eps_list=[0.5, 0.4, 0.3, 0.2],
    min_cluster_size=3,
):

    final = labels_df.copy()
    next_cluster_id = final.Cluster.max() + 1

    for eps in eps_list:
        print(f"\n=== RECLUSTER PASS (eps={eps}) ===")

        # find large clusters
        large_clusters = (
            final[final.Cluster != -1]
            .Cluster.value_counts()
            .loc[lambda s: s > max_size]
            .index.tolist()
        )

        if len(large_clusters) == 0:
            print("No oversized clusters remain.")
            break

        print("Oversized:", large_clusters)

        for cl in large_clusters:
            print(f"  Reclustering cluster {cl} ...")

            genes = final.loc[final.Cluster == cl, "gene_naming"]
            genes = genes.dropna()
            genes = genes[genes.isin(dist_df.index)]

            if len(genes) < 3:
                print("    Too small to recluster; skipping.")
                continue

            # subset matrices
            sub_dist = dist_df.loc[genes, genes].values

            # run HDBSCAN
            reclust = run_hdbscan(
                sub_dist,
                min_cluster_size=min_cluster_size,
                eps=eps
            )

            # remap cluster IDs
            new_ids = []
            for x in reclust:
                if x == -1:
                    new_ids.append(-1)
                else:
                    new_ids.append(next_cluster_id + x)

            next_cluster_id += reclust.max() + 1

            # assign only to reclustered genes
            mask = (final.Cluster == cl) & (final.gene_naming.isin(genes))
            final.loc[mask, "Cluster"] = new_ids

    return final


def reassign_singletons_to_noise(labels):
    labelcounts=labels[labels.Cluster!=-1].Cluster.value_counts().reset_index()
    
    new_labels=labels.copy()
    for i,row in labelcounts.iterrows():
        
        if row['count'] == 1:
            new_labels[new_labels == row['Cluster']] = -1

    return new_labels


def select_top_divergent_cells(
    adata, 
    include, 
    layer="z_norm", 
    neg_label="Negative control",
    group_key="gene_naming",
    top_n=50,
    phase_key="phase"
):
    """
    For each gene in `include`, select up to top_n cells most divergent 
    from the NC centroid, while preserving the original cell-cycle 
    phase distribution within each perturbation.

    Also include the negative controls most similar to the NC centroid.
    """
    
    # 1. Extract expression matrix
    X = adata.X if layer is None else adata.layers[layer]
    obs = adata.obs
    
    # 2. Compute NC centroid
    neg_mask = obs[group_key] == neg_label
    X_neg = X[neg_mask]
    neg_centroid = X_neg.mean(axis=0)
    
    selected_indices = []

    # =========================================================
    #     For each perturbation: top divergent + phase weights
    # =========================================================
    for gene in include:
        if gene == neg_label:
            continue

        mask = obs[group_key] == gene
        if mask.sum() == 0:
            print(f"⚠️ No cells for {gene}")
            continue
        
        idx_gene = np.where(mask)[0]
        X_gene = X[idx_gene]
        
        # Distances to NC centroid
        d = np.linalg.norm(X_gene - neg_centroid, axis=1)
        
        # Rank by divergence (descending)
        order = np.argsort(d)[::-1]
        
        # ===== Cell cycle phase proportions in full dataset =====
        phase_counts = obs.loc[mask, phase_key].value_counts(normalize=True)

        # number to sample per phase
        target_per_phase = (phase_counts * top_n).round().astype(int)
        if target_per_phase.sum() < top_n:
            # adjust (due to rounding)
            deficit = top_n - target_per_phase.sum()
            # give extra cells to the largest phases
            largest_phases = target_per_phase.sort_values(ascending=False).index[:deficit]
            target_per_phase[largest_phases] += 1
        
        # ------------------------------------
        #  Phase-aware top selection
        # ------------------------------------
        df = pd.DataFrame({
            "idx": idx_gene,
            "dist": d,
            phase_key: obs.loc[mask, phase_key].values
        })
        df.sort_values("dist", ascending=False, inplace=True)
        
        selected_gene = []
        for phase, n_take in target_per_phase.items():
            df_phase = df[df[phase_key] == phase]
            take_idx = df_phase.head(n_take)["idx"].tolist()
            selected_gene.extend(take_idx)

        selected_indices.extend(selected_gene)

    # =========================================================
    #  Negative controls: phase-aware selection closest to each
    #  phase-specific NC centroid (correct absolute indexing)
    # =========================================================
    neg_mask = obs[group_key] == neg_label
    neg_idx_abs = np.where(neg_mask)[0]        # absolute indices into adata
    X_neg = X[neg_mask]

    # --- Compute global centroid ---
    centroid = X_neg.mean(axis=0)
    # --- Distance to centroid ---
    d_neg = np.linalg.norm(X_neg - centroid, axis=1)
    # --- How many to select ---
    n_take = min(500, len(d_neg))   # or whatever cap you want
    # --- Select closest cells ---
    order_local = np.argsort(d_neg)[:n_take]
    # ---- Convert local → absolute indices ----
    selected_neg = neg_idx_abs[order_local]
        
    selected_neg=np.random.choice(neg_idx_abs,size = top_n*2,replace=False)
    selected_indices.extend(selected_neg)
    
    # Return final AnnData
    return adata[selected_indices].copy()