#!/usr/bin/env python
# coding: utf-8

# -----------------------------
#   Perturb-seq Distance Analysis (Optimized)
# -----------------------------

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
import scanpy.external as sce
from scipy.spatial.distance import cdist


warnings.filterwarnings('ignore')
sns.set_style('white')
sns.set_palette('deep')

# -----------------------------
#   Paths and directories
# -----------------------------
fig_out = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/figs"
python_out = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/python_outputs"
suffix = "_all"

sc.settings.figdir = fig_out
os.makedirs(python_out, exist_ok=True)
os.makedirs(fig_out, exist_ok=True)

# -----------------------------
#   Load AnnData
# -----------------------------
print("--- 0. Load data ---")

orf_info = pd.read_csv(
    '/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/orf_files/perturb-orf_metadata.csv',
    index_col=0
)
# Define controls
all_controls=orf_info[~orf_info['pert_type'].isin(['Perturbed'])]
neg_controls=all_controls[all_controls.pert_type.str.contains('egative')&(all_controls.Geneid!='GFP')]
pos_controls=all_controls[~all_controls.Geneid.isin(neg_controls.Geneid)]
print('Positive controls')
print(pos_controls.Geneid.unique())


savefile = f"{python_out}/adata_znorm_hvg.h5ad"
adata_raw=ad.read_h5ad(savefile)
print(adata_raw)
print(adata_raw.obs.head(5))

# --- Edit negatives ---
negative_cells = adata_raw.obs[adata_raw.obs['Geneid'].isin(neg_controls.Geneid)].index
adata_raw.obs['gene_naming'] = adata_raw.obs['gene_naming'].astype(str)
adata_raw.obs.loc[negative_cells, 'gene_naming'] = 'Negative control'
adata_raw.obs['pert_type'] = adata_raw.obs['pert_type'].astype(str)



# Filter Geneids that appear at least min_cell times
min_cell = 5
good_genes = adata_raw.obs['gene_naming'].value_counts()
good_genes = good_genes[good_genes >= min_cell].index
adata = adata_raw[adata_raw.obs['gene_naming'].isin(good_genes), :].copy()
print(adata)
print(f'Number of perturbations: {adata.obs.gene_naming.nunique()}')


control_name = "Negative control"
group = 'gene_naming'
#-----------------------------
#  Determine DEGs
#-----------------------------
print("--- I. Determine DEGs ---")
all_degs = []
#from log1norm data

sc.tl.rank_genes_groups(
    adata,
    groupby=group,
    method="wilcoxon",
    key_added="wilcoxon",
    reference=control_name,
    layer = "log1p_norm"
)
for g in adata.obs[group].unique():
    if g != control_name:
        df = sc.get.rank_genes_groups_df(adata, group=g, key="wilcoxon")
        df = df[df.pvals_adj < 0.05]
        df[group] = g
        df['DEG_type'] = "log1p_norm"
        all_degs.append(df)

# #from znorm data
# sc.tl.rank_genes_groups(
#     adata,
#     groupby=group,
#     method="wilcoxon",
#     key_added="wilcoxon",
#     reference=control_name,
#     layer = "z_norm"
# )
# for g in adata.obs[group].unique():
#     if g != control_name:
#         df = sc.get.rank_genes_groups_df(adata, group=g, key="wilcoxon")
#         # df = df[df.pvals_adj < 0.05]
#         df[group] = g
#         df['DEG_type'] = "z_norm"
#         all_degs.append(df)


#from DEGs
if all_degs:
    all_degs = pd.concat(all_degs)
    all_degs.to_csv(f"{python_out}/bulk_summaries_DEGs.csv", index=False)
else:
    print("No significant DEGs found.")

# all_degs=pd.read_csv(f"{python_out}/bulk_summaries_DEGs.csv")


# Select interesting genes
hvgs = all_degs[all_degs.pvals_adj < 0.05].names.to_list()+list(adata.var_names[adata.var_names.str.startswith("MT-")])#+ adata.var[adata.var[[c for c in adata.var.columns if 'total_counts' in c]].sum(axis=1) / adata.obs.shape[0] > 0.25].index.to_list() 
hvgs = list(set(hvgs))
print(f'{len(hvgs)} variable genes detected')
# Subset to interesting genes
adata = adata[:, hvgs].copy()

# Run PCA
sc.tl.pca(adata, layer='z_norm')
sc.pl.pca_variance_ratio(adata, save=".png")
adata.obsm['X_pca']=adata.obsm['X_pca'][:,0:20]

print(f"anndata with PCA saved to: {python_out}")
savefile = f"{python_out}/adata_znorm_hvg_processed.h5ad"
adata.write_h5ad(savefile)


# -----------------------------
#   Compute Distances (Spearman and Pearson)
# -----------------------------
print("--- II. Compute spearman and pearson distances ---")
check = adata.obs[adata.obs.pert_type.str.contains('control')].gene_naming.unique().tolist()+adata.obs[adata.obs.gene_naming.str.contains('E7')|adata.obs.gene_naming.str.contains('6/7')].gene_naming.unique().tolist()

# Pearson
pearson = pt.tl.Distance(metric="pearson_distance", layer_key='z_norm')
pearson_df = pearson.pairwise(adata, groupby=group)
pearson_df.to_csv(f"{python_out}/all_dist_pearson_znorm.csv")
pearson = pt.tl.Distance(metric="pearson_distance", layer_key='z_norm')
pearson_df = pearson.pairwise(adata[adata.obs.gene_naming.isin(check)], groupby=group)
# Plot clustermap
g = sns.clustermap(
    pearson_df.loc[check,check],
    figsize=(30, 30)
)
plt.savefig(f'{fig_out}/clustermap_pearson_check.png', dpi=100)
plt.show()


# Spearman
spearman = pt.tl.Distance(metric="spearman_distance", layer_key='z_norm')
spearman_df = spearman.pairwise(adata, groupby=group)
spearman_df.to_csv(f"{python_out}/all_dist_spearman_znorm.csv")
spearman = pt.tl.Distance(metric="spearman_distance", layer_key='z_norm')
spearman_df = spearman.pairwise(adata[adata.obs.gene_naming.isin(check)], groupby=group)
# Plot clustermap
g = sns.clustermap(
    spearman_df.loc[check,check],
    figsize=(30, 30)
)
plt.savefig(f'{fig_out}/clustermap_spearman_check.png', dpi=100)
plt.show()



# -----------------------------
#   Build pseudobulk
# -----------------------------
print("--- III. Compute Pseudobulk ---")
# 1) Build pseudobulk: mean per perturbation using embedding (Harmony-corrected PCA)
ps = pt.tl.PseudobulkSpace()
psadata = ps.compute(
    adata,
    target_col=group,
    layer_key='z_norm',
    mode="mean"
)


# -----------------------------
#   Energy Distance Test (Parallelized with 4 workers)
# -----------------------------
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import fdrcorrection
from concurrent.futures import ProcessPoolExecutor, as_completed

calc_matrix = 'X_pca'

print("--- V. Energy distance test (parallelized) ---")

def energy_stat(X, Y):
    return (2 * cdist(X, Y).mean() - cdist(X, X).mean() - cdist(Y, Y).mean())

def energy_test(X, Y, n_perm=500, random_state=0):
    rng = np.random.default_rng(random_state)
    n_X, n_Y = len(X), len(Y)
    all_data = np.vstack([X, Y])
    observed = energy_stat(X, Y)

    perm_stats = np.zeros(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(all_data))
        Xp = all_data[idx[:n_X]]
        Yp = all_data[idx[n_X:]]
        perm_stats[i] = energy_stat(Xp, Yp)

    pval = np.mean(perm_stats >= observed)
    return observed, pval


# Subsample up to 200 cells per group to control runtime
adata_sub = adata[
    adata.obs.groupby(group, group_keys=False)
    .apply(lambda x: x.sample(n=min(200, len(x)), random_state=0))
    .index
]
emb = adata_sub.obsm[calc_matrix]
groups = adata_sub.obs[group]
print(f"Running test on {emb.shape[0]:,} subsampled cells using {calc_matrix} {emb.shape[1]:,} dimensions"
      f"across {groups.nunique()} groups...")

# Define control group
ctrl_names = adata_sub.obs[adata_sub.obs.pert_type.str.contains('egative')][group].to_list()
ctrl = emb[groups.isin(ctrl_names)]

# Worker function for one group
def process_group(g):
    if g in ctrl_names:
        return None
    X = emb[groups == g]
    stat, pval = energy_test(X, ctrl, n_perm=500)
    return (g, stat, pval)

# Run in parallel
results = []
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(process_group, g): g for g in groups.unique()}
    for future in as_completed(futures):
        res = future.result()
        if res is not None:
            results.append(res)

energy_df = pd.DataFrame(results, columns=[group, f'edist_to_control', 'edist_pval'])
_, qvals = fdrcorrection(energy_df['edist_pval'])
energy_df['edist_FDR'] = qvals

energy_df.to_csv(f"{python_out}/all_test_edist.csv", index=False)




