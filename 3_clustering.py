# Import standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from adjustText import adjust_text
import gc

# For clustering
from seaborn import clustermap
import hdbscan

# Core scverse libraries
import scanpy as sc
import anndata as ad
import scipy.io as sio 
import scipy.sparse as sp 
import h5py 
import pertpy as pt
from sklearn.manifold import MDS

warnings.filterwarnings('ignore')
sns.set_style('white')
sns.set_palette('deep')

def highlight_labels(clustergrid, highlight_set, color):
    """Bold and color tick labels in a ClusterGrid if they are in highlight_set."""
    for axis in [clustergrid.ax_heatmap]:
        for lbl in axis.get_xticklabels() + axis.get_yticklabels():
            if lbl.get_text() in highlight_set:
                lbl.set_color(color)
                lbl.set_fontweight("bold")

# -----------------------------
# Paths and directories
# -----------------------------
fig_out = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/figs"
python_out = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/python_outputs"
os.makedirs(fig_out, exist_ok=True)
os.makedirs(python_out, exist_ok=True)
sc.settings.figdir = fig_out


# -----------------------------
# Load AnnData and orf info
# -----------------------------
print("--- 0. Load data ---")
savefile = f"/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/python_outputs/adata_znorm_hvg_processed.h5ad"
adata=ad.read_h5ad(savefile)

#degs
deg_df = pd.read_csv(f"{python_out}/bulk_summaries_DEGs.csv")
all_degs=deg_df.names.to_list()

# # Select interesting genes
# sc.pp.highly_variable_genes(adata, layer="log1p_norm", flavor="seurat_v3", batch_key="batch")
# hvgs = adata.var_names[adata.var['highly_variable']].tolist() + adata.var[adata.var[[c for c in adata.var.columns if 'total_counts' in c]].sum(axis=1) / adata.obs.shape[0] > 0.25].index.to_list() + all_degs
# hvgs = list(set(hvgs))
# print(f'{len(hvgs)} variable genes detected')
# # Subset to interesting genes
# adata = adata[:, hvgs] 

orf_info = pd.read_csv(
    '/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/orf_files/perturb-orf_metadata.csv',
    index_col=0
)

# Define controls
all_controls=orf_info[~orf_info['pert_type'].isin(['Perturbed'])]
neg_controls=all_controls[all_controls.pert_type.str.contains('egative')]
pos_controls=all_controls[~all_controls.Geneid.isin(neg_controls.Geneid)]
print('Positive controls')
print(pos_controls.Geneid.unique())


# -----------------------------
# Integrate info from genetic screens
# -----------------------------
print("--- I. Highlight gene hits from genetic screens ---")

def compute_outlier_set(df, feature, control_filter="ctrl", n_std=2):
    """Return a set of genes with feature values > or < mean ± n_std of controls."""
    ctrl_vals = df.loc[df.gene_naming.str.contains(control_filter), feature]
    mean, std = ctrl_vals.mean(), ctrl_vals.std()
    low, high = mean - n_std * std, mean + n_std * std
    mask = (df[feature] < low) | (df[feature] > high)
    print(f'{feature} (n_std={n_std}) upper and lower bounds:')
    print(f'High: {high}')
    print(f'Low: {low}')
    return set(df.loc[mask, "gene_naming"])


#--- Define highlight sets ---
# Controls
negative_ctrl_set = set(neg_controls.gene_naming)
positive_ctrl_set = set(pos_controls.gene_naming)
control_union = negative_ctrl_set | positive_ctrl_set

# Growth-related (using distinct names for clarity)
# Genes related to toxicity (LFC < 0, n_std=2)
growth_set_std2 = compute_outlier_set(orf_info, "growth_LFC", n_std=1.5) | set(orf_info[orf_info.growth_FDR<.05].gene_naming)
toxic_set = orf_info[orf_info.gene_naming.isin(growth_set_std2)&(orf_info.growth_LFC<0)]
toxic_set_genes = set(toxic_set['gene_naming'])
print(f'Toxic genes (LFC<0, n_std=2): {len(toxic_set)}')
print('')

# Genes related to proliferation/growth (LFC > 0, n_std=1.5)
growth_set_std1_5 = compute_outlier_set(orf_info, "growth_LFC", n_std=1) | set(orf_info[orf_info.growth_FDR<.05].gene_naming)
progrow_set = orf_info[orf_info.gene_naming.isin(growth_set_std1_5)&(orf_info.growth_LFC>0)]
progrow_set_genes = set(progrow_set['gene_naming'])
print(f'Pro-growth genes (LFC>0, n_std=1.5): {len(progrow_set)}')
print('')

# Immune-related
mhc_set = compute_outlier_set(orf_info, "MHC_LFC") - control_union
ifn_set = compute_outlier_set(orf_info, "IFN_LFC") - control_union
immune_set = (mhc_set | ifn_set)
print(f'Immune-related genes: {len(immune_set)}')
print('')

# Translation-related
upr_set = compute_outlier_set(orf_info, "UPR_LFC", n_std=3) - control_union
isr_set = compute_outlier_set(orf_info, "UPR/ISR_LFC", n_std=3) - control_union
translation_set = (upr_set | isr_set)
print(f'Translation-related genes: {len(translation_set)}')
print('')


# -----------------------------
# Load DEG and Distance data
# -----------------------------
print("--- III. Merge DEG and Distance data ---")

# Load data
spearman_df= pd.read_csv(f"{python_out}/all_dist_spearman_znorm.csv",index_col=0)
pearson_df= pd.read_csv(f"{python_out}/all_dist_pearson_znorm.csv",index_col=0)
edist_df = pd.read_csv(f'{python_out}/all_test_edist.csv')

spearman_summary = (
    spearman_df.loc[:, spearman_df.columns.isin(negative_ctrl_set)]
    .mean(axis=1)
    .reset_index()
    .rename(columns={"index": "gene_naming", 0: "spearman_to_control"})
)

pearson_summary = (
    pearson_df.loc[:, pearson_df.columns.isin(negative_ctrl_set)]
    .mean(axis=1)
    .reset_index()
    .rename(columns={"index": "gene_naming", 0: "pearson_to_control"})
)


# Summarize DEG counts
deg_summary = (
    deg_df[deg_df["pvals_adj"] < 0.05]
    .groupby('gene_naming')['names']
    .nunique()
    .reset_index()
    .rename(columns={'names': "num_DEG"})
           )
# deg_summary=deg_summary[deg_summary['DEG_type'] == "z_norm"]

edist_summary = edist_df

# Merge all summaries into bulk_summary
bulk_summary=orf_info.copy()
summary_dfs=[spearman_summary, pearson_summary, deg_summary, edist_summary]
for df1 in summary_dfs:
    bulk_summary=bulk_summary.merge(df1, on ='gene_naming', how ='left')
bulk_summary=bulk_summary.drop_duplicates()
print(bulk_summary.columns)


# -----------------------------
# Plot: Growth vs number DEG
# -----------------------------
#make plots with labels
plt.figure(figsize=(7,6))
x = 'num_DEG'
y = 'growth_LFC'

# Explicit hue order: last one ('Negative control') will be drawn on top
sns.scatterplot(
    data=bulk_summary,
    x=x, y=y,
    color = 'lightgrey',
    alpha = 0.5
)

sns.scatterplot(
    data=bulk_summary[bulk_summary.edist_FDR<.05],
    x=x, y=y,
    color = 'indigo',
    alpha = 0.3, label = f'Energy test FDR<.05'
)

sns.scatterplot(
    data=bulk_summary[bulk_summary.pert_type=='Negative control'],
    x=x, y=y, alpha = 0.7,
    color='lightsteelblue', label = f'Negative control'
)
sns.scatterplot(
    data=bulk_summary[bulk_summary.pert_type=='Positive control'],
    x=x, y=y,
    color='firebrick', label = f'Positive control'
)

sns.despine()

texts = []
for _, row in bulk_summary[bulk_summary["pert_type"] == "Positive control"].iterrows():
    texts.append(
        plt.text(row[x], row[y], row["gene_naming"],
                 fontsize=9, color="firebrick")
    )


# Adjust spacing
adjust_text(
    texts,
    arrowprops=dict(arrowstyle='-', lw=0.5, color='grey'),
)

plt.xscale('log')
plt.xlabel("Number of DEGs")
plt.ylabel('Growth Phenotype')
plt.legend(title=None, bbox_to_anchor=(1,1))
plt.savefig(f'{fig_out}/growth_DEG.png', dpi = 200)
plt.show()


adata_hvg = adata.copy()#[:, adata.var['highly_variable']].copy()
# 1) Build pseudobulk: mean per perturbation using embedding (Harmony-corrected PCA)
group='gene_naming'
ps = pt.tl.PseudobulkSpace()
psadata = ps.compute(
    adata_hvg,
    target_col=group,
    layer_key='z_norm',
    mode="mean"
)

psadata_og=psadata.copy()
# -----------------------------
# IV–V–VI unified analysis loop for Spearman & Pearson
# -----------------------------
fig_out_pos = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/figs/pos_ctrl"
os.makedirs(fig_out_pos, exist_ok=True)

fig_out_clust = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/figs/clusters"
os.makedirs(fig_out_clust, exist_ok=True)
def run_corr_analysis(corr_type, corr_df, bulk_summary):
    print(f"\n=== Running downstream analysis for {corr_type.upper()} ===")

    # --- Clustermap ---
    print(f"--- {corr_type}: Making clustermap ---")
    n_labels = corr_df.shape[0]
    figsize = (min(100, n_labels * 0.25 + 15), min(100, n_labels * 0.25 + 10))
    cg = sns.clustermap(corr_df, robust=True, figsize=figsize)
    highlight_labels(cg, immune_set, 'indigo')
    highlight_labels(cg, progrow_set_genes, 'seagreen')
    highlight_labels(cg, toxic_set_genes, 'peru')
    highlight_labels(cg, negative_ctrl_set, 'steelblue')
    highlight_labels(cg, positive_ctrl_set, 'firebrick')
    plt.savefig(f'{fig_out}/clustermap_{corr_type}.png', dpi=50)
    plt.close()

    # --- HDBSCAN clustering ---
    print(f"--- {corr_type}: HDBSCAN clustering ---")
    dist_matrix = corr_df.values
    clusterer = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=3,
        min_samples=1,
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(dist_matrix)
    clusters = pd.DataFrame({
        'gene_naming': corr_df.index,
        f'Cluster_{corr_type}': cluster_labels
    })

    print(f'{clusters[f"Cluster_{corr_type}"].nunique()-1} clusters identified')
    print(f'{clusters[clusters[f"Cluster_{corr_type}"]>0].gene_naming.nunique()} perturbations assigned')
    print(f'{clusters[clusters[f"Cluster_{corr_type}"]==-1].gene_naming.nunique()} unassigned')

    # Merge with bulk summary
    merged = bulk_summary.merge(clusters, on='gene_naming', how='left')
    merged.to_csv(f"{python_out}/all_dist-DEG_cluster_summary_{corr_type}.csv", index=False)
    
    print(f"--- {corr_type}: Creating pseudobulk ---")
    psadata=psadata_og.copy()
    
    # Add metadata
    dist_cols = ['gene_naming','pearson_to_control', 'spearman_to_control','num_DEG','edist_to_control', 'edist_pval', f'Cluster_{corr_type}']
    psadata.obs = psadata.obs.join(
        merged[dist_cols].drop_duplicates().set_index('gene_naming'),
        how='left'
    )
    #save file
    savefile = f"{python_out}/psadata_znorm_summary_{corr_type}-clust.h5ad"
    psadata.write_h5ad(savefile)


    # --- Cluster-level clustermaps ---
    print(f"--- {corr_type}: Generating cluster plots ---")

    fig_out_pos = f"{fig_out}/pos_ctrl_{corr_type}"
    fig_out_clust = f"{fig_out}/clusters_{corr_type}"
    os.makedirs(fig_out_pos, exist_ok=True)
    os.makedirs(fig_out_clust, exist_ok=True)

    controls = adata.obs[
        (adata.obs.pert_type == 'Positive control') |
        adata.obs.gene_naming.str.contains('shuff') |
        adata.obs.gene_naming.str.contains('ctrl')
    ].gene_naming.unique()

    for out_dir, subset in [(fig_out_pos, psadata.obs[psadata.obs.pert_type == 'Positive control']),
                            (fig_out_clust, psadata.obs)]:
        for clust in subset[f'Cluster_{corr_type}'].dropna().unique():
            if clust > 0:
                cluster = subset[subset[f'Cluster_{corr_type}'] == clust].gene_naming.unique()
                plotting = list(controls) + list(cluster)
                
                corr=pt.tl.Distance(metric=f"{corr_type}_distance", layer_key='z_norm')
                sub_df = corr.pairwise(adata[adata.obs.gene_naming.isin(plotting)], groupby=group)

                immune = list(sub_df.index[sub_df.index.isin(immune_set)])
                toxic = list(sub_df.index[sub_df.index.isin(toxic_set_genes)])
                progro = list(sub_df.index[sub_df.index.isin(progrow_set_genes)])

                n_labels = sub_df.shape[0]
                figsize = (min(200, n_labels * 0.2 + 7), min(200, n_labels * 0.2 + 7))
                cg = sns.clustermap(sub_df, robust=True, figsize=figsize)
                highlight_labels(cg, cluster, 'black')
                highlight_labels(cg, immune, 'indigo')
                highlight_labels(cg, progro, 'seagreen')
                highlight_labels(cg, pd.Series(toxic)[~pd.Series(toxic).isin(positive_ctrl_set)], 'peru')
                highlight_labels(cg, negative_ctrl_set, 'steelblue')
                highlight_labels(cg, pd.Series(cluster)[pd.Series(cluster).isin(positive_ctrl_set)], 'firebrick')

                filename = f'{out_dir}/clustermap_{corr_type}_cluster{clust}-size{len(cluster)}.png'
                plt.savefig(filename, dpi=200)
                plt.close()

    print(f"--- {corr_type.upper()} analysis complete ---\n")
    return merged


bulk_summary_znorm = run_corr_analysis("pearson", psadata.X, bulk_summary)


# # Run for both correlation types
# bulk_summary_spearman = run_corr_analysis("spearman", spearman_df, bulk_summary)
# bulk_summary_pearson = run_corr_analysis("pearson", pearson_df, bulk_summary)




# # -----------------------------
# # Compute clustermap from perason matrix
# # -----------------------------
# print("--- IV. Making clustermap from corr matrix---")


# # sensible figure size based on number of labels (avoid enormous plots)
# n_labels = spearman_df.shape[0]
# figsize = (min(500, n_labels * 0.25 + 15), min(500, n_labels * 0.25 + 10))
# cg = sns.clustermap(spearman_df, robust=True, figsize=figsize)
# highlight_labels(cg, immune_set, 'indigo')
# highlight_labels(cg, progrow_set_genes, 'seagreen')
# highlight_labels(cg, toxic_set_genes, 'peru')
# highlight_labels(cg, negative_ctrl_set, 'steelblue')
# highlight_labels(cg, positive_ctrl_set, 'firebrick')
# plt.savefig(f'{fig_out}/clustermap_spearman.png', dpi=50)
# plt.close()


# # -----------------------------
# # Perform Clustering with HDBSCAN
# # -----------------------------

# print("--- V. Clustering with HDBSCAN---")
# # Convert to numpy array
# dist_matrix = spearman_df.values
# # HDBSCAN expects a condensed matrix or squareform if metric='precomputed'
# clusterer = hdbscan.HDBSCAN(
#     metric='precomputed',
#     min_cluster_size=4,
#     min_samples=1,
#     cluster_selection_method='eom'
# )

# cluster_labels = clusterer.fit_predict(dist_matrix)
# clusters = pd.DataFrame({
#     'gene_naming': spearman_df.index,
#     'Cluster': cluster_labels
# })

# print(f'{clusters.Cluster.nunique()-1} Clusters identified')
# print(f'{clusters[clusters.Cluster>0].gene_naming.nunique()} perturbations assigned to cluster')
# print(f'{clusters[clusters.Cluster==-1].gene_naming.nunique()} perturbations with no clusters assigned')

# bulk_summary = bulk_summary.merge(clusters, on='gene_naming', how = 'left')
# out_path = f"{python_out}/all_dist-DEG_cluster_summary_spearman.csv"
# bulk_summary.to_csv(out_path, index=False)




# print("--- V. Plot clusters and clustermap ---")
# fig_out_pos = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/figs/pos_ctrl"
# os.makedirs(fig_out_pos, exist_ok=True)

# #m=0
# controls= adata.obs[(adata.obs.pert_type == 'Positive control')|adata.obs.gene_naming.str.contains('shuff')|adata.obs.gene_naming.str.contains('ctrl')].gene_naming.unique()
# for clust in psadata.obs[psadata.obs.pert_type=='Positive control'].Cluster.unique():
#     if clust>0:
#         cluster= psadata.obs[(psadata.obs.Cluster==clust)&(psadata.obs.Cluster>0)].gene_naming.unique()
#         plotting=list(controls)+list(cluster)

#         sub_df = spearman_df.loc[plotting,plotting]
        
#         immune = list(sub_df.index[sub_df.index.isin(immune_set)])
#         toxic=list(sub_df.index[sub_df.index.isin(toxic_set_genes)])
#         progro=list(sub_df.index[sub_df.index.isin(progrow_set_genes)])

#         # sensible figure size based on number of labels 
#         print(list(cluster))
#         n_labels = sub_df.shape[0]
#         figsize = (min(200, n_labels * 0.2 + 7), min(200, n_labels * 0.2 + 7))
#         cg = sns.clustermap(sub_df, robust=True, figsize=figsize)
        
#         highlight_labels(cg, cluster, 'black')
#         highlight_labels(cg, immune, 'indigo')
#         highlight_labels(cg, progro, 'seagreen')
#         highlight_labels(cg, pd.Series(toxic)[~pd.Series(toxic).isin(positive_ctrl_set)], 'peru')
#         highlight_labels(cg, negative_ctrl_set, 'steelblue')
#         # highlight_labels(cg, positive_ctrl_set, 'black')
#         highlight_labels(cg, cluster[cluster.isin(positive_ctrl_set)], 'firebrick')
        
#         filename = f'{fig_out_pos}/clustermap_spearman_cluster{clust}-size{len(cluster)}.png'
#         print(filename)
#         plt.savefig(filename, dpi=200)
        
#         # if m<20:
#         #     m+=1
#         #     plt.show()
#         # break
#         plt.close()
    
        
# fig_out_clust = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/figs/clusters"
# os.makedirs(fig_out_clust, exist_ok=True)

# controls= adata.obs[(adata.obs.pert_type == 'Positive control')|adata.obs.gene_naming.str.contains('shuff')|adata.obs.gene_naming.str.contains('ctrl')].gene_naming.unique()
# for clust in psadata.obs.Cluster.unique():
#     if clust>0:
#         cluster= psadata.obs[(psadata.obs.Cluster==clust)&(psadata.obs.Cluster>0)].gene_naming.unique()
#         plotting=list(controls)+list(cluster)
#         sub_df = spearman_df.loc[plotting,plotting]
        
#         immune = list(sub_df.index[sub_df.index.isin(immune_set)])
#         toxic=list(sub_df.index[sub_df.index.isin(toxic_set_genes)])
#         progro=list(sub_df.index[sub_df.index.isin(progrow_set_genes)])

#         # sensible figure size based on number of labels 
#         print(list(cluster))
#         n_labels = sub_df.shape[0]
#         figsize = (min(200, n_labels * 0.2 + 7), min(200, n_labels * 0.2 + 7))
#         cg = sns.clustermap(sub_df, robust=True, figsize=figsize)
        
#         highlight_labels(cg, cluster, 'black')
#         highlight_labels(cg, immune, 'indigo')
#         highlight_labels(cg, progro, 'seagreen')
#         highlight_labels(cg, pd.Series(toxic)[~pd.Series(toxic).isin(positive_ctrl_set)], 'peru')
#         highlight_labels(cg, negative_ctrl_set, 'steelblue')
#         # highlight_labels(cg, positive_ctrl_set, 'black')
#         highlight_labels(cg, cluster[cluster.isin(positive_ctrl_set)], 'firebrick')
        
#         filename = f'{fig_out_clust}/clustermap_spearman_cluster{clust}-size{len(cluster)}.png'
#         print(filename)
#         plt.savefig(filename, dpi=200)
        
#         # if m<20:
#         #     m+=1
#         #     plt.show()
#         # break
#         plt.close()
        



