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
import networkx as nx
import pertpy as pt
import gseapy as gp


warnings.filterwarnings('ignore')
sns.set_style('white')
sns.set_palette('deep')

# -----------------------------
# Paths and directories
# -----------------------------
fig_out = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/figs"
python_out = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/test/python_outputs"
os.makedirs(fig_out, exist_ok=True)
os.makedirs(python_out, exist_ok=True)
sc.settings.figdir = fig_out

def highlight_labels(clustergrid, highlight_set, color):
    """Bold and color tick labels in a ClusterGrid if they are in highlight_set."""
    for axis in [clustergrid.ax_heatmap]:
        for lbl in axis.get_xticklabels() + axis.get_yticklabels():
            if lbl.get_text() in highlight_set:
                lbl.set_color(color)
                lbl.set_fontweight("bold")


def balance_enrich_adata(adata, elements, name, random_state=0, balance_negative=False):
    """
    Create a balanced AnnData subset for enrichment analysis.
    
    Parameters
    ----------
    adata : AnnData
        The full AnnData object.
    elements : list
        List of GeneIDs (experimental targets of interest).
    name : str
        The label to assign to the experimental group.
    random_state : int, optional
        Seed for reproducible sampling.
    balance_negative : bool, optional
        If True, also downsample the 'negative' group to the same size.
    
    Returns
    -------
    AnnData
        Balanced subset of AnnData ready for enrichment or DEG analysis.
    """

    # Subset to relevant cells
    neg_adata=adata[adata.obs.gene_naming.str.contains('shuff')|
                    adata.obs.gene_naming.str.contains('GFP')]
    enrich_adata = adata[
        adata.obs.gene_naming.isin(elements)
        | adata.obs.gene_naming.isin(neg_adata.obs.gene_naming)
    ].copy()
    
    # Standardize gene_naming: all controls -> 'negative'
    enrich_adata.obs['gene_naming'] = [
        'negative' if (n in neg_adata.obs.gene_naming.unique()) else name
        for n in enrich_adata.obs.gene_naming
    ]

    
    # Determine minimum number of cells per experimental group
    counts = adata[adata.obs.gene_naming.isin(elements)].obs['gene_naming'].value_counts()
    min_cells = int(counts.min()*.9)
    # print(f"Downsampling to {min_cells} cells per experimental group")
    # Sample equal number of cells for each experimental group
    sampled_idx = []
    for g in enrich_adata.obs.gene_naming.unique():
        idx = enrich_adata.obs.query("gene_naming == @g").sample(
            n=min_cells, random_state=random_state
        ).index
        sampled_idx.extend(idx)

    # Final combined indices
    enrich_adata = enrich_adata[sampled_idx].copy()
    
    

    # print(enrich_adata.obs['gene_naming'].value_counts())
    return enrich_adata


def pathway_enrich(enrich_adata, name='', fdr=.25, figsize=(9,4), show=True, permutation = 300):

    print(f'Processing {name}')

    # Build class labels
    classes = enrich_adata.obs['gene_naming']
    classes = np.where(classes == 'negative', 'negative', 'positive')

    # Prepare data matrix (genes x samples)
    data = enrich_adata.to_df().T
    data.index = enrich_adata.var_names  # ensure gene names preserved

    # Run GSEA
    res = gp.gsea(
        data=data,
        gene_sets='GO_Biological_Process_2021',
        cls=classes,
        permutation_num=permutation,
        permutation_type='phenotype',
        outdir=None,
        method='s2n',
        threads=4,
    )

    res_df = res.res2d.copy()

    # Filter generic terms
    # generic_terms = [
    #     'biological process', 'cellular process', 'regulation of',
    #     'response to stimulus', 'metabolic process', 'system process'
    # ]
    # res_df = res_df[~res_df.Term.str.contains('|'.join(generic_terms), case=False)]
    res_df.Term = res_df.Term.str.split(" \(GO").str[0]

    # Label direction
    res_df['direction'] = np.where(res_df.NES < 0, 'Upregulated', 'Downregulated')
    res_df=res_df[res_df['NOM p-val'] < fdr]

    # Select top enriched terms
    up= (
        res_df[res_df['direction']=='Upregulated'].sort_values('FDR q-val')
        .reset_index(drop=True)
        .head(15).sort_values('NES')
    )
    down= (
        res_df[res_df['direction']=='Downregulated'].sort_values('FDR q-val')
        .reset_index(drop=True)
        .head(7).sort_values('NES')
    )
    top =pd.concat([up,down]).reset_index()
    top['NES'] = np.abs(top['NES'])
    top=top
    

    # Plot if any enriched terms found
    if len(top) > 0:
        if show:
            
            fig, ax = plt.subplots(figsize=figsize)
            sns.barplot(
                data=top,
                x='NES',
                y='Term',
                hue='direction',
                palette={'Upregulated':'seagreen', 
                         'Downregulated':'peru'},
                alpha=0.7,
                ax=ax
            )
            ax.set_xlabel('Enrichment score')
            ax.set_ylabel('')
            ax.set_title(f'{name} GO Pathway enrichment (FDR < {fdr})')
            sns.despine(ax=ax)

            # Place legend just outside the plot box
            ax.legend(
                title='Direction',
                loc='upper left',
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0
            )

            plt.tight_layout()
            plt.show()

        return res_df, fig
    else:
        print(f'No significant pathways found (FDR < {fdr})')
        return ''

#Load data
savefile = f"/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/python_outputs/adata_lognorm_harmony_hvg_cc-regress.h5ad"
adata = ad.read_h5ad(savefile)
savefile = f"/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/python_outputs/psadata_znorm-pca.h5ad"
psadata=sc.read_h5ad(savefile)
psadata.obs['labels'] = [row.gene_naming if 'ositive' in row.pert_type else row.pert_type for i,row in psadata.obs.iterrows()]
print(psadata)

fig_out_pos = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/figs/pos_ctrl"
os.makedirs(fig_out_pos, exist_ok=True)
controls= adata.obs[(adata.obs.pert_type == 'Positive control')|adata.obs.gene_naming.str.contains('shuff')|adata.obs.gene_naming.str.contains('ctrl')].gene_naming.unique()
for clust in psadata.obs[psadata.obs.pert_type=='Positive control'].Cluster.unique():
    if clust>0:
        cluster= psadata.obs[(psadata.obs.Cluster==clust)&(psadata.obs.Cluster>0)].gene_naming.unique()
        plotting=list(controls)+list(cluster)
        sub_adata=adata[adata.obs.gene_naming.isin(plotting)]
        name = f'cluster{clust}-size{len(cluster)}'
        print(list(cluster))
        

        immune = list(sub_df.index[sub_df.index.isin(immune_set)])
        toxic=list(sub_df.index[sub_df.index.isin(toxic_set_genes)])
        progro=list(sub_df.index[sub_df.index.isin(progrow_set_genes)])
        

        # Make clustermap
        distance = pt.tl.Distance(metric="spearman_distance", obsm_key="X_pca_z_norm_harmony")
        sub_df = distance.pairwise(sub_adata, groupby='gene_naming')
        n_labels = sub_df.shape[0]
        figsize = (min(200, n_labels * 0.2 + 7), min(200, n_labels * 0.2 + 7))
        cg = sns.clustermap(sub_df, robust=True, figsize=figsize)
        highlight_labels(cg, cluster, 'black')
        highlight_labels(cg, immune, 'indigo')
        highlight_labels(cg, progro, 'seagreen')
        highlight_labels(cg, pd.Series(toxic)[~pd.Series(toxic).isin(positive_ctrl_set)], 'peru')
        highlight_labels(cg, negative_ctrl_set, 'steelblue')
        highlight_labels(cg, cluster[cluster.isin(positive_ctrl_set)], 'firebrick')
        #savefig
        filename = f'{fig_out_pos}/clustermap_spearman_dist_{name}.png'
        print(filename)
        plt.savefig(filename, dpi=200)
        

        # Perform GSEA
        enrich_adata = balance_enrich_adata(adata, cluster, name, balance_negative=False)
        gsea_df, fig = pathway_enrich(enrich_adata, name=name)
        fig.savefig(f'{fig_out_pos}/{name}_gsea.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        
            
    
        