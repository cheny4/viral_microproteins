#!/usr/bin/env python
# coding: utf-8


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
import gseapy as gseapy

import hdbscan




#Import cell cycle genes
cellcycle ='/lab/solexa_weissman/yhc/viral_ORF/20241209_elledge_150_new-analysis1/postprocess_analysis/regev_lab_cell_cycle_genes.txt'
cell_cycle_genes = [x.strip() for x in open(cellcycle)]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]


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

def highlight_labels(clustergrid, highlight_set, color, bold=True):
    """Bold and color tick labels in a ClusterGrid if they are in highlight_set."""
    for axis in [clustergrid.ax_heatmap]:
        for lbl in axis.get_xticklabels() + axis.get_yticklabels():
            if lbl.get_text() in highlight_set:
                lbl.set_color(color)
                if bold:
                    lbl.set_fontweight("bold")

def balance_enrich_adata(adata, elements, name, random_state=0, balance_negative=True, mincell = 10):
    """
    Creates a balanced AnnData subset for enrichment analysis.

    Parameters
    ----------
    adata : AnnData
        Full AnnData object.
    elements : list
        List of gene_naming values (experimental targets of interest).
    name : str
        Label for the experimental group.
    random_state : int
        Seed for reproducibility.
    balance_negative : bool
        Whether to downsample negative group to match experimental.

    Returns
    -------
    AnnData
        Balanced AnnData object.
    """

    # -----------------------------
    # 1. Identify negatives
    # -----------------------------
    neg_mask = (
        adata.obs.Geneid.str.contains('egative') |
        adata.obs.Geneid.str.contains('shuff')   |
        adata.obs.Geneid.str.contains('ctrl')    |
        adata.obs.Geneid.str.contains('GFP')
    )
    neg_adata = adata[neg_mask]

    # -----------------------------
    # 2. Subset to relevant groups
    # -----------------------------
    enrich_adata = adata[
        adata.obs.gene_naming.isin(elements) |
        adata.obs.gene_naming.isin(neg_adata.obs.gene_naming)
    ].copy()

    # -----------------------------
    # 3. Standardize controls → "negative"
    # -----------------------------
    neg_names = set(neg_adata.obs.gene_naming.unique())
    enrich_adata.obs['gene_naming'] = [
        'negative' if x in neg_names else x
        for x in enrich_adata.obs.gene_naming
    ]

    # -----------------------------
    # 4. Determine downsampling size
    # -----------------------------
    print(enrich_adata.obs['gene_naming'].value_counts())
    exp_counts = enrich_adata.obs.query("gene_naming != 'negative'")['gene_naming'].value_counts()
    min_cells = int(exp_counts.min())

    # -----------------------------
    # 5. Sample experimental groups
    # -----------------------------
    sampled_idx = []
    exp_groups = [g for g in enrich_adata.obs['gene_naming'].unique() if g != 'negative']

    for g in exp_groups:
        cells = enrich_adata.obs[enrich_adata.obs['gene_naming']==g]
        if len(cells)> min_cells*1.1:
            idx = enrich_adata.obs.query("gene_naming == @g").sample(
                n=min_cells,
                random_state=random_state,
                replace=False
            ).index
        else:
            idx=cells.index
        sampled_idx.extend(idx)

    # -----------------------------
    # 6. Sample negative group once
    # -----------------------------
    if balance_negative:
        neg_pool = enrich_adata.obs.query("gene_naming == 'negative'").index
        need_replace = len(neg_pool) < min_cells

        neg_sample = np.random.RandomState(random_state).choice(
            neg_pool,
            size=min_cells,
            replace=need_replace
        )
        sampled_idx.extend(neg_sample)

    # -----------------------------
    # 7. Ensure unique indices
    # -----------------------------
    sampled_idx = list(dict.fromkeys(sampled_idx))

    # -----------------------------
    # 8. Final subset
    # -----------------------------
    enrich_adata = enrich_adata[sampled_idx].copy()

    return enrich_adata



def pathway_enrich(enrich_adata, name='', fdr=.25, figsize=(7,4), show=True, permutation = 300, up_num=10, down_num=10):

    print(f'Processing {name}')
    
    enrich_adata.obs['classes']=['positive' if g !='negative' else 'negative' for g in enrich_adata.obs['gene_naming']]

    # Build class labels
    classes = enrich_adata.obs['classes']
    classes = np.where(classes == 'negative', 'negative', 'positive')

    # Prepare data matrix (genes x samples)
    data = enrich_adata.to_df().T
    data.index = enrich_adata.var_names  # ensure gene names preserved

    # Run GSEA
    res = gseapy.gsea(
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
    res_df['direction'] = np.where(res_df.NES > 0, 'Upregulated', 'Downregulated')
    res_df=res_df[res_df['FDR q-val'] < fdr]

    # Select top enriched terms
    up= (
        res_df[res_df['direction']=='Upregulated'].sort_values('FDR q-val')
        .reset_index(drop=True)
        .head(up_num).sort_values('NES')
    )
    down= (
        res_df[res_df['direction']=='Downregulated'].sort_values('FDR q-val')
        .reset_index(drop=True)
        .head(down_num).sort_values('NES')
    )
    top =pd.concat([up,down]).reset_index()
    top['NES'] = np.abs(top['NES'])
    top=top
    

    # Plot if any enriched terms found
    if len(top) > 0:
        if show:
            figsize=(figsize[0],(figsize[1]/2)+(figsize[1]/2)*(len(top)/(up_num+down_num)))
            fig, ax = plt.subplots(figsize=figsize)
            sns.barplot(
                data=top,
                x='NES',
                y='Term',
                hue='direction',
                palette={'Upregulated':'darkred', 
                         'Downregulated':'navy'},
                alpha=0.7,
                ax=ax
            )
            ax.set_xlabel('Enrichment score')
            ax.set_ylabel('')
            ax.set_title(f'{name} GO Pathway enrichment\n(FDR < {fdr})'.replace(' | ','\n'))
            sns.despine(ax=ax)

            # Place legend just outside the plot box
            ax.legend(
                title='Direction',
                loc='upper left',
                # bbox_to_anchor=(1.02, 1),
                borderaxespad=0
            )
            ax.get_legend().remove() 


            plt.tight_layout()
            plt.show()

        return res_df, fig
    else:
        print(f'No significant pathways found (FDR < {fdr})')
        return '',''

def examine_cell_cycle(
    adata,
    cell_cycle_genes,
    cell_lim=1,
    g='gene_naming',
    order=None,
    highlight_dict=None,
    figsize=(5,4)
):
    """
    Summarize & plot cell-cycle phase proportions per gene (or other grouping column `g`).

    Phase order is fixed to: S → G2M → G1
    Bars are stacked horizontally by phase, with proportions shown per gene.

    Returns
    -------
    adata_subset : AnnData
        Subset of adata corresponding to genes plotted.
    cc_summary : pd.DataFrame
        Summary of cell cycle composition for each gene and phase.
    """

    # --- Step 0: summarize counts ---
    cc = (
        adata.obs
        .loc[:, [g, 'phase']]
        .groupby([g, 'phase'])
        .size()
        .reset_index(name='num_cells')
    )

    # keep only genes with enough cells
    gene_totals = cc.groupby(g)['num_cells'].sum()
    valid_genes = gene_totals[gene_totals > cell_lim].index.tolist()
    cc = cc[cc[g].isin(valid_genes)].copy()

    # fixed phase order: S → G2M → G1
    phase_order = ['S', 'G2M', 'G1']

    # fill missing (gene, phase) pairs with zeros
    full_index = pd.MultiIndex.from_product([valid_genes, phase_order], names=[g, 'phase'])
    cc = cc.set_index([g, 'phase']).reindex(full_index, fill_value=0).reset_index()

    # compute proportions per gene
    cc['proportion'] = cc.groupby(g)['num_cells'].transform(lambda x: x / x.sum())

    # compute growing fraction (S + G2M)
    cc['is_growing_phase'] = cc['phase'].isin(['S', 'G2M']).astype(int)
    growing_by_gene = cc.groupby(g).apply(lambda df: (df['proportion'] * df['is_growing_phase']).sum())
    growing_by_gene = growing_by_gene.rename('growing_fraction')

    # --- Step 1: determine gene order ---
    if order and len(order) > 0:
        provided = [x for x in order if x in growing_by_gene.index]
        remaining = [x for x in growing_by_gene.index if x not in provided]
        remaining_sorted = growing_by_gene.loc[remaining].sort_values(ascending=False).index.tolist()
        gene_order = provided + remaining_sorted
    else:
        gene_order = growing_by_gene.sort_values(ascending=True).index.tolist()

    # --- Step 2: pivot for stacked bar plotting ---
    pivot = (
        cc.pivot_table(index=g, columns='phase', values='proportion', aggfunc='sum')
        .fillna(0)
        .reindex(gene_order)
    )
    # enforce S→G2M→G1 order for columns
    pivot = pivot[[ph for ph in phase_order if ph in pivot.columns]]

    # --- Step 3: plot ---
    phase_colors = {'S': 'indianred', 'G2M': 'lightcoral', 'G1': 'silver'}
    colors = [phase_colors.get(ph, 'grey') for ph in pivot.columns]

    fig, ax = plt.subplots(figsize=(figsize))

    pivot.plot(
        kind='barh',
        stacked=True,
        ax=ax,
        color=colors,
        edgecolor='none',
        width=.75 
    )

    # --- Step 4: annotate total cell counts ---
    total_cells_per_gene = (
        cc.groupby(g)['num_cells'].sum().reindex(gene_order).astype(int)
    )
    for i, gene in enumerate(gene_order):
        ax.text(
            1, i,
            s=str(total_cells_per_gene.loc[gene]),
            transform=ax.get_yaxis_transform(),
            ha='left',
            va='center',
            fontsize=8,
            color='black'
        )

    # --- Step 5: style adjustments ---
    ax.set_xlabel('Proportion')
    ax.set_ylabel('')
    ax.set_xlim(0, 1.01)
    ax.set_yticks(range(len(gene_order)))
    ax.set_yticklabels(gene_order)
    sns.despine(fig=fig, left=False)
    # plt.xticks(rotation=90)

    # reorder legend according to S→G2M→G1
    handles, labels = ax.get_legend_handles_labels()
    legend_order = [ph for ph in phase_order if ph in labels]
    ordered_handles = [handles[labels.index(ph)] for ph in legend_order]
    ax.legend(
        ordered_handles, legend_order,
        bbox_to_anchor=(1, 1), loc='upper left',
        title='Phase'
    )

    # --- Step 6: highlight selected genes ---
    if highlight_dict:
        for label in ax.get_yticklabels():
            gene_label = label.get_text()
            for _, (gene_list, color) in highlight_dict.items():
                if gene_label in gene_list:
                    label.set_color(color)
                    label.set_fontweight('bold')

    plt.tight_layout()

    # --- Step 7: subset adata and return ---
    orf_cells = adata.obs[adata.obs[g].isin(gene_order)].index.tolist()
    adata_subset = adata[adata.obs.index.isin(orf_cells), :].copy()

    cc_summary = cc.merge(growing_by_gene.reset_index(), on=g, how='left')
    return adata_subset, cc_summary
