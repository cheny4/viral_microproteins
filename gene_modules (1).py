#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

warnings.filterwarnings('ignore')
sns.set_style('white')
sns.set_palette('deep')

fig_out="./figs3"
python_out = "./python_outputs3"
suffix='_all'
sc.settings.figdir = fig_out


import os
def rescale_kde(g):
    # rescale the curves in the x direction
    max_height = max([np.max(curve.get_data()[1]) for curve in g.lines])
    for curve in g.lines:
        height = np.max(curve.get_data()[1])
        curve.set_ydata(curve.get_data()[1] / height * max_height)


def examine_cell_cycle(adata, cc_adata, cell_cycle_genes, cell_lim=1):
    
    # Step 1: counts
    cc_summary = (
        adata.obs.groupby(['gene', 'phase'])
        .size()
        .reset_index(name='num_cells')
    )

    # Step 1b: drop genes with no cells at all
    gene_totals = cc_summary.groupby('gene')['num_cells'].sum()
    valid_genes = gene_totals[gene_totals > 0].index
    cc_summary = cc_summary[cc_summary['gene'].isin(valid_genes)]

    # Step 2: proportions
    cc_summary['proportion'] = (
        cc_summary['num_cells'] /
        cc_summary.groupby('gene')['num_cells'].transform('sum')
    )

    # Step 3: growing score (non-G1 phases only)
    cc_summary['growing'] = (cc_summary['phase'] != 'G1') * cc_summary['proportion']

    # Step 4: gene order by total growing proportion
    gene_order = (
        cc_summary.groupby('gene')['growing']
        .sum()
        .sort_values()
        .index.tolist()
    )
    #print(gene_order)

    # Step 5: filter & sort
    cc_summary = cc_summary[cc_summary['gene'].isin(gene_order)]
    cc_summary['gene'] = pd.Categorical(cc_summary['gene'], categories=gene_order, ordered=True)
    cc_summary = cc_summary.sort_values('gene')
    


    plt.figure(figsize=(len(gene_order)*.3+2, 3))

    # Create a stacked bar plot
    ax = sns.histplot(
        data=cc_summary,
        x="gene", hue="phase", weights="proportion",
        multiple="stack", shrink=0.8, hue_order=sorted(cc_summary['phase'].unique())
    )

    # Annotate the total number of cells above each bar
    for gene in gene_order:
        total_cells = cc_summary[cc_summary.gene == gene]['num_cells'].sum()
        gene_idx = list(gene_order).index(gene)
        ax.text(
            x=gene_idx,
            y=1.02,  # Slightly above the top of the bar
            s=f"{total_cells}",
            ha='center',
            va='bottom',
            fontsize=8,
            color='black'
        )

    # Adjust the plot
    plt.xticks(rotation=90)
    sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1))
    sns.despine()
    plt.xlabel("Gene")
    plt.ylabel("Proportion")
    

    print(f"Total number of cells: {cc_summary['num_cells'].sum()}")

    orf_cells = list(adata.obs[adata.obs.gene.isin(gene_order)].index)
    adata=adata[adata.obs.index.isin(orf_cells),:].copy()
    cc_adata=cc_adata[cc_adata.obs.index.isin(orf_cells),:].copy()
    return adata, cc_adata


#isr
isr_genes =[
    "AKT3", "ABCA7", "CREB3", "CEBPA", "CEBPB", "CEBPD", "CEBPE", "BATF", "CEBPG",
    "AGR2", "TMED2", "GCN1", "OMA1", "BATF2", "DDIT3", "EIF2S1", "AKT1", "AKT2",
    "FOS", "PPP1R15A", "ARIH1", "EIF2AK1", "HSPA5", "JUN", "JUNB", "MAF", "EIF2AK4",
    "ATF4", "NCK1", "NFE2", "NFE2L2", "HERC5", "MAP3K20", "QRICH1", "TMEM33", "ATAD3A",
    "IMPACT", "BATF3", "PTPN1", "PTPN2", "CREBZF", "DDRGK1", "BOK", "RPAP2", "FOSL1",
    "NCK2", "PPP1R15B", "EIF2AK3", "NFE2L3", "DELE1", "MAFB"]

#ifn_genes
ifn_genes = [
    "ADAR", "CDC37", "USP18", "TREX1", "TRIM6", "DCST1", "TTLL12", "YTHDF3", "SAMHD1",
    "GIGYF2", "LSM14A", "TBK1", "CNOT7", "UBE2K", "STING1", "IRF3", "IRF7", "USP27X",
    "SMIM30", "MIR21", "MMP12", "OAS1", "OAS3", "YTHDF2", "RBM47", "METTL3", "MAVS",
    "USP29", "PTPN1", "PTPN2", "PTPN6", "PTPN11", "CACTIN", "STAT2", "WNT5A", "MUL1",
    "ZBP1", "TRIM56", "NLRC5", "FADD", "TRIM41", "RNF185", "EIF4E2", "ISG15", "IKBKE",
    "TANK", "IFITM3", "IFITM2", "CR2", "TRIM65", "SIN3A", "SETD2", "IFNE", "IFI27",
    "IFIT1", "IFNA1", "IFNA2", "IFNA4", "IFNA5", "IFNA6", "IFNA7", "IFNA8", "IFNA10",
    "IFNA14", "IFNA16", "IFNA17", "IFNA21", "IFNAR1", "IFNAR2", "IFNB1", "IFNW1",
    "IRAK1", "JAK1", "MX1", "MYD88", "OAS2", "SHFL", "IFNK", "GPR108", "IFIH1", "AZI2",
    "SHMT2", "SMPD1", "SP100", "STAT1", "TRAF3", "TYK2", "IFITM1", "OASL", "CH25H",
    "TBKBP1", "HDAC4"
]

#nfkb_genes
nfkb_genes = [
    "TLR6", "NOD1", "TAB1", "RBCK1", "ZNF268", "EDAR", "COPS8", "RIPK3", "PTP4A3",
    "CHI3L1", "PHB2", "NLRP3", "CHUK", "C1QTNF3", "C1QTNF4", "TRIM6", "TRIM40", "RC3H1",
    "PYDC2", "DAB2IP", "CYLD", "DDX3X", "TRIM60", "DLG1", "AGER", "EDA", "EDN1", "AGO3",
    "NLRC3", "RTKN2", "ADGRG3", "FOXJ1", "TAB2", "SASH1", "MKRN2", "TAB3", "LETMD1",
    "PTPN22", "AGO1", "GREM1", "AMFR", "PDCD4", "PYCARD", "CARD10", "HMGB1", "BIRC2",
    "BIRC3", "IFI35", "APP", "IL1B", "IL12B", "IL18", "RHOA", "LGALS9", "MIR132",
    "MIR149", "MIR15B", "MIR182", "MIR204", "MIR21", "MIR223", "MIR27A", "MIR27B",
    "MIR29B1", "MIR9-1", "NR3C2", "NDUFC2", "NFKB1", "NFKB2", "NFKBIA", "CCN3", "PRDX1",
    "ADIPOR1", "IL23A", "HDAC7", "PHB1", "TREM2", "TERF2IP", "RC3H2", "TRIM44", "LIME1",
    "PPM1A", "PPM1B", "ADISSP", "UACA", "NLRP2", "EIF2AK2", "AKIP1", "TCIM", "MIR508",
    "REL", "RELA", "RELB", "BCL3", "RPS3", "CCL19", "NOD2", "LRRC19", "BMP7", "SPI1",
    "MAP3K7", "TLR3", "TNF", "TRAF2", "TRAF6", "TRIP6", "VCP", "EZR", "MIR766", "TRIM26",
    "LAPTM5", "CARD14", "ZC3H12A", "ZFP91", "ACTN4", "CALR", "TRIM56", "TRIM55",
    "HAVCR2", "TNFSF11", "RIPK1", "TNFSF14", "TNFRSF11A", "TNFRSF10B", "TNFRSF10A",
    "IL18R1", "SPHK1", "CPNE1", "BCL10", "BTRC", "TRIM15", "NOL3", "MAP3K14", "NMI",
    "NLRP12", "CD27", "CD86", "LITAF", "RASSF2", "TNFSF15"
]

upr_genes = [
    "AKT3","BCL2L11","OPTN","PIGBOS1","STUB1","ABCA7","RACK1","CREB3","AGR2","ERN2",
    "OS9","TMED2","COPS5","EDEM2","FICD","PACRG","ATF6B","ASB11","CTH","DAB2IP",
    "CREBRF","DAXX","DDIT3","BHLHA15","EIF2S1","AKT1","AKT2","ERN1","ATF6","UFL1",
    "ABCB10","PPP1R15A","FUT1","HSPB8","TBL2","AMFR","BBC3","SERP1","ERLEC1","ERO1A",
    "HSF1","HSPA1A","HSPA5","HSPD1","SERP2","MIR199A1","DNAJB9","ATF3","ATF4","NCK1",
    "NFE2L2","PRKN","DERL2","BFAR","MBTPS2","TMBIM4","TM7SF3","PIK3R1","DNAJC10",
    "QRICH1","PARP16","TMEM33","ATAD3A","YOD1","SELENOS","EIF2AK2","PARP6","RHBDD2",
    "PTPN1","PTPN2","BAK1","BAX","CREBZF","CCND1","HERPUD2","DDRGK1","BOK","TMBIM6",
    "UMOD","VCP","WFS1","XBP1","MANF","DERL1","PARP8","RPAP2","ERMP1","EDEM3",
    "CDK5RAP3","RHBDD1","NCK2","TMTC4","PPP1R15B","STC2","MBTPS1","CREB3L1","DERL3",
    "VAPB","EIF2AK3","BAG3","RNF7","HERPUD1"
]

#Import cell cycle genes
cellcycle ='/lab/solexa_weissman/yhc/viral_ORF/20241209_elledge_150_new-analysis1/postprocess_analysis/regev_lab_cell_cycle_genes.txt'
cell_cycle_genes = [x.strip() for x in open(cellcycle)]

s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]

# In[ ]:


# adata=ad.read_h5ad('/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis/all_lognorm_hvg_combat_pert_cc-regress.h5ad')
# adata=adata[adata.obs.gene!='GFP11']

# screen3_genes='/lab/weissman_imaging/yhc/validation_screen/python_outputs/MHC_IFN-top_IFN-bottom.csv'
# df=pd.read_csv(screen3_genes)
# df['gene_naming'] =[str(df.iloc[i].virus_name)+'\n'+str(p) if (p!=0) & ~('ORF_' in str(p)) else df.iloc[i].Geneid for i,p in enumerate(df['product'])]

# adata.obs = adata.obs.merge(df[['Geneid', 'gene_naming']],on='Geneid', how='left')

# distance = pt.tl.Distance(metric="euclidean", obsm_key="X_pca")
# df = distance.pairwise(adata, groupby='gene_naming')
# sns.clustermap(df, robust=True, figsize=(50, 50))
# plt.savefig(f'{fig_out}/clustermap_euclidean')
# plt.show()

                


# screen1_genes= '/lab/weissman_imaging/yhc/validation_screen/python_outputs/K562_none_brefeldinA.csv'
# screen2_genes= '/lab/weissman_imaging/yhc/validation_screen/python_outputs/K562_none_thapsigargin.csv'
# screen3_genes='/lab/weissman_imaging/yhc/validation_screen/python_outputs/MHC_IFN-top_IFN-bottom.csv'
# screen4_genes='/lab/weissman_imaging/yhc/validation_screen/python_outputs/MHC_none-top_none-bottom.csv'

# brefA_screen=pd.read_csv(screen1_genes)

# thaps_screen=pd.read_csv(screen2_genes)

# ifn_screen=pd.read_csv(screen3_genes)

# mhc_screen=pd.read_csv(screen4_genes)



# # In[9]:


# sc.tl.score_genes(adata, isr_genes, score_name='isr_score')
# sc.tl.score_genes(adata, ifn_genes, score_name='ifn_score')
# sc.tl.score_genes(adata, nfkb_genes, score_name='nfkb_score')
# sc.tl.score_genes(adata, upr_genes, score_name='upr_score')



# # In[15]:


# brefA_screen


# # In[20]:


# summary_scores=adata.obs.groupby(['Geneid', 'pert_type'])[['isr_score','ifn_score', 'nfkb_score', 'upr_score']].mean().dropna().reset_index()
# summary_scores=summary_scores.merge(ifn_screen[['Geneid','sample', 'LFC','product','virus_name']], on='Geneid')
# summary_scores=summary_scores.rename(columns={'LFC':'IFN_screen'})

# summary_scores=summary_scores.merge(thaps_screen[['Geneid', 'LFC']], on='Geneid')
# summary_scores=summary_scores.rename(columns={'LFC':'ISR_screen'})

# summary_scores=summary_scores.merge(mhc_screen[['Geneid','LFC']], on='Geneid')
# summary_scores=summary_scores.rename(columns={'LFC':'MHC_screen'})

# summary_scores=summary_scores.merge(brefA_screen[['Geneid','LFC']], on='Geneid')
# summary_scores=summary_scores.rename(columns={'LFC':'BrefA_screen'})

# summary_scores['gene_naming'] =[str(summary_scores.iloc[i].virus_name)+'\n'+str(p) if (p!=0) & ~('ORF_' in str(p)) else summary_scores.iloc[i].Geneid for i,p in enumerate(summary_scores['product'])]



# # In[21]:


# summary_scores


# # In[23]:
# show_genes=10


# x = 'ifn_score'
# y='IFN_screen'
# plt.figure(figsize=(10,10))
# sns.scatterplot(summary_scores, 
#                 x = x, 
#                 y =y, hue = 'pert_type', alpha = 0.5)
# sns.scatterplot(summary_scores[summary_scores.pert_type=='Positive control'], 
#                 x = x,
#                 y =y, 
#                 hue = 'pert_type', alpha = 1, legend=None)
# plt.legend(bbox_to_anchor=(1,1), loc='upper left')

# labels=summary_scores[summary_scores.pert_type=='Positive control']
# for i, row in labels.iterrows():
#     plt.text(
#         row[x],  # x coordinate
#         row[y],          # y coordinate
#         row['Geneid'],            # text label
#         fontsize=9,
#         ha='center',             # horizontal alignment
#         va='bottom'               # vertical alignment
#     )
# labels=pd.concat([summary_scores.sort_values(x).reset_index()[0:show_genes],summary_scores.sort_values(x).reset_index()[-show_genes:]])
# for i, row in labels.iterrows():
#     plt.text(
#         row[x],  # x coordinate
#         row[y],          # y coordinate
#         row['gene_naming'],            # text label
#         fontsize=5,
#         ha='center',    # horizontal alignment
#         va='bottom'               # vertical alignment
#     )
# sns.despine()
# plt.savefig(f'{fig_out}/scatter_{x}_{y}.png', dpi = 200, bbox_inches='tight')
# plt.show()


# x = 'nfkb_score'
# y='IFN_screen'
# plt.figure(figsize=(10,10))
# sns.scatterplot(summary_scores, 
#                 x = x, 
#                 y =y, hue = 'pert_type', alpha = 0.5)
# sns.scatterplot(summary_scores[summary_scores.pert_type=='Positive control'], 
#                 x = x,
#                 y =y, 
#                 hue = 'pert_type', alpha = 1, legend=None)
# plt.legend(bbox_to_anchor=(1,1), loc='upper left')

# labels=summary_scores[summary_scores.pert_type=='Positive control']
# for i, row in labels.iterrows():
#     plt.text(
#         row[x],  # x coordinate
#         row[y],          # y coordinate
#         row['Geneid'],            # text label
#         fontsize=9,
#         ha='center',            # horizontal alignment
#         va='bottom'               # vertical alignment
#     )
# labels=pd.concat([summary_scores.sort_values(x).reset_index()[0:show_genes],summary_scores.sort_values(x).reset_index()[-show_genes:]])
# for i, row in labels.iterrows():
#     plt.text(
#         row[x],  # x coordinate
#         row[y],          # y coordinate
#         row['gene_naming'],            # text label
#         fontsize=5,
#         ha='center',             # horizontal alignment
#         va='bottom'               # vertical alignment
#     )
# sns.despine()
# plt.savefig(f'{fig_out}/scatter_{x}_{y}.png', dpi = 200, bbox_inches='tight')
# plt.show()


# x = 'isr_score'
# y='ISR_screen'
# plt.figure(figsize=(10,10))
# sns.scatterplot(summary_scores, 
#                 x = x, 
#                 y =y, hue = 'pert_type', alpha = 0.5)
# sns.scatterplot(summary_scores[summary_scores.pert_type=='Positive control'], 
#                 x = x,
#                 y =y, 
#                 hue = 'pert_type', alpha = 1, legend=None)
# plt.legend(bbox_to_anchor=(1,1), loc='upper left')

# labels=summary_scores[summary_scores.pert_type=='Positive control']
# for i, row in labels.iterrows():
#     plt.text(
#         row[x],  # x coordinate
#         row[y],          # y coordinate
#         row['gene_naming'],            # text label
#         fontsize=9,
#         ha='center',            # horizontal alignment
#         va='bottom'               # vertical alignment
#     )
# labels=pd.concat([summary_scores.sort_values(x).reset_index()[0:show_genes],summary_scores.sort_values(x).reset_index()[-show_genes:]])
# for i, row in labels.iterrows():
#     plt.text(
#         row[x],  # x coordinate
#         row[y],          # y coordinate
#         row['gene_naming'],            # text label
#         fontsize=5,
#         ha='center',               # horizontal alignment
#         va='bottom'               # vertical alignment
#     )
# sns.despine()
# plt.savefig(f'{fig_out}/scatter_{x}_{y}.png', dpi = 200, bbox_inches='tight')
# plt.show()


# x = 'upr_score'
# y='ISR_screen'
# plt.figure(figsize=(10,10))
# sns.scatterplot(summary_scores, 
#                 x = x, 
#                 y =y, hue = 'pert_type', alpha = 0.5)
# sns.scatterplot(summary_scores[summary_scores.pert_type=='Positive control'], 
#                 x = x,
#                 y =y, 
#                 hue = 'pert_type', alpha = 1, legend=None)
# plt.legend(bbox_to_anchor=(1,1), loc='upper left')

# labels=summary_scores[summary_scores.pert_type=='Positive control']
# for i, row in labels.iterrows():
#     plt.text(
#         row[x],  # x coordinate
#         row[y],          # y coordinate
#         row['Geneid'],            # text label
#         fontsize=9,
#         ha='center',               # horizontal alignment
#         va='bottom'               # vertical alignment
#     )
# labels=pd.concat([summary_scores.sort_values(x).reset_index()[0:show_genes],summary_scores.sort_values(x).reset_index()[-show_genes:]])
# for i, row in labels.iterrows():
#     plt.text(
#         row[x],  # x coordinate
#         row[y],          # y coordinate
#         row['gene_naming'],            # text label
#         fontsize=5,
#         ha='center',               # horizontal alignment
#         va='bottom'               # vertical alignment
#     )
# sns.despine()
# plt.savefig(f'{fig_out}/scatter_{x}_{y}.png', dpi = 200, bbox_inches='tight')
# plt.show()

# x = 'isr_score'
# y='BrefA_screen'
# plt.figure(figsize=(10,10))
# sns.scatterplot(summary_scores, 
#                 x = x, 
#                 y =y, hue = 'pert_type', alpha = 0.5)
# sns.scatterplot(summary_scores[summary_scores.pert_type=='Positive control'], 
#                 x = x,
#                 y =y, 
#                 hue = 'pert_type', alpha = 1, legend=None)
# plt.legend(bbox_to_anchor=(1,1), loc='upper left')

# labels=summary_scores[summary_scores.pert_type=='Positive control']
# for i, row in labels.iterrows():
#     plt.text(
#         row[x],  # x coordinate
#         row[y],          # y coordinate
#         row['Geneid'],            # text label
#         fontsize=9,
#         ha='center',              # horizontal alignment
#         va='bottom'               # vertical alignment
#     )
# labels=pd.concat([summary_scores.sort_values(x).reset_index()[0:show_genes],summary_scores.sort_values(x).reset_index()[-show_genes:]])
# for i, row in labels.iterrows():
#     plt.text(
#         row[x],  # x coordinate
#         row[y],          # y coordinate
#         row['gene_naming'],            # text label
#         fontsize=5,
#         ha='center',               # horizontal alignment
#         va='bottom'               # vertical alignment
#     )
# sns.despine()
# plt.savefig(f'{fig_out}/scatter_{x}_{y}.png', dpi = 200, bbox_inches='tight')
# plt.show()


# x='upr_score'
# y='BrefA_screen'
# plt.figure(figsize=(10,10))
# sns.scatterplot(summary_scores, 
#                 x = x, 
#                 y =y, hue = 'pert_type', alpha = 0.5)
# sns.scatterplot(summary_scores[summary_scores.pert_type=='Positive control'], 
#                 x = x,
#                 y =y, 
#                 hue = 'pert_type', alpha = 1, legend=None)
# plt.legend(bbox_to_anchor=(1,1), loc='upper left')

# labels=summary_scores[summary_scores.pert_type=='Positive control']
# for i, row in labels.iterrows():
#     plt.text(
#         row[x],  # x coordinate
#         row[y],          # y coordinate
#         row['Geneid'],            # text label
#         fontsize=9,
#         ha='center',               # horizontal alignment
#         va='bottom'               # vertical alignment
#     )
# labels=pd.concat([summary_scores.sort_values(x).reset_index()[0:show_genes],summary_scores.sort_values(x).reset_index()[-show_genes:]])
# for i, row in labels.iterrows():
#     plt.text(
#         row[x],  # x coordinate
#         row[y],          # y coordinate
#         row['gene_naming'],            # text label
#         fontsize=5,
#         ha='center',               # horizontal alignment
#         va='bottom'               # vertical alignment
#     )
# sns.despine()
# plt.savefig(f'{fig_out}/scatter_{x}_{y}.png', dpi = 200, bbox_inches='tight')
# plt.show()


####Load data
adata=ad.read_h5ad('/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis/all_lognorm_hvg_combat_pert_cc-regress.h5ad')
adata=adata[adata.obs.gene!='GFP11']

screen3_genes='/lab/weissman_imaging/yhc/validation_screen/python_outputs/MHC_IFN-top_IFN-bottom.csv'
df=pd.read_csv(screen3_genes)
df['gene_naming'] =[str(df.iloc[i].virus_name)+'\n'+str(p) if (p!=0) & ~('ORF_' in str(p)) else df.iloc[i].Geneid for i,p in enumerate(df['product'])]
adata.obs = adata.obs.merge(df[['Geneid', 'gene_naming']],on='Geneid', how='left')
adata_raw=adata.copy()


###test on interesting genes
suffix='_potential'

interesting_genes=pd.read_csv('./potential.csv', index_col=0).fillna(0)
interesting_genes['gene_naming'] =[interesting_genes.iloc[i].virus_name+' '+p if (p!=0) & ~('ORF_' in str(p)) else interesting_genes.iloc[i].Geneid for i,p in enumerate(interesting_genes['product'])]
interesting_genes

# save_file = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis/all_lognorm_hvg_combat_pert.h5ad"
# adata = ad.read_h5ad(save_file)
adata = adata[adata.obs.Geneid.isin(list(interesting_genes.Geneid)+['GFP',
                                                                    'TWIST-LONG_translated_YP_401720.3_1491', #BNLF2b
                                                                    'TWIST-LONG_translated_YP_001129513.1_1467' #'BNLF2b'])
                                                                   ])]
print(f'{adata.obs.Geneid.nunique()} perturbations in total')
print(f'{len(adata.obs)} cells in total')

adata.obs.loc[(adata.obs.Geneid=='GFP'),'pert_type'] = 'Negative control'
adata.obs.loc[(adata.obs.Geneid=='GFP'),'gene_naming'] = 'GFP'
adata.obs.loc[(adata.obs.Geneid.isin(['TWIST-LONG_translated_YP_401720.3_1491','TWIST-LONG_translated_YP_001129513.1_1467'])),'gene_naming'] = 'BNLF2b'


# # ## ii) PCA and UMAP
#plot pca on all genes
sc.tl.pca(adata)

##perform clustering and make umap 
sc.pp.neighbors(adata, n_neighbors=20)
sc.tl.umap(adata)
sc.tl.leiden(adata, n_iterations=2)

##plot
#plotting
sc.pl.umap(
    adata,
    color=["GEM_group", "day", "phase", "pert_type", "leiden", "gene_naming"],
    save=f"_regress-cc{suffix}.png",

)

distance = pt.tl.Distance(metric="euclidean", obsm_key="X_pca")
df = distance.pairwise(adata, groupby='gene_naming')
sns.clustermap(df, robust=True, figsize=(30, 30))
plt.savefig(f'{fig_out}/clustermap_euclidean{suffix}', dpi = 200)
plt.show()



# VI. Analyze Positive controls
suffix='_pos'
#also include anything i think would be interesting
#for cell cycle vpr, E7, and BNLF2b might be interesting, so include those too
pos_genes = list(adata.obs[adata.obs.pert_type=='Positive control']["Geneid"].unique())
pos = ['TWIST_translated_NP_056841.1_1072', #HIV2 Vpr
 'TWIST-LONG_translated_NP_057852.2_1344', #HIV1 Vpr
 'TWIST-LONG_translated_NP_041326.1_1486', #HPV16 E7
 'TWIST-LONG_translated_YP_401720.3_1491', #BNLF2b
 'TWIST-LONG_translated_YP_001129513.1_1467' #'BNLF2b'
]
pos_genes = pos_genes + pos                 

adata = adata_raw[adata_raw.obs.Geneid.isin(pos_genes)]
print(suffix)
print(f'{adata.obs.Geneid.nunique()} perturbations in total')
print(f'{len(adata.obs)} cells in total')

adata.obs.loc[(adata.obs.Geneid=='GFP'),'pert_type'] = 'Negative control'
adata.obs = adata.obs.merge(
    interesting_genes[['Geneid', 'gene_naming']],
    how='left',
    on='Geneid',
    sort=False).set_index(adata.obs.index)
adata.obs.loc[(adata.obs.Geneid=='GFP'),'gene_naming'] = 'GFP'

# # ## ii) PCA and UMAP
#plot pca on all genes
sc.tl.pca(adata)

##perform clustering and make umap 
sc.pp.neighbors(adata, n_neighbors=20)
sc.tl.umap(adata)
sc.tl.leiden(adata, n_iterations=2)

##plot
#plotting
sc.pl.umap(
    adata,
    color=["GEM_group", "day", "phase", "pert_type", "leiden", "gene_naming"],
    save=f"_regress-cc{suffix}.png",

)
distance = pt.tl.Distance(metric="euclidean", obsm_key="X_pca")
df = distance.pairwise(adata, groupby='gene_naming')
sns.clustermap(df, robust=True, figsize=(30, 30))
plt.savefig(f'{fig_out}/clustermap_euclidean{suffix}', dpi = 200)
plt.show()



# ###test on interesting genes
# suffix='_potential'

# interesting_genes=pd.read_csv('./potential.csv', index_col=0).fillna(0)
# interesting_genes['gene_naming'] =[interesting_genes.iloc[i].virus_name+' '+p if (p!=0) & ~('ORF_' in str(p)) else interesting_genes.iloc[i].Geneid for i,p in enumerate(interesting_genes['product'])]
# interesting_genes

# # save_file = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis/all_lognorm_hvg_combat_pert.h5ad"
# # adata = ad.read_h5ad(save_file)
# adata = adata[adata.obs.Geneid.isin(list(interesting_genes.Geneid)+['GFP',
#                                                                     'TWIST-LONG_translated_YP_401720.3_1491', #BNLF2b
#                                                                     'TWIST-LONG_translated_YP_001129513.1_1467' #'BNLF2b'])
#                                                                    ])]
# print(f'{adata.obs.Geneid.nunique()} perturbations in total')
# print(f'{len(adata.obs)} cells in total')

# adata.obs.loc[(adata.obs.Geneid=='GFP'),'pert_type'] = 'Negative control'
# adata.obs = adata.obs.merge(
#     interesting_genes[['Geneid', 'gene_naming']],
#     how='left',
#     on='Geneid',
#     sort=False).set_index(adata.obs.index)
# adata.obs.loc[(adata.obs.Geneid=='GFP'),'gene_naming'] = 'GFP'
# adata.obs.loc[(adata.obs.Geneid.isin(['TWIST-LONG_translated_YP_401720.3_1491','TWIST-LONG_translated_YP_001129513.1_1467'])),'gene_naming'] = 'BNLF2b'


# #plot cell cycle
# cc_genes_in_data = [x for x in cell_cycle_genes if x in adata.var_names]
# cc_adata=adata[:, cc_genes_in_data]
# cc_adata.obs['gene']=cc_adata.obs['gene_naming']
# cc_adata.obs['Geneid']=cc_adata.obs['gene_naming']

# adata_new=adata.copy()
# adata_new.obs['gene']=adata_new.obs['gene_naming']
# adata_new.obs['Geneid']=adata_new.obs['gene_naming']

# adata_new, cc_adata=examine_cell_cycle(adata_new, cc_adata, cc_genes_in_data, cell_lim=0)
# plt.savefig(f'./{fig_out}/cc_score{suffix}.png', bbox_inches = 'tight', dpi = 300)

# # # ## ii) PCA and UMAP

# # #plot pca on all genes
# sc.tl.pca(adata)

# ##perform clustering and make umap 
# sc.pp.neighbors(adata, metric='cosine')
# sc.tl.umap(adata)
# sc.tl.leiden(adata, n_iterations=2)
# sc.pl.umap(
#     adata,
#     color=["GEM_group", "day", "phase", "pert_type", "leiden"],
#     save=f"{suffix}.png",

# )

# #perform pseudobulk
# ps = pt.tl.CentroidSpace()
# psadata = ps.compute(adata, target_col="gene_naming", embedding_key="X_umap")
# sc.pl.umap(
#     psadata, 
#     color=["gene_naming"], 
#     legend_loc="on data",
#     legend_fontsize='xx-small',
#     save=f"_pseudobulk-name{suffix}.png",
# )
# ps = pt.tl.CentroidSpace()
# psadata = ps.compute(adata, target_col="gene", embedding_key="X_umap")
# sc.pl.umap(
#     psadata, 
#     color=["gene"], 
#     legend_loc="on data",
#     legend_fontsize='xx-small',
#     save=f"_pseudobulk-id{suffix}.png",
# )



# #try regressing out cell cycle
# sc.pp.regress_out(adata, ['S_score', 'G2M_score'])

# # # Optional: Rescale the data after regression
# # # This can be useful as regress_out can shift the mean expression of genes to 0
# sc.pp.scale(adata)

# #plot pca on all genes
# sc.tl.pca(adata)

# ##perform clustering and make umap 
# sc.pp.neighbors(adata, n_neighbors=20)
# sc.tl.umap(adata)
# sc.tl.leiden(adata, n_iterations=2)


# ##plot
# #plotting
# sc.pl.pca(
#     adata,
#     color=["GEM_group", "day", "phase", "pert_type", "leiden"],
#     save=f"_regress-cc{suffix}.png",  # will save in ./figs/ if set in sc.settings.figdir

# )

# sc.pl.umap(
#     adata,
#     color=["GEM_group", "day", "phase", "pert_type", "leiden"],
#     save=f"_regress-cc{suffix}.png",

# )


# #perform pseudobulk
# ps = pt.tl.CentroidSpace()
# psadata = ps.compute(adata, target_col="gene_naming", embedding_key="X_umap")
# sc.pl.umap(
#     psadata, 
#     color=["gene_naming"], 
#     legend_loc="on data",
#     legend_fontsize='xx-small',
#     save=f"_pseudobulk-name{suffix}.png",
# )
# ps = pt.tl.CentroidSpace()
# psadata = ps.compute(adata, target_col="gene", embedding_key="X_umap")
# sc.pl.umap(
#     psadata, 
#     color=["gene_naming"], 
#     legend_loc="on data",
#     legend_fontsize='xx-small',
#     save=f"_pseudobulk-id{suffix}.png",
# )
