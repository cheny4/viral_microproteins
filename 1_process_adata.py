#!/usr/bin/env python
# coding: utf-8

# Import standard libraries
import os, warnings, pickle, gc
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import scipy.sparse as sp
from sklearn.utils.sparsefuncs import mean_variance_axis
import seaborn as sns
import pertpy as pt


warnings.filterwarnings('ignore')
sns.set_style('white')
sns.set_palette('deep')


warnings.filterwarnings('ignore')
sns.set_style('white')
sns.set_palette('deep')

# === I. Define directories ===
fig_out = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/figs"
python_out = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/python_outputs"
os.makedirs(fig_out, exist_ok=True)
os.makedirs(python_out, exist_ok=True)
sc.settings.figdir = fig_out

# === II. Import gene and cell metadata ===
orf_info = pd.read_csv(
    '/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/orf_files/perturb-orf_metadata.csv',
    index_col=0
)

cellcycle = '/lab/solexa_weissman/yhc/viral_ORF/20241209_elledge_150_new-analysis1/postprocess_analysis/regev_lab_cell_cycle_genes.txt'
cell_cycle_genes = [x.strip() for x in open(cellcycle)]
s_genes, g2m_genes = cell_cycle_genes[:43], cell_cycle_genes[43:]

singlets_filtered = pd.read_csv(
    '/lab/weissman_scratch/yhc/20250815_perturb_analysis/scripts/tests/singlets.csv'
)

singlets_filtered = singlets_filtered.merge(orf_info, on='Geneid', how='left')
singlet_map = (
    singlets_filtered[singlets_filtered.Geneid != 'GFP11']
    .set_index('cell_barcode')
)

# === III. Load and process individual .pkl datasets ===
gex_dir = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/scripts/tests"
hvgs_all = list(s_genes)+list(g2m_genes)
all_adata = []

print("--- I. Merging Anndata objects ---")
for root, dirs, files in os.walk(gex_dir):
    for file in files:
        if not file.endswith('.pkl'):
            continue

        file_path = os.path.join(root, file)
        print(f'Processing {file_path}')
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # --- Merge GEX data ---
        adata_list = []
        for key, subdata in data.items():
            a = subdata['GEX'][0]
            a.obs['sample'] = key
            adata_list.append(a)
        adata = adata_list[0].concatenate(adata_list[1:], index_unique=None) if len(adata_list) > 1 else adata_list[0]

        # Subset to singlets and add metadata
        adata = adata[adata.obs_names.isin(singlet_map.index)]
        adata.obs['cell_barcode'] = list(adata.obs.index)
        adata.obs = adata.obs.merge(singlets_filtered, on='cell_barcode',how='left')
        adata.obs=adata.obs.set_index('cell_barcode')

        # --- QC ---
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
        adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

        # --- Log normalization ---
        print("--- Normalizing ---")
        normed = sc.pp.normalize_total(adata, target_sum=10000, inplace=False)
        adata.layers["log1p_norm"] = sc.pp.log1p(normed["X"], copy=True)

        # --- Z normalization (vs negative controls) ---
        ctrl = adata[adata.obs.pert_type == 'Negative control']
        if sp.issparse(ctrl.layers["log1p_norm"]):
            ctrl_means, ctrl_vars = mean_variance_axis(ctrl.layers["log1p_norm"], axis=0)
            ctrl_std = np.sqrt(ctrl_vars)
        else:
            ctrl_means = np.asarray(ctrl.layers["log1p_norm"].mean(axis=0)).ravel()
            ctrl_std = np.asarray(ctrl.layers["log1p_norm"].std(axis=0)).ravel()
        ctrl_std[ctrl_std == 0] = 1.0
        adata.layers["z_norm"] = (adata.layers["log1p_norm"] - ctrl_means) / ctrl_std
        # Clip extreme z-scores
        adata.layers['z_norm'] = np.clip(adata.layers['z_norm'], a_min=None, a_max=10)

        
        # --- Cell cycle scoring ---
        print("--- Scoring cell cycle ---")
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes, layer="log1p_norm")
        all_adata.append(adata)

        # --- HVGs ---
        sc.pp.highly_variable_genes(adata, layer="log1p_norm", flavor="seurat_v3",
                                    n_top_genes=3000, batch_key="Batch")
        hvgs = adata.var_names[adata.var['highly_variable']].tolist()
        hvgs += s_genes + g2m_genes
        hvgs = [g for g in hvgs if g in adata.var_names]
        hvgs = list(dict.fromkeys(hvgs))  # preserves order, removes duplicates
        hvgs_all += hvgs
        
#         # --- Regress out + PCA for both normalization types ---
#         for layer in ["log1p_norm", "z_norm"]:
#             print(f"--- Regressing + PCA ({layer}) ---")
#             sc.pp.regress_out(adata_hvgs, keys=["S_score", "G2M_score"], layer=layer)
#             sc.pp.scale(adata_hvgs, max_value=10, layer=layer)
#             sc.tl.pca(adata_hvgs, svd_solver="arpack", n_comps=50, layer=layer)

#             adata.obsm[f"X_pca_{layer}"] = adata_hvgs.obsm["X_pca"].copy()
#         all_adata_regress.append(adata)

# === IV. Concatenate all batches ===
adata = all_adata[0].concatenate(all_adata[1:], index_unique=None) if len(all_adata) > 1 else all_adata[0]
savefile = f"{python_out}/adata_znorm.h5ad"
adata.write_h5ad(savefile)
print(f"Saved normalized dataset to {savefile} ({adata.shape})")

# === V. Subset to HVGs and save ===
hvgs_all = list(set(hvgs_all))+list(adata.var_names[adata.var_names.str.startswith("MT-")])
adata = adata[:, adata.var_names.isin(hvgs_all)]
savefile = f"{python_out}/adata_znorm_hvg.h5ad"
adata.write_h5ad(savefile)
print(f"Processing complete. Saved to {savefile} ({adata.shape})")



# # === V. Batch correction with Harmony ===
# print("--- Batch correction with Harmony ---")
# for layer in ["log1p_norm", "z_norm"]:
#     basis = f"X_pca_{layer}"
#     adj_basis = f"{basis}_harmony"
#     sce.pp.harmony_integrate(adata, 'batch', basis=basis, adjusted_basis=adj_basis)

#     # before Harmony
#     sc.pp.neighbors(adata, use_rep=basis, n_neighbors=20, method='umap', metric='correlation', n_pcs=20)
#     sc.tl.umap(adata)
#     adata.obsm[f"X_umap_no-harmony_{layer}"] = adata.obsm["X_umap"].copy()
#     sc.pl.umap(adata, color=["GEM_group", 'day', "phase", "pert_type"], save=f"_no-harmony_{layer}.png")

#     # after Harmony
#     sc.pp.neighbors(adata, use_rep=adj_basis, n_neighbors=20, method='umap', metric='correlation', n_pcs=20)
#     sc.tl.umap(adata)
#     adata.obsm[f"X_umap_harmony_{layer}"] = adata.obsm["X_umap"].copy()
#     sc.pl.umap(adata, color=["GEM_group", 'day', "phase", "pert_type"], save=f"_harmony_{layer}.png")


