# --------------------------------------------------------------------------------------
# MDE Implementation for Perturbation Analysis (Based on Paper Methodology)
# This script outlines the data preparation and the exact pymde calls required
# to replicate the "Minimum Distortion Embedding of strong perturbations."
# --------------------------------------------------------------------------------------

# NOTE: This script requires the 'pymde' library (pip install pymde)
# and 'scikit-learn'.

import numpy as np
import pandas as pd
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pymde
import scanpy as sc
import anndata as ad
import torch

sns.set_style('white')
sns.set_palette('deep')

# -----------------------------
# Paths and directories
# -----------------------------
fig_out = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/figs"
python_out = "/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/python_outputs"
os.makedirs(fig_out, exist_ok=True)
os.makedirs(python_out, exist_ok=True)
sc.settings.figdir = fig_out


# --- Configuration Constants (From Paper) ---
N_NEIGHBORS = 7
REDUCTION_DIM_HIGH = 20
REDUCTION_DIM_LOW = 2
RANDOM_STATE=42

# Set the random seed for NumPy
np.random.seed(RANDOM_STATE)

# Set the random seed for PyTorch
torch.manual_seed(RANDOM_STATE)


# -----------------------------
# Load AnnData and orf info
# -----------------------------
print("--- I. Load AnnData and screen data ---")
savefile = f"/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/python_outputs/psadata_znorm_summary_pearson-clust.h5ad"
psadata = ad.read_h5ad(savefile)
psadata.obs['labels'] = [row.gene_naming if 'ositive' in row.pert_type else row.pert_type for i,row in psadata.obs.iterrows()]
psadata.obs['Cluster']=psadata.obs['Cluster_pearson']
print(psadata)

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
X_normalized = psadata.X

# ------------------------------------------------------------------------------
# Initialization using Spectral Embedding
# ------------------------------------------------------------------------------

print("\n--- 2. Computing Spectral Embedding Initialization (20D) ---")

# Spectral Embedding is used to provide a good initial guess for the MDE solver.
# This prevents the solver from getting stuck in poor local minima.
# Parameters are taken directly from gwps

se_initializer = SpectralEmbedding(
    n_components=REDUCTION_DIM_HIGH,
    affinity='nearest_neighbors',
    n_neighbors=N_NEIGHBORS,
    eigen_solver='arpack',
    random_state=RANDOM_STATE,
    n_jobs=-1 # Use all available cores
)

# Fit and transform the Z-normalized data
print("Running Spectral Embedding (Initialization)...")
X_init_20d = se_initializer.fit_transform(X_normalized)
print(f"Spectral Embedding initialization shape: {X_init_20d.shape}")

# ------------------------------------------------------------------------------
# 3. Run Minimum Distortion Embedding (MDE)
# ------------------------------------------------------------------------------

print("\n--- 3. Running Minimum Distortion Embedding (MDE) ---")

# Create the MDE object
mde = pymde.preserve_neighbors(
    X_init_20d,
    embedding_dim=REDUCTION_DIM_HIGH,
    n_neighbors=N_NEIGHBORS,
    repulsive_fraction=0.5,
    init="random",
    verbose=False
)

# Solve for the embedding
embedding_20d = mde.embed(verbose=False).numpy()
print(f"MDE embedding shape: {embedding_20d.shape}")

# Store the result in AnnData
psadata.obsm['X_mde_20d'] = embedding_20d

sc.pp.neighbors(psadata, n_neighbors=3, use_rep='X_mde_20d')
sc.tl.umap(psadata)
sc.pl.umap(psadata, 
           color=['Cluster_pearson','num_DEG', 'total_counts','growth_LFC', 'labels'], 
           frameon=False,
           save="_bulk_mde_20d.png")


# ------------------------------------------------------------------------------
# 4. Low-Dimensional MDE (2D) for Visualization 
# ------------------------------------------------------------------------------

print("\n--- 4. Running Low-Dimensional MDE (2D) ---")

# Create the MDE object
mde = pymde.preserve_neighbors(
    X_init_20d,
    embedding_dim=REDUCTION_DIM_LOW,
    n_neighbors=N_NEIGHBORS,
    repulsive_fraction=0.5,
    init="random",
    verbose=False
)

# Solve for the embedding
embedding_2d = mde.embed(verbose=False).numpy()
print(f"MDE embedding shape: {embedding_2d.shape}")

# Store the result in AnnData
psadata.obsm['X_mde'] = embedding_2d

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Red → white → blue diverging colormap centered at 0
growth_cmap = mcolors.LinearSegmentedColormap.from_list(
    "growth_cmap", ["blue", "white", "red"], N=50
)

sc.pl.embedding(
    psadata,
    basis="X_mde",
    color=["Cluster_pearson", 
           "num_DEG", 
           'pert_type' ],
    frameon=False,
    save="_bulk_mde_2d_basic.png"
)

sc.pl.embedding(
    psadata,
    basis="X_mde",
    color="growth_LFC",
    cmap=growth_cmap,
    vcenter=0,   # ensures white corresponds to 0
    frameon=False,
    save="_bulk_mde_2d_growth.png"
)


print("\n--- Plotting ---")
import matplotlib.pyplot as plt
from adjustText import adjust_text

def plot_mde_labels(df, 
                    color = 'lightgrey',
                    labels="", 
                    labelling = True,
                    s=20, alpha=0.6, linewidth=0,
                    legend_name = '', size = 10
                   ):
    """
    Plot MDE embedding with custom label and color rules.

    """

    # Extract coordinates
    x, y = df[:, 0], df[:, 1]
    
    # Metadata
    

    # --- Plot ---
    plt.scatter(x, y, c=color, s=s,alpha = alpha,linewidth=linewidth,label=legend_name)

    # Add text for non-'Perturbed' labels
    if labelling:
        labels = labels.astype(str)
        if len(labels)>0:
            texts=[]
            for i, lbl in enumerate(labels):
                texts.append(plt.text(
                    x[i], y[i], lbl, fontsize=size, color=color,
                    # ha="center", va="bottom"
                ))
            return texts

    
    
    
controls= psadata.obs[(psadata.obs.pert_type == 'Positive control')|psadata.obs.gene_naming.str.contains('shuff')].gene_naming.unique()
e7 = psadata.obs[psadata.obs.gene_naming.str.contains('E7')
               &psadata.obs.virus_name.str.contains('apillomavirus')].gene_naming.unique()
e7 = psadata.obs[psadata.obs.gene_naming.isin(e7)&(psadata.obs.Cluster>0)].gene_naming.unique()
e7_clust= psadata.obs[psadata.obs.gene_naming.isin(e7)&(psadata.obs.Cluster>0)].Cluster.unique()
nonclust=psadata.obs[psadata.obs.Cluster.isin(e7_clust)
                     &~psadata.obs.gene_naming.isin(e7)
                    &(psadata.obs.virus_name.str.contains('apillomavirus')==False)].gene_naming.unique()
bnlf2b = psadata.obs[psadata.obs.gene_naming.str.contains('BNLF2b')].gene_naming.unique()
plotting=list(controls)+list(e7) + list(bnlf2b) + list(nonclust)

plt.figure(figsize=(8,7))
all_text = []


#plot everything first
perturbed = psadata[psadata.obs.pert_type=='Perturbed']
plot_mde_labels(
    perturbed.obsm['X_mde'],
    labelling = False,
)

#plot everything in plotting
perturbed = psadata[psadata.obs.gene_naming.isin(plotting)]
plot_mde_labels(
    perturbed.obsm['X_mde'],
    color='darkgrey',
    labelling = False,
)


#plot negative controls
neg_control = psadata[psadata.obs.gene_naming.str.contains('shuff')]
t = plot_mde_labels(
    neg_control.obsm['X_mde'],
    color = 'steelblue',
    labelling = True,
    labels=neg_control.obs.gene_naming,
    legend_name = 'Negative control'
)
all_text+=t

#plot positive controls
pos_control = psadata[psadata.obs.pert_type.str.contains('ositive')]
t = plot_mde_labels(
    pos_control.obsm['X_mde'],
    color = 'black',
    labelling = True,
    labels=pos_control.obs.gene_naming,
    legend_name = 'Positive control',
    s=20
)
all_text+=t

#plot E7
e = psadata[psadata.obs.gene_naming.isin(e7)]
plot_mde_labels(
    e.obsm['X_mde'],
    labelling = False,
    color = 'seagreen',
    legend_name = 'HPV E7',
    alpha = 0.5,
    s=20
)
b = psadata[psadata.obs.gene_naming.str.contains('type 16 ')&psadata.obs.gene_naming.isin(e7)]
b.obs['labels'] = [g.replace(' | ',' \n ') for g in  b.obs.gene_naming]
t=plot_mde_labels(
    b.obsm['X_mde'],
    labelling = True,
    color = 'seagreen',
    labels=b.obs.labels,
    alpha = 1,
    s=20
)
all_text+=t


#plot bnlf2b
b = psadata[psadata.obs.gene_naming.str.contains('BNLF2b')]
b.obs['labels'] = [g.replace(' | ',' \n ') for g in  b.obs.gene_naming]
t=plot_mde_labels(
    b.obsm['X_mde'],
    labelling = True,
    color = 'firebrick',
    labels=b.obs.labels,
    alpha = 1,
    s=20
)
all_text+=t

#plot e7 clust
b = psadata[psadata.obs.gene_naming.str.contains('6/7')]
b.obs['labels'] = [g.replace(' | ',' \n ') for g in  b.obs.gene_naming]
t=plot_mde_labels(
    b.obsm['X_mde'],
    labelling = True,
    color = 'indigo',
    labels=b.obs.labels,
    alpha = 1,
    s=20
)
all_text+=t
                                                  
adjust_text(all_text, arrowprops=dict(arrowstyle="-", color='grey', lw=1))



plt.legend(bbox_to_anchor=(1,1))
plt.axis("off")
plt.tight_layout()


# Save
savepath = f"/lab/weissman_scratch/yhc/20250815_perturb_analysis/postprocess_analysis_new/figs/growth_MDE_plot"
plt.savefig(savepath, dpi=300)
plt.show()
print(f"✅ Saved labeled MDE plot to {savepath}")



