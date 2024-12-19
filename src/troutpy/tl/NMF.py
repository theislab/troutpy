import numpy as np
import pandas as pd
import sys 
import pandas as pd
import numpy as np
import scanpy as sc
import os
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def apply_nmf_to_adata(adata, n_components=20, subsample_percentage=1.0,save=False,output_path:str='',random_state=None):
    """
    Applies Non-Negative Matrix Factorization (NMF) to an AnnData object.

    This function performs NMF on the expression matrix (`adata.X`) to extract 
    a reduced number of latent factors that describe the gene expression profiles 
    of cells. The number of factors is specified by `n_components`. Optionally, 
    the data can be subsampled before applying NMF.
    """
    
    # Extract the cell count matrix (X) from AnnData object
    # Assuming that adata.X contains the raw counts for cells
    sc.pp.subsample(adata, subsample_percentage)
    counts = adata.X.copy()
    
    # Perform NMF with 20 factors
    nmf_model = NMF(n_components=n_components, init='random', random_state=42)
    W = nmf_model.fit_transform(counts)  # Cell factors
    H = nmf_model.components_  # Gene loadings
    
    # Add NMF results to the AnnData object
    adata.obsm['W_nmf'] = W  # Add the cell factors to the AnnData object
    adata.uns['H_nmf'] = H 
    if save:
        H=pd.DataFrame(adata.uns['H_nmf'],columns=adata.var.index)
        H.to_parquet(os.path.join(output_path,'factor_loadings_H_per_gene.parquet'))
        W=pd.DataFrame(adata.obsm['W_nmf'],index=adata.obs.index)
        W.to_parquet(os.path.join(output_path,'factor_scores_W_per_cell.parquet')) 
    return adata

def nmf(
    sdata, layer='extracellular_transcripts_enriched', 
    feature_key='feature_name', bin_key='bin_id', 
    density_table_key='segmentation_free_table', 
    n_components=20, subsample_percentage=0.1,
    random_state=None,all=False):

    """
    Applies Non-negative Matrix Factorization (NMF) on filtered data based on feature_name and bin_id.

    Parameters:
    ----------
    sdata : spatial data object
        Input spatial data containing transcript and bin data.
  
    layer : str, optional
        Layer name of the data that contains extracellular transcripts (default: 'extracellular_transcripts_enriched').
    
    feature_key : str, optional
        Column name for the transcript feature (default: 'feature_name').
    
    bin_key : str, optional
        Column name for bin IDs (default: 'bin_id').
    
    density_table_key : str, optional
        Key to retrieve the density table from sdata (default: 'segmentation_free_table').
    
    n_components : int, optional
        Number of components for NMF (default: 20).
   
    subsample_percentage : float, optional
        Percentage of data to use for NMF (default: 0.1).
   
    random_state : int, optional
        Random state for NMF initialization for reproducibility (default: None).

    Returns:
    -------
    sdata : Updated spatial data object with NMF components stored.
    """
    if all==False:
    # Extract the DataFrame with feature_name and bin_id
         df = sdata.points[layer][[feature_key, bin_key]].compute()
         # Filter the density table to include only the relevant bin_ids and feature_names
         filtered_bin_ids = df[bin_key].astype(int).astype(str).unique()
         filtered_feature_name_ids = df[feature_key].astype(str).unique()
         # Filter adata_density to only include the bins and features present in df
         adata_density_raw = sdata[density_table_key]
         adata_density = adata_density_raw[adata_density_raw.obs.index.astype(str).isin(filtered_bin_ids),:]
         adata_density = adata_density[:, adata_density.var.index.astype(str).isin(filtered_feature_name_ids)]
         # Retrieve the segmentation-free density table
    else:
        adata_density = sdata[density_table_key]
    # Apply NMF to filtered data
    adata_nmf = apply_nmf_to_adata(
        adata_density, 
        n_components=n_components, 
        subsample_percentage=subsample_percentage, 
        random_state=random_state
    ) # This function adds adata.obsm['W_nmf'] and adata.uns['H_nmf']
    
    # Store the NMF results in the spatial data
    sdata['nmf_data'] = adata_nmf
    
    return sdata

def apply_exrna_factors_to_cells(sdata, layer_factors='nmf_data'):
    """Applies extracellular RNA (exRNA) factor loadings to cellular annotation data based on NMF factors.

    This function extracts extracellular RNA data and associated NMF factor loadings, intersects the gene annotations between the extracellular data and the cellular data, and applies the NMF factors to annotate the cellular data with exRNA-related factors.

    Parameters:
    sdata (AnnData): The AnnData object containing both extracellular and cellular data.
    layer_factors (str, optional): The key in `sdata` that contains the extracellular RNA data with NMF factors. Default is 'nmf_data'.
    
    Returns:
    AnnData: The updated `sdata` object with annotated cellular data that includes the applied exRNA factors as new columns.
    
    Notes:
    The function assumes that the extracellular RNA data is stored in `sdata[layer_factors]` and that the NMF factor loadings are stored in the `uns` attribute of the extracellular dataset as 'H_nmf'. The factor scores are added to the `obs` attribute of the cellular data.
    """
    
    # Extract extracellular data and cellular annotations
    adata_extracellular_with_nmf = sdata[layer_factors]
    adata_annotated_cellular = sdata['table']
    
    # Retrieve NMF factor loadings (H matrix) from extracellular data
    H = adata_extracellular_with_nmf.uns['H_nmf']
    
    # Get gene names from both datasets
    genes_spots2region = adata_extracellular_with_nmf.var_names
    genes_annotated = adata_annotated_cellular.var_names
    
    # Get the intersection of genes between the extracellular and cellular datasets
    common_genes = genes_annotated.intersection(genes_spots2region)
    
    # Filter both datasets to retain only the common genes
    adata_annotated_cellular = adata_annotated_cellular[:, common_genes]
    H_filtered = H[:, np.isin(genes_spots2region, common_genes)]  # Filtered NMF factor loadings for common genes
    
    # Apply NMF factors to the annotated cellular dataset
    # Calculate the W matrix by multiplying the cellular data (X) with the filtered NMF loadings (H)
    W_annotated = adata_annotated_cellular.X @ H_filtered.T
    
    # Store the factors in the 'obsm' attribute of the AnnData object
    adata_annotated_cellular.obsm['factors'] = pd.DataFrame(W_annotated, index=adata_annotated_cellular.obs.index)
    
    # Add each factor as a new column in the 'obs' attribute of the cellular dataset
    for factor in range(W_annotated.shape[1]):
        adata_annotated_cellular.obs[f'NMF_factor_{factor + 1}'] = W_annotated[:, factor]
    
    # Update the 'table' in the sdata object with the annotated cellular data
    sdata['table'] = adata_annotated_cellular
    
    return sdata
