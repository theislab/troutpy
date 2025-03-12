import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from spatialdata import SpatialData


def latent_factor(
    sdata: SpatialData,
    method="NMF",
    layer: str = "segmentation_free_table",
    n_components: int = 20,
    copy: bool | None = None,
    random_state=None,
    **kwargs,
):
    """
    Applies latent_factor_identification to an AnnData object to reduce the dimensionality of gene expression data.

    Parameters
    ----------
    adata
        The AnnData object containing the gene expression matrix (`adata.X`) along with cell and gene annotations.
    n_components
        The number of components (latent factors) to extract from the NMF model.
    save
        If True, the factor loadings (`H`) and factor scores (`W`) will be saved as Parquet files to the specified output path.
    random_state (int, optional)
        The random seed used for initializing the NMF model. If None, the random seed is not fixed.
    copy
        Wether to save the spatialdata object as a new object
    method


    Returns
    -------
    sdata
        The input spatialdata object with the NMF results added:`spatialdata[layer].obsm['W_nmf']` contains the cell factors (factor scores for each cell) and `spatialdata[layer].uns['H_nmf']` contains the gene loadings (factor loadings for each gene).

    Notes
    -----
    - The NMF algorithm is initialized using a random method for factorization (`init='random'`).
    - The function assumes that the expression matrix (`adata.X`) contains raw gene expression counts.
    """
    # Extract the cell count matrix (X) from AnnData object
    adata = sdata[layer]
    counts = adata.X.copy()

    # Perform NMF with the specified number of components
    if method == "NMF":
        nmf_model = NMF(n_components=n_components, init="random", random_state=random_state)
        cell_loadings = nmf_model.fit_transform(counts, **kwargs)  # Cell factors
        gene_loadings = nmf_model.components_  # Gene loadings
    elif method == "LDA":
        lda = LatentDirichletAllocation(n_components=10, random_state=42)
        lda.fit(counts, **kwargs)  # expression_matrix: cells × genes or spots × genes
        cell_loadings = lda.transform(counts)
        gene_loadings = lda.components_  # Gene loadings

    # Add NMF results to the AnnData object
    adata.obsm["cell_loadings"] = cell_loadings  # Add the cell factors to the AnnData object
    adata.varm["gene_loadings"] = gene_loadings.transpose()

    # Optionally save the factor loadings and scores to disk
    sdata[layer] = adata

    return sdata if copy else None


def apply_exrna_factors_to_cells(
    sdata: SpatialData, extracellular_layer: str = "segmentation_free_table", cellular_layer: str = "table", copy: bool | None = None
) -> SpatialData:
    """
    Extracts extracellular RNA data and associated NMF factor loadings, intersects the gene annotations between the extracellular data and the cellular data, and applies the NMF factors to annotate the cellular data with exRNA-related factors.

    Parameters
    ----------
    sdata
        The AnnData object containing both extracellular and cellular data.
    layer_factors
        The key in `sdata` that contains the extracellular RNA data with NMF factors. Default is 'nmf_data'.
    copy
        Wether to save the `sdata` object in a separate object

    Returns
    -------
    sdata
        The updated `sdata` object with annotated cellular data that includes the applied exRNA factors as new columns.

    Notes
    -----
    The function assumes that the extracellular RNA data is stored in `sdata[layer_factors]` and that the NMF factor loadings are stored in the `uns` attribute of the extracellular dataset as 'H_nmf'. The factor scores are added to the `obs` attribute of the cellular data.
    """
    # Extract extracellular data and cellular annotations
    adata_extracellular_with_factors = sdata[extracellular_layer]
    adata_annotated_cellular = sdata[cellular_layer]

    # Retrieve NMF factor loadings (H matrix) from extracellular data
    H = adata_extracellular_with_factors.varm["gene_loadings"].transpose()  # type: ignore

    # Get gene names from both datasets
    genes_selected = adata_extracellular_with_factors.var_names
    genes_annotated = adata_annotated_cellular.var_names

    # Get the intersection of genes between the extracellular and cellular datasets
    common_genes = genes_annotated.intersection(genes_selected)

    # Filter both datasets to retain only the common genes
    adata_annotated_cellular = adata_annotated_cellular[:, common_genes]
    H_filtered = H[:, np.isin(genes_selected, common_genes)]  # Filtered NMF factor loadings for common genes

    # Apply NMF factors to the annotated cellular dataset
    # Calculate the W matrix by multiplying the cellular data (X) with the filtered NMF loadings (H)
    W_annotated = adata_annotated_cellular.X @ H_filtered.T

    # Store the factors in the 'obsm' attribute of the AnnData object
    adata_annotated_cellular.obsm["factors_cell_loadings"] = pd.DataFrame(W_annotated, index=adata_annotated_cellular.obs.index)

    #### Not include factors as obs.
    # Add each factor as a new column in the 'obs' attribute of the cellular dataset
    # for factor in range(W_annotated.shape[1]):
    #    adata_annotated_cellular.obs[f"Factor_{factor + 1}"] = W_annotated[:, factor]

    # Update the 'table' in the sdata object with the annotated cellular data
    sdata[cellular_layer] = adata_annotated_cellular

    return sdata if copy else None
