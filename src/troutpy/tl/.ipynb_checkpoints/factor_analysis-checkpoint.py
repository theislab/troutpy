import warnings
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from spatialdata import SpatialData
import spatialdata


def latent_factor(
    sdata: spatialdata.SpatialData,
    method: str = "NMF",
    layer: str = "segmentation_free_table",
    n_components: int = 20,
    copy: bool | None = None,
    random_state=None,
    drvi_model_path: str = None,
    **kwargs,
):
    """
    Applies latent factor identification (NMF, LDA, or DRVI) to reduce dimensionality of gene expression data.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object with the specified layer containing AnnData.
    method : str
        One of "NMF", "LDA", or "DRVI".
    layer : str
        The AnnData layer in SpatialData to operate on.
    n_components : int
        Number of latent dimensions (ignored if DRVI model is loaded).
    copy : bool
        If True, return modified SpatialData. If False, operate in-place.
    random_state : int or None
        Random seed.
    drvi_model_path : str
        Path to a pretrained DRVI model.
    kwargs : dict
        Additional parameters for DRVI:
        - encoder_dims : list[int]
        - decoder_dims : list[int]
        - n_epochs : int
        - kl_warmup : int
        - is_count_data : bool
        - early_stopping : bool
        - accelerator : str
        - devices : int

    Returns
    -------
    sdata : SpatialData or None
        Modified SpatialData object or None if copy=False.
    """
    adata = sdata[layer]
    counts = adata.X.copy()

    if method == "NMF":
        model = NMF(n_components=n_components, init="random", random_state=random_state)
        cell_loadings = model.fit_transform(counts, **kwargs)
        gene_loadings = model.components_

    elif method == "LDA":
        lda = LatentDirichletAllocation(n_components=n_components, random_state=random_state)
        lda.fit(counts, **kwargs)
        cell_loadings = lda.transform(counts)
        gene_loadings = lda.components_

    elif method == "DRVI":
        try:
            from drvi.model import DRVI
            from drvi.utils.tools import (
                calculate_differential_vars,
                set_latent_dimension_stats,
                traverse_latent,
            )
        except ImportError as err:
            raise ImportError("The 'drvi' package is required for method='DRVI'.Please install it with: pip install drvi") from err

        # DRVI-specific parameters
        encoder_dims = kwargs.pop("encoder_dims", [128, 128])
        decoder_dims = kwargs.pop("decoder_dims", [128, 128])
        n_epochs = kwargs.pop("n_epochs", 400)
        kl_warmup = kwargs.pop("kl_warmup", n_epochs)
        is_count_data = kwargs.pop("is_count_data", True)
        early_stopping = kwargs.pop("early_stopping", False)
        accelerator = kwargs.pop("accelerator", None)
        devices = kwargs.pop("devices", None)

        adata.layers["counts"] = adata.X
        DRVI.setup_anndata(adata, layer="counts", is_count_data=is_count_data)

        if drvi_model_path:
            model = DRVI.load(drvi_model_path, adata)
        else:
            model = DRVI(
                adata,
                n_latent=n_components,
                encoder_dims=encoder_dims,
                decoder_dims=decoder_dims,
            )

            train_args = {
                "max_epochs": n_epochs,
                "early_stopping": early_stopping,
                "plan_kwargs": {"n_epochs_kl_warmup": kl_warmup},
            }
            if accelerator:
                train_args["accelerator"] = accelerator
            if devices:
                train_args["devices"] = devices

            model.train(**train_args)

        # Get latent representation and analyze gene effects
        latent = model.get_latent_representation()
        embed = ad.AnnData(X=latent, obs=adata.obs.copy())
        set_latent_dimension_stats(model, embed)

        traverse_adata = traverse_latent(model, embed, n_samples=20, max_noise_std=0.0)
        calculate_differential_vars(traverse_adata)

        cell_loadings = latent
        gene_loadings_pos = traverse_adata.varm["combined_score_traverse_effect_pos"]
        gene_loadings_neg = traverse_adata.varm["combined_score_traverse_effect_neg"]
        gene_loadings = combine_loadings_arrays(gene_loadings_pos, gene_loadings_neg)

    else:
        raise ValueError(f"Unsupported method: {method}. Choose from ['NMF', 'LDA', 'DRVI'].")

    # Store results
    adata.obsm["cell_loadings"] = cell_loadings
    if method == "DRVI":
        adata.varm["gene_loadings_positive"] = gene_loadings_pos
        adata.varm["gene_loadings_negative"] = gene_loadings_neg
    adata.varm["gene_loadings"] = gene_loadings

    sdata[layer] = adata
    return sdata if copy else None

def combine_loadings_arrays(gene_loadings_pos: np.ndarray, gene_loadings_neg: np.ndarray) -> np.ndarray:
    """Combines positive and negative gene loading arrays from DRVI analysis"""
    if gene_loadings_pos.shape != gene_loadings_neg.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Boolean masks
    pos_nonzero = gene_loadings_pos != 0
    neg_nonzero = gene_loadings_neg != 0
    conflict_mask = pos_nonzero & neg_nonzero

    # Raise warning if conflicts found
    if np.any(conflict_mask):
        conflict_indices = np.argwhere(conflict_mask)
        warnings.warn(f"Conflicts found at {len(conflict_indices)} locations. "
        f"Example: {conflict_indices[:5].tolist()}",stacklevel=2)

    # Initialize combined output
    combined = np.zeros_like(gene_loadings_pos)
    combined[pos_nonzero] = gene_loadings_pos[pos_nonzero]
    combined[neg_nonzero] = -gene_loadings_neg[neg_nonzero]

    return combined



def factors_to_cells(
    sdata: SpatialData, extracellular_layer: str = "segmentation_free_table", cellular_layer: str = "table", copy: bool | None = None
) -> SpatialData:
    """
    Extracts extracellular RNA data and associated NMF factor loadings, intersects the gene annotations between the extracellular data and the cellular data, and applies the NMF factors to annotate the cellular data with exRNA-related factors.

    Parameters
    ----------
    sdata: spatialdata.SpatialData
        The AnnData object containing both extracellular and cellular data.
    layer_factors: str
        The key in `sdata` that contains the extracellular RNA data with NMF factors. Default is 'nmf_data'.
    copy: bool
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
