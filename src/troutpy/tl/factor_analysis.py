import warnings

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from spatialdata import SpatialData


def latent_factor(
    sdata: SpatialData,
    method: str = "NMF",
    layer: str = "segmentation_free_table",
    n_components: int = 20,
    copy: bool | None = None,
    random_state=None,
    drvi_model_path: str | None = None,
    **kwargs,
):
    """Apply latent factor identification (NMF, LDA, or DRVI) to reduce the dimensionality of gene expression data.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object with the specified layer containing AnnData.
    method : str, optional
        One of ``"NMF"``, ``"LDA"``, or ``"DRVI"``. Defaults to ``"NMF"``.
    layer : str, optional
        The AnnData layer in ``sdata`` to operate on. Defaults to ``"segmentation_free_table"``.
    n_components : int, optional
        Number of latent dimensions (ignored if a pretrained DRVI model is loaded). Defaults to ``20``.
    copy : bool, optional
        If truthy, return the modified SpatialData object; otherwise modify ``sdata`` in place
        and return ``None``.
    random_state : int, optional
        Random seed for ``"NMF"`` and ``"LDA"``.
    drvi_model_path : str, optional
        Path to a pretrained DRVI model to load instead of training a new one.
    **kwargs
        Additional parameters. For ``"NMF"``/``"LDA"`` these are forwarded to
        ``model.fit_transform``/``model.fit``. For ``"DRVI"`` the following are popped before
        constructing/training the model: ``encoder_dims`` (list of int), ``decoder_dims``
        (list of int), ``n_epochs`` (int), ``kl_warmup`` (int), ``is_count_data`` (bool),
        ``early_stopping`` (bool), ``accelerator`` (str), ``devices`` (int).

    Returns
    -------
    spatialdata.SpatialData or None
        Modified SpatialData object if ``copy`` is truthy; otherwise ``None``.
    """
    adata = sdata[layer]
    counts = adata.X.copy()

    if method == "NMF":
        model = NMF(n_components=n_components, init="random", random_state=random_state)
        cell_loadings = model.fit_transform(counts, **kwargs)
        gene_loadings = model.components_.T

    elif method == "LDA":
        lda = LatentDirichletAllocation(n_components=n_components, random_state=random_state)
        lda.fit(counts, **kwargs)
        cell_loadings = lda.transform(counts)
        gene_loadings = lda.components_.T

    elif method == "DRVI":
        try:
            from drvi.model import DRVI
            from drvi.utils.tools import (
                calculate_differential_vars,
                set_latent_dimension_stats,
                traverse_latent,
            )
        except ImportError as err:
            raise ImportError("The 'drvi' package is required for method='DRVI'. Please install it with: pip install troutpy[factor-analysis]") from err

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
    """Combine positive and negative gene loading arrays from DRVI into a single signed matrix.

    Elements present only in ``gene_loadings_pos`` are kept as positive values;
    elements present only in ``gene_loadings_neg`` are stored as negative values.
    Positions non-zero in both arrays trigger a warning and the negative loading
    takes precedence.

    Parameters
    ----------
    gene_loadings_pos : numpy.ndarray
        2-D array of non-negative gene loading magnitudes for dimensions where genes
        are positively associated (output of DRVI ``traverse_latent``).
    gene_loadings_neg : numpy.ndarray
        2-D array of non-negative gene loading magnitudes for dimensions where genes
        are negatively associated; must have the same shape as ``gene_loadings_pos``.

    Returns
    -------
    combined : numpy.ndarray
        Signed loading array of the same shape, where positive-only entries are
        positive and negative-only entries are negated.

    Raises
    ------
    ValueError
        If ``gene_loadings_pos`` and ``gene_loadings_neg`` have different shapes.
    """
    if gene_loadings_pos.shape != gene_loadings_neg.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Boolean masks
    pos_nonzero = gene_loadings_pos != 0
    neg_nonzero = gene_loadings_neg != 0
    conflict_mask = pos_nonzero & neg_nonzero

    # Raise warning if conflicts found
    if np.any(conflict_mask):
        conflict_indices = np.argwhere(conflict_mask)
        warnings.warn(f"Conflicts found at {len(conflict_indices)} locations. Example: {conflict_indices[:5].tolist()}", stacklevel=2)

    # Initialize combined output
    combined = np.zeros_like(gene_loadings_pos)
    combined[pos_nonzero] = gene_loadings_pos[pos_nonzero]
    combined[neg_nonzero] = -gene_loadings_neg[neg_nonzero]

    return combined



def factors_to_cells(
    sdata: SpatialData, extracellular_layer: str = "segmentation_free_table", cellular_layer: str = "table", copy: bool | None = None
) -> SpatialData:
    """Project extracellular-RNA factor loadings onto the cellular table for shared genes.

    Intersects the gene annotations between the extracellular and cellular tables, and
    multiplies the cellular expression matrix by the (gene-subset) factor loadings to
    obtain per-cell factor scores.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object containing both the extracellular and cellular tables.
    extracellular_layer : str, optional
        Key in ``sdata`` for the extracellular table with NMF/LDA/DRVI gene loadings in
        ``.varm["gene_loadings"]``. Defaults to ``"segmentation_free_table"``.
    cellular_layer : str, optional
        Key in ``sdata`` for the cellular table to annotate. Defaults to ``"table"``.
    copy : bool, optional
        If truthy, return the modified SpatialData object; otherwise modify ``sdata`` in
        place and return ``None``.

    Returns
    -------
    spatialdata.SpatialData or None
        ``sdata`` with ``sdata[cellular_layer].obsm["factors_cell_loadings"]`` set to the
        per-cell factor scores (restricted to genes shared with ``extracellular_layer``) if
        ``copy`` is truthy; otherwise ``None``.
    """
    # Extract extracellular data and cellular annotations
    adata_extracellular_with_factors = sdata[extracellular_layer]
    adata_annotated_cellular = sdata[cellular_layer]

    # Retrieve NMF factor loadings (H matrix) from extracellular data
    H = adata_extracellular_with_factors.varm["gene_loadings"].transpose()

    genes_selected = adata_extracellular_with_factors.var_names
    genes_annotated = adata_annotated_cellular.var_names
    common_genes = genes_annotated.intersection(genes_selected)

    adata_annotated_cellular = adata_annotated_cellular[:, common_genes]
    H_filtered = H[:, np.isin(genes_selected, common_genes)]

    W_annotated = adata_annotated_cellular.X @ H_filtered.T
    adata_annotated_cellular.obsm["factors_cell_loadings"] = pd.DataFrame(W_annotated, index=adata_annotated_cellular.obs.index)

    sdata[cellular_layer] = adata_annotated_cellular

    return sdata if copy else None
