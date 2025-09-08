import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from spatialdata import SpatialData
from tqdm import tqdm

warnings.filterwarnings("ignore")


def create_xrna_metadata(sdata: SpatialData, layer: str = "transcripts", gene_key: str = "gene", copy: bool = False) -> SpatialData | None:
    """
    Creates a new table within the SpatialData object that contains a 'gene' column with the unique gene names extracted from the specified points layer.

    Parameters
    ----------
    sdata: spatialdata.SpatialData
        The SpatialData object to modify.
    layer: str
        The name of the layer in `sdata.points` from which to extract gene names.Default is 'transcripts'.
    gene_key: str
        The key in the `layer` dataframe that contains the gene names.Default is 'feature_name'.
    copy: bool
        If `True`, returns a copy of the `SpatialData` object with the new table added.

    Returns
    -------
    sdata: spatialdata.SpatialData
        If `copy` is `True`, returns a copy of the modified `SpatialData` object. Otherwise, returns `None`.

    """
    # Check if the specified points layer exists
    if layer not in sdata.points:
        raise ValueError(f"Points layer '{layer}' not found in sdata.points.")

    # Extract unique gene names from the specified points layer
    points_data = sdata.points[layer]
    if gene_key not in points_data.columns:
        raise ValueError(f"The specified points layer '{layer}' does not contain a '{gene_key}' column.")

    unique_genes = points_data[gene_key].compute().unique().astype(str)

    # Create a DataFrame for unique genes
    gene_metadata = pd.DataFrame(index=unique_genes)

    # Convert to AnnData and then to SpatialData table model
    exrna_adata = sc.AnnData(var=gene_metadata)
    metadata_table = sd.models.TableModel.parse(exrna_adata)

    # Add the new table to the SpatialData object
    sdata.tables["xrna_metadata"] = metadata_table

    print(f"Added 'xrna_metadata' table with {len(unique_genes)} unique genes to the SpatialData object.")

    # Return copy or modify in place
    return sdata if copy else None


# deprecated
def compute_source_cells(sdata: SpatialData, expression_threshold=1, gene_key="gene", layer="transcripts", copy=False):
    """
    Compute the source of extracellular RNA by linking detected extracellular transcripts to specific cell types in the spatial data.

    Parameters
    ----------
    sdata: spatialdata.SpatialData
        The input spatial data object containing spatial transcriptomics data.
    expression_threshold: float
        Threshold for filtering transcripts based on expression levels.
    gene_key: str
        Column name for gene identifiers in the transcripts data.
    layer: str
        Layer in `sdata.points` containing the transcript information.
    copy: bool
        If True, returns a modified copy of the spatial data object. Otherwise, modifies in place.

    Returns
    -------
    sdata: spatialdata.SpatialData
        The modified spatial data object with added `source` metadata if `copy=True`. Otherwise, modifies the input object in place and returns None.
    """
    # Create a copy of the table containing spatial transcriptomics data
    adata = sdata["table"].copy()
    adata.X = adata.layers["raw"]  # Use the 'raw' layer for calculations

    # Generate a binary matrix where values above the threshold are set to True
    adata_bin = adata.copy()
    adata_bin.X = adata_bin.X > expression_threshold

    # Compute the proportion of cells expressing each feature per cell type
    proportions = get_proportion_expressed_per_cell_type(adata_bin, cell_type_key="cell type")

    # Ensure the necessary `xrna_metadata` is present in `sdata`
    if "xrna_metadata" not in sdata:
        create_xrna_metadata(sdata, layer=layer)

    # Create an output DataFrame and store computed proportions
    outtable = pd.DataFrame(index=sdata["xrna_metadata"].var.index)
    sdata["xrna_metadata"].varm["source"] = outtable.join(proportions).to_numpy()

    # Return the modified SpatialData object or None based on the `copy` parameter
    return sdata.copy() if copy else None


# deprecated
def distance_to_source_cell(
    sdata: SpatialData,
    layer="transcripts",
    xcoord="x",
    ycoord="y",
    xcellcoord="x_centroid",
    ycellcoord="y_centroid",
    gene_key="gene",
    copy=False,
):
    """
    It computes the distance from each extracellular RNA transcript to the nearest source cell based on their spatial coordinates. The function uses a KDTree to efficiently find the closest cell to each transcript, storing the results in the `sdata` object.

    Parameters
    ----------
    sdata: spatialdata.SpatialData
        The AnnData object containing both transcript and cellular data.
    layer: str
        The layer in `sdata` containing the transcript data. Default is 'transcripts'.
    xcoord: str
        The column name in the transcript data for the x-coordinate. Default is 'x'.
    ycoord: str
        The column name in the transcript data for the y-coordinate. Default is 'y'.
    xcellcoord: str
        The column name in the cellular data for the x-coordinate of cell centroids. Default is 'x_centroid'.
    ycellcoord: str
        The column name in the cellular data for the y-coordinate of cell centroids. Default is 'y_centroid'.
    gene_key: str
        The column name for the gene identifier. Default is 'feature_name'.
    copy: str
        Whether to return a copy of the `sdata` object with updated distances, or modify in place. Default is False.

    Returns
    -------
    AnnData or None: anndata.AnnData
        If `copy` is True, returns the updated `sdata` object. Otherwise, modifies `sdata` in place and returns None.

    Notes
    -----
    The function assumes that the transcript data contains a column `transcript_id` and that the cellular data contains cell centroids for spatial coordinates. The KDTree algorithm is used to compute the closest cell for each transcript. The resulting distances are stored in the `distance_to_source_cell` column of the `sdata` object's transcript layer, and the closest source cell is stored in the `closest_source_cell` column. The median distance for each gene is also added to the `xrna_metadata` in the `var` attribute of `sdata`.
    """
    # Extract transcript and cellular data
    adata_bin = sdata["table"].copy()
    adata_bin.X = sdata["table"].layers["raw"]
    adata_bin.obs["x_centroid"] = [sp[0] for sp in adata_bin.obsm["spatial"]]
    adata_bin.obs["y_centroid"] = [sp[1] for sp in adata_bin.obsm["spatial"]]
    transcripts = sdata.points[layer].compute()
    extracellular_transcripts = transcripts[transcripts["extracellular"]]

    # Initialize lists to store results
    tranid = []
    dist = []
    cellids = []

    # Loop through each gene in the cellular data
    for gene_of_interest in tqdm(adata_bin.var_names):
        gene_idx = np.where(adata_bin.var_names == gene_of_interest)[0][0]
        adata_filtered = adata_bin[adata_bin.X[:, gene_idx] > 0]
        extracellular_transcripts_filtered = extracellular_transcripts[extracellular_transcripts[gene_key] == gene_of_interest].copy()

        # Only proceed if there are positive cells for the gene of interest
        if (adata_filtered.n_obs > 0) & (extracellular_transcripts_filtered.shape[0] > 0):
            # Extract coordinates of cells and transcripts
            cell_coords = np.array([adata_filtered.obs[xcellcoord], adata_filtered.obs[ycellcoord]]).T
            transcript_coords = np.array([extracellular_transcripts_filtered[xcoord], extracellular_transcripts_filtered[ycoord]]).T

            # Compute KDTree for nearest cell
            tree = KDTree(cell_coords)
            distances, closest_cells_indices = tree.query(transcript_coords, k=1)

            # Append results to lists
            tranid.extend(extracellular_transcripts_filtered["transcript_id"])
            dist.extend([d[0] for d in distances])
            cell_ids = adata_filtered.obs["cell_id"].values[closest_cells_indices.flatten()]
            cellids.extend(c[0] for c in cell_ids.reshape(closest_cells_indices.shape))

    # Create a dictionary to map transcript IDs to distances and cell IDs
    id2dist = dict(zip(tranid, dist, strict=False))
    id2closeid = dict(zip(tranid, cellids, strict=False))

    # Store the results in the DataFrame
    transcripts["distance_to_source_cell"] = transcripts["transcript_id"].map(id2dist)
    transcripts["closest_source_cell"] = transcripts["transcript_id"].map(id2closeid)
    sdata.points[layer] = sd.models.PointsModel.parse(transcripts)

    # Add median distance_to_source_cell
    dist_to_source = transcripts.loc[:, [gene_key, "distance_to_source_cell"]].groupby(gene_key).median()
    dist_to_source.columns = ["median_distance_to_source_cell"]
    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(dist_to_source)

    return sdata.copy() if copy else None


# deprecated
def compute_distant_cells_proportion(sdata: SpatialData, gene_key: str = "gene", threshold: int = 30, copy: bool = False):
    """
    Compute the proportion of transcripts for each gene that are located beyond a specified distance (in um) from their closest source cell, and add the result to the metadata of the SpatialData object.

    Parameters
    ----------
    sdata
        A SpatialData object containing the spatial omics data.
    gene_key
        Column name in the transcript data representing gene identifiers. Default is 'feature_name'.
    threshold
        The distance threshold (in micrometers) to calculate the proportion of transcripts farther away from their closest source cell. Default is 30.

    Returns
    -------
    None
    """
    data = sdata["source_score"].obs
    total_distant = pd.crosstab(data[gene_key], data["distance"] > threshold)
    prop_distant = total_distant.div(total_distant.sum(axis=1), axis=0)
    gene2distantprop = dict(zip(prop_distant.index, prop_distant[True], strict=False))

    from troutpy.tl.quantify_xrna import create_xrna_metadata

    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_xrna_metadata(sdata, layer="transcripts")
    sdata["xrna_metadata"].var["distant_from_source_proportion"] = sdata["xrna_metadata"].var.index.map(gene2distantprop)

    return sdata.copy() if copy else None


def get_proportion_expressed_per_cell_type(adata: SpatialData, cell_type_key="cell type"):
    """
    Calculate the proportion of expression for each feature (gene) per cell type.

    Parameters
    ----------
    adata: anndata.AnnData
        An AnnData object containing the single-cell or spatial transcriptomics dataset.The `obs` attribute should contain cell type annotations.
    cell_type_key: str
        The key in `adata.obs` corresponding to cell type annotations, by default 'cell type'.

    Returns
    -------
    proportion: pandas.DataFrame
        A DataFrame where rows correspond to features (genes) and columns correspond to cell types. Each entry represents the mean expression of the feature in the specified cell type.
    """
    cell_types = adata.obs[cell_type_key].unique().dropna()
    proportions = pd.DataFrame(index=adata.var_names, columns=cell_types)
    for cell_type in cell_types:
        proportions[cell_type] = adata[adata.obs[cell_type_key] == cell_type].X.mean(axis=0).T
    return proportions


def compute_source_score(
    sdata,
    layer: str = "transcripts",
    gene_key: str = "gene",
    coord_keys: list = None,
    lambda_decay: float = 0.1,
    copy: bool = False,
    celltype_key: str = "cell type",
    n_jobs: int = -1,
):
    """
    Compute a source score for extracellular transcripts based on nearby cell types and gene expression profiles, using exponential distance decay.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object with a transcript layer and an AnnData table.
    layer : str
        Name of the layer in sdata.points containing transcripts. Default is "transcripts".
    gene_key : str
        Column name in transcript table corresponding to gene names. Default is "gene".
    coord_keys : list
        List of coordinate column names to use (e.g., ["x", "y"]). Default is ["x", "y"].
    lambda_decay : float
        Decay rate for the exponential function applied to distances. Default is 0.1.
    copy : bool
        If True, return a copy of the SpatialData object with results added. Default is False.
    celltype_key : str
        Column name in adata.obs specifying cell type labels. Default is "cell type".
    n_jobs : int
        Number of parallel jobs to run (-1 uses all processors). Default is -1.

    Returns
    -------
    SpatialData or None
        Updated SpatialData object with added transcript-level and cell-level source scores.
        Returns None if copy=False (modifies sdata in place).
    """
    if coord_keys is None:
        coord_keys = ["x", "y"]
    xcoord, ycoord = coord_keys[:2]

    transcripts = sdata.points[layer].compute()
    adata = sdata["table"]
    coord_cells = adata.obsm["spatial"]

    cells = extract_expression_matrix(adata)
    cell_types = adata.obs[celltype_key]
    all_cell_types = cell_types.unique()

    extracellular_transcripts = get_extracellular_transcripts(transcripts)

    genes_to_process = extracellular_transcripts[gene_key].unique()

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_gene)(gene, extracellular_transcripts, cells, coord_cells, cell_types, gene_key, xcoord, ycoord, lambda_decay, all_cell_types)
        for gene in tqdm(genes_to_process, desc="Processing genes")
    )

    probabilities_table, closet_cell_table, cell_source_table = build_result_tables(
        results, extracellular_transcripts, adata, genes_to_process, all_cell_types
    )

    store_results_in_sdata(sdata, probabilities_table, closet_cell_table, extracellular_transcripts, xcoord, ycoord, gene_key, cell_source_table)

    return sdata.copy() if copy else None


def extract_expression_matrix(adata):
    """Extract expression matrix from AnnData object."""
    return pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        index=adata.obs_names,
        columns=adata.var_names,
    )


def get_extracellular_transcripts(transcripts):
    """Filter for extracellular transcripts."""
    if "extracellular" not in transcripts.columns:
        raise ValueError("Column 'extracellular' missing from transcript table.")
    extracellular_transcripts = transcripts[transcripts["extracellular"]]
    if extracellular_transcripts.empty:
        raise ValueError("No extracellular transcripts found.")
    return extracellular_transcripts


def process_gene(gene, extracellular_transcripts, cells, coord_cells, cell_types, gene_key, xcoord, ycoord, lambda_decay, all_cell_types):
    """Process a single gene to compute transcript scores and closest cells."""
    gene_transcripts = extracellular_transcripts[extracellular_transcripts[gene_key] == gene]
    if gene not in cells.columns:
        return None

    gene_mask = cells[gene] > 0
    if not gene_mask.any() or gene_transcripts.empty:
        return None

    gene_cells = cells.loc[gene_mask]
    coord_cells_filt = coord_cells[gene_mask]
    cell_types_filt = cell_types[gene_mask]
    transcript_coords = gene_transcripts[[xcoord, ycoord]].to_numpy()

    tree = cKDTree(coord_cells_filt)
    distances, cell_indices = tree.query(transcript_coords, k=len(gene_cells))

    distances_min = distances[:, 0]
    closest_idxs = cell_indices[:, 0]
    closest_cell_ids = gene_cells.index[closest_idxs].to_numpy()
    closest_cell_types = cell_types_filt.iloc[closest_idxs].to_numpy()

    cell_exprs = gene_cells[gene].to_numpy()
    exp_decay = np.exp(-lambda_decay * distances)
    scores = exp_decay * cell_exprs[cell_indices]

    ### this part
    residual = 1e-6  # small stabilizing term
    row_sums = scores.sum(axis=1, keepdims=True) + residual
    scores = scores / row_sums
    #scores=scores.div(scores.sum(axis=1),axis=0)

    gene_prob_df = compute_probability_table(scores, cell_indices, cell_types_filt, all_cell_types, gene_transcripts)
    gene_closest_df = build_closest_cell_table(distances_min, closest_cell_ids, closest_cell_types, gene_transcripts)
    gene_by_cell_scores = sum_scores_per_cell(gene_cells, scores, cell_indices)

    return gene_prob_df, gene_closest_df, gene_by_cell_scores


def compute_probability_table(scores, cell_indices, cell_types_filt, all_cell_types, gene_transcripts):
    """Build transcript-by-celltype score probability table."""
    n_transcripts = len(gene_transcripts)
    prob_results = np.zeros((n_transcripts, len(all_cell_types)), dtype=float)
    cell_type_to_idx = {ct: i for i, ct in enumerate(all_cell_types)}

    for i in range(n_transcripts):
        types_i = cell_types_filt.iloc[cell_indices[i]].to_numpy()
        scores_i = scores[i]
        type_indices = np.array([cell_type_to_idx[t] for t in types_i])
        score_sum = np.bincount(type_indices, weights=scores_i, minlength=len(all_cell_types))
        prob_results[i, :] = score_sum

    return pd.DataFrame(prob_results, index=gene_transcripts.index, columns=all_cell_types)


def build_closest_cell_table(distances, closest_cell_ids, closest_cell_types, gene_transcripts):
    """Construct DataFrame with closest cell and cell type info."""
    return pd.DataFrame(
        {"distance": distances, "closest_cell": closest_cell_ids, "closest_celltype": closest_cell_types},
        index=gene_transcripts.index,
    )


def sum_scores_per_cell(gene_cells, scores, cell_indices):
    """Aggregate transcript scores per cell."""
    all_cell_ids = gene_cells.index.to_numpy()
    gene_by_cell_scores = pd.Series(0.0, index=all_cell_ids)

    for i in range(len(scores)):
        cell_idx_i = cell_indices[i]
        scores_i = scores[i]
        cell_ids_i = gene_cells.index[cell_idx_i]
        gene_by_cell_scores[cell_ids_i] += scores_i

    return gene_by_cell_scores


def build_result_tables(results, extracellular_transcripts, adata, genes_to_process, all_cell_types):
    """Initialize and fill summary tables for all genes."""
    prob_table = pd.DataFrame(0, index=extracellular_transcripts.index, columns=all_cell_types, dtype=float)
    closest_table = pd.DataFrame(0, index=extracellular_transcripts.index, columns=["closest_cell", "closest_celltype", "distance"])
    cell_score_table = pd.DataFrame(0, index=adata.obs_names, columns=genes_to_process)

    for i, res in enumerate(results):
        if res is None:
            continue
        gene_prob_df, gene_closest_df, gene_by_cell_df = res
        prob_table.loc[gene_prob_df.index, :] = gene_prob_df
        closest_table.loc[gene_closest_df.index, :] = gene_closest_df
        cell_score_table.loc[gene_by_cell_df.index, genes_to_process[i]] = gene_by_cell_df.values.flatten()

    return prob_table, closest_table, cell_score_table


def store_results_in_sdata(sdata, prob_table, closest_table, extracellular_transcripts, xcoord, ycoord, gene_key, cell_source_table):
    """Store transcript- and cell-level results in the sdata object."""
    prob_adata = sc.AnnData(prob_table)
    prob_adata.obs[gene_key] = extracellular_transcripts[gene_key].values
    prob_adata.obs["distance"] = closest_table["distance"].astype(float).values
    prob_adata.obs["closest_cell"] = closest_table["closest_cell"].astype(str).values
    prob_adata.obs["closest_celltype"] = closest_table["closest_celltype"].astype(str).values
    prob_adata.obsm["spatial"] = extracellular_transcripts[[xcoord, ycoord]].to_numpy()
    sdata.tables["source_score"] = prob_adata

    sdata["table"].obs["urna_source_score"] = list(np.sum(cell_source_table, axis=1))
    try:
        adata=sdata['table']
        raw_expr=adata.layers['raw']
        cell_expr_sum = np.array(np.sum(raw_expr.todense(), axis=1)).flatten()
        sdata["table"].obs['normalized_urna_source_score'] = adata.obs['urna_source_score'] / cell_expr_sum
    except KeyError:
        print('Normalized urna source score could not be computed')
    


def compute_contribution_score(sdata):
    """Compute a normalized extracellular RNA (uRNA) contribution score for each cell.

    For each gene, a cell's contribution is weighted as extracellular proportionand divided by the number of cells expressing that gene.

    Parameters
    ----------
    sdata : dict
        A spatialdata object with keys 'table' and 'xrna_metadata'.
    """
    adata = sdata["table"]
    gene_meta = sdata["xrna_metadata"].var

    raw_expr = adata.layers["raw"]
    if hasattr(raw_expr, "toarray"):
        raw_expr = raw_expr.toarray()

    genes = adata.var_names
    gene_meta = gene_meta.loc[gene_meta.index.intersection(genes)]

    if gene_meta.empty:
        raise ValueError("No common genes found between adata and gene metadata.")

    common_genes = gene_meta.index
    gene_indices = [adata.var_names.get_loc(g) for g in common_genes]
    raw_expr = raw_expr[:, gene_indices]  # shape (n_cells, n_common_genes)

    # Reorder gene_meta to match
    gene_meta = gene_meta.loc[common_genes]
    extracellular_weights = gene_meta["count"].values  # shape (n_genes,)

    n_cells, n_genes = raw_expr.shape
    contribution_matrix = np.zeros_like(raw_expr, dtype=float)

    # For each gene, calculate normalized extracellular contribution
    for i in range(n_genes):
        gene_expr = raw_expr[:, i]
        expressing_cells = gene_expr > 0
        n_expressing = expressing_cells.sum()

        if n_expressing > 0:
            weight = extracellular_weights[i] / n_expressing
            contribution_matrix[expressing_cells, i] = weight

    # Cell-wise sum of contributions across genes
    score = contribution_matrix.sum(axis=1)

    adata.obs["urna_contribution_score"] = score
    adata.obs["normalized_urna_contribution_score"]=score/np.sum(raw_expr,axis=1)
