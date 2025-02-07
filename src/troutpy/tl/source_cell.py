import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from sklearn.neighbors import KDTree
from spatialdata import SpatialData
from tqdm import tqdm

warnings.filterwarnings("ignore")


def create_xrna_metadata(
    sdata: SpatialData, points_layer: str = "transcripts", gene_key: str = "feature_name", copy: bool = False
) -> SpatialData | None:
    """
    Creates a new table within the SpatialData object that contains a 'gene' column with the unique gene names extracted from the specified points layer.

    Parameters
    ----------
    - sdata (SpatialData)
        The SpatialData object to modify.
    - points_layer (str, optional)
        The name of the layer in `sdata.points` from which to extract gene names.Default is 'transcripts'.
    - gene_key (str, optional)
        The key in the `points_layer` dataframe that contains the gene names.Default is 'feature_name'.
    - copy (bool)
        If `True`, returns a copy of the `SpatialData` object with the new table added.

    Returns
    -------
    - sdata (SpatialData)
        If `copy` is `True`, returns a copy of the modified `SpatialData` object. Otherwise, returns `None`.

    """
    # Check if the specified points layer exists
    if points_layer not in sdata.points:
        raise ValueError(f"Points layer '{points_layer}' not found in sdata.points.")

    # Extract unique gene names from the specified points layer
    points_data = sdata.points[points_layer]
    if gene_key not in points_data.columns:
        raise ValueError(f"The specified points layer '{points_layer}' does not contain a '{gene_key}' column.")

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


def compute_source_cells(sdata: SpatialData, expression_threshold=1, gene_id_column="feature_name", layer="transcripts", copy=False):
    """
    Compute the source of extracellular RNA by linking detected extracellular transcripts to specific cell types in the spatial data.

    Parameters
    ----------
    - sdata (SpatialData object)
        The input spatial data object containing spatial transcriptomics data.
    - expression_threshold (float, optional, default=1)
        Threshold for filtering transcripts based on expression levels.
    - gene_id_column (str, optional, default='feature_name')
        Column name for gene identifiers in the transcripts data.
    - layer (str, optional, default='transcripts')
        Layer in `sdata.points` containing the transcript information.
    - copy (bool, optional, default=False)
        If True, returns a modified copy of the spatial data object. Otherwise, modifies in place.

    Returns
    -------
    - sdata (SpatialData)
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
        create_xrna_metadata(sdata, points_layer="transcripts")

    # Create an output DataFrame and store computed proportions
    outtable = pd.DataFrame(index=sdata["xrna_metadata"].var.index)
    sdata["xrna_metadata"].varm["source"] = outtable.join(proportions).to_numpy()

    # Return the modified SpatialData object or None based on the `copy` parameter
    return sdata.copy() if copy else None


def distance_to_source_cell(
    sdata: SpatialData,
    layer="transcripts",
    xcoord="x",
    ycoord="y",
    xcellcoord="x_centroid",
    ycellcoord="y_centroid",
    gene_id_column="feature_name",
    copy=False,
):
    """
    It computes the distance from each extracellular RNA transcript to the nearest source cell based on their spatial coordinates. The function uses a KDTree to efficiently find the closest cell to each transcript, storing the results in the `sdata` object.

    Parameters
    ----------
    - sdata (AnnData)
        The AnnData object containing both transcript and cellular data.
    - layer (str, optional)
        The layer in `sdata` containing the transcript data. Default is 'transcripts'.
    - xcoord (str, optional)
        The column name in the transcript data for the x-coordinate. Default is 'x'.
    - ycoord (str, optional)
        The column name in the transcript data for the y-coordinate. Default is 'y'.
    - xcellcoord (str, optional)
        The column name in the cellular data for the x-coordinate of cell centroids. Default is 'x_centroid'.
    - ycellcoord (str, optional)
        The column name in the cellular data for the y-coordinate of cell centroids. Default is 'y_centroid'.
    - gene_id_column (str, optional)
        The column name for the gene identifier. Default is 'feature_name'.
    - copy (bool, optional)
        Whether to return a copy of the `sdata` object with updated distances, or modify in place. Default is False.

    Returns
    -------
    - AnnData or None
        If `copy` is True, returns the updated `sdata` object. Otherwise, modifies `sdata` in place and returns None.

    Notes
    -----
    - The function assumes that the transcript data contains a column `transcript_id` and that the cellular data contains cell centroids for spatial coordinates. The KDTree algorithm is used to compute the closest cell for each transcript. The resulting distances are stored in the `distance_to_source_cell` column of the `sdata` object's transcript layer, and the closest source cell is stored in the `closest_source_cell` column. The median distance for each gene is also added to the `xrna_metadata` in the `var` attribute of `sdata`.
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
        extracellular_transcripts_filtered = extracellular_transcripts[extracellular_transcripts[gene_id_column] == gene_of_interest].copy()

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
    dist_to_source = transcripts.loc[:, [gene_id_column, "distance_to_source_cell"]].groupby(gene_id_column).median()
    dist_to_source.columns = ["median_distance_to_source_cell"]
    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(dist_to_source)

    return sdata.copy() if copy else None


def compute_distant_cells_proportion(sdata: SpatialData, gene_id_column: str = "feature_name", threshold: int = 30, copy: bool = False):
    """
    Compute the proportion of transcripts for each gene that are located beyond a specified distance (in um) from their closest source cell, and add the result to the metadata of the SpatialData object.

    Parameters
    ----------
    sdata
        A SpatialData object containing the spatial omics data.
    gene_id_column
        Column name in the transcript data representing gene identifiers. Default is 'feature_name'.
    threshold
        The distance threshold (in micrometers) to calculate the proportion of transcripts farther away from their closest source cell. Default is 30.

    Returns
    -------
    None
    """
    data = sdata["source_score"].obs
    total_distant = pd.crosstab(data[gene_id_column], data["distance"] > threshold)
    prop_distant = total_distant.div(total_distant.sum(axis=1), axis=0)
    gene2distantprop = dict(zip(prop_distant.index, prop_distant[True], strict=False))

    from troutpy.tl.quantify_xrna import create_xrna_metadata

    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_xrna_metadata(sdata, points_layer="transcripts")
    sdata["xrna_metadata"].var["distant_from_source_proportion"] = sdata["xrna_metadata"].var.index.map(gene2distantprop)

    return sdata.copy() if copy else None


def get_proportion_expressed_per_cell_type(adata: SpatialData, cell_type_key="cell type"):
    """
    Calculate the proportion of expression for each feature (gene) per cell type.

    Parameters
    ----------
    - adata (AnnData)
        An AnnData object containing the single-cell or spatial transcriptomics dataset.The `obs` attribute should contain cell type annotations.
    - cell_type_key (str, optional)
        The key in `adata.obs` corresponding to cell type annotations, by default 'cell type'.

    Returns
    -------
    - proportions (pd.DataFrame)
        A DataFrame where rows correspond to features (genes) and columns correspond to cell types. Each entry represents the mean expression of the feature in the specified cell type.
    """
    cell_types = adata.obs[cell_type_key].unique().dropna()
    proportions = pd.DataFrame(index=adata.var_names, columns=cell_types)
    for cell_type in cell_types:
        proportions[cell_type] = adata[adata.obs[cell_type_key] == cell_type].X.mean(axis=0).T
    return proportions


def compute_source_score(
    sdata: SpatialData,
    layer: str = "transcripts",
    gene_id_column: str = "feature_name",
    xcoord: str = "x",
    ycoord: str = "y",
    lambda_decay: float = 0.1,
    copy: bool = False,
    celltype_key: str = "cell type",
):
    """
    Computes the probabilities of each extracellular transcript originating from a specific cell.

    Parameters
    ----------
    sdata
        The input spatial data object.
    layer
        The layer in `sdata.points` containing the transcript data. Default is 'transcripts'.
    gene_id_column
        Column name in the transcript data representing gene identifiers. Default is 'feature_name'.
    xcoord, ycoord
        Column names for spatial coordinates of transcripts and cell centroids.
    lambda_decay
        The exponential decay factor for distances.
    copy
        If True, returns a modified copy of the SpatialData object.
    celltype_key
        Key for cell type annotations in the cell table.

    Returns
    -------
    sdata
        Updated SpatialData object with source scores added in sdata.tables['source_score'], including distance to closest cell
    """
    transcripts = sdata.points[layer].compute()
    cells = sdata["table"].to_df()
    coord_cells = sdata["table"].obsm["spatial"]
    cell_types = sdata["table"].obs[celltype_key]
    all_cell_types = cell_types.unique()

    # Ensure necessary columns exist
    required_cols = [xcoord, ycoord, gene_id_column]
    for col in required_cols:
        if col not in transcripts.columns and col not in cells.columns:
            raise ValueError(f"Required column '{col}' is missing.")

    # Filter for extracellular transcripts only
    extracellular_transcripts = transcripts[transcripts["extracellular"] == False]
    probabilities_table = pd.DataFrame(0, index=extracellular_transcripts.index, columns=all_cell_types, dtype=float)

    # Precompute KDTree for all cell coordinates
    closet_cell_table = pd.DataFrame(0, index=extracellular_transcripts.index, columns=["closest_cell", "closest_celltype", "distance"])

    # Process each gene
    for gene in tqdm(cells.columns.unique(), desc="Processing genes"):
        # Filter transcripts and cells for the current gene
        gene_transcripts = extracellular_transcripts[extracellular_transcripts[gene_id_column] == gene]

        gene_mask = cells[gene] > 0  # Mask for cells expressing the gene
        if not gene_mask.any() or gene_transcripts.empty:
            continue
        transcript_coords = gene_transcripts[[xcoord, ycoord]].to_numpy()
        gene_cells = cells.loc[gene_mask]
        coord_cells_filt = coord_cells[gene_mask]
        cell_types_filt = cell_types[gene_mask]

        # Compute distances between transcripts and filtered cells
        kdtree = KDTree(coord_cells_filt)
        distances, cell_indices = kdtree.query(transcript_coords, k=len(coord_cells_filt))

        # compute_min_dist
        distances_min = np.min(distances, axis=1)
        distances_idx = cell_indices[:, 0]
        cell_id_min = gene_cells.index[distances_idx]
        cell_type_min = cell_types_filt[distances_idx]

        # Get the gene-specific expression values
        cell_exprs = gene_cells[gene].to_numpy()

        # Compute exponential decay scores
        exp_decay = np.exp(-lambda_decay * distances)
        scores = exp_decay * cell_exprs[cell_indices]

        # Aggregate scores by cell type
        for i, transcript_idx in enumerate(gene_transcripts.index):
            cell_indices_i = cell_indices[i]
            scores_i = scores[i]
            types_i = cell_types_filt.iloc[cell_indices_i].to_numpy()

            # add min_distance
            closet_cell_table.loc[transcript_idx, "distance"] = distances_min[i]
            closet_cell_table.loc[transcript_idx, "closest_cell"] = cell_id_min[i]
            closet_cell_table.loc[transcript_idx, "closest_celltype"] = cell_type_min[i]
            for cell_type in all_cell_types:
                probabilities_table.loc[transcript_idx, cell_type] = scores_i[types_i == cell_type].sum()

    # We format all probabilieies as an anndata, stored in sdata.tables['source_score']
    prob_table = sc.AnnData(probabilities_table)
    prob_table.obs[gene_id_column] = list(extracellular_transcripts[gene_id_column])
    prob_table.obs["distance"] = list(closet_cell_table["distance"].astype(float))
    prob_table.obs["feature_name"] = list(closet_cell_table["feature_name"].astype(str))
    prob_table.obs["closest_cell"] = list(closet_cell_table["closest_cell"].astype(str))
    prob_table.obs["closest_celltype"] = list(closet_cell_table["closest_celltype"].astype(str))

    prob_table.obsm["spatial"] = extracellular_transcripts[[xcoord, ycoord]].to_numpy()
    sdata.tables["source_score"] = prob_table
    return sdata.copy() if copy else None
