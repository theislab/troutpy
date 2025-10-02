import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from sklearn.neighbors import KDTree
from spatialdata import SpatialData
from tqdm import tqdm


# deprecated
def calculate_target_cells(
    sdata: SpatialData,
    layer: str = "transcripts",
    xcoord: str = "x",
    ycoord: str = "y",
    xcellcoord: str = "x_centroid",
    ycellcoord: str = "y_centroid",
    celltype_key: str = "cell type",
    gene_id_key: str = "gene",
    copy: bool = False,
) -> SpatialData | None:
    """
    Identifies the nearest cell to each transcript based on spatial coordinates and annotates the transcript data with the ID, cell type, and distance to the closest cell.

    Parameters
    ----------
    sdata: spatialdata.SpatialData
        SpatialData object containing spatial and transcript data.
    layer: str
        The layer in `sdata.points` containing transcript data. Default is 'transcripts'.
    xcoord: str
        Column name for the x-coordinate of transcripts. Default is 'x'.
    ycoord: str
        Column name for the y-coordinate of transcripts. Default is 'y'.
    xcellcoord: str
        Column name for the x-coordinate of cell centroids. Default is 'x_centroid'.
    ycellcoord: str
        Column name for the y-coordinate of cell centroids. Default is 'y_centroid'.
    celltype_key: str
        Column name in `adata.obs` that contains cell type annotations. Default is 'cell type'.
    gene_id_key: str
        Column name in `sdata.points[layer]` that contains gene identity. Default is 'gene'.
    copy: bool
        If True, returns a copy of the modified SpatialData object. Default is False.

    Returns
    -------
    Optional (SpatialData)
        Modified SpatialData object with updated transcript annotations if `copy=True`.Otherwise, updates are made in place, and None is returned.
    """
    # Copy AnnData object from the SpatialData table
    adata = sdata["table"].copy()

    # Use the 'raw' layer for transcript data
    adata.X = sdata["table"].layers["raw"]

    # Extract x and y centroid coordinates from cell data
    adata.obs[xcellcoord] = [sp[0] for sp in adata.obsm["spatial"]]
    adata.obs[ycellcoord] = [sp[1] for sp in adata.obsm["spatial"]]

    # Extract transcript data from the specified layer
    transcripts = sdata.points[layer].compute()

    # Extract cell and transcript spatial coordinates
    cell_coords = np.array([adata.obs[xcellcoord], adata.obs[ycellcoord]]).T
    transcript_coords = np.array([transcripts[xcoord], transcripts[ycoord]]).T

    # Initialize arrays to store the closest cell indices and distances
    closest_cells = np.zeros(len(transcripts), dtype=int)
    distances = np.zeros(len(transcripts))

    # Calculate the closest cell and distance for each transcript
    for i in tqdm(range(len(transcript_coords)), desc="Calculating closest cells"):
        dist = np.linalg.norm(cell_coords - transcript_coords[i], axis=1)
        closest_cells[i] = np.argmin(dist)
        distances[i] = np.min(dist)

    # Annotate the transcript DataFrame with the closest cell information
    transcripts["closest_target_cell"] = adata.obs.index[closest_cells].values
    transcripts["closest_target_cell_type"] = adata.obs[celltype_key].values[closest_cells]
    transcripts["distance_to_target_cell"] = distances

    # Update the SpatialData object with the modified transcript data
    sdata.points[layer] = sd.models.PointsModel.parse(transcripts)

    extracellular_transcripts = transcripts[transcripts["extracellular"]]
    # Compute cross-tabulation between features and cell types (raw counts)
    celltype_by_feature_raw = pd.crosstab(extracellular_transcripts[gene_id_key], extracellular_transcripts["closest_target_cell_type"])
    # Normalize by the total number of each feature (row-wise normalization)
    celltype_by_feature = celltype_by_feature_raw.div(celltype_by_feature_raw.sum(axis=1), axis=0)

    # Create an output DataFrame and store computed proportions
    outtable = pd.DataFrame(index=sdata["xrna_metadata"].var.index)
    sdata["xrna_metadata"].varm["target"] = outtable.join(celltype_by_feature).to_numpy()

    # Return a copy of the modified SpatialData object if requested
    return sdata.copy() if copy else None


# deprecated
def define_target_by_celltype(sdata: SpatialData, layer="transcripts", closest_celltype_key="closest_target_cell_type", feature_key="gene"):
    """
    It calculates a cross-tabulation between features (e.g., extracellular transcripts) and cell types,and then normalizes the result to provide the proportion of each feature associated with each cell type.

    Parameters
    ----------
    sdata: spatialdata.SpatialData
        A spatial data object that contains transcript and cell type information. The relevant data is accessed from the `sdata.points[layer]`
    layer: str
        The key for the layer in `sdata.points` that contains the transcript data (default: 'extracellular_transcripts').
    celltype_key: str
        The column name representing cell types in the transcript data (default: 'cell type').
    feature_key: str
        The column name representing the feature (e.g., transcript or gene) in the transcript data (default: 'gene').

    Returns
    -------
    celltype_by_feature: pandas.DataFrame
        A pandas DataFrame where the rows represent features (e.g., transcripts), and the columns represent cell types. Each entry in the DataFrame is the proportion of that feature associated with the respective cell type.
    """
    # Extract transcript data from the specified layer
    transcripts = sdata.points[layer][[feature_key, closest_celltype_key]].compute()
    # Compute cross-tabulation between features and cell types (raw counts)
    celltype_by_feature_raw = pd.crosstab(transcripts[feature_key], transcripts[closest_celltype_key])
    # Normalize by the total number of each feature (row-wise normalization)
    celltype_by_feature = celltype_by_feature_raw.div(celltype_by_feature_raw.sum(axis=1), axis=0)

    return celltype_by_feature

def compute_target_score(
    sdata,
    layer: str = "transcripts",
    gene_key: str = "gene",
    coords_key: list = None,  # type: ignore
    lambda_decay: float = 0.1,
    copy: bool = False,
    celltype_key: str = "cell type",
    k_neighbors: int = 50,
    batch_size: int = 100_000,
):
    """
    Computes target scores for extracellular transcripts including distance and closest cell info.

    Parameters
    ----------
    sdata: SpatialData
        SpatialData object containing transcript points and cell table.
    layer: str
        Transcript layer in sdata.points.
    gene_key: str
        Column in transcript table for gene names.
    coords_key: list
        List of coordinate column names, e.g., ["x", "y"].
    lambda_decay: float
        Exponential decay factor for distance.
    copy: bool
        Return a copy of sdata if True.
    celltype_key: str
        Column in adata.obs with cell type labels.
    k_neighbors: int
        Number of nearest cells to consider per transcript.
    batch_size: int
        Process transcripts in batches for memory efficiency.

    Returns
    -------
    SpatialData
        Updated sdata with 'target_score' table including per-transcript distance, closest cell, closest_cell_type.
    """
    if coords_key is None:
        coords_key = ["x", "y"]
    xcoord, ycoord = coords_key

    # Extract transcript data
    transcripts = sdata.points[layer].compute()
    adata = sdata["table"]
    coord_cells = adata.obsm["spatial"]
    cell_types = adata.obs[celltype_key]
    all_cell_types = cell_types.unique()

    extracellular_transcripts = transcripts[transcripts["extracellular"]]
    transcript_coords = extracellular_transcripts[[xcoord, ycoord]].to_numpy()

    # Output tables
    target_scores_table = pd.DataFrame(
        0, index=extracellular_transcripts.index, columns=all_cell_types, dtype=float
    )
    closest_cell_info = pd.DataFrame(
        index=extracellular_transcripts.index,
        columns=["distance", "closest_cell", "closest_cell_type"],
        dtype=object
    )

    # KDTree on cell centroids
    kdtree = KDTree(coord_cells)

    n_transcripts = transcript_coords.shape[0]
    print(f"Computing target scores for {n_transcripts:,} transcripts in batches of {batch_size}...")

    for start in tqdm(range(0, n_transcripts, batch_size)):
        end = min(start + batch_size, n_transcripts)
        coords_batch = transcript_coords[start:end]
        indices_batch = extracellular_transcripts.index[start:end]

        # Query KDTree for k nearest neighbors
        distances, cell_indices = kdtree.query(coords_batch, k=k_neighbors)
        exp_decay = np.exp(-lambda_decay * distances)

        for i, transcript_idx in enumerate(indices_batch):
            cell_indices_i = cell_indices[i]
            scores_i = exp_decay[i]
            types_i = cell_types.iloc[cell_indices_i].to_numpy()

            # Compute target scores per cell type
            for cell_type in all_cell_types:
                target_scores_table.loc[transcript_idx, cell_type] = scores_i[types_i == cell_type].sum()

            # Store closest cell info
            min_idx = np.argmin(distances[i])
            closest_cell_idx = cell_indices_i[min_idx]
            closest_cell_info.loc[transcript_idx, "distance"] = distances[i][min_idx]
            closest_cell_info.loc[transcript_idx, "closest_cell"] = adata.obs_names[closest_cell_idx]
            closest_cell_info.loc[transcript_idx, "closest_cell_type"] = cell_types.iloc[closest_cell_idx]

    # Normalize to probabilities
    residual = 1e-6
    row_sums = target_scores_table.sum(axis=1) + residual
    target_scores_table = target_scores_table.div(row_sums, axis=0)

    # Store results as AnnData
    prob_table = sc.AnnData(target_scores_table)
    prob_table.obs[gene_key] = extracellular_transcripts[gene_key].astype(str).values
    prob_table.obsm["spatial"] = transcript_coords

    # Store closest cell info in .obs as well
    for col in ["distance", "closest_cell", "closest_cell_type"]:
        prob_table.obs[col] = closest_cell_info[col].values

    sdata.tables["target_score"] = prob_table

    return sdata.copy() if copy else None

