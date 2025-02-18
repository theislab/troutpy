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
    gene_id_key: str = "feature_name",
    copy: bool = False,
) -> SpatialData | None:
    """
    Identifies the nearest cell to each transcript based on spatial coordinates and annotates the transcript data with the ID, cell type, and distance to the closest cell.

    Parameters
    ----------
    sdata (SpatialData)
        SpatialData object containing spatial and transcript data.
    layer (str, optional)
        The layer in `sdata.points` containing transcript data. Default is 'transcripts'.
    xcoord (str, optional)
        Column name for the x-coordinate of transcripts. Default is 'x'.
    ycoord (str, optional)
        Column name for the y-coordinate of transcripts. Default is 'y'.
    xcellcoord (str, optional)
        Column name for the x-coordinate of cell centroids. Default is 'x_centroid'.
    ycellcoord (str, optional)
        Column name for the y-coordinate of cell centroids. Default is 'y_centroid'.
    celltype_key (str, optional)
        Column name in `adata.obs` that contains cell type annotations. Default is 'cell type'.
    gene_id_key (str, optional)
        Column name in `sdata.points[layer]` that contains gene identity. Default is 'feature_name'.
    copy (bool, optional)
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
def define_target_by_celltype(sdata: SpatialData, layer="transcripts", closest_celltype_key="closest_target_cell_type", feature_key="feature_name"):
    """
    It calculates a cross-tabulation between features (e.g., extracellular transcripts) and cell types,and then normalizes the result to provide the proportion of each feature associated with each cell type.

    Parameters
    ----------
    - sdata (SpatialData): A spatial data object that contains transcript and cell type information. The relevant data is accessed from the `sdata.points[layer]`
    - layer (str, optional): The key for the layer in `sdata.points` that contains the transcript data (default: 'extracellular_transcripts').
    - celltype_key (str, optional): The column name representing cell types in the transcript data (default: 'cell type').
    - feature_key (str, optional): The column name representing the feature (e.g., transcript or gene) in the transcript data (default: 'feature_name').

    Returns
    -------
    pd.DataFrame: A pandas DataFrame where the rows represent features (e.g., transcripts), and the columns represent cell types. Each entry in the DataFrame is the proportion of that feature associated with the respective cell type.
    """
    # Extract transcript data from the specified layer
    transcripts = sdata.points[layer][[feature_key, closest_celltype_key]].compute()
    # Compute cross-tabulation between features and cell types (raw counts)
    celltype_by_feature_raw = pd.crosstab(transcripts[feature_key], transcripts[closest_celltype_key])
    # Normalize by the total number of each feature (row-wise normalization)
    celltype_by_feature = celltype_by_feature_raw.div(celltype_by_feature_raw.sum(axis=1), axis=0)

    return celltype_by_feature


def compute_target_score(
    sdata: SpatialData,
    layer: str = "transcripts",
    gene_key: str = "feature_name",
    coords_key: list = None,  # type: ignore
    lambda_decay=0.1,
    copy=False,
    celltype_key="cell type",
):
    """
    Computes the scores for each extracellular transcript targeting a specific cell type.

    Parameters
    ----------
    - sdata (SpatialData)
        The input spatial data object.
    - layer (str, optional)
        The layer in `sdata.points` containing the transcript data. Default is 'transcripts'.
    - gene_key (str, optional)
        Column name in the transcript data representing gene identifiers. Default is 'feature_name'.
    - coords_key
        Column names for spatial coordinates of transcripts and cell centroids.
    - lambda_decay (float, optional)
        The exponential decay factor for distances.
    - copy (bool, optional)
        If True, returns a modified copy of the SpatialData object.
    - celltype_key (str, optional)
        Key for cell type annotations in the cell table.

    Returns
    -------
    - sdata (SpatialData)
        Updated SpatialData object with target cell scores added.
    """
    # define xcoord and ycoord
    if coords_key is None:
        coords_key = ["x", "y"]
    xcoord = coords_key[0]
    ycoord = coords_key[1]

    # Extract transcript and cellular data
    transcripts = sdata.points[layer].compute()
    cells = sdata["table"].to_df()
    coord_cells = sdata["table"].obsm["spatial"]
    cell_types = sdata["table"].obs[celltype_key]
    all_cell_types = cell_types.unique()

    # Ensure necessary columns exist
    required_cols = [xcoord, ycoord]
    for col in required_cols:
        if col not in transcripts.columns and col not in cells.columns:
            raise ValueError(f"Required column '{col}' is missing.")

    # Filter for extracellular transcripts only
    extracellular_transcripts = transcripts[transcripts["extracellular"]]
    target_scores_table = pd.DataFrame(0, index=extracellular_transcripts.index, columns=all_cell_types, dtype=float)

    # Precompute KDTree for all cell coordinates
    kdtree = KDTree(coord_cells)

    # Compute scores for each transcript
    transcript_coords = extracellular_transcripts[[xcoord, ycoord]].to_numpy()
    distances, cell_indices = kdtree.query(transcript_coords, k=len(coord_cells))

    # Compute exponential decay scores for proximity
    exp_decay = np.exp(-lambda_decay * distances)

    # Aggregate scores by cell type
    for i, transcript_idx in enumerate(tqdm(extracellular_transcripts.index)):
        cell_indices_i = cell_indices[i]
        scores_i = exp_decay[i]
        types_i = cell_types.iloc[cell_indices_i].to_numpy()

        for cell_type in all_cell_types:
            target_scores_table.loc[transcript_idx, cell_type] = scores_i[types_i == cell_type].sum()

    # probabilities_table['feature_name']=extracellular_transcripts['feature_name']
    prob_table = sc.AnnData(target_scores_table)
    prob_table.obs[gene_key] = list(extracellular_transcripts[gene_key].astype(str))
    prob_table.obsm["spatial"] = extracellular_transcripts[[xcoord, ycoord]].to_numpy()
    sdata.tables["target_score"] = prob_table

    return sdata.copy() if copy else None
