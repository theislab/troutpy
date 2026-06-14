from copy import deepcopy

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
    """Find the nearest cell to each transcript and annotate it with that cell's ID, type, and distance.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object containing spatial and transcript data.
    layer : str, optional
        The layer in ``sdata.points`` containing transcript data. Defaults to ``"transcripts"``.
    xcoord : str, optional
        Column name for the x-coordinate of transcripts. Defaults to ``"x"``.
    ycoord : str, optional
        Column name for the y-coordinate of transcripts. Defaults to ``"y"``.
    xcellcoord : str, optional
        Column name for the x-coordinate of cell centroids. Defaults to ``"x_centroid"``.
    ycellcoord : str, optional
        Column name for the y-coordinate of cell centroids. Defaults to ``"y_centroid"``.
    celltype_key : str, optional
        Column name in ``adata.obs`` that contains cell type annotations. Defaults to ``"cell type"``.
    gene_id_key : str, optional
        Column name in ``sdata.points[layer]`` that contains gene identity. Defaults to ``"gene"``.
    copy : bool, optional
        If ``True``, returns a copy of the modified SpatialData object. Defaults to ``False``.

    Returns
    -------
    spatialdata.SpatialData or None
        Modified SpatialData object with updated transcript annotations if ``copy=True``;
        otherwise updates are made in place and ``None`` is returned.
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
    return deepcopy(sdata) if copy else None


# deprecated
def define_target_by_celltype(sdata: SpatialData, layer="transcripts", closest_celltype_key="closest_target_cell_type", feature_key="gene"):
    """Compute the per-gene proportion of extracellular transcripts assigned to each cell type.

    Cross-tabulates ``feature_key`` against ``closest_celltype_key`` and normalizes each row
    (gene) to sum to 1.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        A spatial data object that contains transcript and cell type information, accessed
        from ``sdata.points[layer]``.
    layer : str, optional
        Key of the layer in ``sdata.points`` that contains the transcript data. Defaults to
        ``"transcripts"``.
    closest_celltype_key : str, optional
        Column name holding the cell type of each transcript's closest target cell. Defaults
        to ``"closest_target_cell_type"``.
    feature_key : str, optional
        Column name representing the feature (e.g., gene) in the transcript data. Defaults
        to ``"gene"``.

    Returns
    -------
    pandas.DataFrame
        DataFrame where rows are features (e.g., genes) and columns are cell types, with each
        entry giving the proportion of that feature's extracellular transcripts associated
        with the respective cell type.
    """
    transcripts = sdata.points[layer][[feature_key, closest_celltype_key]].compute()
    celltype_by_feature_raw = pd.crosstab(transcripts[feature_key], transcripts[closest_celltype_key])
    celltype_by_feature = celltype_by_feature_raw.div(celltype_by_feature_raw.sum(axis=1), axis=0)

    return celltype_by_feature


def compute_target_score(
    sdata,
    layer: str = "transcripts",
    gene_key: str = "gene",
    coords_key: list | None = None,
    lambda_decay: float = 0.1,
    copy: bool = False,
    celltype_key: str = "cell type",
    k_neighbors: int = 50,
    batch_size: int = 100_000,
):
    """Compute, for every extracellular transcript, a per-cell-type target score plus its closest cell.

    For each transcript, the ``k_neighbors`` nearest cells (by centroid distance) are
    weighted by ``exp(-lambda_decay * distance)`` and summed per cell type to produce a
    target-score distribution, alongside the single closest cell and its type.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing transcript points and cell table.
    layer : str, optional
        Transcript layer in ``sdata.points``. Defaults to ``"transcripts"``.
    gene_key : str, optional
        Column in the transcript table for gene names. Defaults to ``"gene"``.
    coords_key : list of str, optional
        Coordinate column names, e.g. ``["x", "y"]``. Defaults to ``["x", "y"]``.
    lambda_decay : float, optional
        Exponential decay factor applied to neighbor distances. Defaults to ``0.1``.
    copy : bool, optional
        If ``True``, return a copy of ``sdata``. Defaults to ``False``.
    celltype_key : str, optional
        Column in ``adata.obs`` with cell type labels. Defaults to ``"cell type"``.
    k_neighbors : int, optional
        Number of nearest cells to consider per transcript. Defaults to ``50``.
    batch_size : int, optional
        Number of transcripts to process per batch. Defaults to ``100_000``.

    Returns
    -------
    SpatialData or None
        ``sdata`` with a ``"target_score"`` table (per-transcript per-cell-type scores plus
        ``distance``, ``closest_cell``, and ``closest_cell_type`` columns in ``.obs``) if
        ``copy=False``; a copy of it if ``copy=True``.
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

    return deepcopy(sdata) if copy else None

