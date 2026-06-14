import os
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
from spatialdata import SpatialData
from tqdm import tqdm


# deprecated
def get_number_of_communication_genes(
    source_proportions: pd.DataFrame,
    target_proportions: pd.DataFrame,
    source_proportion_threshold: float = 0.2,
    target_proportion_threshold: float = 0.2,
) -> pd.DataFrame:
    """Compute the number of significantly shared genes between each pair of source and target cell types.

    Parameters
    ----------
    source_proportions : pandas.DataFrame
        Proportion of cells expressing each gene (rows), per source cell type (columns).
    target_proportions : pandas.DataFrame
        Proportion of cells that are the closest cell to transcripts of each gene
        (rows), per target cell type (columns).
    source_proportion_threshold : float, optional
        Minimum proportion for a cell type to be considered a significant source of a
        gene. Defaults to ``0.2``.
    target_proportion_threshold : float, optional
        Minimum proportion for a cell type to be considered a significant target of a
        gene. Defaults to ``0.2``.

    Returns
    -------
    pandas.DataFrame
        Square matrix (source cell types x target cell types) with the number of
        genes shared between each pair.
    """
    source_binary = source_proportions > source_proportion_threshold
    target_binary = target_proportions > target_proportion_threshold

    number_interactions_df = pd.DataFrame(index=source_binary.columns, columns=target_binary.columns)

    for col in source_binary.columns:
        sig_gene_source = source_binary.index[source_binary[col]]
        for col2 in target_binary.columns:
            sig_gene_target = target_binary.index[target_binary[col2]]
            number_interactions_df.loc[col, col2] = len(set(sig_gene_source).intersection(sig_gene_target))

    number_interactions_df = number_interactions_df[number_interactions_df.index]
    number_interactions_df.columns.name = "Target cell type"
    number_interactions_df.index.name = "Source cell type"
    return number_interactions_df


# deprecated
def get_gene_interaction_strength(
    source_proportions: pd.DataFrame,
    target_proportions: pd.DataFrame,
    gene_symbol: str = "",
    return_interactions: bool = False,
    save: bool = False,
    output_path: str = "",
    format: str = "pdf",
) -> pd.DataFrame:
    """Compute and plot the interaction strength between source and target cell types for one gene.

    The interaction matrix is the outer product of the gene's proportion in each
    source cell type and its proportion in each target cell type.

    Parameters
    ----------
    source_proportions : pandas.DataFrame
        Proportion of the gene (rows) in each source cell type (columns).
    target_proportions : pandas.DataFrame
        Proportion of the gene (rows) in each target cell type (columns).
    gene_symbol : str, optional
        Gene for which the interaction strength is computed. Defaults to ``""``.
    return_interactions : bool, optional
        Currently unused. Defaults to ``False``.
    save : bool, optional
        If ``True``, save the plot to ``output_path/figures``. Defaults to ``False``.
    output_path : str, optional
        Directory in which the ``figures`` subdirectory is created when ``save=True``.
        Defaults to ``""``.
    format : str, optional
        File format used when ``save=True`` (e.g. ``"pdf"``, ``"png"``). Defaults to
        ``"pdf"``.

    Returns
    -------
    pandas.DataFrame
        Interaction strength matrix (source cell types x target cell types) for
        ``gene_symbol``.
    """
    target_proportions = target_proportions[source_proportions.columns]

    source_proportions_vals = source_proportions.loc[gene_symbol].values[:, None]
    target_proportions_vals = target_proportions.loc[gene_symbol].values[None, :]

    interactions = source_proportions_vals @ target_proportions_vals

    plt.title(f"exotranscriptomic {gene_symbol} exchange", fontweight="bold")

    if save:
        figures_path = os.path.join(output_path, "figures")
        os.makedirs(figures_path, exist_ok=True)
        plt.savefig(os.path.join(figures_path, f"communication_profile_{gene_symbol}.{format}"))

    plt.show()
    return pd.DataFrame(interactions, index=source_proportions.columns, columns=target_proportions.columns)


def communication_strength(sdata: SpatialData, source_layer: str = "source_score", target_layer: str = "target_score", copy: bool = False):
    """Compute a 3D interaction strength matrix by multiplying per-transcript source and target scores.

    For each extracellular transcript, the outer product of its source-cell-type
    probability vector and its target-cell-type probability vector is computed.
    The resulting matrix of shape ``(n_transcripts, n_source_types, n_target_types)``
    is stored in ``sdata["source_score"].uns["interaction_strength"]``.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object containing pre-computed ``source_score`` and
        ``target_score`` tables in ``sdata.tables``.
    source_layer : str, optional
        Key for the source-score AnnData in ``sdata.tables``.
        Defaults to ``"source_score"``.
    target_layer : str, optional
        Key for the target-score AnnData in ``sdata.tables``.
        Defaults to ``"target_score"``.
    copy : bool, optional
        If ``True``, return the updated SpatialData object; otherwise modify in
        place and return ``None``. Defaults to ``False``.

    Returns
    -------
    spatialdata.SpatialData or None
        Updated SpatialData with ``interaction_strength`` stored in
        ``sdata["source_score"].uns`` if ``copy=True``; otherwise ``None``.
    """
    source_table = sdata.tables[source_layer]
    target_table = sdata.tables[target_layer]
    interaction_strength = np.empty((source_table.shape[0], source_table.shape[1], target_table.shape[1]))

    for i, id in enumerate(tqdm(source_table.obs.index)):
        matmul_result = np.dot(np.array(source_table[id].X).T, np.array(target_table[id].X))
        interaction_strength[i, :, :] = matmul_result

    sdata["source_score"].uns["interaction_strength"] = interaction_strength

    return deepcopy(sdata) if copy else None


def gene_specific_interactions(sdata, copy: bool = False, gene_key: str = "gene"):
    """Aggregate per-transcript interaction scores into gene-level mean interaction matrices.

    Groups the 3-D ``interaction_strength`` array stored in
    ``sdata["source_score"].uns`` by gene (using ``gene_key`` from
    ``source_score.obs``) and computes the mean matrix per gene, yielding a
    ``(n_genes, n_source_types, n_target_types)`` array.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object containing a pre-computed ``"source_score"`` table with
        ``uns["interaction_strength"]`` (produced by :func:`communication_strength`).
    copy : bool, optional
        If ``True``, return the updated SpatialData object; otherwise modify in
        place and return ``None``. Defaults to ``False``.
    gene_key : str, optional
        Column in ``sdata["source_score"].obs`` containing gene identifiers.
        Defaults to ``"gene"``.

    Returns
    -------
    spatialdata.SpatialData or None
        Updated SpatialData with ``sdata["source_score"].uns["gene_interaction_strength"]``
        (shape ``(n_genes, n_source_types, n_target_types)``) and
        ``sdata["source_score"].uns["gene_interaction_names"]`` (gene name array)
        if ``copy=True``; otherwise ``None``.
    """
    try:
        interaction_strength = sdata["source_score"].uns["interaction_strength"]
        source_table = sdata["source_score"]
    except KeyError as err:
        raise KeyError("Interaction strength has not been computed. Run troutpy.tl.communication_strength first.") from err

    categories = list(source_table.obs[gene_key])  # Extract categories
    unique_cats = np.unique(categories)  # Get unique categories

    # Initialize result 3D matrix (num_categories, H, W)
    result_matrix = np.zeros((len(unique_cats), interaction_strength.shape[1], interaction_strength.shape[2]))

    # Iterate over unique categories
    for i, cat in enumerate(tqdm(unique_cats, desc="Processing categories")):
        mask = np.array(categories) == cat  # Boolean mask
        subset = interaction_strength[mask, :, :]  # Extract relevant slices
        result_matrix[i] = np.mean(subset, axis=0)  # Compute mean over axis 0

    # save resulting matrix
    sdata["source_score"].uns["gene_interaction_strength"] = result_matrix
    sdata["source_score"].uns["gene_interaction_names"] = unique_cats

    return deepcopy(sdata) if copy else None


def cell_contacts_with_urna_sources(
    sdata,
    spatial_key: str = "spatial",
    cell_type_key: str = "cell type",
    distance: float = 50,
    copy: bool = False,
    uns_prefix: str = "cell_contact",
):
    """Compute cell-type x cell-type contact-count matrices based on spatial and uRNA-mediated neighborhoods.

    Three matrices are computed: ``"cell_body"`` (cells within ``distance`` of each
    other), ``"combined"`` (cell-body neighbors plus cells connected via an
    extracellular transcript whose source cell is not already a spatial neighbor of
    its target cell), and ``"urna_specific"`` (``combined`` minus ``cell_body``).
    They are stored in ``sdata["table"].uns`` under
    ``f"{uns_prefix}_cell_body"``, ``f"{uns_prefix}_combined"``, and
    ``f"{uns_prefix}_urna_specific"``.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object containing ``sdata["table"]`` (AnnData with cell type
        annotations and spatial coordinates), ``sdata["target_score"]`` (AnnData with
        ``closest_cell`` and ``distance`` in ``.obs``), and ``sdata["source_score"]``
        (AnnData with ``closest_cell`` in ``.obs``).
    spatial_key : str, optional
        Key in ``sdata["table"].obsm`` containing spatial coordinates. Defaults to
        ``"spatial"``.
    cell_type_key : str, optional
        Column in ``sdata["table"].obs`` with cell type labels. Defaults to
        ``"cell type"``.
    distance : float, optional
        Radius used to define spatial neighborhoods. Defaults to ``50``.
    copy : bool, optional
        If ``True``, return the matrices. Otherwise modify ``sdata`` in place and
        return ``None``. Defaults to ``False``.
    uns_prefix : str, optional
        Prefix for the keys under which matrices are saved in ``sdata["table"].uns``.
        Defaults to ``"cell_contact"``.

    Returns
    -------
    dict of pandas.DataFrame or None
        Dictionary with ``"cell_body"``, ``"combined"``, and ``"urna_specific"``
        contact-count DataFrames (cell types x cell types) if ``copy=True``;
        otherwise ``None``.
    """
    adata = sdata["table"]
    cell_ids = adata.obs_names.values
    cell_types = adata.obs[cell_type_key].values
    unique_cell_types = np.unique(cell_types)

    # Spatial neighbors
    coords = np.asarray(adata.obsm[spatial_key])
    tree = BallTree(coords)
    neighbors_idx = tree.query_radius(coords, r=distance)
    neighbors_dict = {cell_id: cell_ids[neighbors] for cell_id, neighbors in zip(cell_ids, neighbors_idx, strict=False)}

    # Merge source and target transcript info
    target_obs = sdata["target_score"].obs[["closest_cell", "distance"]].copy()
    target_obs.rename(columns={"closest_cell": "target_cell"}, inplace=True)
    source_obs = sdata["source_score"].obs[["closest_cell"]].copy()
    source_obs.rename(columns={"closest_cell": "source_cell"}, inplace=True)

    trans_df = target_obs.join(source_obs, how="inner")
    trans_df = trans_df[trans_df["distance"] <= distance]

    # Keep only transcripts whose source cell is not already a spatial neighbor of the target
    def keep_transcript(row):
        target = row["target_cell"]
        source = row["source_cell"]
        return (source != target) and (source not in neighbors_dict[target])

    filtered_trans = trans_df[trans_df.apply(keep_transcript, axis=1)]

    # Extend spatial neighborhoods with uRNA-connected cells
    extended_neighbors_dict = {cell_id: list(neighs) for cell_id, neighs in neighbors_dict.items()}
    for target_cell, group in filtered_trans.groupby("target_cell"):
        sources = group["source_cell"].values
        extended_neighbors_dict[target_cell] = np.unique(np.concatenate([extended_neighbors_dict[target_cell], sources]))

    cell_type_map = dict(zip(cell_ids, cell_types, strict=False))

    def count_matrix_from_neighbors(neigh_dict):
        mat = pd.DataFrame(0, index=unique_cell_types, columns=unique_cell_types, dtype=int)
        for source_cell, neighbors in neigh_dict.items():
            source_type = cell_type_map[source_cell]
            for target_cell in neighbors:
                try:
                    target_type = cell_type_map[target_cell]
                    mat.loc[source_type, target_type] += 1
                except KeyError:
                    pass
        return mat

    cell_body_mat = count_matrix_from_neighbors(neighbors_dict)
    combined_mat = count_matrix_from_neighbors(extended_neighbors_dict)
    urna_specific_mat = combined_mat - cell_body_mat
    urna_specific_mat[urna_specific_mat < 0] = 0

    adata.uns[f"{uns_prefix}_cell_body"] = cell_body_mat
    adata.uns[f"{uns_prefix}_combined"] = combined_mat
    adata.uns[f"{uns_prefix}_urna_specific"] = urna_specific_mat

    results = {"cell_body": cell_body_mat, "combined": combined_mat, "urna_specific": urna_specific_mat}
    return results if copy else None


def celltype_contact_matrix(
    sdata,
    cell_type_key: str = "leiden",
    radius: float = 50.0,
    min_score: float = 0.1,
    gene_list: str | list[str] | None = None,
    gene_key: str = "gene",
    normalize: bool = False,
    cell_types: list[str] | None = None,
    tile_size: float = 1000.0,
    store_in_sdata: bool = False,
    obsp_key: str = "transcript_contacts",
    obs_key: str = "contact_cell_ids",
) -> pd.DataFrame:
    """Compute a cell-type x cell-type contact matrix from the spatial proximity of "owned" transcripts.

    Each transcript is assigned to an owner cell: either the cell it physically
    overlaps (``overlaps_cell``), or, if unassigned but with ``assignment_score >=
    min_score``, its ``predicted_parent`` cell from
    ``sdata.tables["source_score"]``. Pairs of owned transcripts within ``radius`` of
    each other (found tile-by-tile for memory efficiency) define a directed
    cell-cell contact, which is then aggregated into a ``cell_type_key x
    cell_type_key`` count matrix.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object with a ``"table"`` AnnData (cell metadata, including
        ``cell_id`` and ``cell_type_key`` in ``.obs``), a ``"source_score"`` table
        with ``predicted_parent`` and ``assignment_score`` in ``.obs``, and a
        ``"transcripts"`` points layer with ``x``, ``y``, ``cell_id``,
        ``overlaps_cell``, and ``gene_key`` columns.
    cell_type_key : str, optional
        Column in ``sdata["table"].obs`` with cell type labels. Defaults to
        ``"leiden"``.
    radius : float, optional
        Maximum distance between two owned transcripts for their owner cells to be
        considered in contact. Defaults to ``50.0``.
    min_score : float, optional
        Minimum ``assignment_score`` for an unassigned transcript to be attributed to
        its ``predicted_parent`` cell. Defaults to ``0.1``.
    gene_list : str or list of str, optional
        If given, restrict the analysis to transcripts of this gene (or genes).
        Defaults to ``None`` (all genes).
    gene_key : str, optional
        Column in the transcripts table holding gene identity. Defaults to ``"gene"``.
    normalize : bool, optional
        If ``True``, normalize each row of the output matrix to sum to 1. Defaults to
        ``False``.
    cell_types : list of str, optional
        If given, restrict owners and contacts to these cell types. Defaults to
        ``None`` (all cell types).
    tile_size : float, optional
        Side length of the square tiles used to find nearby transcript pairs.
        Defaults to ``1000.0``.
    store_in_sdata : bool, optional
        If ``True``, also store a per-cell sparse adjacency matrix in
        ``sdata["table"].obsp[obsp_key]`` and a per-cell neighbor-id string in
        ``sdata["table"].obs[obs_key]``. Defaults to ``False``.
    obsp_key : str, optional
        Key for the sparse adjacency matrix in ``sdata["table"].obsp`` when
        ``store_in_sdata=True``. Defaults to ``"transcript_contacts"``.
    obs_key : str, optional
        Key for the per-cell neighbor-id string in ``sdata["table"].obs`` when
        ``store_in_sdata=True``. Defaults to ``"contact_cell_ids"``.

    Returns
    -------
    pandas.DataFrame
        ``cell_type_key x cell_type_key`` matrix of directed contact counts (or
        row-normalized proportions if ``normalize=True``).
    """
    cells = sdata["table"]
    res_obs = sdata.tables["source_score"].obs.copy()
    transcripts = sdata.points["transcripts"].compute()

    transcripts.index = transcripts.index.astype(int)
    res_obs.index = res_obs.index.astype(int)

    if gene_list is not None:
        if isinstance(gene_list, str):
            gene_list = [gene_list]
        transcripts = transcripts[transcripts[gene_key].isin(gene_list)].copy()
        if transcripts.empty:
            print(f"No transcripts found for gene(s): {gene_list}")
            return pd.DataFrame()

    # Cell type map and integer index of each cell (needed for the obsp matrix)
    cell_obs = cells.obs.copy()
    cell_obs["cell_id"] = cell_obs["cell_id"].astype(str).str.strip()
    cell_type_map = dict(zip(cell_obs["cell_id"], cell_obs[cell_type_key].astype(str), strict=False))
    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_obs["cell_id"])}
    n_cells = len(cell_obs)

    allowed = set(cell_types) if cell_types is not None else None

    print("Assigning transcript owners...")
    transcripts = transcripts.join(res_obs[["predicted_parent", "assignment_score"]], how="left")
    transcripts["cell_id_str"] = transcripts["cell_id"].astype(str).str.strip()
    transcripts["parent_id_str"] = transcripts["predicted_parent"].astype(str).str.strip()
    transcripts["overlaps_cell_bool"] = transcripts["overlaps_cell"].astype(str).str.lower() == "true"

    is_body = transcripts["overlaps_cell_bool"] & transcripts["cell_id_str"].isin(cell_type_map)
    is_halo = ~transcripts["overlaps_cell_bool"] & (transcripts["assignment_score"] >= min_score) & transcripts["parent_id_str"].isin(cell_type_map)

    transcripts["owner"] = np.where(is_body, transcripts["cell_id_str"], np.where(is_halo, transcripts["parent_id_str"], np.nan))

    owned = transcripts[transcripts["owner"].notna()].copy()
    if allowed is not None:
        owned = owned[owned["owner"].map(cell_type_map).isin(allowed)]

    coords = owned[["x", "y"]].values
    owners = owned["owner"].values

    print(f"Owned transcripts: {len(owned):,} / {len(transcripts):,} total")

    # Find nearby transcript pairs tile-by-tile to limit peak memory usage
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    x_starts = np.arange(x_min, x_max, tile_size)
    y_starts = np.arange(y_min, y_max, tile_size)
    tiles = [(x0, y0) for x0 in x_starts for y0 in y_starts]

    all_cell_pairs = set()  # directed (cell_id_a, cell_id_b) pairs

    if store_in_sdata:
        adj = lil_matrix((n_cells, n_cells), dtype=np.int8)

    for x0, y0 in tqdm(tiles, desc=f"Processing tiles ({len(x_starts)}x{len(y_starts)} grid)", unit="tile"):
        x1 = x0 + tile_size
        y1 = y0 + tile_size

        mask = (coords[:, 0] >= x0 - radius) & (coords[:, 0] < x1 + radius) & (coords[:, 1] >= y0 - radius) & (coords[:, 1] < y1 + radius)
        tile_coords = coords[mask]
        tile_owners = owners[mask]

        if len(tile_coords) < 2:
            continue

        tree = cKDTree(tile_coords)
        pairs = tree.query_pairs(r=radius, output_type="ndarray")

        if len(pairs) == 0:
            continue

        src_owner = tile_owners[pairs[:, 0]]
        nb_owner = tile_owners[pairs[:, 1]]

        diff = src_owner != nb_owner
        src_owner, nb_owner = src_owner[diff], nb_owner[diff]

        if len(src_owner) == 0:
            continue

        for a, b in zip(src_owner, nb_owner, strict=False):
            all_cell_pairs.add((a, b))
            all_cell_pairs.add((b, a))

            if store_in_sdata:
                i = cell_id_to_idx.get(a)
                j = cell_id_to_idx.get(b)
                if i is not None and j is not None:
                    adj[i, j] = 1
                    adj[j, i] = 1

    print(f"Unique directed cell-cell contacts found: {len(all_cell_pairs):,}")

    if store_in_sdata:
        print("Storing contacts in sdata['table']...")

        sdata["table"].obsp[obsp_key] = adj.tocsr()

        neighbours = defaultdict(set)
        for a, b in all_cell_pairs:
            neighbours[a].add(b)

        sdata["table"].obs[obs_key] = [";".join(sorted(neighbours[cid])) if cid in neighbours else "" for cid in cell_obs["cell_id"]]

        print(f"  -> sdata['table'].obsp['{obsp_key}']  shape: {sdata['table'].obsp[obsp_key].shape}")
        print(f"  -> sdata['table'].obs['{obs_key}']   (first 3): {sdata['table'].obs[obs_key].head(3).tolist()}")

    if not all_cell_pairs:
        all_types = sorted({cell_type_map[c] for c in owned["owner"].unique() if c in cell_type_map})
        return pd.DataFrame(0, index=all_types, columns=all_types)

    print("Building contact matrix...")
    cell_pairs = pd.DataFrame(list(all_cell_pairs), columns=["src", "nb"])
    cell_pairs["src_type"] = cell_pairs["src"].map(cell_type_map)
    cell_pairs["nb_type"] = cell_pairs["nb"].map(cell_type_map)
    cell_pairs = cell_pairs.dropna(subset=["src_type", "nb_type"])

    if allowed is not None:
        cell_pairs = cell_pairs[cell_pairs["src_type"].isin(allowed) & cell_pairs["nb_type"].isin(allowed)]

    all_types = sorted(cell_pairs[["src_type", "nb_type"]].stack().unique())
    type_index = {t: i for i, t in enumerate(all_types)}
    n = len(all_types)

    contact_matrix = np.zeros((n, n), dtype=int)
    src_idx = cell_pairs["src_type"].map(type_index).values.astype(int)
    nb_idx = cell_pairs["nb_type"].map(type_index).values.astype(int)
    np.add.at(contact_matrix, (src_idx, nb_idx), 1)

    df = pd.DataFrame(contact_matrix, index=all_types, columns=all_types)
    df.index.name = "source_cell_type"
    df.columns.name = "neighbour_cell_type"

    if normalize:
        row_sums = df.sum(axis=1).replace(0, np.nan)
        df = df.div(row_sums, axis=0).fillna(0)

    print("Done.")
    return df
