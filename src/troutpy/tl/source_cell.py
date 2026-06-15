import warnings

import numba as nb
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree, cKDTree
from spatialdata import SpatialData
from spatialdata.models import TableModel
from tqdm import tqdm

warnings.filterwarnings("ignore")


def create_xrna_metadata(sdata: SpatialData, layer: str = "transcripts", gene_key: str = "gene", copy: bool = False) -> SpatialData | None:
    """Create the ``"xrna_metadata"`` table holding the unique genes found in a transcripts layer.

    Parameters
    ----------
    sdata
        The SpatialData object to modify.
    layer
        The name of the layer in `sdata.points` from which to extract gene names.
    gene_key
        The column in `sdata.points[layer]` that contains the gene names.
    copy
        If `True`, return a copy of `sdata` with the new table added. Otherwise modify in place.

    Returns
    -------
    If `copy=True`, a copy of the modified `sdata`. Otherwise `None`, modifying `sdata` in place.
    """
    if layer not in sdata.points:
        raise ValueError(f"Points layer '{layer}' not found in sdata.points.")

    points_data = sdata.points[layer]
    if gene_key not in points_data.columns:
        raise ValueError(f"The specified points layer '{layer}' does not contain a '{gene_key}' column.")

    unique_genes = points_data[gene_key].compute().unique().astype(str)
    gene_metadata = pd.DataFrame(index=unique_genes)

    exrna_adata = sc.AnnData(var=gene_metadata)
    metadata_table = TableModel.parse(exrna_adata)
    sdata.tables["xrna_metadata"] = metadata_table

    print(f"Added 'xrna_metadata' table with {len(unique_genes)} unique genes to the SpatialData object.")

    return sdata if copy else None


def get_proportion_expressed_per_cell_type(adata: sc.AnnData, cell_type_key: str = "cell type") -> pd.DataFrame:
    """Calculate the mean expression of each feature (gene) per cell type.

    Parameters
    ----------
    adata
        AnnData object containing the single-cell or spatial transcriptomics dataset.
        The `obs` attribute should contain cell type annotations.
    cell_type_key
        Key in `adata.obs` corresponding to cell type annotations.

    Returns
    -------
    DataFrame where rows correspond to features (genes) and columns correspond to cell
    types. Each entry represents the mean expression of the feature in the
    specified cell type.
    """
    cell_types = adata.obs[cell_type_key].unique().dropna()
    proportions = pd.DataFrame(index=adata.var_names, columns=cell_types)
    for cell_type in cell_types:
        proportions[cell_type] = adata[adata.obs[cell_type_key] == cell_type].X.mean(axis=0).T
    return proportions


def extract_expression_matrix(adata):
    """Extract the expression matrix from an AnnData object as a dense pandas DataFrame.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object whose ``.X`` matrix (dense or sparse) will be extracted.

    Returns
    -------
    pandas.DataFrame
        Dense expression matrix with observation names as the index and variable
        names as columns.
    """
    return pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        index=adata.obs_names,
        columns=adata.var_names,
    )


def get_extracellular_transcripts(transcripts):
    """Filter the transcript table to retain only extracellular transcripts.

    Parameters
    ----------
    transcripts : pandas.DataFrame
        Full transcript table. Must contain an ``"extracellular"`` boolean column.

    Returns
    -------
    pandas.DataFrame
        Subset of ``transcripts`` where ``extracellular`` is ``True``.

    Raises
    ------
    ValueError
        If the ``"extracellular"`` column is absent or no extracellular transcripts
        are found.
    """
    if "extracellular" not in transcripts.columns:
        raise ValueError("Column 'extracellular' missing from transcript table.")
    extracellular_transcripts = transcripts[transcripts["extracellular"]]
    if extracellular_transcripts.empty:
        raise ValueError("No extracellular transcripts found.")
    return extracellular_transcripts


def process_gene(
    gene_name: str,
    gene_transcripts: pd.DataFrame,
    gene_col_sparse,
    xcoord: str,
    ycoord: str,
    lambda_decay: float,
    all_cell_types: np.ndarray,
    cell_type_to_idx: dict,
    all_cell_ids: np.ndarray,
    high_res_tree: cKDTree,
    it_cell_ids: np.ndarray,
    cell_id_to_int_idx: dict,
    cell_type_mapping: dict,
):
    """Assign one gene's extracellular transcripts to nearby expressing cells.

    For each extracellular transcript, the nearest intracellular transcripts (from
    `high_res_tree`) are queried, filtered to cells expressing `gene_name`, and
    converted into per-cell-type and per-cell scores using exponential distance decay.

    Parameters
    ----------
    gene_name
        Name of the gene being processed.
    gene_transcripts
        Extracellular transcript table for this gene.
    gene_col_sparse
        Sparse column vector of this gene's expression across all cells.
    xcoord
        Column in `gene_transcripts` for the x-coordinate.
    ycoord
        Column in `gene_transcripts` for the y-coordinate.
    lambda_decay
        Decay rate for the exponential distance weighting.
    all_cell_types
        Ordered array of all unique cell-type labels defining the score-matrix columns.
    cell_type_to_idx
        Mapping from cell-type label to its column index in `all_cell_types`.
    all_cell_ids
        Observation names of all cells, defining the index of the returned per-cell scores.
    high_res_tree
        KDTree built on intracellular transcript coordinates.
    it_cell_ids
        Cell IDs corresponding to each point in `high_res_tree`.
    cell_id_to_int_idx
        Mapping from cell ID to its row index in `gene_col_sparse`.
    cell_type_mapping
        Mapping from cell ID to cell-type label.

    Returns
    -------
    prob_df : pandas.DataFrame
        Transcript x cell-type score matrix for this gene.
    closest_df : pandas.DataFrame
        Distance, ID, and cell type of the closest expression-filtered cell per transcript.
    gene_by_cell_scores : pandas.Series
        Cumulative score per cell for this gene.
    gene_name : str
        The gene name, echoed back for downstream bookkeeping.
    """
    gene_expr = gene_col_sparse.toarray().flatten()
    t_coords = gene_transcripts[[xcoord, ycoord]].values

    k_search = min(200, len(it_cell_ids))
    dists, indices = high_res_tree.query(t_coords, k=k_search)

    n_transcripts = len(gene_transcripts)
    prob_mat = np.zeros((n_transcripts, len(all_cell_types)))
    gene_by_cell_scores = pd.Series(0.0, index=all_cell_ids)

    closest_data = []

    for i in range(n_transcripts):
        neighbor_transcript_indices = indices[i]
        neighbor_dists = dists[i]
        neighbor_cell_ids = it_cell_ids[neighbor_transcript_indices]

        abs_closest_cell = neighbor_cell_ids[0]
        abs_closest_dist = neighbor_dists[0]
        abs_closest_type = cell_type_mapping.get(abs_closest_cell, "unknown")

        unique_cells = {}
        for cell_id, d in zip(neighbor_cell_ids, neighbor_dists, strict=False):
            if cell_id not in unique_cells:
                cell_idx = cell_id_to_int_idx.get(cell_id)
                if cell_idx is not None and gene_expr[cell_idx] > 0:
                    unique_cells[cell_id] = (d, cell_idx)
            if len(unique_cells) >= 30:
                break

        # Fall back to the nearest physical cell so downstream diffusion analysis
        # never sees a NaN distance, even when no nearby cell expresses this gene.
        if not unique_cells:
            closest_data.append([abs_closest_dist, abs_closest_cell, abs_closest_type])
            continue

        cell_ids = list(unique_cells.keys())
        dists_to_use = np.array([v[0] for v in unique_cells.values()])
        idxs_to_use = np.array([v[1] for v in unique_cells.values()])

        cell_weights = np.exp(-lambda_decay * dists_to_use) * gene_expr[idxs_to_use]
        total_weight = cell_weights.sum() + 1e-9
        norm_weights = cell_weights / total_weight

        for cell_id, weight in zip(cell_ids, norm_weights, strict=False):
            c_type = cell_type_mapping.get(cell_id, "unknown")
            if c_type in cell_type_to_idx:
                prob_mat[i, cell_type_to_idx[c_type]] += weight
            gene_by_cell_scores[cell_id] += weight

        closest_data.append([dists_to_use[0], cell_ids[0], cell_type_mapping.get(cell_ids[0], "unknown")])

    prob_df = pd.DataFrame(prob_mat, columns=all_cell_types, index=gene_transcripts.index)
    closest_df = pd.DataFrame(closest_data, columns=["distance", "closest_cell", "closest_celltype"], index=gene_transcripts.index)

    return prob_df, closest_df, gene_by_cell_scores, gene_name


def store_results_in_sdata(
    sdata: SpatialData,
    prob_table: pd.DataFrame,
    closest_table: pd.DataFrame,
    et: pd.DataFrame,
    x: str,
    y: str,
    gene_key: str,
    cell_source_table: pd.DataFrame,
) -> None:
    """Store source-score results in the SpatialData object.

    Creates a ``"source_score"`` AnnData table (transcripts x cell-types) in
    ``sdata.tables``, adds per-transcript metadata (gene, distance, closest cell),
    and writes a cell-level summary score and its normalized variant to
    ``sdata["table"].obs``.

    Parameters
    ----------
    sdata
        SpatialData object to update in place.
    prob_table
        Transcript x cell-type probability DataFrame.
    closest_table
        DataFrame with columns ``"distance"``, ``"closest_cell"``, and
        ``"closest_celltype"`` indexed by extracellular transcript index.
    et
        Extracellular transcript table used to attach spatial coordinates and gene labels.
    x
        Column name for the x-coordinate in `et`.
    y
        Column name for the y-coordinate in `et`.
    gene_key
        Column name in `et` containing gene identifiers.
    cell_source_table
        Cells x genes matrix of per-cell cumulative source scores.

    Returns
    -------
    None
    """
    prob_adata = sc.AnnData(prob_table.values)
    prob_adata.var_names = prob_table.columns
    prob_adata.obs_names = prob_table.index

    prob_adata.obs[gene_key] = et[gene_key].values
    prob_adata.obs["distance"] = closest_table["distance"].values
    prob_adata.obs["closest_cell"] = closest_table["closest_cell"].values
    prob_adata.obs["closest_celltype"] = closest_table["closest_celltype"].values
    prob_adata.obsm["spatial"] = et[[x, y]].to_numpy()

    sdata.tables["source_score"] = prob_adata
    sdata["table"].obs["urna_source_score"] = cell_source_table.sum(axis=1).values

    try:
        adata = sdata["table"]
        if "raw" in adata.layers:
            raw_sums = np.array(adata.layers["raw"].sum(axis=1)).flatten()
            sdata["table"].obs["normalized_urna_source_score"] = adata.obs["urna_source_score"] / (raw_sums + 1e-9)
    except Exception as e:  # noqa: BLE001 - best-effort: skip normalization if "raw" layer / sums are unusable
        print(f"Could not compute normalized scores: {e}")


def compute_probability_table(scores, cell_indices, cell_types_filt, all_cell_types, gene_transcripts):
    """Build a transcript × cell-type score probability table from per-transcript neighbor scores.

    Parameters
    ----------
    scores : numpy.ndarray
        2-D array of shape ``(n_transcripts, k)`` with exponential-decay weights for
        the k nearest neighbor cells.
    cell_indices : numpy.ndarray
        2-D integer array of shape ``(n_transcripts, k)`` mapping each weight to a
        row in ``cell_types_filt``.
    cell_types_filt : pandas.Series
        Cell-type label for each cell, indexed by cell observation name.
    all_cell_types : array-like
        Ordered sequence of all unique cell-type labels defining the column order.
    gene_transcripts : pandas.DataFrame
        Extracellular transcript table for the current gene; its index becomes the
        row index of the output.

    Returns
    -------
    pandas.DataFrame
        Transcript × cell-type probability matrix indexed by ``gene_transcripts.index``.
    """
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
    """Construct a DataFrame with the closest-cell distance, ID, and cell-type for each transcript.

    Parameters
    ----------
    distances : array-like
        Distance from each transcript to its single closest expressing cell.
    closest_cell_ids : array-like
        Cell observation names corresponding to the closest cell per transcript.
    closest_cell_types : array-like
        Cell-type labels for the closest cell per transcript.
    gene_transcripts : pandas.DataFrame
        Extracellular transcript table for the current gene; its index is used as
        the output row index.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``"distance"``, ``"closest_cell"``, and
        ``"closest_celltype"``, indexed by ``gene_transcripts.index``.
    """
    return pd.DataFrame(
        {"distance": distances, "closest_cell": closest_cell_ids, "closest_celltype": closest_cell_types},
        index=gene_transcripts.index,
    )


def sum_scores_per_cell(gene_cells, scores, cell_indices):
    """Aggregate per-transcript source scores into a per-cell total for a single gene.

    Parameters
    ----------
    gene_cells : pandas.DataFrame
        Cell table slice for the gene; its index provides the cell observation names.
    scores : numpy.ndarray
        2-D array of shape ``(n_transcripts, k)`` with per-transcript neighbor weights.
    cell_indices : numpy.ndarray
        2-D integer array of shape ``(n_transcripts, k)`` mapping weights to rows of
        ``gene_cells``.

    Returns
    -------
    gene_by_cell_scores : pandas.Series
        Cumulative source score per cell, indexed by cell observation name.
    """
    all_cell_ids = gene_cells.index.to_numpy()
    gene_by_cell_scores = pd.Series(0.0, index=all_cell_ids)

    for i in range(len(scores)):
        cell_idx_i = cell_indices[i]
        scores_i = scores[i]
        cell_ids_i = gene_cells.index[cell_idx_i]
        gene_by_cell_scores[cell_ids_i] += scores_i

    return gene_by_cell_scores


def build_result_tables(results, extracellular_transcripts, adata, genes_to_process, all_cell_types):
    """Assemble per-gene results into transcript- and cell-level summary tables.

    Parameters
    ----------
    results : list
        Per-gene results, each a 3-tuple ``(gene_prob_df, gene_closest_df, gene_by_cell_df)``
        as produced by the score-computation loop, or `None` for skipped genes.
    extracellular_transcripts : pandas.DataFrame
        Full extracellular transcript table; its index defines the row index of
        `prob_table` and `closest_table`.
    adata : anndata.AnnData
        Cell table; its observation names define the index of `cell_score_table`.
    genes_to_process : list
        Gene names corresponding to the entries in `results`.
    all_cell_types : array-like
        Ordered sequence of all unique cell-type labels defining `prob_table`'s columns.

    Returns
    -------
    prob_table : pandas.DataFrame
        Transcript x cell-type probability matrix.
    closest_table : pandas.DataFrame
        Per-transcript closest-cell distance, ID, and cell type.
    cell_score_table : pandas.DataFrame
        Cell x gene matrix of cumulative per-cell source scores.
    """
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


def compute_contribution_score(sdata):
    """Compute a normalised extracellular RNA (uRNA) contribution score for each cell.

    For each gene, a cell's contribution is weighted by the total extracellular count
    of that gene divided by the number of cells expressing it. Scores are summed
    across genes and also normalised by each cell's total raw expression. Results
    are written to ``sdata["table"].obs``.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object containing a ``"table"`` AnnData (with a ``"raw"`` layer)
        and an ``"xrna_metadata"`` table whose ``var`` DataFrame includes a
        ``"count"`` column with per-gene extracellular transcript counts.

    Returns
    -------
    None
        Scores are added in place to ``sdata["table"].obs`` as
        ``"urna_contribution_score"`` and ``"normalized_urna_contribution_score"``.

    Raises
    ------
    ValueError
        If no common genes are found between the cell table and the gene metadata.
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
    adata.obs["normalized_urna_contribution_score"] = score / np.sum(raw_expr, axis=1)


def adaptative_source_score(
    sdata: SpatialData,
    max_dist: float = 200,
    lmbda: float = 0.1,
    max_k: int = 15,
    ambient_floor: float = 1.5,
    signal_threshold: float = 3.0,
    residual: float = 0.1,
    cell_type_col: str = "leiden",
    copy: bool = False,
) -> SpatialData | None:
    """Score extracellular transcripts against nearby cells using an adaptive neighborhood size.

    For each extracellular transcript, a context gene set is built from its `k_adaptive`
    nearest extracellular neighbors (where `k_adaptive` grows with the transcript's
    ``"enrichment_over_random"``), then candidate parent cells within `max_dist` are
    weighted by how well their expression matches that context and by exponential
    distance decay.

    Parameters
    ----------
    sdata
        SpatialData object containing a ``"table"`` AnnData and a ``"transcripts"``
        points layer with ``"overlaps_cell"``, ``"enrichment_over_random"``, ``"gene"``,
        ``"x"``, and ``"y"`` columns.
    max_dist
        Maximum distance (in coordinate units) to search for candidate parent cells.
    lmbda
        Decay rate for the exponential distance weighting.
    max_k
        Maximum number of extracellular neighbors used to build the context gene set.
    ambient_floor
        ``"enrichment_over_random"`` value at or below which `k_adaptive` is 1.
    signal_threshold
        ``"enrichment_over_random"`` value at or above which `k_adaptive` is `max_k`.
    residual
        Constant added to the denominator when normalizing candidate weights, preventing
        division by zero and capping the maximum normalized score.
    cell_type_col
        Column in ``sdata["table"].obs`` containing cell-type annotations.
    copy
        If `True`, return `sdata`. Otherwise modify in place and return `None`.

    Returns
    -------
    If `copy=True`, `sdata`. Otherwise `None`, modifying `sdata` in place.

    Notes
    -----
    Adds a ``"source_score"`` table to `sdata`; ``"urna_source_score"`` /
    ``"normalized_urna_source_score"`` columns to ``sdata["table"].obs``; and
    ``"aggregated_source_score"`` / ``"mean_assignment_score"`` columns to
    ``sdata["xrna_metadata"].var``. This is one of two overlapping source-scoring
    implementations in this module, alongside :func:`adaptative_source_score_optimized`.
    """
    cells = sdata["table"]
    transcripts = sdata.points["transcripts"].compute()

    cell_types = cells.obs[cell_type_col].values
    unique_types = np.unique(cell_types)

    common_genes = list(cells.var_names.intersection(transcripts["gene"].unique()))
    gene_to_idx = {g: i for i, g in enumerate(common_genes)}
    cell_exp = csr_matrix(cells[:, common_genes].X > 0)

    is_extra = ~transcripts["overlaps_cell"]
    et_df = transcripts[is_extra].copy()

    cell_tree = KDTree(cells.obsm["spatial"])
    et_tree = KDTree(et_df[["x", "y"]].values)

    enrichment = et_df["enrichment_over_random"].values
    coords = et_df[["x", "y"]].values
    genes = et_df["gene"].values

    prob_mat = np.zeros((len(et_df), len(unique_types)))
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    cell_source_scores = pd.Series(0.0, index=cells.obs_names)

    results = []

    for i in tqdm(range(len(et_df)), desc="Adaptive Source Scoring"):
        p_coord, p_enrich, p_gene = coords[i], enrichment[i], genes[i]

        if p_enrich <= ambient_floor:
            k_adaptive = 1
        elif p_enrich >= signal_threshold:
            k_adaptive = max_k
        else:
            fraction = (p_enrich - ambient_floor) / (signal_threshold - ambient_floor)
            k_adaptive = int(1 + (max_k - 1) * fraction)

        if k_adaptive <= 1:
            query_genes = [p_gene]
        else:
            _, neighbors = et_tree.query(p_coord, k=k_adaptive)
            query_genes = list(set(genes[neighbors]))

        query_mask = np.zeros(len(common_genes), dtype=bool)
        for g in query_genes:
            if g in gene_to_idx:
                query_mask[gene_to_idx[g]] = True
        total_bits = query_mask.sum()

        dists, idxs = cell_tree.query(p_coord, k=10, distance_upper_bound=max_dist)
        valid = np.isfinite(dists)

        best_cell_id, best_cell_score, best_dist = "unassigned", 0.0, 0.0

        if np.any(valid):
            raw_weights = []
            for d, c_idx in zip(dists[valid], idxs[valid], strict=False):
                c_mask = cell_exp[c_idx].toarray().flatten()
                match = (np.logical_and(query_mask, c_mask).sum() / total_bits) if total_bits > 0 else 0
                weight = match * np.exp(-lmbda * d)

                raw_weights.append({"weight": weight, "id": cells.obs_names[c_idx], "type": cell_types[c_idx], "dist": d})

            total_sum = sum(w["weight"] for w in raw_weights) + residual

            for w_dict in raw_weights:
                norm_score = w_dict["weight"] / total_sum
                prob_mat[i, type_to_idx[w_dict["type"]]] += norm_score
                cell_source_scores[w_dict["id"]] += norm_score

                if norm_score > best_cell_score:
                    best_cell_score = norm_score
                    best_cell_id = w_dict["id"]
                    best_dist = w_dict["dist"]

        results.append({"predicted_parent": best_cell_id, "distance": best_dist})

    # The "source_score" table is the primary container for per-transcript results
    prob_adata = sc.AnnData(prob_mat)
    prob_adata.var_names = unique_types
    prob_adata.obs_names = et_df.index

    res_df = pd.DataFrame(results, index=et_df.index)
    prob_adata.obs["gene"] = et_df["gene"].values
    prob_adata.obs["predicted_parent"] = res_df["predicted_parent"].values
    prob_adata.obs["distance_to_source"] = res_df["distance"].values
    prob_adata.obs["assignment_score"] = prob_mat.max(axis=1)
    prob_adata.obsm["spatial"] = et_df[["x", "y"]].to_numpy()

    sdata.tables["source_score"] = prob_adata

    sdata["table"].obs["urna_source_score"] = cell_source_scores.values
    try:
        raw_sums = np.array(sdata["table"].X.sum(axis=1)).flatten()
        sdata["table"].obs["normalized_urna_source_score"] = sdata["table"].obs["urna_source_score"] / (raw_sums + 1e-9)
    except Exception:  # noqa: BLE001 - best-effort: skip normalization if X sums are unusable
        pass

    agg_source = prob_adata.obs.assign(score=np.array(prob_adata.X.sum(axis=1)).flatten()).groupby("gene")["score"].mean()
    mean_assign = prob_adata.obs.groupby("gene")["assignment_score"].mean()

    sdata["xrna_metadata"].var["aggregated_source_score"] = sdata["xrna_metadata"].var.index.map(agg_source)
    sdata["xrna_metadata"].var["mean_assignment_score"] = sdata["xrna_metadata"].var.index.map(mean_assign)

    return sdata if copy else None


@nb.njit(parallel=True, fastmath=True)
def _core_scoring_engine_chunk(
    et_coords,
    et_gene_idxs,
    et_neighs,
    k_vals,
    shell_coords,
    shell_row_indices,
    shell_cell_types_idx,
    shell_neighbor_indices,
    shell_neighbor_ptr,
    csr_data,
    csr_indices,
    csr_indptr,
    lmbda,
    residual,
    n_types,
):
    """Score a chunk of extracellular transcripts against candidate parent cells (numba kernel).

    For each transcript, candidate cells from its spatial "shell" neighborhood are
    weighted by the fraction of the transcript's neighbor genes (`et_neighs`,
    truncated to `k_vals`) found in that cell's expression profile (via binary
    search over `csr_indices`) combined with exponential distance decay. Used by
    :func:`adaptative_source_score_optimized`.

    Returns
    -------
    prob_mat : numpy.ndarray
        Transcript x cell-type matrix of summed normalized weights.
    best_parent_rows : numpy.ndarray
        Row index (into the cell table) of the highest-weighted candidate per transcript, or -1.
    best_scores : numpy.ndarray
        Normalized weight of the highest-weighted candidate per transcript.
    best_distances : numpy.ndarray
        Distance to the highest-weighted candidate per transcript.
    """
    n_transcripts = len(et_coords)
    prob_mat = np.zeros((n_transcripts, n_types), dtype=np.float32)
    best_parent_rows = np.full(n_transcripts, -1, dtype=np.int32)
    best_scores = np.zeros(n_transcripts, dtype=np.float32)
    best_distances = np.zeros(n_transcripts, dtype=np.float32)

    for i in nb.prange(n_transcripts):
        # Context genes from this transcript's nearest extracellular neighbors
        k = k_vals[i]
        q_genes = [et_gene_idxs[idx] for idx in et_neighs[i, :k] if et_gene_idxs[idx] != -1]
        n_q = len(q_genes)
        if n_q == 0:
            continue

        # Candidate parent cells from the precomputed spatial shell
        s_start, s_end = shell_neighbor_ptr[i], shell_neighbor_ptr[i + 1]
        if s_start == s_end:
            continue

        sum_w = residual
        n_candidates = s_end - s_start
        weights = np.zeros(n_candidates, dtype=np.float32)
        dists = np.zeros(n_candidates, dtype=np.float32)

        for sj_idx in range(n_candidates):
            s_idx = shell_neighbor_indices[s_start + sj_idx]
            c_row = shell_row_indices[s_idx]

            dx = et_coords[i, 0] - shell_coords[s_idx, 0]
            dy = et_coords[i, 1] - shell_coords[s_idx, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            dists[sj_idx] = dist

            # Binary search the cell's CSR row for matches against the context genes
            match_count = 0
            r_start, r_end = csr_indptr[c_row], csr_indptr[c_row + 1]
            for g_idx in q_genes:
                low, high = r_start, r_end - 1
                while low <= high:
                    mid = (low + high) // 2
                    if csr_indices[mid] < g_idx:
                        low = mid + 1
                    elif csr_indices[mid] > g_idx:
                        high = mid - 1
                    else:
                        match_count += 1
                        break

            w = (match_count / n_q) * np.exp(-lmbda * dist)
            weights[sj_idx] = w
            sum_w += w

        # Normalize weights and record per-cell-type sums and the best parent
        max_w_norm = -1.0
        for sj_idx in range(n_candidates):
            norm_w = weights[sj_idx] / sum_w
            if norm_w > 0:
                s_idx = shell_neighbor_indices[s_start + sj_idx]
                t_idx = shell_cell_types_idx[s_idx]
                prob_mat[i, t_idx] += norm_w

                if norm_w > max_w_norm:
                    max_w_norm = norm_w
                    best_parent_rows[i] = shell_row_indices[s_idx]
                    best_scores[i] = norm_w
                    best_distances[i] = dists[sj_idx]

    return prob_mat, best_parent_rows, best_scores, best_distances


def adaptative_source_score_optimized(
    sdata: SpatialData,
    chunk_size: int = 100000,
    max_dist: float = 100,
    lmbda: float = 0.1,
    max_k: int = 10,
    ambient_floor: float = 1.0,
    signal_threshold: float = 10,
    residual: float = 0.1,
    cell_type_col: str = "leiden",
    copy: bool = False,
) -> SpatialData | None:
    """Score extracellular transcripts against nearby cells using a chunked, numba-accelerated kernel.

    A faster, chunked reimplementation of :func:`adaptative_source_score`. Cells are
    represented by a "shell" of their outermost assigned transcripts, and each
    extracellular transcript is scored against shell points within `max_dist` using
    `_core_scoring_engine_chunk`, processing `chunk_size` transcripts at a time.

    Parameters
    ----------
    sdata
        SpatialData object containing a ``"table"`` AnnData (with ``"cell_id"`` in
        ``.obs``) and a ``"transcripts"`` points layer with ``"cell_id"``,
        ``"enrichment_over_random"``, ``"gene"``, ``"x"``, and ``"y"`` columns.
    chunk_size
        Number of extracellular transcripts processed per chunk.
    max_dist
        Maximum distance (in coordinate units) to search for candidate parent cells.
    lmbda
        Decay rate for the exponential distance weighting.
    max_k
        Maximum number of extracellular neighbors used to build the context gene set.
    ambient_floor
        ``"enrichment_over_random"`` value at or below which `k_adaptive` is 1.
    signal_threshold
        ``"enrichment_over_random"`` value at or above which `k_adaptive` is `max_k`.
    residual
        Constant added to the denominator when normalizing candidate weights.
    cell_type_col
        Column in ``sdata["table"].obs`` containing cell-type annotations.
    copy
        If `True`, return `sdata`. Otherwise modify in place and return `None`.

    Returns
    -------
    If `copy=True`, `sdata`. Otherwise `None`, modifying `sdata` in place.

    Notes
    -----
    Adds a ``"source_score"`` table to `sdata`; an ``"urna_source_score"`` column to
    ``sdata["table"].obs``; and ``"aggregated_source_score"`` /
    ``"mean_assignment_score"`` columns to ``sdata["xrna_metadata"].var``. This is one
    of two overlapping source-scoring implementations in this module, alongside
    :func:`adaptative_source_score`.
    """
    print("Aligning data and cleaning IDs...")
    cells = sdata["table"]
    obs = cells.obs.copy()
    obs["cell_id"] = obs["cell_id"].astype(str).str.strip()

    unique_types = np.sort(obs[cell_type_col].unique())
    type_to_idx = {t: i for i, t in enumerate(unique_types)}

    transcripts = sdata.points["transcripts"].compute()
    transcripts["cell_id"] = transcripts["cell_id"].astype(str).str.strip()

    valid_ids = set(obs["cell_id"])
    is_assigned = transcripts["cell_id"].isin(valid_ids)

    et_df = transcripts[~is_assigned].copy()
    assigned_df = transcripts[is_assigned].copy()

    # Build a global "shell" of the outermost transcripts per cell, used as candidate
    # parent locations for the kernel's spatial neighbor search.
    print("Building global spatial shell...")
    shell_idx = pd.concat(
        [
            assigned_df.groupby("cell_id")["x"].idxmin(),
            assigned_df.groupby("cell_id")["x"].idxmax(),
            assigned_df.groupby("cell_id")["y"].idxmin(),
            assigned_df.groupby("cell_id")["y"].idxmax(),
        ]
    ).unique()
    shell_df = assigned_df.loc[shell_idx].copy()

    cell_id_to_row = {cid: i for i, cid in enumerate(obs["cell_id"])}
    shell_coords = shell_df[["x", "y"]].values.astype(np.float32)
    shell_row_indices = np.array([cell_id_to_row[cid] for cid in shell_df["cell_id"]], dtype=np.int32)
    shell_type_indices = np.array([type_to_idx[obs.iloc[r][cell_type_col]] for r in shell_row_indices], dtype=np.int32)
    shell_tree = cKDTree(shell_coords)

    et_coords = et_df[["x", "y"]].values.astype(np.float32)
    gene_to_idx = {g: i for i, g in enumerate(cells.var_names)}
    et_gene_idxs = np.array([gene_to_idx.get(g, -1) for g in et_df["gene"]], dtype=np.int32)

    enrich = et_df["enrichment_over_random"].values
    k_vals = np.clip(1 + (max_k - 1) * (enrich - ambient_floor) / (signal_threshold - ambient_floor), 1, max_k).astype(int)

    print("Building transcript spatial tree...")
    et_tree = cKDTree(et_coords)
    csr = csr_matrix(cells.X)

    n_et = len(et_coords)
    all_best_rows = np.zeros(n_et, dtype=np.int32)
    all_best_scores = np.zeros(n_et, dtype=np.float32)
    all_best_distances = np.zeros(n_et, dtype=np.float32)
    all_prob_mat = np.zeros((n_et, len(unique_types)), dtype=np.float32)

    print(f"Processing {n_et} transcripts in {int(np.ceil(n_et / chunk_size))} chunks...")

    for start in tqdm(range(0, n_et, chunk_size)):
        end = min(start + chunk_size, n_et)

        chunk_coords = et_coords[start:end]
        _, chunk_neighs = et_tree.query(chunk_coords, k=max_k)
        chunk_shell_neighs = shell_tree.query_ball_point(chunk_coords, r=max_dist)

        s_flat, s_ptr = [], [0]
        for n in chunk_shell_neighs:
            s_flat.extend(n)
            s_ptr.append(len(s_flat))

        p_mat, b_rows, b_scores, b_dists = _core_scoring_engine_chunk(
            chunk_coords,
            et_gene_idxs,
            chunk_neighs,
            k_vals[start:end],
            shell_coords,
            shell_row_indices,
            shell_type_indices,
            np.array(s_flat, dtype=np.int32),
            np.array(s_ptr, dtype=np.int32),
            csr.data,
            csr.indices,
            csr.indptr,
            lmbda,
            residual,
            len(unique_types),
        )

        all_prob_mat[start:end] = p_mat
        all_best_rows[start:end] = b_rows
        all_best_scores[start:end] = b_scores
        all_best_distances[start:end] = b_dists

    print("Reconstructing results into AnnData...")
    res_adata = sc.AnnData(X=all_prob_mat, obs=pd.DataFrame(index=et_df.index))
    res_adata.var_names = unique_types
    res_adata.obs["gene"] = et_df["gene"].values
    res_adata.obs["predicted_parent"] = [obs["cell_id"].values[r] if r != -1 else "unassigned" for r in all_best_rows]
    res_adata.obs["assignment_score"] = all_best_scores
    res_adata.obs["distance_to_source"] = all_best_distances
    res_adata.obsm["spatial"] = et_coords

    sdata.tables["source_score"] = res_adata

    print("Updating xrna_metadata and cell scores...")
    agg_source = res_adata.obs.assign(score=res_adata.X.sum(axis=1)).groupby("gene")["score"].mean()
    mean_assign = res_adata.obs.groupby("gene")["assignment_score"].mean()

    sdata["xrna_metadata"].var["aggregated_source_score"] = sdata["xrna_metadata"].var.index.map(agg_source)
    sdata["xrna_metadata"].var["mean_assignment_score"] = sdata["xrna_metadata"].var.index.map(mean_assign)

    cell_scores = np.zeros(len(obs))
    valid_mask = all_best_rows != -1
    np.add.at(cell_scores, all_best_rows[valid_mask], all_best_scores[valid_mask])
    sdata["table"].obs["urna_source_score"] = cell_scores

    print(f"Success: {np.sum(valid_mask)} transcripts assigned.")
    return sdata if copy else None
