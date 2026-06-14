import os

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import ConvexHull, KDTree, QhullError, cKDTree
from scipy.stats import poisson
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from spatialdata import SpatialData
from spatialdata.models import PointsModel, TableModel
from tqdm import tqdm


# deprecated
def colocalization_proportion(
    sdata: SpatialData, outpath: str, threshold_colocalized: int = 1, filename: str = "proportion_of_grouped_exRNA.parquet", save: bool = True
):
    """Calculate the proportion of colocalized transcripts for each gene.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object with an ``"extracellular_transcripts_enriched"`` points layer
        (``feature_name``, ``bin_id`` columns) and a ``"segmentation_free_table"`` table
        whose ``.X`` holds per-bin transcript counts per gene.
    outpath : str
        Directory path where the output file should be saved.
    threshold_colocalized : int, optional
        Minimum per-bin count for a transcript to be considered colocalized. Defaults to ``1``.
    filename : str, optional
        Name of the output Parquet file. Defaults to ``"proportion_of_grouped_exRNA.parquet"``.
    save : bool, optional
        If ``True``, write ``coloc`` to ``outpath/filename`` as a Parquet file. Defaults to ``True``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by gene with a single ``"proportion_of_colocalized"`` column.
    """
    df = sdata.points["extracellular_transcripts_enriched"][["feature_name", "bin_id"]].compute()
    adata_density_raw = sdata["segmentation_free_table"]

    filtered_bin_ids = df["bin_id"].astype(str).unique()
    filtered_feature_name_ids = df["feature_name"].astype(str).unique()
    adata_density = adata_density_raw[adata_density_raw.obs.index.isin(filtered_bin_ids)]
    adata_density = adata_density[:, adata_density.var.index.isin(filtered_feature_name_ids)]

    dense_matrix = adata_density.X.todense()

    positive_counts = np.sum(dense_matrix > 0, axis=0)
    colocalized_counts = np.sum(dense_matrix > threshold_colocalized, axis=0)
    proportions = np.divide(colocalized_counts, positive_counts, where=(positive_counts > 0))

    coloc = pd.DataFrame(
        data=proportions.A1,
        index=adata_density.var.index,
        columns=["proportion_of_colocalized"],
    )

    os.makedirs(outpath, exist_ok=True)
    if save:
        filepath = os.path.join(outpath, filename)
        coloc.to_parquet(filepath)

    return coloc


def density_similarity(
    sdata, radius=10.0, ambient_floor=1.5, n_cells_for_model=500, process_all=False, segmentation_key="overlaps_cell", prob_density=0.5
):
    """Classify transcripts by local-density enrichment relative to a Bayesian cell/background model.

    For every transcript (or, if ``process_all=False``, only those with
    ``segmentation_key`` false), counts neighboring transcripts within ``radius`` and
    compares that count to two reference distributions: a Poisson background model fit
    from the global transcript density, and an empirical PMF of local counts built from
    a random subset of ``n_cells_for_model`` cells. The resulting Bayesian posterior and
    enrichment ratio are used to assign each transcript to an ``enrichment_class``.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object with a ``"transcripts"`` points layer containing ``x``, ``y``,
        ``cell_id``, and ``segmentation_key`` columns.
    radius : float, optional
        Radius (in coordinate units) used to count local neighboring transcripts. Defaults
        to ``10.0``.
    ambient_floor : float, optional
        Minimum enrichment-over-random ratio for a transcript to be labeled
        ``"Moderate Density"``. Defaults to ``1.5``.
    n_cells_for_model : int, optional
        Number of cells randomly sampled to build the empirical cell-density PMF.
        Defaults to ``500``.
    process_all : bool, optional
        If ``True``, compute local counts for all transcripts; otherwise only for those
        where ``segmentation_key`` is ``False``. Defaults to ``False``.
    segmentation_key : str, optional
        Boolean column identifying transcripts that overlap a segmented cell. Defaults to
        ``"overlaps_cell"``.
    prob_density : float, optional
        Minimum Bayesian posterior probability for a transcript to be labeled
        ``"High Density"``. Defaults to ``0.5``.

    Returns
    -------
    spatialdata.SpatialData
        ``sdata`` with ``enrichment_class``, ``density_similarity_score``,
        ``enrichment_over_random``, and ``local_count_in_radius`` columns added to the
        ``"transcripts"`` layer (modified in place).
    """
    transcripts_df = sdata.points["transcripts"].compute()
    is_extra = ~transcripts_df[segmentation_key]
    is_cell = (~is_extra) & (transcripts_df["cell_id"].notna()) & (transcripts_df["cell_id"] != "UNASSIGNED")

    mask_to_process = np.ones(len(transcripts_df), dtype=bool) if process_all else is_extra
    all_coords = transcripts_df[["x", "y"]].values

    # local neighbor counts within `radius` for every transcript that needs processing
    print(f"Calculating local counts for {mask_to_process.sum()} transcripts...")
    nn_engine = NearestNeighbors(radius=radius, n_jobs=-1).fit(all_coords)
    rng_query = nn_engine.radius_neighbors(all_coords[mask_to_process], return_distance=False)
    local_counts = np.array([len(res) for res in rng_query])

    # empirical local-count PMF for cells, built from a random subset for speed
    print(f"Subsampling {n_cells_for_model} cells to build likelihood distribution...")
    valid_cell_ids = transcripts_df[is_cell]["cell_id"].unique()
    sampled_ids = np.random.choice(valid_cell_ids, size=min(len(valid_cell_ids), n_cells_for_model), replace=False)

    if process_all:
        cell_model_counts = local_counts[transcripts_df[mask_to_process]["cell_id"].isin(sampled_ids)]
    else:
        sample_coords = transcripts_df[transcripts_df["cell_id"].isin(sampled_ids)][["x", "y"]].values
        rng_sample = nn_engine.radius_neighbors(sample_coords, return_distance=False)
        cell_model_counts = np.array([len(res) for res in rng_sample])

    max_c = max(local_counts.max(), cell_model_counts.max())
    cell_pmf_counts = np.bincount(cell_model_counts, minlength=max_c + 1)
    cell_pmf = cell_pmf_counts / (cell_pmf_counts.sum() + 1e-12)

    # global Poisson background model
    total_hull = ConvexHull(all_coords)
    urna_count = is_extra.sum()
    global_lambda = urna_count / (total_hull.volume + 1e-12)
    circle_area = np.pi * (radius**2)
    expected_random_count = global_lambda * circle_area

    # Bayesian posterior P(Cell | count), assuming P(Cell) = P(Background) = 0.5
    likelihood_bg = poisson.pmf(local_counts, mu=expected_random_count)
    likelihood_cell = cell_pmf[local_counts]
    denom = likelihood_cell + likelihood_bg + 1e-12
    prob_cell = likelihood_cell / denom

    # enrichment ratio and classification
    enrichment = local_counts / (expected_random_count + 1e-9)
    classes = np.array(["Background"] * len(local_counts), dtype=object)
    classes[enrichment > ambient_floor] = "Moderate Density"
    classes[prob_cell > prob_density] = "High Density"

    transcripts_df["enrichment_class"] = "Intracellular"
    transcripts_df.loc[mask_to_process, "enrichment_class"] = classes
    if process_all:
        transcripts_df.loc[is_cell, "enrichment_class"] = "Intracellular"

    transcripts_df.loc[mask_to_process, "density_similarity_score"] = prob_cell
    transcripts_df.loc[mask_to_process, "enrichment_over_random"] = enrichment
    transcripts_df.loc[mask_to_process, "local_count_in_radius"] = local_counts

    sdata.points["transcripts"] = PointsModel.parse(transcripts_df)

    print("--- Analysis Complete ---")
    print(f"Expected Background Count: {expected_random_count:.2f}")
    print(f"Cell Model Median Count: {np.median(cell_model_counts):.2f}")

    return sdata


def calculate_heuristic_radius_by_cells(sdata, k=10, n_cells=500, segmentation_key="overlaps_cell"):
    """Estimate a neighbor-distance radius from the local geometry of a sample of cells.

    Randomly selects ``n_cells`` cells and computes the mean distance to each
    transcript's ``k``-th nearest neighbor among the transcripts of those cells.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object with a ``"transcripts"`` points layer containing ``x``, ``y``,
        ``cell_id``, and ``segmentation_key`` columns.
    k : int, optional
        Neighbor rank used for the distance heuristic. Defaults to ``10``.
    n_cells : int, optional
        Number of cells to randomly sample. Defaults to ``500``.
    segmentation_key : str, optional
        Boolean column used to restrict the sampled transcripts to those that overlap a
        cell. Defaults to ``"overlaps_cell"``.

    Returns
    -------
    float
        Mean distance to the ``k``-th nearest neighbor across the sampled transcripts.

    Raises
    ------
    ValueError
        If no valid (non-``UNASSIGNED``) cells are found, or if fewer than ``k + 1``
        transcripts remain after filtering by ``segmentation_key``.
    """
    transcripts_dask = sdata.points["transcripts"]

    print("Identifying unique cells...")
    all_cell_ids = transcripts_dask["cell_id"].unique().compute()
    valid_cells = [c for c in all_cell_ids if c not in [None, "UNASSIGNED", "nan", np.nan]]

    if len(valid_cells) == 0:
        raise ValueError(f"No valid cells found in transcripts with key '{segmentation_key}'")

    selected_cell_ids = np.random.choice(valid_cells, size=min(len(valid_cells), n_cells), replace=False)

    print(f"Loading transcripts for {len(selected_cell_ids)} cells...")
    subset_df = transcripts_dask[transcripts_dask["cell_id"].isin(selected_cell_ids)].compute()
    subset_coords = subset_df[subset_df[segmentation_key]][["x", "y"]].values

    if len(subset_coords) < k + 1:
        raise ValueError("Not enough transcripts found in selected cells to calculate neighbors.")

    tree = cKDTree(subset_coords)
    distances, _ = tree.query(subset_coords, k=k + 1, workers=-1)
    mean_radius = np.mean(distances[:, -1])

    print(f"Heuristic Radius from cell-subset: {mean_radius:.4f}")
    return mean_radius


def identify_density_k_neighbors(sdata, k_range=range(3, 61, 2), crop_size=400, d_threshold=0.8, segmentation_key="overlaps_cell"):
    """Find and visualize the smallest neighbor count ``k`` that separates cell from extracellular transcript density.

    Crops a square region around the median transcript position, computes a
    k-nearest-neighbor density (``k / (pi * r_k^2)``) for every ``k`` in ``k_range``, and
    measures the Cohen's d effect size between the densities of transcripts inside vs.
    outside cells. A plot of effect size vs. ``k`` is shown.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object with a ``"transcripts"`` points layer containing ``x``, ``y``,
        and ``segmentation_key`` columns.
    k_range : range or iterable of int, optional
        Candidate neighbor counts to evaluate. Defaults to ``range(3, 61, 2)``.
    crop_size : float, optional
        Side length of the square region (centered on the median transcript position)
        used for the analysis. Defaults to ``400``.
    d_threshold : float, optional
        Minimum Cohen's d effect size for ``k`` to be considered separating. Defaults to
        ``0.8``.
    segmentation_key : str, optional
        Boolean column identifying transcripts that overlap a segmented cell. Defaults to
        ``"overlaps_cell"``.

    Returns
    -------
    int
        The smallest ``k`` in ``k_range`` whose Cohen's d reaches ``d_threshold``, or the
        largest value in ``k_range`` if none does.
    """
    df = sdata.points["transcripts"].compute()
    cx, cy = df["x"].median(), df["y"].median()
    half = crop_size / 2
    subset = df[df["x"].between(cx - half, cx + half) & df["y"].between(cy - half, cy + half)].copy()

    coords = subset[["x", "y"]].values
    is_cell = subset[segmentation_key].values

    knn = NearestNeighbors(n_neighbors=max(k_range), n_jobs=-1).fit(coords)
    dist, _ = knn.kneighbors(coords)

    metrics = []
    minimal_k = None

    for k in k_range:
        densities = k / (np.pi * (dist[:, k - 1] ** 2 + 1e-6))

        c_vals = densities[is_cell]
        e_vals = densities[~is_cell]

        # Cohen's d effect size
        mean_diff = np.mean(c_vals) - np.mean(e_vals)
        pooled_std = np.sqrt((np.var(c_vals) + np.var(e_vals)) / 2)
        d = mean_diff / (pooled_std + 1e-6)

        metrics.append({"k": k, "cohens_d": d})

        if minimal_k is None and d >= d_threshold:
            minimal_k = k

    metrics_df = pd.DataFrame(metrics)

    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df["k"], metrics_df["cohens_d"], marker="o", color="black", label="Separation (Cohen's d)")
    plt.axhline(y=d_threshold, color="red", linestyle="--", label=f"Threshold (d={d_threshold})")

    if minimal_k:
        plt.axvline(x=minimal_k, color="green", alpha=0.5, label=f"Minimal k={minimal_k}")
        plt.scatter(minimal_k, d_threshold, color="green", s=100, zorder=5)

    plt.title("Finding Minimal k: Separation of Cell vs. Extracellular Density")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Effect Size (Separation Power)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    selected_k = minimal_k if minimal_k else max(k_range)
    print(f"Suggested Minimal k: {selected_k}")

    return selected_k


def segment_protrusions(
    sdata: SpatialData,
    layer: str = "transcripts",
    table_name: str = "structure_table",
    cluster_eps: float = 1.0,
    min_samples: int = 10,
    connectivity_threshold: float = 2.0,
    max_distance: float = 30,
    global_distance: float = 500,
    n_neighbors: int = 50,
    lmbda: float = 0.01,
    copy: bool = False,
) -> SpatialData | None:
    """Identify extracellular RNA structures, quantify their morphology, and assign them to parent cells.

    Transcripts in ``sdata.points[layer]`` that are either classified as ``"High
    Density"`` and not overlapping a cell, or intracellular-like but not overlapping
    a cell, are clustered with DBSCAN into candidate structures (e.g. protrusions).
    For each resulting structure this computes:

    - Morphology (area, perimeter, circularity) from the convex hull of its
      transcripts, and whether it is physically connected to a segmented cell.
    - An individual ``parent_score`` for the most likely parent cell, based on
      gene-expression overlap weighted by an exponential distance decay, plus an
      ``assignment_confidence`` margin and an ``is_ambiguous`` flag.
    - A ``neighborhood_score`` describing how much of the structure's gene content
      is explained by the collective expression of its nearby cells.

    Results are stored as a new table ``sdata.tables[table_name]``, and
    ``sdata.points[layer]`` is annotated with a ``structure_id`` column (``-1`` for
    transcripts not assigned to any structure).

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object with a ``layer`` points table containing ``x``, ``y``,
        ``gene``, ``enrichment_class``, ``extracellular``, and ``overlaps_cell``
        columns, and a ``"table"`` AnnData with ``.obsm["spatial"]``.
    layer : str, optional
        Points layer to read transcripts from and annotate with ``structure_id``.
        Defaults to ``"transcripts"``.
    table_name : str, optional
        Key under which the resulting structure table is stored in ``sdata.tables``.
        Defaults to ``"structure_table"``.
    cluster_eps : float, optional
        DBSCAN ``eps`` (maximum distance between two points to be considered
        neighbors) used to cluster candidate structure transcripts. Defaults to ``1.0``.
    min_samples : int, optional
        DBSCAN ``min_samples`` required to form a structure. Defaults to ``10``.
    connectivity_threshold : float, optional
        Maximum distance to a cell-assigned transcript for a structure to be
        considered physically connected to a cell. Defaults to ``2.0``.
    max_distance : float, optional
        Currently unused; reserved for future filtering by structure-to-cell distance.
        Defaults to ``30``.
    global_distance : float, optional
        Maximum distance for a cell to be considered a neighbor of a structure when
        computing parent and neighborhood scores. Defaults to ``500``.
    n_neighbors : int, optional
        Number of nearby cells to consider per structure for parent assignment.
        Defaults to ``50``.
    lmbda : float, optional
        Exponential decay rate applied to structure-to-cell distance when scoring
        candidate parent cells. Defaults to ``0.01``.
    copy : bool, optional
        If ``True``, return the modified SpatialData object; otherwise modify
        ``sdata`` in place and return ``None``. Defaults to ``False``.

    Returns
    -------
    spatialdata.SpatialData or None
        ``sdata`` with ``sdata.tables[table_name]`` containing per-structure
        morphology, ``predicted_parent``, ``parent_score``, ``neighborhood_score``,
        ``assignment_confidence``, and ``is_ambiguous`` columns, and
        ``sdata.points[layer]["structure_id"]`` set, if ``copy=True``; otherwise
        ``None``.
    """
    print(f"--- Step 1: Extracting structures from '{layer}' ---")
    df = sdata.points[layer].compute().copy()

    cond_hd = (df["enrichment_class"] == "High Density") & ~df["overlaps_cell"]
    cond_cl = ~df["extracellular"] & ~df["overlaps_cell"]

    df["point_source"] = "other"
    df.loc[cond_hd, "point_source"] = "high_density_source"
    df.loc[cond_cl, "point_source"] = "cell_like_source"

    target_df = df[cond_hd | cond_cl].copy()

    if target_df.empty:
        print("Exiting: No transcripts found matching criteria.")
        return sdata if copy else None

    coords = target_df[["x", "y"]].values
    dbscan = DBSCAN(eps=cluster_eps, min_samples=min_samples, n_jobs=-1)
    target_df["structure_id"] = dbscan.fit_predict(coords)

    structures_df = target_df[target_df["structure_id"] != -1].copy()
    if structures_df.empty:
        print("Exiting: No clusters formed.")
        return sdata if copy else None

    # Check whether each structure is near an already-segmented cell
    cell_assigned_df = df[df["overlaps_cell"]]
    is_connected_map = {}
    if not cell_assigned_df.empty:
        anchor_tree = cKDTree(cell_assigned_df[["x", "y"]].values)
        for sid, group in structures_df.groupby("structure_id"):
            cluster_coords = group[["x", "y"]].values
            dist, _ = anchor_tree.query(cluster_coords, k=1, distance_upper_bound=connectivity_threshold)
            is_connected_map[sid] = np.any(np.isfinite(dist))
    else:
        is_connected_map = dict.fromkeys(structures_df["structure_id"].unique(), False)

    print("--- Step 2: Computing morphology and metrics ---")
    counts = structures_df.groupby(["structure_id", "gene"]).size().unstack(fill_value=0)

    morphology = []
    for sid, group in structures_df.groupby("structure_id"):
        pts = group[["x", "y"]].values
        area, perimeter, circularity = np.nan, np.nan, np.nan
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                area = hull.volume  # 2D area (ConvexHull.volume is area in 2D)
                perimeter = hull.area  # 2D perimeter (ConvexHull.area is perimeter in 2D)
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter**2)
            except QhullError:
                pass
        morphology.append({"structure_id": sid, "area": area, "perimeter": perimeter, "circularity": circularity})

    morph_df = pd.DataFrame(morphology).set_index("structure_id")

    nature_counts = structures_df.groupby(["structure_id", "point_source"]).size().unstack(fill_value=0)
    for col in ["high_density_source", "cell_like_source"]:
        if col not in nature_counts.columns:
            nature_counts[col] = 0
    dominant_nature = np.where(
        nature_counts["high_density_source"] > nature_counts["cell_like_source"], "High-Density Structure", "Cell-Like Structure"
    )

    metrics_to_compute = ["cosine_similarity", "cosine_sim_stainings", "enrichment_over_random", "local_density_area"]
    available_metrics = [m for m in metrics_to_compute if m in structures_df.columns]
    struct_metrics = structures_df.groupby("structure_id")[available_metrics].max() if available_metrics else pd.DataFrame(index=counts.index)

    struct_stats = structures_df.groupby("structure_id").agg({"x": "mean", "y": "mean"})
    struct_stats["structure_label"] = dominant_nature
    struct_stats["n_transcripts"] = counts.sum(axis=1)
    struct_stats["is_physically_connected"] = struct_stats.index.map(is_connected_map)
    struct_stats = struct_stats.join(morph_df).join(struct_metrics)

    adata_struct = ad.AnnData(X=counts.values.astype(np.float32), obs=struct_stats, var=pd.DataFrame(index=counts.columns))
    adata_struct.obs_names = [f"struct_{i}" for i in counts.index]
    adata_struct.obsm["spatial"] = struct_stats[["x", "y"]].values

    print(f"--- Step 3: Assignment logic for {len(adata_struct)} structures ---")
    cells = sdata["table"]
    common_genes = list(cells.var_names.intersection(adata_struct.var_names))
    cell_exp = csr_matrix(cells[:, common_genes].X > 0)
    prot_exp = csr_matrix(adata_struct[:, common_genes].X > 0)
    cell_tree = KDTree(cells.obsm["spatial"])

    parent_ids = np.full(adata_struct.n_obs, "unassigned", dtype=object)
    parent_scores = np.zeros(adata_struct.n_obs)
    neighborhood_scores = np.zeros(adata_struct.n_obs)
    assignment_conf = np.zeros(adata_struct.n_obs)
    is_ambiguous = np.zeros(adata_struct.n_obs, dtype=bool)

    for i in tqdm(range(adata_struct.n_obs), desc="Analyzing Parents"):
        p_mask = prot_exp[i].toarray().flatten()
        total_p_genes = p_mask.sum()
        if total_p_genes == 0:
            continue

        d_g, i_g = cell_tree.query(adata_struct.obsm["spatial"][i], k=n_neighbors, distance_upper_bound=global_distance)
        valid = np.isfinite(d_g)

        if np.any(valid):
            # Neighborhood score: collective explanation by nearby cells
            neighbor_masks = cell_exp[i_g[valid]].toarray()
            combined_mask = neighbor_masks.sum(axis=0) > 0
            neighborhood_scores[i] = np.logical_and(p_mask, combined_mask).sum() / total_p_genes

            # Individual parent score for each nearby cell
            candidates = []
            for g_idx, d in zip(i_g[valid], d_g[valid], strict=False):
                c_mask = cell_exp[g_idx].toarray().flatten()
                match = np.logical_and(p_mask, c_mask).sum() / total_p_genes
                score = match * np.exp(-lmbda * d)
                if adata_struct.obs["is_physically_connected"].iloc[i]:
                    score *= 1.2
                candidates.append({"score": score, "id": cells.obs_names[g_idx]})

            candidates.sort(key=lambda x: x["score"], reverse=True)
            if candidates:
                best = candidates[0]
                parent_ids[i], parent_scores[i] = best["id"], best["score"]
                if len(candidates) > 1:
                    assignment_conf[i] = best["score"] - candidates[1]["score"]
                    is_ambiguous[i] = (candidates[1]["score"] / (best["score"] + 1e-9)) > 0.8

    adata_struct.obs["predicted_parent"] = parent_ids
    adata_struct.obs["parent_score"] = parent_scores
    adata_struct.obs["neighborhood_score"] = neighborhood_scores
    adata_struct.obs["assignment_confidence"] = assignment_conf
    adata_struct.obs["is_ambiguous"] = is_ambiguous

    print("--- Step 4: Updating sdata ---")
    df["structure_id"] = -1
    df.loc[structures_df.index, "structure_id"] = structures_df["structure_id"]
    sdata.points[layer] = PointsModel.parse(df)

    adata_struct.obs["region"] = layer
    adata_struct.obs["instance_id"] = counts.index.values.astype(int)
    sdata.tables[table_name] = TableModel.parse(adata_struct, region=layer, region_key="region", instance_key="instance_id")

    print(f"Success! Registered '{table_name}' with morphological and parent metrics.")
    return sdata if copy else None
