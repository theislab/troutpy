import copy as cp
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pls
import seaborn as sns
import spatialdata as sd
from scipy.interpolate import interp1d
from scipy.sparse import coo_matrix
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from spatialdata import SpatialData


def compute_extracellular_counts(data_extracell):
    """Compute observed and expected transcript counts, log fold-ratio over baseline, and codeword category per gene for extracellular transcripts.

    Parameters
    ----------
    data_extracell : pandas.DataFrame
        DataFrame of extracellular transcripts. Must contain a ``feature_name`` column
        with gene identifiers and a ``codeword_category`` column with probe category labels.

    Returns
    -------
    extracellular_counts : pandas.DataFrame
        DataFrame indexed by gene name with columns:

        - ``observed`` : int — raw transcript count per gene.
        - ``expected`` : int — uniform expected count (total / number of genes).
        - ``fold_ratio`` : float — log(observed / expected).
        - ``codeword_category`` : str — probe category for each gene.
    """
    extracellular_counts = data_extracell.groupby("feature_name").count()
    extracellular_counts = pd.DataFrame({"observed": extracellular_counts.iloc[:, 0]})
    extracellular_counts["expected"] = int(extracellular_counts["observed"].sum() / extracellular_counts.shape[0])

    # Calculate fold ratios
    extracellular_counts["fold_ratio"] = np.log(extracellular_counts["observed"] / extracellular_counts["expected"])

    # Map gene categories
    gene2cat = dict(zip(data_extracell["feature_name"], data_extracell["codeword_category"], strict=False))
    extracellular_counts["codeword_category"] = extracellular_counts.index.map(gene2cat)

    return extracellular_counts


def define_urna_probability(sdata, p_threshold=0.5, copy=False):
    """Classify transcripts as extracellular (uRNA) via a Bayesian probability from grid-interpolated KDEs.

    Fits Gaussian KDEs of ``cosine_similarity`` for transcripts inside cells versus
    background transcripts outside cells, evaluates both on a shared grid to derive
    ``P(uRNA | cosine_similarity)``, and interpolates this probability back onto every
    transcript.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object with a ``"transcripts"`` points layer containing
        ``overlaps_cell``, ``cosine_similarity``, and ``enrichment_class`` columns.
    p_threshold : float, optional
        Minimum ``prob_is_urna`` value for a transcript to be classified as
        extracellular. Defaults to ``0.5``.
    copy : bool, optional
        If ``True``, return the modified SpatialData object. Otherwise modify ``sdata``
        in place and return ``None``. Defaults to ``False``.

    Returns
    -------
    spatialdata.SpatialData or None
        SpatialData with ``prob_is_urna`` and ``extracellular`` columns added to the
        ``"transcripts"`` layer if ``copy=True``; otherwise ``None``. If fewer than 10
        reference transcripts are available inside or outside cells, ``sdata`` is
        returned unchanged.
    """
    df = sdata.points["transcripts"].compute()

    # Reference populations: transcripts inside cells vs. background transcripts outside cells
    mask_inside = df["overlaps_cell"]
    mask_outside = ~df["overlaps_cell"] & (df["enrichment_class"] == "Background")

    scores_in = df.loc[mask_inside, "cosine_similarity"].dropna().values
    scores_out = df.loc[mask_outside, "cosine_similarity"].dropna().values

    if len(scores_in) < 10 or len(scores_out) < 10:
        print("Warning: Insufficient data for KDE.")
        return sdata if copy else None

    kde_in = gaussian_kde(scores_in)
    kde_out = gaussian_kde(scores_out)

    # Evaluate both KDEs on a shared grid and derive P(uRNA | score)
    all_scores = df["cosine_similarity"].values
    valid_mask = np.isfinite(all_scores)
    actual_scores = all_scores[valid_mask]

    grid = np.linspace(np.nanmin(all_scores), np.nanmax(all_scores), 500)
    li_in_grid = kde_in(grid)
    li_out_grid = kde_out(grid)
    prob_grid = li_out_grid / (li_out_grid + li_in_grid + 1e-10)

    # Interpolate the grid probabilities back onto each transcript
    prob_is_urna = np.full(len(df), np.nan)
    prob_is_urna[valid_mask] = np.interp(actual_scores, grid, prob_grid)

    df["prob_is_urna"] = prob_is_urna
    df["extracellular"] = ~df["overlaps_cell"] & (df["prob_is_urna"] > p_threshold)

    sdata.points["transcripts"] = sd.models.PointsModel.parse(df)
    return sdata if copy else None


def define_urna(
    sdata: sd.SpatialData,
    layer: str = "transcripts",
    method: str = "segmentation_free",
    min_prop_of_extracellular: float = 0.8,
    unassigned_tag: str = "UNASSIGNED",
    copy: bool = False,
    prob_threshold: float = 0.5,  # Replaces percentile_threshold for probabilistic logic
):
    """Identify extracellular RNA (uRNA) transcripts by classifying each transcript based on the specified segmentation method.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object containing a points layer with transcript coordinates and
        metadata columns used by the chosen method.
    layer : str, optional
        Key in ``sdata.points`` that holds the transcript data. Defaults to ``"transcripts"``.
    method : str, optional
        Classification strategy. One of:

        - ``"spots2regions"`` — uses segmentation-free cluster proportions.
        - ``"sainsc"`` — uses the Bayesian ``prob_is_urna`` column produced by SAiNSC.
        - ``"nuclei"`` — marks transcripts outside nucleus overlap as extracellular.
        - ``"cells"`` — marks transcripts with ``cell_id == unassigned_tag`` as extracellular.

        Defaults to ``"segmentation_free"``.
    min_prop_of_extracellular : float, optional
        Minimum proportion of transcripts in a cluster that must be outside cells for the
        cluster to be considered extracellular (used by ``"spots2regions"``). Defaults to ``0.8``.
    unassigned_tag : str, optional
        String value in the ``cell_id`` column that indicates an unassigned transcript.
        Defaults to ``"UNASSIGNED"``.
    copy : bool, optional
        If ``True``, return the modified SpatialData object. If ``False``, modify in place
        and return ``None``. Defaults to ``False``.
    prob_threshold : float, optional
        Minimum ``prob_is_urna`` value for a transcript to be classified as extracellular
        (used by ``"sainsc"``). Defaults to ``0.5``.

    Returns
    -------
    spatialdata.SpatialData or None
        Modified SpatialData with an ``extracellular`` boolean column added to the points
        layer if ``copy=True``; otherwise ``None``.

    Raises
    ------
    KeyError
        If ``method="sainsc"`` and the ``prob_is_urna`` column is absent.
    ValueError
        If an unsupported ``method`` string is provided.
    """
    # Compute the data layer from the spatial data object
    data = sdata.points[layer].compute()

    # Method: Segmentation-free clustering (spots2regions)
    if method == "spots2regions":
        data["overlaps_cell"] = (data["cell_id"] != unassigned_tag).astype(int)
        overlapping_cell = pd.crosstab(data["segmentation_free_clusters"], data["overlaps_cell"])
        cluster_totals = overlapping_cell.sum(axis=1)
        cluster_proportions = overlapping_cell.div(cluster_totals, axis=0)
        extracellular_clusters = cluster_proportions[cluster_proportions.loc[:, 0] >= min_prop_of_extracellular].index
        data["extracellular"] = ~data["segmentation_free_clusters"].isin(extracellular_clusters)

    # Method: Using SAiNSC Bayesian Probability
    elif method == "sainsc":
        if "prob_is_urna" not in data.columns:
            raise KeyError("Column 'prob_is_urna' not found. Ensure you ran the updated SAiNSC pipeline first.")

        # Extracellular (uRNA): physically outside a cell boundary AND P(uRNA) above threshold
        data["extracellular"] = ~data["overlaps_cell"] & (data["prob_is_urna"] > prob_threshold)

        n_urna = data["extracellular"].sum()
        print(f"Defined {n_urna} URNA transcripts using Bayesian P > {prob_threshold}")

    # Method: Based on nuclei overlap
    elif method == "nuclei":
        data["extracellular"] = data["overlaps_nucleus"] != 1

    # Method: Based on cell assignment
    elif method == "cells":
        data["extracellular"] = data["cell_id"] == unassigned_tag

    else:
        raise ValueError(f"Unknown method: {method}. Supported: 'spots2regions', 'sainsc', 'nuclei', 'cells'.")

    # Update the spatial data object
    sdata.points[layer] = sd.models.PointsModel.parse(data)

    return sdata if copy else None


def filter_urna(
    sdata,
    min_counts=None,
    min_extracellular_proportion=None,
    control_probe=None,
    min_logfoldratio_over_noise=None,
    max_p_val_noise=0.05,
    min_morani=None,
    gene_key="feature_name",
    filter_cellular=False,
    copy=False,
    genes_in_segmented=False,
    table_key="table",
):
    """Filter xRNA genes based on quality and significance thresholds, retaining only genes that pass all active criteria.

    All threshold parameters are optional; only those explicitly set are applied (AND logic).

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object containing ``xrna_metadata``, ``transcripts``, and optionally
        ``segmentation_free_table``, ``source_score``, and ``target_score`` layers.
    min_counts : int or None, optional
        Minimum total transcript count per gene. Defaults to ``None`` (not applied).
    min_extracellular_proportion : float or None, optional
        Minimum fraction of transcripts classified as extracellular. Defaults to ``None``.
    control_probe : bool or None, optional
        If ``False``, exclude control probes. Defaults to ``None`` (not applied).
    min_logfoldratio_over_noise : float or None, optional
        Minimum log fold-ratio over the noise baseline. Defaults to ``None``.
    max_p_val_noise : float or None, optional
        Maximum p-value against the Poisson noise model; genes with
        ``p_val_noise >= max_p_val_noise`` are removed. Defaults to ``0.05``.
    min_morani : float or None, optional
        Minimum Moran's I spatial autocorrelation score. Defaults to ``None``.
    gene_key : str, optional
        Column in the transcripts layer containing gene identifiers.
        Defaults to ``"feature_name"``.
    filter_cellular : bool, optional
        If ``True``, also subset the cellular ``"table"`` layer to selected genes.
        Defaults to ``False``.
    copy : bool, optional
        If ``True``, operate on a copy of ``sdata`` and return it.
        Otherwise modify in place and return ``None``. Defaults to ``False``.
    genes_in_segmented : bool, optional
        If ``True``, further restrict to genes present in the segmented ``table`` layer.
        Defaults to ``False``.
    table_key : str, optional
        Key for the segmented cell-by-gene table used when ``genes_in_segmented=True``.
        Defaults to ``"table"``.

    Returns
    -------
    spatialdata.SpatialData or None
        Filtered SpatialData object if ``copy=True``; otherwise ``None``.
    """
    if copy:
        sdata = cp.deepcopy(sdata)

    # Access the metadata DataFrame
    var_df = sdata["xrna_metadata"].var

    # Initialize a mask that selects all genes
    mask = pd.Series(True, index=var_df.index)

    # Apply filters sequentially (Intersection/AND logic)
    if min_counts is not None:
        mask &= var_df["count"] > min_counts

    if min_extracellular_proportion is not None:
        mask &= var_df["extracellular_proportion"] > min_extracellular_proportion

    if control_probe is False:
        col = "is_control" if "is_control" in var_df.columns else "control_probe"
        mask &= ~var_df[col]

    if min_logfoldratio_over_noise is not None:
        col = "log2_fc_over_noise" if "log2_fc_over_noise" in var_df.columns else "logfoldratio_over_noise"
        mask &= var_df[col] > min_logfoldratio_over_noise

    if max_p_val_noise is not None:
        # Keep genes that are significantly different from noise (p < threshold)
        mask &= var_df["p_val_noise"] < max_p_val_noise

    if min_morani is not None:
        mask &= var_df["moran_I"] > min_morani

    if genes_in_segmented and table_key in sdata:
        mask &= var_df.index.isin(sdata[table_key].var.index)

    # Final list of genes to keep
    selected_genes = var_df.index[mask]

    # --- Apply Filtering to sdata objects ---

    # Filter transcripts (Point Data)
    transcripts = sdata["transcripts"]
    transcript_mask = transcripts[gene_key].isin(selected_genes)
    filtered_transcripts = transcripts[transcript_mask]
    # Boolean-indexing a dask DataFrame can drop its `.attrs` (e.g. after a deepcopy of
    # sdata under newer dask versions), so PointsModel validation fails on reassignment
    # unless we restore (or default) the "transform" attr.
    filtered_transcripts.attrs.update(transcripts.attrs or {"transform": {"global": sd.transformations.Identity()}})
    sdata["transcripts"] = filtered_transcripts

    # Filter Tables (AnnData objects)
    for key in ["segmentation_free_table", "xrna_metadata"]:
        if key in sdata and hasattr(sdata[key], "var"):
            try:
                # Slicing AnnData: [obs, var]
                sdata[key] = sdata[key][:, sdata[key].var.index.isin(selected_genes)]
            except Exception:  # noqa: BLE001 - best-effort: leave table untouched if it can't be subset by gene
                pass

    # Filter score tables (Observation-based)
    for key in ["source_score", "target_score"]:
        if key in sdata:
            try:
                sdata[key] = sdata[key][sdata[key].obs[gene_key].isin(selected_genes), :]
            except Exception:  # noqa: BLE001 - best-effort: leave table untouched if it can't be subset by gene
                pass

    # Filter cellular table if requested
    if filter_cellular and "table" in sdata:
        sdata["table"] = sdata["table"][:, sdata["table"].var.index.isin(selected_genes)]

    return sdata if copy else None


def process_gene(group, shape):
    """Build a sparse ``(Y, X)`` count matrix for a single gene's binned transcripts.

    Parameters
    ----------
    group : polars.DataFrame
        Per-gene partition with ``"gene"``, ``"bin_x"``, and ``"bin_y"`` columns.
    shape : tuple of int
        Output matrix shape as ``(n_rows, n_cols)``, i.e. ``(height, width)``.

    Returns
    -------
    gene_name : str
        The gene name for this partition.
    sparse_matrix : scipy.sparse.csr_matrix
        Sparse matrix of transcript counts with shape ``shape``.
    """
    gene_name = group["gene"][0]

    # Matrix row = Y (vertical), matrix col = X (horizontal); each entry is one transcript
    y_idx = group["bin_y"].to_numpy()
    x_idx = group["bin_x"].to_numpy()
    counts = np.ones(len(x_idx), dtype=np.uint32)

    sparse_matrix = coo_matrix((counts, (y_idx, x_idx)), shape=shape).tocsr()
    return gene_name, sparse_matrix


def process_dataframe(df: pls.DataFrame, binsize: float, n_threads: int = 4):
    """Bin transcripts into a spatial grid and build a per-gene sparse count matrix.

    Coordinates are divided by ``binsize``, floored to integer bins, and shifted so
    that bin indices start at zero. Each gene's binned transcripts are then converted
    to a sparse ``(n_rows, n_cols)`` count matrix in parallel.

    Parameters
    ----------
    df : polars.DataFrame
        Transcript table with ``"x"``, ``"y"``, and ``"gene"`` columns.
    binsize : float
        Size of each spatial bin, in the same units as ``"x"``/``"y"``.
    n_threads : int, optional
        Number of worker processes used to build the per-gene matrices. Defaults to ``4``.

    Returns
    -------
    results : dict
        Mapping from gene name to its sparse :class:`scipy.sparse.csr_matrix` count matrix.
    shape : tuple of int
        Shape ``(n_rows, n_cols)`` shared by all matrices in ``results``.
    df : polars.DataFrame
        Input DataFrame with added ``"bin_x"``/``"bin_y"`` integer bin columns and
        ``"gene"`` cast to a categorical dtype.
    """
    # Compute bin coordinates
    df = df.with_columns(
        [
            (pls.col("x") / binsize).floor().alias("bin_x"),
            (pls.col("y") / binsize).floor().alias("bin_y"),
        ]
    )

    # Shift coordinates so bins start at zero (required for matrix indices)
    min_x, min_y = df["bin_x"].min(), df["bin_y"].min()
    df = df.with_columns(
        [
            (pls.col("bin_x") - min_x).cast(pls.Int32),
            (pls.col("bin_y") - min_y).cast(pls.Int32),
        ]
    )

    # Shape: (height, width) -> (rows, cols) -> (Y, X)
    n_cols = int(df["bin_x"].max() + 1)
    n_rows = int(df["bin_y"].max() + 1)
    shape = (n_rows, n_cols)

    df = df.with_columns(df["gene"].cast(pls.Categorical))

    # Partition by gene and build each gene's sparse matrix in parallel
    gene_groups = df.partition_by("gene", maintain_order=False)
    process_gene_partial = partial(process_gene, shape=shape)

    with Pool(n_threads) as pool:
        gene_results = pool.map(process_gene_partial, gene_groups)

    results = dict(gene_results)

    return results, shape, df


def segmentation_free_sainsc(
    sdata,
    binsize=3,
    celltype_key="leiden",
    background_filter=0.4,
    gaussian_kernel_key=2.5,
    n_threads=16,
    resolution=1000,
    return_sainsc=False,
    copy=False,
    default_cell_type="unknown",
    default_numeric=np.nan,
):
    """Assign a per-bin cell type and uRNA probability via SAiNSC segmentation-free analysis.

    Bins all transcripts on a regular grid using :class:`sainsc.LazyKDE`, computes a
    cosine-similarity map between each bin's local expression and cell-type signatures
    derived from ``sdata["table"]``, and assigns each bin to its closest cell type.
    A Bayesian probability of being extracellular (``prob_is_urna``) is then derived
    from the cosine-similarity distributions of bins overlapping vs. not overlapping
    segmented cells, and the per-bin results are mapped back onto every transcript.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object with a ``"transcripts"`` points layer (``"gene"``, ``"x"``,
        ``"y"``, ``"overlaps_cell"``) and a ``"table"`` AnnData with cell-type
        annotations in ``obs[celltype_key]``.
    binsize : float, optional
        Spatial bin size, in micrometres, used by SAiNSC. Defaults to ``3``.
    celltype_key : str, optional
        Column in ``sdata["table"].obs`` containing cell-type labels used to build
        reference signatures. Defaults to ``"leiden"``.
    background_filter : float, optional
        Total-mRNA KDE threshold below which bins are treated as background by
        :meth:`sainsc.LazyKDE.filter_background`. Defaults to ``0.4``.
    gaussian_kernel_key : float, optional
        Bandwidth (in micrometres) of the Gaussian kernel used to smooth total mRNA
        counts. Defaults to ``2.5``.
    n_threads : int, optional
        Number of threads/processes used by SAiNSC and bin assignment. Defaults to ``16``.
    resolution : int, optional
        Resolution (in nanometres per pixel) passed to :meth:`sainsc.LazyKDE.from_dataframe`.
        Defaults to ``1000``.
    return_sainsc : bool, optional
        If ``True``, additionally return a dict with the underlying ``LazyKDE`` instance
        (``"brain"``) and the per-bin results table (``"output_df"``). Defaults to ``False``.
    copy : bool, optional
        If ``True``, return the modified SpatialData object. Otherwise modify ``sdata``
        in place and return ``None``. Ignored if ``return_sainsc=True``. Defaults to ``False``.
    default_cell_type : str, optional
        Reserved for future use; currently has no effect. Defaults to ``"unknown"``.
    default_numeric : float, optional
        Reserved for future use; currently has no effect. Defaults to ``numpy.nan``.

    Returns
    -------
    spatialdata.SpatialData or None
        SpatialData with ``closest_cell_type``, ``cosine_similarity``, ``prob_is_urna``,
        and ``extracellular`` columns added to the ``"transcripts"`` layer if
        ``copy=True``; otherwise ``None``.
    tuple[spatialdata.SpatialData, dict]
        If ``return_sainsc=True``, returns ``(sdata, {"brain": LazyKDE, "output_df": pandas.DataFrame})``
        instead of the above.
    """
    try:
        from sainsc import LazyKDE
    except ImportError as err:
        raise ImportError(
            "The 'sainsc' package is required for segmentation_free_sainsc. Please install it with: pip install troutpy[segmentation-free]"
        ) from err

    # --- 1. Prepare data ---
    transcripts_all = sdata.points["transcripts"].compute().reset_index(drop=True)
    transcripts_full = pls.from_pandas(transcripts_all[["gene", "x", "y"]].copy())

    # --- 2. Signatures & engine ---
    adata = sdata["table"]
    signatures = adata.to_df().assign(ct=adata.obs[celltype_key]).groupby("ct").mean().T
    common_genes = list(set(transcripts_all["gene"]).intersection(signatures.index))
    signatures = signatures.loc[common_genes, :]

    brain = LazyKDE.from_dataframe(
        transcripts_full.filter(pls.col("gene").is_in(common_genes)),
        resolution=resolution,
        binsize=binsize,
        n_threads=n_threads,
    )
    brain.calculate_total_mRNA()
    brain.gaussian_kernel(gaussian_kernel_key, unit="um")
    brain.calculate_total_mRNA_KDE()
    brain.filter_background(background_filter)
    brain.assign_celltype(signatures, log=True)

    # --- 3. Matrix realignment: transpose matrices so they are (Y, X) ---
    sim_matrix = brain.cosine_similarity.T
    ct_map = brain.celltype_map.T
    n_rows, n_cols = sim_matrix.shape

    output_df = pd.DataFrame(
        {
            "bin_id": np.arange(n_rows * n_cols),
            "ct_idx": ct_map.flatten(),
            "cosine_similarity": sim_matrix.flatten(),
        }
    )

    # --- 4. Bin alignment & Bayesian logic ---
    _, _, t2b_pl = process_dataframe(transcripts_full, binsize=binsize, n_threads=n_threads)
    t2b = t2b_pl.to_pandas()

    # Row-major indexing: (y * total_columns) + x
    t2b["bin_id"] = (t2b["bin_y"] * n_cols) + t2b["bin_x"]
    t2b["overlaps_cell"] = transcripts_all["overlaps_cell"].values

    bin_gt = t2b.groupby("bin_id")["overlaps_cell"].mean()
    output_df["is_cellular_gt"] = output_df["bin_id"].map(bin_gt) > 0.5

    # Bayesian grid KDE
    scores_in = output_df.loc[output_df["is_cellular_gt"], "cosine_similarity"].dropna().values
    scores_out = output_df.loc[~output_df["is_cellular_gt"], "cosine_similarity"].dropna().values

    if len(scores_in) > 10 and len(scores_out) > 10:
        kin, kout = gaussian_kde(scores_in), gaussian_kde(scores_out)
        grid = np.linspace(output_df["cosine_similarity"].min(), output_df["cosine_similarity"].max(), 500)
        p_grid = kout(grid) / (kout(grid) + kin(grid) + 1e-10)
        output_df["prob_is_urna"] = np.interp(output_df["cosine_similarity"], grid, p_grid)
    else:
        output_df["prob_is_urna"] = 0.5

    # --- 5. Merge & update sdata ---
    num2ct = dict(zip(range(len(brain.celltypes)), brain.celltypes, strict=False))
    output_df["closest_cell_type"] = output_df["ct_idx"].map(num2ct)

    t2b = t2b.merge(output_df, on="bin_id", how="left")

    transcripts_all["closest_cell_type"] = t2b["closest_cell_type"].values
    transcripts_all["cosine_similarity"] = t2b["cosine_similarity"].values
    transcripts_all["prob_is_urna"] = t2b["prob_is_urna"].values
    transcripts_all["extracellular"] = ~transcripts_all["overlaps_cell"] & (transcripts_all["prob_is_urna"] > 0.5)

    sdata.points["transcripts"] = sd.models.PointsModel.parse(transcripts_all)

    if return_sainsc:
        return sdata, {"brain": brain, "output_df": output_df}
    return sdata if copy else None


def add_morphological_metrics(sdata: SpatialData, labels_key: str = "cell_labels", copy: bool = False) -> SpatialData | None:
    """Extract cell morphology metrics from a labels layer and add them to ``sdata.table.obs``.

    Computes region properties (area, perimeter, solidity, axis lengths) via
    ``skimage.measure.regionprops_table`` and derives circularity, protrusion length,
    and morphological complexity. Existing overlapping columns are replaced.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object that must contain a ``"table"`` AnnData and the specified
        labels layer.
    labels_key : str
        Key in ``sdata.labels`` identifying the segmentation label image.
    copy : bool, optional
        If ``True``, operate on a copy of ``sdata`` and return it.
        Otherwise modify in place and return ``None``. Defaults to ``False``.

    Returns
    -------
    spatialdata.SpatialData or None
        Updated SpatialData with morphology columns added to ``sdata.table.obs``
        if ``copy=True``; otherwise ``None``.

    Raises
    ------
    ValueError
        If ``sdata`` does not contain a ``"table"`` AnnData object.
    """
    try:
        from skimage.measure import regionprops_table
    except ImportError as err:
        raise ImportError(
            "The 'scikit-image' package is required for add_morphological_metrics. Please install it with: pip install troutpy[morphology]"
        ) from err

    # 1. Handle Copy Logic
    obj = cp.deepcopy(sdata) if copy else sdata

    # 2. Extract labels array
    label_element = obj.labels[labels_key]
    if hasattr(label_element, "scale0"):
        labels_array = label_element["scale0"].image.values
    else:
        labels_array = label_element.values

    # 3. Dimensionality alignment
    labels_image = np.array(labels_array).squeeze()
    if labels_image.ndim > 3:
        labels_image = labels_image[0]

    # 4. Property Extraction
    properties = ["label", "area", "perimeter", "solidity", "major_axis_length", "equivalent_diameter"]

    props = regionprops_table(labels_image, properties=properties)
    df_morph = pd.DataFrame(props)

    # 5. Calculate Metrics
    df_morph["circularity"] = (4 * np.pi * df_morph["area"]) / (df_morph["perimeter"] ** 2)
    df_morph["protrusion_length"] = (df_morph["major_axis_length"] - df_morph["equivalent_diameter"]) / 2
    df_morph["morphological_complexity"] = 1 - df_morph["solidity"]

    # 6. Merge with Table
    if obj.table is not None:
        # Standardize the morphology index to string
        df_morph["label"] = df_morph["label"].astype(str)
        df_morph = df_morph.set_index("label")

        # Prepare the existing obs table
        # We ensure the index is string to match our label IDs
        existing_obs = obj.table.obs.copy()
        existing_obs.index = existing_obs.index.astype(str)

        # Drop overlapping columns so the join below doesn't raise a ValueError
        cols_to_drop = [c for c in df_morph.columns if c in existing_obs.columns]
        if cols_to_drop:
            existing_obs = existing_obs.drop(columns=cols_to_drop)

        # Join the new metrics
        obj.table.obs = existing_obs.join(df_morph, how="left")

        # Final safety check: if 'cell_id' column exists but isn't the index,
        # ensure it matches the strings too
        if "cell_id" in obj.table.obs.columns:
            obj.table.obs["cell_id"] = obj.table.obs["cell_id"].astype(str)
    else:
        raise ValueError("SpatialData object must contain a 'table'.")

    return obj if copy else None


def find_optimal_segmentation_free_bin_size(
    sdata,
    bin_sizes=(1, 2, 4, 8, 16, 32),
    cell_type_key="leiden",
    roi_size=120,
    min_div=0.85,
):
    """Find the optimal bin size for transcript aggregation by maximizing separability between intracellular and extracellular gene signatures.

    The function computes cosine similarity between binned transcript counts and known
    cell-type signatures, then uses Jensen-Shannon (JS) Divergence to identify the bin
    size that best distinguishes "cellular" signals from background noise. A spatial
    visualization is produced automatically.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object containing a ``"table"`` AnnData with cell-type annotations
        and a ``"transcripts"`` points layer with ``"gene"``, ``"x"``, ``"y"``, and
        ``"overlaps_cell"`` columns.
    bin_sizes : tuple of int, optional
        Pixel/unit sizes to test for spatial binning. Defaults to ``(1, 2, 4, 8, 16, 32)``.
    cell_type_key : str, optional
        Key in ``sdata["table"].obs`` containing cell-type labels used to build reference
        signatures. Defaults to ``"leiden"``.
    roi_size : int, optional
        Side length (in the same units as transcript coordinates) of the square Region of
        Interest centred on the median transcript position. Defaults to ``120``.
    min_div : float, optional
        Fraction of the maximum observed JS Divergence used as the acceptance threshold
        when selecting the optimal bin size. Defaults to ``0.85``.

    Returns
    -------
    results_df : pandas.DataFrame
        Long-format DataFrame with columns ``"bin_x"``, ``"bin_y"``, ``"cosine_sim"``,
        ``"is_cellular"``, and ``"bin_size"`` for every bin across all tested bin sizes.
    metrics_df : pandas.DataFrame
        DataFrame with columns ``"bin_size"`` and ``"js_divergence"`` for each tested size.
    optimal_bin : int
        The smallest bin size that achieves at least ``min_div`` of the maximum observed
        JS Divergence.
    """
    # --- 1. signatures & roi setup ---
    adata = sdata["table"]
    expr = adata.to_df()
    expr["cell_type_label"] = adata.obs[cell_type_key].values.astype(str)
    sig_df = expr.groupby("cell_type_label").mean()
    signatures = sig_df.values
    gene_names = sig_df.columns.tolist()
    gene_map = {gene: i for i, gene in enumerate(gene_names)}

    points_ref = sdata.points["transcripts"][["x", "y"]].compute()
    cx, cy = points_ref["x"].median(), points_ref["y"].median()
    x_range = (cx - roi_size / 2, cx + roi_size / 2)
    y_range = (cy - roi_size / 2, cy + roi_size / 2)

    points = sdata.points["transcripts"].compute()
    mask = (points["x"].between(*x_range)) & (points["y"].between(*y_range))
    transcripts = points[mask].copy()
    transcripts["gene_idx"] = transcripts["gene"].map(gene_map)
    df_pl = pls.from_pandas(transcripts.dropna(subset=["gene_idx"]))

    all_results = []
    metrics_list = []
    spatial_snapshots = {}

    # --- 2. iterative analysis ---
    for b in bin_sizes:
        binned = df_pl.with_columns(
            [
                ((pls.col("x") - x_range[0]) / b).floor().cast(pls.Int32).alias("bin_x"),
                ((pls.col("y") - y_range[0]) / b).floor().cast(pls.Int32).alias("bin_y"),
            ]
        )

        bin_agg = binned.group_by(["bin_x", "bin_y"]).agg(
            [
                pls.col("gene_idx"),
                pls.col("overlaps_cell").mean().alias("cell_fraction"),
            ]
        )

        # cosine similarity to cell-type signatures
        bin_gene_indices = bin_agg["gene_idx"].to_list()
        bin_count_matrix = np.zeros((len(bin_gene_indices), len(gene_names)))
        for i, idxs in enumerate(bin_gene_indices):
            bin_count_matrix[i, :] = np.bincount(idxs, minlength=len(gene_names))

        sims = np.max(cosine_similarity(bin_count_matrix, signatures), axis=1)

        df_bins = pd.DataFrame(
            {
                "bin_x": bin_agg["bin_x"],
                "bin_y": bin_agg["bin_y"],
                "cosine_sim": sims,
                "is_cellular": bin_agg["cell_fraction"] > 0.5,
                "bin_size": b,
            }
        )
        all_results.append(df_bins)
        spatial_snapshots[b] = df_bins

        # JS divergence between intracellular and extracellular score distributions
        intra = df_bins.loc[df_bins["is_cellular"], "cosine_sim"]
        extra = df_bins.loc[~df_bins["is_cellular"], "cosine_sim"]

        div = 0.0
        if len(intra) > 5 and len(extra) > 5:
            p, _ = np.histogram(intra, bins=50, range=(0, 1), density=True)
            q, _ = np.histogram(extra, bins=50, range=(0, 1), density=True)
            div = jensenshannon(p + 1e-9, q + 1e-9, base=2) ** 2
        metrics_list.append({"bin_size": b, "js_divergence": div})

    metrics_df = pd.DataFrame(metrics_list)
    results_df = pd.concat(all_results)

    # smallest bin achieving `min_div` of the max separability observed
    threshold = metrics_df["js_divergence"].max() * min_div
    optimal_bin = metrics_df[metrics_df["js_divergence"] >= threshold]["bin_size"].min()

    # --- 3. visualization ---
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)

    # metric plot
    ax0 = fig.add_subplot(gs[0, 0])
    sns.lineplot(data=metrics_df, x="bin_size", y="js_divergence", marker="o", ax=ax0)
    ax0.axvline(optimal_bin, color="green", ls="--")
    ax0.set_title("Separability (JS Divergence)")

    # global KDE for optimal bin
    ax1 = fig.add_subplot(gs[0, 1:])
    sns.kdeplot(
        data=spatial_snapshots[optimal_bin],
        x="cosine_sim",
        hue="is_cellular",
        fill=True,
        palette={True: "#2ecc71", False: "#e74c3c"},
        ax=ax1,
    )
    ax1.set_title(f"Score Distribution @ Bin Size {optimal_bin}")

    # spatial ROI maps
    snap = spatial_snapshots[optimal_bin]
    vmax = snap["cosine_sim"].quantile(0.95)

    titles = ["Total ROI Scores", "Intracellular (Target)", "Extracellular (Background)"]
    masks = [None, True, False]

    for i, (title, m) in enumerate(zip(titles, masks, strict=False)):
        ax = fig.add_subplot(gs[1, i])
        d = snap if m is None else snap[snap["is_cellular"] == m]

        im = ax.scatter(d["bin_x"], d["bin_y"], c=d["cosine_sim"], cmap="magma", s=15, marker="s", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    return results_df, metrics_df, optimal_bin


def define_urna_probability_stainings(
    sdata: sd.SpatialData,
    intensity_adata_key: str = "image_intensity_per_transcript",
    prob_threshold: float = 0.5,
    copy: bool = False,
):
    """Classify transcripts as extracellular (uRNA) from image-staining intensity similarity.

    Builds per-cell-type staining-intensity signatures from transcripts confidently
    assigned to a cell type, computes the cosine similarity of every transcript's
    staining-intensity profile to its closest matching signature, and uses
    grid-interpolated KDEs (as in :func:`define_urna_probability`) to derive
    ``P(uRNA | cosine_similarity)``.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object with a ``"transcripts"`` points layer containing
        ``overlaps_cell``, ``closest_cell_type``, and ``enrichment_class`` columns, and
        a table at ``intensity_adata_key`` with one row per transcript and the
        staining-intensity matrix in ``.X``.
    intensity_adata_key : str, optional
        Key of the table in ``sdata`` holding per-transcript image-intensity values.
        Defaults to ``"image_intensity_per_transcript"``.
    prob_threshold : float, optional
        Minimum ``prob_is_urna_stainings`` value for a transcript outside a cell to be
        classified as extracellular. Defaults to ``0.5``.
    copy : bool, optional
        If ``True``, operate on and return a deep copy of ``sdata``. Otherwise modify
        ``sdata`` in place and return ``None``. Defaults to ``False``.

    Returns
    -------
    spatialdata.SpatialData or None
        SpatialData with ``cosine_sim_stainings``, ``prob_is_urna_stainings``, and
        ``extracellular_stainings`` columns added to the ``"transcripts"`` layer if
        ``copy=True``; otherwise ``None``.
    """
    if copy:
        sdata = cp.deepcopy(sdata)

    transcripts = sdata.points["transcripts"].compute()
    adata = sdata[intensity_adata_key]
    intensity_matrix = adata.X

    # cell-type staining signatures from transcripts confidently assigned to a cell type
    cell_mask = transcripts["overlaps_cell"] & transcripts["closest_cell_type"].notna() & (transcripts["closest_cell_type"] != "UNASSIGNED")

    subset_transcripts = transcripts[cell_mask]
    ct_categories = subset_transcripts["closest_cell_type"].astype("category")
    indicator_matrix = pd.get_dummies(ct_categories).values.T

    sig_matrix = indicator_matrix @ intensity_matrix[cell_mask.values]
    sig_matrix = sig_matrix / indicator_matrix.sum(axis=1)[:, None]

    # cosine similarity of each transcript's staining profile to its closest signature
    print("Calculating cosine similarities...")
    all_similarities = cosine_similarity(intensity_matrix, sig_matrix)
    max_sims = all_similarities.max(axis=1)
    transcripts["cosine_sim_stainings"] = max_sims

    # reference populations: transcripts inside cells vs. background transcripts outside cells
    scores_inside = max_sims[transcripts["overlaps_cell"].values]
    outside_baseline_mask = ~transcripts["overlaps_cell"] & (transcripts["enrichment_class"] == "Background")
    scores_outside = max_sims[outside_baseline_mask.values]

    if len(scores_inside) > 20 and len(scores_outside) > 20:
        print(f"Calculating Bayesian distributions using {len(scores_outside)} background transcripts...")

        grid = np.linspace(0, 1, 500)
        kde_in = gaussian_kde(scores_inside)(grid)
        kde_out = gaussian_kde(scores_outside)(grid)

        prob_grid = kde_out / (kde_out + kde_in + 1e-10)
        f_interp = interp1d(grid, prob_grid, kind="linear", fill_value="extrapolate")

        transcripts["prob_is_urna_stainings"] = f_interp(max_sims)

        threshold_idx = np.argmin(np.abs(prob_grid - 0.5))
        print(f"Probabilistic boundary (P=0.5) at similarity: {grid[threshold_idx]:.4f}")
    else:
        print("Warning: Insufficient background/cell transcripts for KDE. Setting probability to 0.")
        transcripts["prob_is_urna_stainings"] = 0.0

    transcripts["extracellular_stainings"] = ~transcripts["overlaps_cell"] & (transcripts["prob_is_urna_stainings"] > prob_threshold)

    sdata.points["transcripts"] = sd.models.PointsModel.parse(transcripts)

    return sdata if copy else None


def get_transcript_categories(sdata, layer="transcripts", struct_table_key="structure_table", metadata_key="xrna_metadata"):
    """Classify transcripts into a hierarchy of intracellular/extracellular categories.

    Transcripts are split, in order, into: intracellular; cell-like (``extracellular`` is
    ``False`` despite being outside a cell), split by structural connectivity; high-density
    extracellular structures, split by connectivity; noise-spectrum genes (high
    ``fdr_noise`` in ``metadata_key``); and the remaining diffuse extracellular transcripts,
    split into diffusion-compatible and -incompatible genes based on the Kolmogorov-Smirnov
    p-value (``ks_pval``) in ``metadata_key``.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object containing the ``layer`` points layer, the ``struct_table_key``
        table with an ``is_physically_connected`` column indexed by ``"struct_<id>"``, and
        the ``metadata_key`` table whose ``.var`` holds the ``fdr_noise`` and ``ks_pval``
        columns.
    layer : str, optional
        Points layer with ``overlaps_cell``, ``extracellular``, ``enrichment_class``,
        ``structure_id``, and ``gene`` columns. Defaults to ``"transcripts"``.
    struct_table_key : str, optional
        Key of the table in ``sdata`` describing extracellular structures. Defaults to
        ``"structure_table"``.
    metadata_key : str, optional
        Key of the table in ``sdata`` holding per-gene uRNA metadata. Defaults to
        ``"xrna_metadata"``.

    Returns
    -------
    pandas.Series
        Transcript counts per category: ``"Intracellular"``, ``"Cell-Like Connected"``,
        ``"Cell-Like Unconnected"``, ``"High-Density Connected"``,
        ``"High-Density Unconnected"``, ``"Noise Spectrum"``, ``"Diffusion Compatible"``,
        and ``"Diffusion Incompatible"``.
    """
    df = sdata.points[layer].compute().copy()
    meta = sdata[metadata_key].var
    struct_obs = sdata[struct_table_key].obs

    # map each transcript's structure_id (int) to "struct_N" (str) connectivity flags
    conn_map = struct_obs["is_physically_connected"].to_dict()

    def map_connectivity(sid):
        if sid == -1:
            return False
        return conn_map.get(f"struct_{int(sid)}", False)

    df["is_conn"] = df["structure_id"].apply(map_connectivity)

    noise_genes = meta.index[meta["fdr_noise"] > 0.05] if "fdr_noise" in meta.columns else pd.Index([])

    if "ks_pval" in meta.columns:
        diff_comp_genes = meta.index[meta["ks_pval"] > 0.05]
    else:
        print(
            "Warning: 'ks_pval' column not found in metadata (likely due to zero remaining extracellular transcripts). Skipping diffusion compatibility filter."
        )
        diff_comp_genes = pd.Index([])

    counts = {}

    # A. intracellular
    intra_mask = df["overlaps_cell"]
    counts["Intracellular"] = intra_mask.sum()
    df_ext = df[~intra_mask]

    # B. cell-like (overlaps_cell=False and extracellular=False)
    cell_like_mask = ~df_ext["extracellular"]
    df_cl = df_ext[cell_like_mask]
    counts["Cell-Like Connected"] = df_cl["is_conn"].sum()
    counts["Cell-Like Unconnected"] = (~df_cl["is_conn"]).sum()

    # remaining are strictly extracellular
    df_rem = df_ext[~cell_like_mask]

    # C. high density (extracellular=True & enrichment_class=High Density)
    hd_mask = df_rem["enrichment_class"] == "High Density"
    df_hd = df_rem[hd_mask]
    counts["High-Density Connected"] = df_hd["is_conn"].sum()
    counts["High-Density Unconnected"] = (~df_hd["is_conn"]).sum()

    # transcripts not in high-density structures
    df_diffuse = df_rem[~hd_mask]

    # D. noise spectrum (FDR > 0.05)
    noise_mask = df_diffuse["gene"].isin(noise_genes)
    counts["Noise Spectrum"] = noise_mask.sum()
    df_signal = df_diffuse[~noise_mask]

    # E. diffusion compatibility (KS > 0.05); if diff_comp_genes is empty, everything is incompatible
    diff_mask = df_signal["gene"].isin(diff_comp_genes)
    counts["Diffusion Compatible"] = diff_mask.sum()
    counts["Diffusion Incompatible"] = (~diff_mask).sum()

    return pd.Series(counts)
