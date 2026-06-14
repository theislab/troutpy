import anndata as ad
import dask.dataframe as dd
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean
from scipy.stats import poisson, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from spatialdata import SpatialData
from spatialdata.models import TableModel
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def spatial_variability(
    sdata: SpatialData,
    coord_keys: list[str] | None = None,
    gene_key: str = "gene",
    n_neighbors: int = 10,
    kde_resolution: int = 1000,
    square_size: int = 20,
    n_threads: int = 1,
    method: str = "moran",
    copy: bool = False,
):
    """Compute spatial variability of extracellular RNA using Moran's I (or another autocorrelation statistic).

    Extracellular transcripts are binned onto a spatial grid with ``LazyKDE``, and
    spatial autocorrelation is computed per gene on the resulting grid using
    :mod:`squidpy`. Results are stored in ``sdata["xrna_metadata"].var``.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object with a ``"transcripts"`` points layer containing
        ``coord_keys``, ``"extracellular"``, and ``gene_key`` columns.
    coord_keys : list of str, optional
        Spatial coordinate column names. Defaults to ``["x", "y"]``.
    gene_key : str, optional
        Column in the transcript layer with gene identifiers. Defaults to ``"gene"``.
    n_neighbors : int, optional
        Number of neighbors used to build the spatial neighbor graph. Defaults to ``10``.
    kde_resolution : int, optional
        Grid resolution passed to ``LazyKDE``. Defaults to ``1000``.
    square_size : int, optional
        Bin size (in coordinate units) for the KDE grid. Defaults to ``20``.
    n_threads : int, optional
        Number of threads for LazyKDE processing. Defaults to ``1``.
    method : str, optional
        Spatial autocorrelation statistic passed to
        :func:`squidpy.gr.spatial_autocorr`. Defaults to ``"moran"``.
    copy : bool, optional
        If ``True``, return the updated SpatialData object; otherwise modify in
        place and return ``None``. Defaults to ``False``.

    Returns
    -------
    spatialdata.SpatialData or None
        Updated SpatialData with ``f"{method}_*"`` columns added to
        ``sdata["xrna_metadata"].var`` if ``copy=True``; otherwise ``None``.
    """
    try:
        from sainsc import LazyKDE
    except ImportError as err:
        raise ImportError("The 'sainsc' package is required for spatial_variability. Please install it with: pip install troutpy[spatial-stats]") from err
    try:
        import squidpy as sq
    except ImportError as err:
        raise ImportError("The 'squidpy' package is required for spatial_variability. Please install it with: pip install troutpy[spatial-stats]") from err

    if coord_keys is None:
        coord_keys = ["x", "y"]

    # Extract extracellular transcripts and bin them onto a spatial grid
    data = sdata.points["transcripts"][coord_keys + ["extracellular", gene_key]].compute()
    data = data[data["extracellular"]]
    data[gene_key] = data[gene_key].astype(str)
    data.columns = ["x", "y", "extracellular", "gene"]

    trans = pl.from_pandas(data)
    embryo = LazyKDE.from_dataframe(trans, resolution=kde_resolution, binsize=square_size, n_threads=n_threads)

    # Extract per-gene counts on the grid
    expr = embryo.counts.get(embryo.counts.genes()[0]).todense()
    allres = np.zeros([expr.size, len(embryo.counts.genes())])

    for n, gene in enumerate(tqdm(embryo.counts.genes(), desc="Extracting gene counts")):
        allres[:, n] = embryo.counts.get(gene).todense().flatten()

    x_coords, y_coords = np.meshgrid(np.arange(expr.shape[1]), np.arange(expr.shape[0]))

    adata = sc.AnnData(allres)
    adata.var.index = embryo.counts.genes()
    adata.obs["x"] = x_coords.flatten()
    adata.obs["y"] = y_coords.flatten()
    adata.obsm["spatial"] = np.array(adata.obs.loc[:, ["x", "y"]])

    sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors)
    sq.gr.spatial_autocorr(adata, mode=method, genes=adata.var_names)

    svg_df = pd.DataFrame(adata.uns["moranI"])
    svg_df.columns = [method + "_" + str(g) for g in svg_df.columns]
    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_urna_metadata(sdata, gene_key=gene_key)

    for column in svg_df.columns:
        if column in sdata["xrna_metadata"].var.columns:
            sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.drop([column], axis=1)

    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(svg_df)

    return sdata if copy else None


def create_urna_metadata(sdata: SpatialData, layer: str = "transcripts", gene_key: str = "gene", copy: bool = False) -> SpatialData | None:
    """Create the ``"xrna_metadata"`` table holding the unique genes found in a transcripts layer.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        The SpatialData object to modify.
    layer : str, optional
        Layer in ``sdata.points`` from which to extract gene names. Defaults to
        ``"transcripts"``.
    gene_key : str, optional
        Column in ``sdata.points[layer]`` that contains the gene names. Defaults to
        ``"gene"``.
    copy : bool, optional
        If ``True``, return a copy of the modified SpatialData object. Defaults to
        ``False``.

    Returns
    -------
    spatialdata.SpatialData or None
        A copy of the modified SpatialData object if ``copy=True``; otherwise
        ``sdata`` is modified in place (a new ``"xrna_metadata"`` table is added)
        and ``None`` is returned.

    Raises
    ------
    ValueError
        If ``layer`` is not in ``sdata.points``, or if ``gene_key`` is not a column
        of ``sdata.points[layer]``.
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


def quantify_overexpression(
    sdata: SpatialData,
    codeword_key: str,
    control_codewords: list,
    gene_key: str = "gene",
    layer: str = "transcripts",
    copy: bool = False,
) -> SpatialData:
    """Quantify gene overexpression relative to a Poisson noise model derived from control codewords.

    For each gene, computes the observed count, log fold-change over the noise
    baseline (mean control count), a one-sided Poisson survival-function p-value,
    and Benjamini–Hochberg FDR correction. Results are stored in
    ``sdata["xrna_metadata"].var``.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object containing a points layer with transcripts and an
        ``"xrna_metadata"`` table (created if absent).
    codeword_key : str
        Column in the transcript points layer that holds the codeword category
        used to identify control probes.
    control_codewords : list of str
        Codeword category values that identify control (noise) probes.
    gene_key : str, optional
        Column in the transcript points layer containing gene identifiers.
        Defaults to ``"gene"``.
    layer : str, optional
        Key in ``sdata.points`` holding the transcript data. Defaults to ``"transcripts"``.
    copy : bool, optional
        If ``True``, return the updated SpatialData object; otherwise modify in
        place and return ``None``. Defaults to ``False``.

    Returns
    -------
    spatialdata.SpatialData or None
        Updated SpatialData with columns ``"count"``, ``"logfoldchange_over_noise"``,
        ``"p_val_noise"``, ``"is_control"``, and ``"fdr_noise"`` added to
        ``sdata["xrna_metadata"].var`` if ``copy=True``; otherwise ``None``.
    """
    data = sdata.points[layer][np.unique([codeword_key, gene_key]).tolist()].compute()

    if isinstance(control_codewords, str):
        control_codewords = [control_codewords]

    # Map each gene to whether it is a control probe
    is_control_mask = data[codeword_key].isin(control_codewords)
    n2c = dict(zip(data[gene_key], is_control_mask, strict=False))
    gene_counts = data[gene_key].value_counts()

    # Noise baseline (lambda) from control probe counts
    control_obs = [gene_counts[g] for g in gene_counts.index if n2c.get(g, True)]
    lambda_noise = np.mean(control_obs) if control_obs else 1e-6

    results = []
    for g, count in gene_counts.items():
        # p = P(Noise >= observed_count)
        p_val = poisson.sf(count - 1, lambda_noise)

        # Log fold change over noise, with a pseudocount of 1 to avoid log(0)
        lfc = np.log((count + 1) / (lambda_noise + 1))

        results.append(
            {
                "gene": g,
                "count": count,
                "logfoldchange_over_noise": lfc,
                "p_val_noise": p_val,
                "is_control": n2c.get(g, True),
            }
        )

    scores_df = pd.DataFrame(results).set_index("gene")

    _, fdr_adj, _, _ = multipletests(scores_df["p_val_noise"], method="fdr_bh")
    scores_df["fdr_noise"] = fdr_adj

    if "xrna_metadata" not in sdata:
        var_df = pd.DataFrame(index=gene_counts.index)
        sdata["xrna_metadata"] = TableModel.parse(ad.AnnData(None, var=var_df))

    existing_cols = sdata["xrna_metadata"].var.columns
    for col in scores_df.columns:
        if col in existing_cols:
            sdata["xrna_metadata"].var.drop(columns=[col], inplace=True)

    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(scores_df)

    return sdata if copy else None


def extracellular_enrichment(sdata: SpatialData, gene_key: str = "gene", copy: bool = False, layer: str = "transcripts") -> SpatialData | None:
    """Compute the proportion of extracellular vs. intracellular transcripts per gene.

    For each gene, calculates the proportion of transcripts classified as extracellular
    and intracellular, the fold ratio between the two, and its log fold change. Results
    are stored as new columns in ``sdata["xrna_metadata"].var``.

    Parameters
    ----------
    sdata
        SpatialData object. ``sdata.points[layer]`` must contain a gene identifier column
        (``gene_key``) and a boolean ``extracellular`` column.
    gene_key
        Column in ``sdata.points[layer]`` containing gene identifiers.
    copy
        If `True`, return a modified copy of `sdata`. Otherwise modify in place.
    layer
        Key of the points element in `sdata` containing the transcripts.

    Returns
    -------
    If `copy=True`, a modified copy of `sdata`. Otherwise `None`, modifying `sdata` in place.

    Notes
    -----
    If ``sdata["xrna_metadata"]`` does not exist, it is created via :func:`create_urna_metadata`.
    """
    data = sdata.points[layer][[gene_key, "extracellular"]].compute()

    feature_inout = pd.crosstab(data[gene_key], data["extracellular"])
    norm_counts = feature_inout.div(feature_inout.sum(axis=0), axis=1)
    norm_counts["extracellular_foldratio"] = norm_counts[False] / norm_counts[True]

    extracellular_proportion = feature_inout.div(feature_inout.sum(axis=1), axis=0)
    extracellular_proportion.columns = extracellular_proportion.columns.map({True: "extracellular_proportion", False: "intracellular_proportion"})
    extracellular_proportion["logfoldratio_extracellular"] = np.log(norm_counts["extracellular_foldratio"])

    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_urna_metadata(sdata, layer=layer, gene_key=gene_key)

    existing_cols = sdata["xrna_metadata"].var.columns
    for col in extracellular_proportion.columns:
        if col in existing_cols:
            sdata["xrna_metadata"].var.drop(columns=[col], inplace=True)
    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(extracellular_proportion)

    return sdata if copy else None


def spatial_colocalization(
    sdata: SpatialData,
    coord_keys: list[str] | None = None,
    gene_key: str = "gene",
    resolution: int = 1000,
    square_size: int = 20,
    n_threads: int = 1,
    threshold_colocalized: int = 1,
    copy: bool = False,
):
    """Compute the proportion of spatially colocalized extracellular transcripts per gene.

    Uses kernel density estimation (LazyKDE) to bin transcripts into a spatial grid,
    then calculates for each gene the fraction of bins whose count exceeds
    ``threshold_colocalized``. Results are stored in
    ``sdata["xrna_metadata"].var["proportion_of_colocalized"]``.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object with a ``"transcripts"`` points layer containing
        ``"extracellular"`` and ``gene_key`` columns, and an ``"xrna_metadata"``
        table (created automatically if absent).
    coord_keys : list of str, optional
        Spatial coordinate column names. Defaults to ``["x", "y"]``.
    gene_key : str, optional
        Column in the transcript layer with gene identifiers. Defaults to ``"gene"``.
    resolution : int, optional
        Grid resolution passed to ``LazyKDE``. Defaults to ``1000``.
    square_size : int, optional
        Bin size (in coordinate units) for the KDE grid. Defaults to ``20``.
    n_threads : int, optional
        Number of threads for LazyKDE processing. Defaults to ``1``.
    threshold_colocalized : int, optional
        Minimum per-bin count for a bin to be considered colocalized.
        Defaults to ``1``.
    copy : bool, optional
        If ``True``, return the updated SpatialData object; otherwise modify in
        place and return ``None``. Defaults to ``False``.

    Returns
    -------
    spatialdata.SpatialData or None
        Updated SpatialData with ``"proportion_of_colocalized"`` added to
        ``sdata["xrna_metadata"].var`` if ``copy=True``; otherwise ``None``.
    """
    try:
        from sainsc import LazyKDE
    except ImportError as err:
        raise ImportError("The 'sainsc' package is required for spatial_colocalization. Please install it with: pip install troutpy[spatial-stats]") from err

    if coord_keys is None:
        coord_keys = ["x", "y"]

    data = sdata.points["transcripts"][coord_keys + ["extracellular", gene_key]].compute()
    data = data[data["extracellular"]]
    data[gene_key] = data[gene_key].astype(str)
    data.columns = ["x", "y", "extracellular", "gene"]

    trans = pl.from_pandas(data)
    embryo = LazyKDE.from_dataframe(trans, resolution=resolution, binsize=square_size, n_threads=n_threads)

    expr = embryo.counts.get(embryo.counts.genes()[0]).todense()
    allres = np.zeros([expr.size, len(embryo.counts.genes())])

    for n, gene in enumerate(tqdm(embryo.counts.genes(), desc="Extracting gene counts")):
        allres[:, n] = embryo.counts.get(gene).todense().flatten()

    x_coords, y_coords = np.meshgrid(np.arange(expr.shape[1]), np.arange(expr.shape[0]))

    adata = sc.AnnData(allres)
    adata.var.index = embryo.counts.genes()
    adata.obs["x"] = x_coords.flatten()
    adata.obs["y"] = y_coords.flatten()
    adata.obsm["spatial"] = np.array(adata.obs.loc[:, ["x", "y"]])

    # Proportion of gene-positive bins that exceed the colocalization threshold
    positive_counts = np.sum(adata.X > 0, axis=0)
    colocalized_counts = np.sum(adata.X > threshold_colocalized, axis=0)
    proportions = np.divide(colocalized_counts, positive_counts, where=(positive_counts > 0))
    coloc = pd.DataFrame(data=proportions, index=adata.var.index, columns=["proportion_of_colocalized"])
    for column in coloc.columns:
        if column in sdata["xrna_metadata"].var.columns:
            sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.drop([column], axis=1)

    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(coloc)

    return sdata if copy else None


def in_out_correlation(
    sdata: SpatialData,
    extracellular_layer: str = "segmentation_free_table",
    cellular_layer: str = "table",
    n_neighbors: int = 5,
    copy: bool | None = None,
) -> SpatialData | None:
    """Compute the correlation between intracellular and extracellular gene expression.

    For each cell, extracellular expression is aggregated as the mean over its
    `n_neighbors` nearest extracellular bins, then correlated (Spearman) against
    the cell's intracellular expression for each shared gene. Results are stored
    in ``sdata["xrna_metadata"].var["in_out_spearmanR"]`` and
    ``sdata["xrna_metadata"].var["in_out_pvalue"]``.

    Parameters
    ----------
    sdata
        SpatialData object containing both the extracellular and cellular AnnData tables.
    extracellular_layer
        Key of the extracellular AnnData table in `sdata`.
    cellular_layer
        Key of the cellular AnnData table in `sdata`.
    n_neighbors
        Number of nearest extracellular bins to aggregate per cell.
    copy
        If `True`, return a modified copy of `sdata`. Otherwise modify in place.

    Returns
    -------
    If `copy=True`, a modified copy of `sdata`. Otherwise `None`, modifying `sdata` in place.
    """
    try:
        adata_extracellular = sdata[extracellular_layer]
    except KeyError:
        print(
            "Extracellular layer not found. Please make ensure ´extracellular_layer´ and that extracellular grouping has been performed. Otherwise, please run trouty.tl.aggregate_extracellular_transcripts"
        )
    adata_cellular = sdata[cellular_layer]

    coords_cellular = adata_cellular.obsm["spatial"]
    coords_extracellular = adata_extracellular.obsm["spatial"]

    shared_genes = adata_cellular.var_names.intersection(adata_extracellular.var_names)
    adata_cellular = adata_cellular[:, shared_genes]
    adata_extracellular = adata_extracellular[:, shared_genes]

    extracellular_tree = cKDTree(coords_extracellular)
    _, nearest_indices = extracellular_tree.query(coords_cellular, k=n_neighbors)

    expr_cellular = adata_cellular.X
    expr_extracellular = adata_extracellular.X

    if not isinstance(expr_cellular, np.ndarray):
        expr_cellular = expr_cellular.toarray()
    if not isinstance(expr_extracellular, np.ndarray):
        expr_extracellular = expr_extracellular.toarray()

    aggregated_extracellular = np.array([expr_extracellular[nearest_indices[i]].mean(axis=0) for i in range(expr_cellular.shape[0])])

    correlations = []
    for i, gene in enumerate(shared_genes):
        cell_expr = expr_cellular[:, i]
        ext_expr = aggregated_extracellular[:, i]

        # Skip the Spearman test for genes with no expression on either side
        if np.any(cell_expr) and np.any(ext_expr):
            corr, pval = spearmanr(cell_expr, ext_expr)
        else:
            corr, pval = np.nan, np.nan

        correlations.append([gene, corr, pval])

    correlation_results = pd.DataFrame(correlations, columns=["Gene", "SpearmanR", "PValue"])
    correlation_results.set_index("Gene", inplace=True)

    gene2spearman = dict(zip(correlation_results.index, correlation_results["SpearmanR"], strict=False))
    gene2pval = dict(zip(correlation_results.index, correlation_results["PValue"], strict=False))
    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_urna_metadata(sdata)
    sdata["xrna_metadata"].var["in_out_spearmanR"] = sdata["xrna_metadata"].var.index.map(gene2spearman)
    sdata["xrna_metadata"].var["in_out_pvalue"] = sdata["xrna_metadata"].var.index.map(gene2pval)

    return sdata if copy else None


def assess_diffusion(
    sdata: SpatialData,
    gene_key: str = "gene",
    distance_key: str = "distance_to_source",
    filters: dict | None = None,
    min_transcripts: int = 15,
    copy: bool = False,
):
    """Fit a 2D Rayleigh diffusion model to transcript distances from their source cell.

    For each gene, fits a Rayleigh distribution to the distances of its transcripts
    from their assigned source cell (``sdata["source_score"].obs[distance_key]``) and
    evaluates the fit against the empirical distribution. Results are stored as new
    columns in ``sdata["xrna_metadata"].var``.

    Parameters
    ----------
    sdata
        SpatialData object with ``"source_score"`` and ``"transcripts"`` tables.
    gene_key
        Column in ``sdata["source_score"].obs`` containing gene identifiers.
    distance_key
        Column in ``sdata["source_score"].obs`` containing the distance of each
        transcript from its source cell.
    filters
        Optional filters applied to ``sdata["transcripts"]`` before matching against
        `source_score`. Each entry maps a column name to either a value to match, or a
        `(value, False)` tuple to exclude transcripts equal to that value, e.g.
        ``{"extracellular": True, "enrichment_class": ("High Density", False)}``.
    min_transcripts
        Minimum number of transcripts required for a gene to be fitted.
    copy
        If `True`, return a modified copy of `sdata`. Otherwise modify in place.

    Returns
    -------
    If `copy=True`, a modified copy of `sdata`. Otherwise `None`, modifying `sdata` in place.

    Notes
    -----
    Adds the columns ``ks_stat``, ``ks_pval``, ``lr_stat``, ``mean_displacement``,
    ``n_transcripts``, ``sigma_est`` and ``-log_ks_pval`` to
    ``sdata["xrna_metadata"].var`` for genes that pass `min_transcripts`.
    """
    if "source_score" not in sdata or "transcripts" not in sdata:
        raise KeyError("Required tables 'source_score' or 'transcripts' missing from sdata.")

    filter_cols = list(filters.keys()) if filters else []
    print("--- Diffusion Analysis Debug ---")
    print(f"Initial transcripts: {len(sdata['transcripts'])}")

    t_df_subset = sdata["transcripts"][filter_cols].compute()

    valid_mask = np.ones(len(t_df_subset), dtype=bool)
    if filters:
        for col, criterion in filters.items():
            if isinstance(criterion, tuple) and len(criterion) == 2 and criterion[1] is False:
                valid_mask &= t_df_subset[col] != criterion[0]
            else:
                valid_mask &= t_df_subset[col] == criterion

    # Cast to string so transcript IDs match source_score's index dtype
    valid_ids = t_df_subset.index[valid_mask].astype(str)
    print(f"Transcripts passing filters: {len(valid_ids)}")

    source_obs = sdata["source_score"].obs
    if hasattr(source_obs, "compute"):
        source_obs = source_obs.compute()
    source_obs.index = source_obs.index.astype(str)

    shared_ids = source_obs.index.intersection(valid_ids)
    print(f"Transcripts shared with source_score: {len(shared_ids)}")

    if len(shared_ids) == 0:
        print("RESULT: No transcripts matching criteria. Check filter stringency or column names.")
        return sdata if copy else None

    filtered_obs = source_obs.loc[shared_ids]
    results = []

    for gene, group in filtered_obs.groupby(gene_key):
        distances = group[distance_key].dropna().values

        if len(distances) < min_transcripts:
            continue

        try:
            # Rayleigh is the physical model for 2D diffusion from a point source
            param = stats.rayleigh.fit(distances)
            ks_stat, ks_pval = stats.kstest(distances, "rayleigh", args=param)

            # Likelihood ratio of the Rayleigh fit vs. a kernel density estimate
            log_likelihood_ray = np.sum(stats.rayleigh.logpdf(distances, *param))
            log_likelihood_emp = np.sum(stats.gaussian_kde(distances).logpdf(distances))
            lr_stat = -2 * (log_likelihood_ray - log_likelihood_emp)

            results.append(
                {
                    "gene": gene,
                    "ks_stat": ks_stat,
                    "ks_pval": ks_pval,
                    "lr_stat": lr_stat,
                    "mean_displacement": np.mean(distances),
                    "n_transcripts": len(distances),
                    "sigma_est": param[1],
                }
            )
        except Exception:  # noqa: BLE001
            # Skip genes that fail fitting (e.g. zero variance)
            continue

    if not results:
        print(f"RESULT: No genes passed the min_transcripts threshold of {min_transcripts}.")
        return sdata if copy else None

    diff_df = pd.DataFrame(results).set_index("gene")
    diff_df["-log_ks_pval"] = -np.log10(diff_df["ks_pval"].clip(lower=1e-300))

    if "xrna_metadata" in sdata:
        if hasattr(sdata["xrna_metadata"].var, "compute"):
            sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.compute()

        existing_cols = [c for c in diff_df.columns if c in sdata["xrna_metadata"].var.columns]
        if existing_cols:
            sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.drop(columns=existing_cols)

        sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(diff_df)
        print(f"SUCCESS: Statistics added for {len(diff_df)} genes.")

    return sdata if copy else None


def cluster_distribution_from_source(
    sdata: SpatialData, gene_key: str = "gene", distance_key: str = "distance", n_clusters: int = 3, n_bins: int = 20, copy: bool = False
):
    """Cluster genes by the distribution of their transcripts' distances to source cells.

    For each gene in ``sdata["source_score"].obs``, computes a normalized histogram
    (with `n_bins` bins) of `distance_key` values, standardizes these histogram
    vectors, and clusters them with KMeans. Results are stored in
    ``sdata["xrna_metadata"].var["kmeans_distribution"]``.

    Parameters
    ----------
    sdata
        SpatialData object containing a `"source_score"` table with an `obs` DataFrame.
    gene_key
        Column in `sdata["source_score"].obs` containing gene identifiers.
    distance_key
        Column in `sdata["source_score"].obs` containing the distance from the source cell.
    n_clusters
        Number of KMeans clusters to form.
    n_bins
        Number of histogram bins used to represent each gene's distance distribution.
    copy
        If `True`, return a modified copy of `sdata`. Otherwise modify in place.

    Returns
    -------
    If `copy=True`, a modified copy of `sdata`. Otherwise `None`, modifying `sdata` in place.
    """
    obs_df = sdata["source_score"].obs

    # Common bin edges across genes, assuming distances start at 0
    global_max = obs_df[distance_key].max()
    bin_edges = np.linspace(0, global_max, n_bins + 1)

    gene_hist_features = {}
    for gene, group in obs_df.groupby(gene_key):
        distances = group[distance_key].dropna().values
        if len(distances) == 0:
            continue
        counts, _ = np.histogram(distances, bins=bin_edges)
        norm_counts = counts / counts.sum() if counts.sum() > 0 else counts
        gene_hist_features[gene] = norm_counts

    hist_df = pd.DataFrame.from_dict(gene_hist_features, orient="index")

    scaler = StandardScaler()
    hist_scaled = scaler.fit_transform(hist_df.values)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(hist_scaled)

    gene_cluster_dict = dict(zip(hist_df.index, clusters, strict=False))
    if "xrna_metadata" not in sdata:
        create_urna_metadata(sdata, gene_key=gene_key)

    sdata["xrna_metadata"].var["kmeans_distribution"] = list(sdata["xrna_metadata"].var.index.map(gene_cluster_dict))

    return sdata if copy else None


def compute_js_divergence(P: np.ndarray, Q: np.ndarray, eps: float = 1e-10) -> float:
    """Compute the Jensen-Shannon divergence between two probability distributions.

    Parameters
    ----------
    P
        First probability distribution.
    Q
        Second probability distribution, same shape as `P`.
    eps
        Small constant added to `P` and `Q` to avoid division by zero.

    Returns
    -------
    The Jensen-Shannon divergence between `P` and `Q`.
    """
    P, Q = P + eps, Q + eps
    M = 0.5 * (P + Q)
    return 0.5 * (np.sum(P * np.log(P / M)) + np.sum(Q * np.log(Q / M)))


def compare_intra_extra_distribution(
    sdata: SpatialData,
    layer: str = "transcripts",
    gene_key: str = "gene",
    copy: bool = False,
    coord_keys: list[str] | None = None,
    n_bins: int = 30,
) -> SpatialData | None:
    """Compare the spatial distribution of intracellular and extracellular transcripts per gene.

    For each gene, computes the centroid shift between its intracellular and
    extracellular transcripts, and the correlation and Jensen-Shannon divergence
    between their spatial density histograms. Results are stored in
    ``sdata["xrna_metadata"].var``.

    Parameters
    ----------
    sdata
        SpatialData object containing transcript locations and metadata.
    layer
        Key of the points element in `sdata` containing the transcripts.
    gene_key
        Column in `sdata[layer]` containing gene identifiers.
    copy
        If `True`, return a modified copy of `sdata`. Otherwise modify in place.
    coord_keys
        Column names for the x and y transcript coordinates. Defaults to `["x", "y"]`.
    n_bins
        Number of bins per axis for the 2D spatial density histograms.

    Returns
    -------
    If `copy=True`, a modified copy of `sdata`. Otherwise `None`, modifying `sdata` in place.

    Notes
    -----
    Genes with fewer than 5 intracellular or 5 extracellular transcripts are skipped.
    """
    if coord_keys is None:
        coord_keys = ["x", "y"]
    transcripts_df = sdata[layer]
    if isinstance(transcripts_df, dd.DataFrame):
        transcripts_df = transcripts_df.compute()

    results = []

    for gene, gene_transcripts in transcripts_df.groupby(gene_key):
        intracellular = gene_transcripts[~gene_transcripts["extracellular"]]
        extracellular = gene_transcripts[gene_transcripts["extracellular"]]

        if intracellular.shape[0] < 5 or extracellular.shape[0] < 5:
            continue

        intracellular_centroid = intracellular[coord_keys].mean().values
        extracellular_centroid = extracellular[coord_keys].mean().values
        centroid_shift_distance = euclidean(intracellular_centroid, extracellular_centroid)

        all_coords = gene_transcripts[coord_keys].values
        x_min, y_min = np.min(all_coords, axis=0)
        x_max, y_max = np.max(all_coords, axis=0)
        range_x, range_y = (x_min, x_max), (y_min, y_max)

        hist_intra, _, _ = np.histogram2d(intracellular[coord_keys[0]], intracellular[coord_keys[1]], bins=n_bins, range=[range_x, range_y])
        hist_extra, _, _ = np.histogram2d(extracellular[coord_keys[0]], extracellular[coord_keys[1]], bins=n_bins, range=[range_x, range_y])

        P_intra = hist_intra / np.sum(hist_intra)
        P_extra = hist_extra / np.sum(hist_extra)

        intra_flat = P_intra.flatten()
        extra_flat = P_extra.flatten()

        spatial_density_correlation = np.corrcoef(intra_flat, extra_flat)[0, 1]
        spatial_js_divergence = compute_js_divergence(intra_flat, extra_flat)

        results.append(
            {
                gene_key: gene,
                "centroid_shift_distance": centroid_shift_distance,
                "spatial_density_correlation": spatial_density_correlation,
                "spatial_js_divergence": spatial_js_divergence,
            }
        )

    results_df = pd.DataFrame(results).set_index(gene_key)

    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_urna_metadata(sdata, layer="transcripts")

    for column in results_df.columns:
        if column in sdata["xrna_metadata"].var.columns:
            sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.drop(columns=[column])

    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(results_df)
    return sdata if copy else None
