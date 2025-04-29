import dask.dataframe as dd
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats
import spatialdata as sd
import squidpy as sq
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean
from scipy.stats import poisson, spearmanr
from spatialdata import SpatialData

from ..pp.aggregate import aggregate_extracellular_transcripts


def spatial_variability(
    sdata: SpatialData,
    layer: str = "transcripts",
    gene_key: str = "gene",
    aggr_method: str = "bin",
    square_size: float = 50,
    key_added: str | None = None,
    n_neighbors: int = 10,
    method: str = "moran",
    copy: bool = False,
):
    """
    Computes spatial variability of extracellular RNA using Moran's I.

    Parameters
    ----------
    sdata
        The spatial data object.
    layer
        The key to access transcript coordinates in sdata.
    gene_key
        Column name where the gene assigned to each transcript is stored
    aggr_method
        Strategy employed to aggregate extracellular transcripts
    square_size
        The size of each square grid bin.
    key_added
        Name of the table where to store the grouped extracellular transcripts .Default is 'segmentation_free_table'
    n_neighbors
        Number of neighbors to use for computing spatial neighbors (default: 10).
    method
        The mode for spatial autocorrelation computation (default: "moran").
    copy
        Wether to return the sdata as a new object

    Returns
    -------
    - sdata(SpatialData)
        Sdata containing Moran's I values for each gene, indexed by gene names.
    """
    try:
        sdata["segmentation_free_table"]
        print("Using precomputed segmentation free table")
    except:
        print("Computing segmentation-free aggregation of extracellular transcripts...")
        aggregate_extracellular_transcripts(sdata, layer, gene_key, aggr_method, square_size, copy, key_added)

    adata = sc.AnnData(sdata["segmentation_free_table"])
    adata.var.index = sdata["segmentation_free_table"].var_names
    adata.obsm["spatial"] = sdata["segmentation_free_table"].obsm["spatial"]

    sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors)
    sq.gr.spatial_autocorr(adata, mode=method, genes=adata.var_names)

    # Extract Moran's I values
    svg_df = pd.DataFrame(adata.uns["moranI"])
    svg_df.columns = [method + "_" + str(g) for g in svg_df.columns]
    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_xrna_metadata(sdata, layer="transcripts", gene_key=gene_key)

    for column in svg_df.columns:
        if column in sdata["xrna_metadata"].var.columns:
            sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.drop([column], axis=1)

    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(svg_df)

    return sdata if copy else None


def create_xrna_metadata(sdata: SpatialData, layer: str = "transcripts", gene_key: str = "gene", copy: bool = False) -> SpatialData | None:
    """
    Creates a new table within the SpatialData object that contains a 'gene' column with the unique gene names extracted from the specified points layer.

    Parameters
    ----------
    - sdata (SpatialData)
        The SpatialData object to modify.
    - layer (str, optional)
        The name of the layer in `sdata.points` from which to extract gene names. Default is 'transcripts'.
    - gene_key (str, optional)
        The key in the `layer` dataframe that contains the gene names.Default is 'feature_name'.
    - copy
        If `True`, returns a copy of the `SpatialData` object with the new table added.

    Returns
    -------
    - SpatialData | None
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


def quantify_overexpression(
    sdata: SpatialData,
    codeword_key: str = "control_probe",
    control_codewords: list = [True],
    gene_key: str = "gene",
    layer: str = "transcripts",
    percentile_threshold: float = 100,
    copy=False,
) -> SpatialData:
    """
    Compare counts per gene with counts per non-gene feature. Additionally, perform a Poisson test
    to check if gene extracellular expression is significantly higher than controls.
    """
    # Compute the data
    data = sdata.points[layer][np.unique(["extracellular", codeword_key, gene_key]).tolist()].compute()

    # Ensure control_codewords is a list
    if isinstance(control_codewords, str):
        control_codewords = [control_codewords]

    # Get counts per control feature
    is_control = list(data[codeword_key].isin(control_codewords))
    control_data = data[data[codeword_key].isin(control_codewords)]
    # gene_data = data[~data[codeword_key].isin(control_codewords)]

    control_counts = control_data[gene_key].value_counts()
    gene_counts = data[gene_key].value_counts()  # we include all because we will test for all
    n2c = dict(zip(data[gene_key], is_control, strict=False))
    bool_control = []
    for g in gene_counts.keys():
        try:
            bool_control.append(n2c[g])
        except:
            bool_control.append(True)

    # Compute percentile-based threshold
    threshold = np.percentile(control_counts.values, percentile_threshold)

    # Calculate log fold ratio
    scores_per_gene = pd.DataFrame({"count": gene_counts, "logfoldratio_over_noise": np.log(gene_counts / threshold), "control_probe": bool_control})

    # Perform Poisson test
    p_values = {}
    control_mean = np.mean(control_counts.values)  # Expected background noise
    for gene in gene_counts.index:
        observed_count = gene_counts[gene]
        p_value = 1 - poisson.cdf(observed_count - 1, control_mean)  # One-tailed test
        p_values[gene] = p_value

    scores_per_gene["p_value_Poisson"] = scores_per_gene.index.map(p_values)

    # Store results in xrna_metadata
    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_xrna_metadata(sdata, layer=layer, gene_key=gene_key)

    # select columns that will be overwritten
    existing_cols = sdata["xrna_metadata"].var.columns
    for col in scores_per_gene.columns:
        if col in existing_cols:
            sdata["xrna_metadata"].var.drop(columns=[col], inplace=True)

    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(scores_per_gene)

    return sdata if copy else None


def extracellular_enrichment(sdata: SpatialData, gene_key: str = "gene", copy: bool = False, layer: str = "transcripts"):
    """
    Computes the proportion of transcripts classified as extracellular or intracellular for each gene and calculates additional metrics, including log fold change of extracellular to intracellular proportions. The results are integrated into the `sdata` object under the 'xrna_metadata' layer.

    Parameters
    ----------
    sdata
        An spatialData object containing spatial transcriptomics data. The `points` attribute should include a 'transcripts' DataFrame with columns for gene IDs (specified by `gene_key`) and a boolean 'extracellular' column indicating whether each transcript is classified as extracellular.
    gene_key
        The name of the column in the 'transcripts' DataFrame containing gene identifiers. Defaults to 'feature_name'.
    copy
        Whether to return a modified copy of the input `sdata` object. If `False`, the input object is modified in place. Defaults to `False`.

    Returns
    -------
    If `copy=True`, returns a modified copy of the input `sdata` object with updated metadata. Otherwise, modifies `sdata` in place and returns `None`.

    Notes
    -----
    - The function assumes that the `sdata` object has a 'points' layer containing a 'transcripts' DataFrame.
    - If the 'xrna_metadata' attribute does not exist in `sdata`, it will be created using the `create_xrna_metadata` function.
    """
    # Extract and compute the required data
    data = sdata.points[layer][[gene_key, "extracellular"]].compute()

    # Create a crosstab to count occurrences of intracellular and extracellular transcripts
    feature_inout = pd.crosstab(data[gene_key], data["extracellular"])
    norm_counts = feature_inout.div(feature_inout.sum(axis=0), axis=1)
    norm_counts["extracellular_foldratio"] = norm_counts[False] / norm_counts[True]

    extracellular_proportion = feature_inout.div(feature_inout.sum(axis=1), axis=0)
    extracellular_proportion.columns = extracellular_proportion.columns.map({True: "extracellular_proportion", False: "intracellular_proportion"})
    extracellular_proportion["logfoldratio_extracellular"] = np.log(norm_counts["extracellular_foldratio"])

    # Ensure the 'xrna_metadata' attribute exists
    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_xrna_metadata(sdata, layer=layer, gene_key=gene_key)

    # select columns that will be overwritten
    existing_cols = sdata["xrna_metadata"].var.columns
    for col in extracellular_proportion.columns:
        if col in existing_cols:
            sdata["xrna_metadata"].var.drop(columns=[col], inplace=True)
    # Join the results to the metadata
    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(extracellular_proportion)

    return sdata if copy else None


def spatial_colocalization(
    sdata: SpatialData,
    extracellular_layer: str = "segmentation_free_table",
    threshold_colocalized: int = 1,
    copy: bool = False,
) -> SpatialData | None:
    """
    Computes the proportion of colocalized transcripts for extracellular RNA
    based on counts stored in a specified AnnData layer.

    Parameters
    ----------
    sdata : SpatialData
        The spatial transcriptomics dataset in SpatialData format.

    extracellular_layer : str, default="segmentation_free_table"
        The key in `sdata` pointing to an AnnData object containing extracellular transcript counts.

    threshold_colocalized : int, default=1
        Minimum count of a gene at a spot to consider it as colocalized.

    copy : bool, default=False
        If True, returns a modified copy of sdata, otherwise modifies in place.

    Returns
    -------
    SpatialData or None
        Modified SpatialData with `proportion_of_colocalized` values added to
        `sdata['xrna_metadata'].var`.
    """
    import numpy as np
    import pandas as pd

    if extracellular_layer not in sdata:
        raise KeyError(f"'{extracellular_layer}' not found in sdata.")

    adata = sdata[extracellular_layer]
    if not isinstance(adata, sc.AnnData):
        raise TypeError(f"'{extracellular_layer}' must be an AnnData object.")

    X = adata.X
    if hasattr(X, "toarray"):  # handle sparse matrices
        X = X.toarray()

    positive_counts = np.sum(X > 0, axis=0)  # total spots where gene is expressed
    colocalized_counts = np.sum(X > threshold_colocalized, axis=0)  # spots exceeding threshold

    proportions = np.divide(colocalized_counts, positive_counts, where=(positive_counts > 0))  # avoid div by zero

    coloc_df = pd.DataFrame(data=proportions, index=adata.var.index, columns=["proportion_of_colocalized"])

    if "xrna_metadata" not in sdata:
        raise KeyError("'xrna_metadata' must exist in sdata to store metrics.")

    # Remove old if present
    if "proportion_of_colocalized" in sdata["xrna_metadata"].var.columns:
        sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.drop(columns=["proportion_of_colocalized"])

    # Add new
    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(coloc_df)

    return sdata if copy else None


def in_out_correlation(
    sdata, extracellular_layer: str = "segmentation_free_table", cellular_layer: str = "table", n_neighbors: int = 5, copy: bool | None = None
):
    """Computes the correlation between intracellular and extracellular gene expressionusing k-nearest extracellular bins.

    Parameters
    ----------
    sdata : SpatialData
        A SpatialData object containing both extracellular and cellular AnnData objects.

    extracellular_layer : str, optional (default="segmentation_free_table")
        Key for the extracellular AnnData object in sdata.

    cellular_layer : str, optional (default="table")
        Key for the cellular AnnData object in sdata.

    n_neighbors : int, optional (default=5)
        Number of nearest extracellular bins to consider for aggregation.

    Returns
    -------
    correlation_results : pd.DataFrame
        A DataFrame containing correlation values for each gene, with gene names as the indexand columns ['SpearmanR', 'PValue'].
    """
    try:
        adata_extracellular = sdata[extracellular_layer]
    except:
        KeyError(
            "Extracellular layer not found. Please make ensure ´extracellular_layer´ and that extracellular grouping has been performed. Otherwise, please run trouty.tl.aggregate_extracellular_transcripts"
        )
    adata_cellular = sdata[cellular_layer]

    # Extract spatial coordinates
    coords_cellular = adata_cellular.obsm["spatial"]
    coords_extracellular = adata_extracellular.obsm["spatial"]

    # Match genes present in both datasets and reindex
    shared_genes = adata_cellular.var_names.intersection(adata_extracellular.var_names)
    adata_cellular = adata_cellular[:, shared_genes]
    adata_extracellular = adata_extracellular[:, shared_genes]

    # Build KDTree for fast nearest-neighbor lookup
    extracellular_tree = cKDTree(coords_extracellular)

    # Find k-nearest extracellular bins for each cell
    _, nearest_indices = extracellular_tree.query(coords_cellular, k=n_neighbors)

    # Extract gene expression matrices
    expr_cellular = adata_cellular.X  # (cells × genes)
    expr_extracellular = adata_extracellular.X  # (bins × genes)

    # Ensure we're working with dense matrices (if sparse)
    if not isinstance(expr_cellular, np.ndarray):
        expr_cellular = expr_cellular.toarray()
    if not isinstance(expr_extracellular, np.ndarray):
        expr_extracellular = expr_extracellular.toarray()

    # Aggregate extracellular expression for each cell (mean of k-nearest bins)
    aggregated_extracellular = np.array(
        [expr_extracellular[nearest_indices[i]].mean(axis=0) for i in range(expr_cellular.shape[0])]
    )  # Shape: (cells × genes)

    # Compute correlation for each gene
    correlations = []

    for i, gene in enumerate(shared_genes):
        # Get expression values for this gene
        cell_expr = expr_cellular[:, i]
        ext_expr = aggregated_extracellular[:, i]

        # Compute Spearman correlation (skip if all-zero)
        if np.any(cell_expr) and np.any(ext_expr):
            corr, pval = spearmanr(cell_expr, ext_expr)
        else:
            corr, pval = np.nan, np.nan

        correlations.append([gene, corr, pval])

    # Convert results into a DataFrame
    correlation_results = pd.DataFrame(correlations, columns=["Gene", "SpearmanR", "PValue"])
    correlation_results.set_index("Gene", inplace=True)

    gene2spearman = dict(zip(correlation_results.index, correlation_results["SpearmanR"], strict=False))
    gene2pval = dict(zip(correlation_results.index, correlation_results["PValue"], strict=False))  # type: ignore
    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_xrna_metadata(sdata)
    sdata["xrna_metadata"].var["in_out_spearmanR"] = sdata["xrna_metadata"].var.index.map(gene2spearman)
    sdata["xrna_metadata"].var["in_out_pvalue"] = sdata["xrna_metadata"].var.index.map(gene2pval)

    return sdata if copy else None


def assess_diffussion(sdata: SpatialData, gene_key: str = "gene", distance_key: str = "distance", copy: bool = False):
    """
    Computes goodness-of-fit metrics for the diffusion pattern of extracellular RNA by testing against a Rayleigh distribution.
    Also estimates the diffusion coefficient (D) based on the mean squared displacement (MSD).

    Parameters
    ----------
        sdata (SpatialData): The spatial transcriptomics dataset.
        gene_key (str): The key for gene/transcript names in source_score.obs.
        distance_key (str): The key for RNA displacement distances.
        copy (bool): Whether to return a modified copy of sdata.

    Returns
    -------
        If copy=True, returns a modified SpatialData object with results in sdata['xrna_metadata']. Otherwise, modifies sdata in place.
    """
    results = []

    if "source_score" not in sdata:
        KeyError("Source_score info not found. Please calculate source cells.")

    for gene, group in sdata["source_score"].obs.groupby(gene_key):
        distances = group[distance_key].dropna().values
        if len(distances) < 10 or np.sum(distances) == 0:
            continue  # Skip genes with very few extracellular transcripts

        # Fit a Rayleigh distribution
        param = stats.rayleigh.fit(distances)
        fitted_cdf = stats.rayleigh.cdf(np.sort(distances), *param)

        # Goodness-of-Fit Tests
        ks_stat, ks_pval = stats.kstest(distances, "rayleigh", args=param)
        ad_stat, _, _ = stats.anderson(distances, dist="expon")

        # Likelihood Ratio Test
        log_likelihood_rayleigh = np.sum(stats.rayleigh.logpdf(distances, *param))
        log_likelihood_empirical = np.sum(stats.gaussian_kde(distances).logpdf(distances))
        lr_stat = -2 * (log_likelihood_rayleigh - log_likelihood_empirical)

        # Estimate Diffusion Coefficient (D)
        mean_distances = np.mean(distances)

        results.append(
            {
                "gene": gene,
                "ks_stat": ks_stat,
                "ks_pval": ks_pval,
                "ad_stat": ad_stat,
                "lr_stat": lr_stat,
                "mean_displacement": mean_distances,  # Add estimated diffusion coefficient
            }
        )

    diffusion_results = pd.DataFrame(results).set_index("gene")
    diffusion_results["-log_ks_pval"] = list(-np.log(diffusion_results["ks_pval"].clip(lower=1e-300)))

    # Ensure `xrna_metadata` exists in sdata
    if "xrna_metadata" not in sdata:
        create_xrna_metadata(sdata, layer="transcripts")

    for column in diffusion_results.columns:
        if column in sdata["xrna_metadata"].var.columns:
            sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.drop([column], axis=1)

    # Merge results into `xrna_metadata.var`
    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(diffusion_results)

    return sdata if copy else None


def cluster_distribution_from_source(
    sdata: SpatialData,
    gene_key: str = "gene",
    distance_key: str = "distance",
    n_clusters: int = 3,
    n_bins: int = 20,
    copy=False,
    layer: str = "transcripts",
):
    """
    Clusters genes based on the distribution of distances of extracellular transcripts
    from their source cell.

    For each gene in sdata['source_score'].obs, the function computes a normalized histogram
    (using n_bins) over the distance range. These histogram vectors are then standardized
    and clustered using KMeans.

    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing a 'source_score' layer with an obs DataFrame.
    gene_key : str, default "feature_name"
        Column name that contains the gene names.
    distance_key : str, default "distance"
        Column name that contains the distance from the source cell.
    n_clusters : int, default 3
        Number of clusters to form.
    n_bins : int, default 20
        Number of bins for the histogram representation.

    Returns
    -------
    gene_cluster_df : DataFrame
        A DataFrame with columns 'gene' and 'cluster' indicating the cluster assignment.
    hist_df : DataFrame
        A DataFrame where each row is a gene and the columns are the normalized histogram counts.
    bin_edges : ndarray
        The bin edges used for the histograms.
    """
    # Get the observation DataFrame from the 'source_score' layer.
    obs_df = sdata["source_score"].obs

    # Set common bin edges across all genes based on the global range of distances.
    global_max = obs_df[distance_key].max()
    # Assume distances start at 0.
    bin_edges = np.linspace(0, global_max, n_bins + 1)

    gene_hist_features = {}

    # Compute a normalized histogram for each gene.
    for gene, group in obs_df.groupby(gene_key):
        distances = group[distance_key].dropna().values
        if len(distances) == 0:
            continue
        counts, _ = np.histogram(distances, bins=bin_edges)
        # Normalize the histogram so that the sum equals 1.
        norm_counts = counts / counts.sum() if counts.sum() > 0 else counts
        gene_hist_features[gene] = norm_counts

    # Create a DataFrame where rows are genes and columns represent the binned histogram.
    hist_df = pd.DataFrame.from_dict(gene_hist_features, orient="index")

    # Standardize the histogram features.
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    hist_scaled = scaler.fit_transform(hist_df.values)

    # Cluster using KMeans.
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(hist_scaled)

    gene_cluster_dict = dict(zip(hist_df.index, clusters, strict=False))
    if "xrna_metadata" not in sdata:
        create_xrna_metadata(sdata, layer=layer, gene_key=gene_key)

    # Merge results into `xrna_metadata.var`
    sdata["xrna_metadata"].var["kmeans_distribution"] = list(sdata["xrna_metadata"].var.index.map(gene_cluster_dict))

    return sdata if copy else None


def compute_js_divergence(P, Q, eps=1e-10):
    """
    Compute the Jensen-Shannon divergence between two probability distributions.
    """
    P, Q = P + eps, Q + eps  # Avoid division by zero
    M = 0.5 * (P + Q)
    return 0.5 * (np.sum(P * np.log(P / M)) + np.sum(Q * np.log(Q / M)))


### warning- this is very sensitive to high density areas. Please reconsider
def compare_intra_extra_distribution(
    sdata,
    layer: str = "transcripts",
    gene_key: str = "gene",
    copy: bool = False,
    coord_keys: list = ["x", "y"],
    n_bins: int = 30,
):
    """
    Compare the spatial distribution of intracellular and extracellular transcripts for each gene.

    Parameters
    ----------
    - sdata: SpatialData object containing transcript locations and metadata.
    - layer: str, layer within sdata.points where transcripts are stored (default: "transcripts").
    - gene_key: str, column name where the gene name is stored (default: "feature_name").
    - copy: bool, whether to return a modified copy of sdata (default: False).
    - coord_keys: list of str, column names containing x and y transcript positions (default: ["x", "y"]).
    - n_bins: int, number of bins for the 2D histograms (default: 30).

    Returns
    -------
    - If copy=True, returns a DataFrame with computed metrics.
    - If copy=False, updates sdata["xrna_metadata"].var with the computed metrics.
    """
    transcripts_df = sdata[layer]  # Extract transcript data
    if isinstance(transcripts_df, dd.DataFrame):
        transcripts_df = transcripts_df.compute()  # Convert to Pandas if Dask

    results = []

    for gene, gene_transcripts in transcripts_df.groupby(gene_key):
        intracellular = gene_transcripts[~gene_transcripts["extracellular"]]
        extracellular = gene_transcripts[gene_transcripts["extracellular"]]

        # Skip genes with too few points for reliable estimation
        if intracellular.shape[0] < 5 or extracellular.shape[0] < 5:
            continue

        # Compute centroids for each group
        intracellular_centroid = intracellular[coord_keys].mean().values
        extracellular_centroid = extracellular[coord_keys].mean().values
        centroid_shift_distance = euclidean(intracellular_centroid, extracellular_centroid)

        # Define common histogram range
        all_coords = gene_transcripts[coord_keys].values
        x_min, y_min = np.min(all_coords, axis=0)
        x_max, y_max = np.max(all_coords, axis=0)
        range_x, range_y = (x_min, x_max), (y_min, y_max)

        # Compute 2D histograms for intracellular and extracellular groups
        hist_intra, _, _ = np.histogram2d(intracellular[coord_keys[0]], intracellular[coord_keys[1]], bins=n_bins, range=[range_x, range_y])
        hist_extra, _, _ = np.histogram2d(extracellular[coord_keys[0]], extracellular[coord_keys[1]], bins=n_bins, range=[range_x, range_y])

        # Normalize histograms to probability distributions
        P_intra = hist_intra / np.sum(hist_intra)
        P_extra = hist_extra / np.sum(hist_extra)

        # Flatten distributions
        intra_flat = P_intra.flatten()
        extra_flat = P_extra.flatten()

        # Compute correlation and divergence metrics
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

    # Store results in sdata["xrna_metadata"] if not copying
    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_xrna_metadata(sdata, layer=layer, gene_key=gene_key)

    # Remove existing columns before adding new ones
    for column in results_df.columns:
        if column in sdata["xrna_metadata"].var.columns:
            sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.drop(columns=[column])

    # Add the new computed metrics
    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(results_df)
    return sdata if copy else None
