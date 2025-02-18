import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import spatialdata as sd
import squidpy as sq
from sainsc import LazyKDE
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from spatialdata import SpatialData
from tqdm import tqdm


def spatial_variability(
    sdata: SpatialData,
    coord_keys: list = None,  # type: ignore
    gene_key: str = "feature_name",
    n_neighbors: int = 10,
    kde_resolution: int = 1000,
    square_size: int = 20,
    n_threads: int = 1,
    method: str = "moran",
    copy: bool = False,
):
    """
    Computes spatial variability of extracellular RNA using Moran's I.

    Parameters
    ----------
    sdata
        The spatial transcriptomics dataset in SpatialData format.
    coord_keys
        The keys for spatial coordinates in the dataset (default: ['x', 'y']).
    gene_key
        The key for gene identifiers in the dataset (default: 'feature_name').
    n_neighbors
        Number of neighbors to use for computing spatial neighbors (default: 10).
    kde_resolution
        The kde_resolution for kernel density estimation (default: 1000).
    square_size
        The square_size for kernel density estimation (default: 20).
    n_threads
        The number of threads for LazyKDE processing (default: 1).
    method
        The mode for spatial autocorrelation computation (default: "moran").

    Returns
    -------
    - sdata(SpatialData)
        Sdata containing Moran's I values for each gene, indexed by gene names.
    """
    # Step 1: Extract and preprocess data
    data = sdata.points["transcripts"][coord_keys + ["extracellular", gene_key]].compute()
    data = data[data["extracellular"]]
    data[gene_key] = data[gene_key].astype(str)

    # Rename columns for clarity
    newnames = ["x", "y", "extracellular", "gene"]
    data.columns = newnames

    # Convert to Polars DataFrame for LazyKDE processing
    trans = pl.from_pandas(data)

    # Step 2: Compute kernel density estimates
    embryo = LazyKDE.from_dataframe(trans, resolution=kde_resolution, square_size=square_size, n_threads=n_threads)

    # Step 3: Extract counts for all genes
    expr = embryo.counts.get(embryo.counts.genes()[0]).todense()
    allres = np.zeros([expr.size, len(embryo.counts.genes())])

    for n, gene in enumerate(tqdm(embryo.counts.genes(), desc="Extracting gene counts")):
        allres[:, n] = embryo.counts.get(gene).todense().flatten()

    # Create spatial grid coordinates
    x_coords, y_coords = np.meshgrid(np.arange(expr.shape[1]), np.arange(expr.shape[0]))

    # Step 4: Create AnnData object
    adata = sc.AnnData(allres)
    adata.var.index = embryo.counts.genes()
    adata.obs["x"] = x_coords.flatten()
    adata.obs["y"] = y_coords.flatten()
    adata.obsm["spatial"] = np.array(adata.obs.loc[:, ["x", "y"]])

    # Step 5: Compute spatial neighbors and Moran's I
    sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors)
    sq.gr.spatial_autocorr(adata, mode=method, genes=adata.var_names)

    # Extract Moran's I values
    svg_df = pd.DataFrame(adata.uns["moranI"])
    svg_df.columns = [method + "_" + str(g) for g in svg_df.columns]
    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_xrna_metadata(sdata, layer="transcripts")
    for column in svg_df.columns:
        if column in sdata["xrna_metadata"].var.columns:
            sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.drop([column], axis=1)

    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(svg_df)

    return sdata if copy else None


def create_xrna_metadata(sdata: SpatialData, layer: str = "transcripts", gene_key: str = "feature_name", copy: bool = False) -> SpatialData | None:
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
    codeword_key: str,
    control_codewords: list,
    gene_key: str = "feature_name",
    layer: str = "transcripts",
    percentile_threshold: float = 100,
    copy=False,
) -> SpatialData:
    """
    Compare counts per gene with counts per non-gene feature. We define a threshold as the 'percentile_threshold' counts of non-gene counts (e.g. 'percentile_threshold = 100' corresponds to the maximum number of counts observed in any non-gene feature). Any gene whose counts are above the threshold are considered overexpressed.

    Parameters
    ----------
    sdata
        The spatial data object holding points and transcript data.
    codeword_key
        Column name that holds codeword category.
    control_codewords
        Name(s) of codewords that correspond to controls based on which noise threshold will be defined.
    gene_key
        Column that holds name of gene (/ or feature) that is being detected.
    percentile_threshold
        Percentile used to define overexpression threshold. Defaults to 100.
    save
        Whether to save outputs to file. Defaults to True.

    Returns
    -------
    sdata
        The updated sdata object with scores per gene DataFrame, and the calculated threshold.
    """
    # Compute the data from the Dask DataFrame
    data = sdata.points[layer][np.unique(["extracellular", codeword_key, gene_key])].compute()
    # data = data[data["extracellular"]]

    # Ensure control_codewords is a list
    if isinstance(control_codewords, str):
        control_codewords = [control_codewords]
    assert isinstance(control_codewords, list), f"control_codewords should be a list but has type: {type(control_codewords)}"
    print(data.shape)
    # Get counts per control feature
    counts_per_nongene = data.loc[list(data.loc[:, codeword_key].isin(control_codewords)), gene_key].value_counts().to_frame().reset_index()
    threshold = np.percentile(counts_per_nongene.loc[:, "count"].values, percentile_threshold)

    # create dict
    gene2genestatus = dict(zip(data[gene_key], data[codeword_key].isin(control_codewords), strict=False))

    # Get counts per gene
    scores_per_gene = data[gene_key].value_counts().to_frame()
    scores_per_gene.columns = ["count"]
    scores_per_gene["control_probe"] = scores_per_gene.index.map(gene2genestatus)
    scores_per_gene.loc[:, "logfoldratio_over_noise"] = np.log(scores_per_gene.loc[:, "count"] / threshold)
    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_xrna_metadata(sdata, layer="transcripts")

    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(scores_per_gene)
    sdata["xrna_metadata"].var["control_probe"] = sdata["xrna_metadata"].var["control_probe"].fillna(False)

    return sdata if copy else None


def extracellular_enrichment(sdata: SpatialData, gene_key: str = "feature_name", copy: bool = False):
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
    data = sdata.points["transcripts"][[gene_key, "extracellular"]].compute()

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
        create_xrna_metadata(sdata, layer="transcripts")

    # Join the results to the metadata
    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(extracellular_proportion)

    return sdata if copy else None


def spatial_colocalization(
    sdata: SpatialData,
    coord_keys: list = ["x", "y"],  # noqa: B006
    gene_key: str = "feature_name",
    resolution: int = 1000,
    square_size: int = 20,
    n_threads: int = 1,
    threshold_colocalized: int = 1,
    copy: bool = False,
):
    """
    Computes spatial variability of extracellular RNA using Moran's I.

    Parameters
    ----------
    sdata
        The spatial transcriptomics dataset in SpatialData format.
    coord_keys
        The keys for spatial coordinates in the dataset (default: ['x', 'y']).
    gene_key
        The key for gene identifiers in the dataset (default: 'feature_name').
    n_neighbors
        Number of neighbors to use for computing spatial neighbors (default: 10).
    resolution
        The resolution for kernel density estimation (default: 1000).
    square_size
        The square_size for kernel density estimation (default: 20).
    n_threads
        The number of threads for LazyKDE processing (default: 1).
    method
        The mode for spatial autocorrelation computation (default: "moran").
    threshold_colocalized
        Minimum expression of two genes to consider them as colocalized if expressed together

    Returns
    -------
    - sdata
        A spatialdata object containing Moran's I values for each gene, in sdata.xrna_metadata.var, indexed by gene names.
    """
    # Step 1: Extract and preprocess data
    data = sdata.points["transcripts"][coord_keys + ["extracellular", gene_key]].compute()
    data = data[data["extracellular"]]
    data[gene_key] = data[gene_key].astype(str)

    # Rename columns for clarity
    newnames = ["x", "y", "extracellular", "gene"]
    data.columns = newnames

    # Convert to Polars DataFrame for LazyKDE processing
    trans = pl.from_pandas(data)

    # Step 2: Compute kernel density estimates
    embryo = LazyKDE.from_dataframe(trans, resolution=resolution, square_size=square_size, n_threads=n_threads)

    # Step 3: Extract counts for all genes
    expr = embryo.counts.get(embryo.counts.genes()[0]).todense()
    allres = np.zeros([expr.size, len(embryo.counts.genes())])

    for n, gene in enumerate(tqdm(embryo.counts.genes(), desc="Extracting gene counts")):
        allres[:, n] = embryo.counts.get(gene).todense().flatten()

    # Create spatial grid coordinates
    x_coords, y_coords = np.meshgrid(np.arange(expr.shape[1]), np.arange(expr.shape[0]))

    # Step 4: Create AnnData object
    adata = sc.AnnData(allres)
    adata.var.index = embryo.counts.genes()
    adata.obs["x"] = x_coords.flatten()
    adata.obs["y"] = y_coords.flatten()
    adata.obsm["spatial"] = np.array(adata.obs.loc[:, ["x", "y"]])

    # Calculate positive and colocalized counts for each gene
    positive_counts = np.sum(adata.X > 0, axis=0)  # Count non-zero (positive) values per gene
    colocalized_counts = np.sum(adata.X > threshold_colocalized, axis=0)  # Colocalized counts per gene
    # Calculate the proportion of colocalized transcripts
    proportions = np.divide(colocalized_counts, positive_counts, where=(positive_counts > 0))  # Avoid div by zero
    # Create the result DataFrame
    coloc = pd.DataFrame(data=proportions, index=adata.var.index, columns=["proportion_of_colocalized"])
    for column in coloc.columns:
        if column in sdata["xrna_metadata"].var.columns:
            sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.drop([column], axis=1)
    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(coloc)

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
        create_xrna_metadata(sdata, layer="transcripts")
    sdata["xrna_metadata"].var["in_out_spearmanR"] = sdata["xrna_metadata"].var.index.map(gene2spearman)
    sdata["xrna_metadata"].var["in_out_pvalue"] = sdata["xrna_metadata"].var.index.map(gene2pval)

    return sdata if copy else None
