import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import spatialdata as sd
import squidpy as sq
from sainsc import LazyKDE
from spatialdata import SpatialData
from tqdm import tqdm


def spatial_variability(
    sdata: SpatialData,
    coords_keys: list = None,  # type: ignore
    gene_id_key: str = "feature_name",
    n_neighbors: int = 10,
    resolution: int = 1000,
    binsize: int = 20,
    n_threads: int = 1,
    spatial_autocorr_mode: str = "moran",
    copy: bool = False,
):
    """
    Computes spatial variability of extracellular RNA using Moran's I.

    Parameters
    ----------
    sdata
        The spatial transcriptomics dataset in SpatialData format.
    coords_keys
        The keys for spatial coordinates in the dataset (default: ['x', 'y']).
    gene_id_key
        The key for gene identifiers in the dataset (default: 'feature_name').
    n_neighbors
        Number of neighbors to use for computing spatial neighbors (default: 10).
    resolution
        The resolution for kernel density estimation (default: 1000).
    binsize
        The binsize for kernel density estimation (default: 20).
    n_threads
        The number of threads for LazyKDE processing (default: 1).
    spatial_autocorr_mode
        The mode for spatial autocorrelation computation (default: "moran").

    Returns
    -------
    - sdata(SpatialData)
        Sdata containing Moran's I values for each gene, indexed by gene names.
    """
    # Step 1: Extract and preprocess data
    data = sdata.points["transcripts"][coords_keys + ["extracellular", gene_id_key]].compute()
    data = data[data["extracellular"]]
    data[gene_id_key] = data[gene_id_key].astype(str)

    # Rename columns for clarity
    newnames = ["x", "y", "extracellular", "gene"]
    data.columns = newnames

    # Convert to Polars DataFrame for LazyKDE processing
    trans = pl.from_pandas(data)

    # Step 2: Compute kernel density estimates
    embryo = LazyKDE.from_dataframe(trans, resolution=resolution, binsize=binsize, n_threads=n_threads)

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
    sq.gr.spatial_autocorr(adata, mode=spatial_autocorr_mode, genes=adata.var_names)

    # Extract Moran's I values
    svg_df = pd.DataFrame(adata.uns["moranI"])
    svg_df.columns = [spatial_autocorr_mode + "_" + str(g) for g in svg_df.columns]
    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_xrna_metadata(sdata, points_layer="transcripts")
    for column in svg_df.columns:
        if column in sdata["xrna_metadata"].var.columns:
            sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.drop([column], axis=1)

    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(svg_df)

    return sdata if copy else None


def create_xrna_metadata(
    sdata: SpatialData, points_layer: str = "transcripts", gene_key: str = "feature_name", copy: bool = False
) -> SpatialData | None:
    """
    Creates a new table within the SpatialData object that contains a 'gene' column with the unique gene names extracted from the specified points layer.

    Parameters
    ----------
    - sdata (SpatialData)
        The SpatialData object to modify.
    - points_layer (str, optional)
        The name of the layer in `sdata.points` from which to extract gene names. Default is 'transcripts'.
    - gene_key (str, optional)
        The key in the `points_layer` dataframe that contains the gene names.Default is 'feature_name'.
    - copy
        If `True`, returns a copy of the `SpatialData` object with the new table added.

    Returns
    -------
    - SpatialData | None
        If `copy` is `True`, returns a copy of the modified `SpatialData` object. Otherwise, returns `None`.
    """
    # Check if the specified points layer exists
    if points_layer not in sdata.points:
        raise ValueError(f"Points layer '{points_layer}' not found in sdata.points.")

    # Extract unique gene names from the specified points layer
    points_data = sdata.points[points_layer]
    if gene_key not in points_data.columns:
        raise ValueError(f"The specified points layer '{points_layer}' does not contain a '{gene_key}' column.")

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
    codeword_column: str,
    control_codewords: list,
    gene_id_column: str = "feature_name",
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
    codeword_column
        Column name that holds codeword category.
    control_codewords
        Name(s) of codewords that correspond to controls based on which noise threshold will be defined.
    gene_id_column
        Column that holds name of gene (/ or feature) that is being detected.
    percentile_threshold
        Percentile used to define overexpression threshold. Defaults to 100.
    save
        Whether to save outputs to file. Defaults to True.
    saving_path
        Path to directory that files should be saved in. Defaults to "".

    Returns
    -------
    sdata
        The updated sdata object with scores per gene DataFrame, and the calculated threshold.
    """
    # Compute the data from the Dask DataFrame
    data = sdata.points[layer][["extracellular", codeword_column, gene_id_column]].compute()
    data = data[data["extracellular"]]

    # Ensure control_codewords is a list
    if isinstance(control_codewords, str):
        control_codewords = [control_codewords]
    assert isinstance(control_codewords, list), f"control_codewords should be a list but has type: {type(control_codewords)}"

    # Get counts per control feature
    counts_per_nongene = data.loc[(data.loc[:, codeword_column].isin(control_codewords)), gene_id_column].value_counts().to_frame().reset_index()
    threshold = np.percentile(counts_per_nongene.loc[:, "count"].values, percentile_threshold)

    # create dict
    gene2genestatus = dict(zip(data[gene_id_column], data[codeword_column].isin(control_codewords), strict=False))

    # Get counts per gene
    scores_per_gene = data[gene_id_column].value_counts().to_frame()
    scores_per_gene.columns = ["count"]
    scores_per_gene["control_probe"] = scores_per_gene.index.map(gene2genestatus)
    scores_per_gene.loc[:, "logfoldratio_over_noise"] = np.log(scores_per_gene.loc[:, "count"] / threshold)
    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_xrna_metadata(sdata, points_layer="transcripts")

    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(scores_per_gene)
    sdata["xrna_metadata"].var["control_probe"] = sdata["xrna_metadata"].var["control_probe"].fillna(False)

    return sdata if copy else None


def extracellular_enrichment(sdata: SpatialData, gene_id_column: str = "feature_name", copy: bool = False):
    """
    Computes the proportion of transcripts classified as extracellular or intracellular for each gene and calculates additional metrics, including log fold change of extracellular to intracellular proportions. The results are integrated into the `sdata` object under the 'xrna_metadata' layer.

    Parameters
    ----------
    sdata
        An spatialData object containing spatial transcriptomics data. The `points` attribute should include a 'transcripts' DataFrame with columns for gene IDs (specified by `gene_id_column`) and a boolean 'extracellular' column indicating whether each transcript is classified as extracellular.
    gene_id_column
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
    data = sdata.points["transcripts"][[gene_id_column, "extracellular"]].compute()

    # Create a crosstab to count occurrences of intracellular and extracellular transcripts
    feature_inout = pd.crosstab(data[gene_id_column], data["extracellular"])
    norm_counts = feature_inout.div(feature_inout.sum(axis=0), axis=1)
    norm_counts["extracellular_foldratio"] = norm_counts[False] / norm_counts[True]

    extracellular_proportion = feature_inout.div(feature_inout.sum(axis=1), axis=0)
    extracellular_proportion.columns = extracellular_proportion.columns.map({True: "intracellular_proportion", False: "extracellular_proportion"})
    extracellular_proportion["logfoldratio_extracellular"] = np.log(norm_counts["extracellular_foldratio"])

    # Ensure the 'xrna_metadata' attribute exists
    try:
        sdata["xrna_metadata"]
    except KeyError:
        create_xrna_metadata(sdata, points_layer="transcripts")

    # Join the results to the metadata
    sdata["xrna_metadata"].var = sdata["xrna_metadata"].var.join(extracellular_proportion)

    return sdata if copy else None


def spatial_colocalization(
    sdata: SpatialData,
    coords_keys: list = ["x", "y"],  # noqa: B006
    gene_id_key: str = "feature_name",
    resolution: int = 1000,
    binsize: int = 20,
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
    coords_keys
        The keys for spatial coordinates in the dataset (default: ['x', 'y']).
    gene_id_key
        The key for gene identifiers in the dataset (default: 'feature_name').
    n_neighbors
        Number of neighbors to use for computing spatial neighbors (default: 10).
    resolution
        The resolution for kernel density estimation (default: 1000).
    binsize
        The binsize for kernel density estimation (default: 20).
    n_threads
        The number of threads for LazyKDE processing (default: 1).
    spatial_autocorr_mode
        The mode for spatial autocorrelation computation (default: "moran").

    Returns
    -------
    - sdata
        A DataFrame containing Moran's I values for each gene, indexed by gene names.
    """
    # Step 1: Extract and preprocess data
    data = sdata.points["transcripts"][coords_keys + ["extracellular", gene_id_key]].compute()
    data = data[data["extracellular"]]
    data[gene_id_key] = data[gene_id_key].astype(str)

    # Rename columns for clarity
    newnames = ["x", "y", "extracellular", "gene"]
    data.columns = newnames

    # Convert to Polars DataFrame for LazyKDE processing
    trans = pl.from_pandas(data)

    # Step 2: Compute kernel density estimates
    embryo = LazyKDE.from_dataframe(trans, resolution=resolution, binsize=binsize, n_threads=n_threads)

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

    threshold_colocalized = 1
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
