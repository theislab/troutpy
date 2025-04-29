from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import polars as pl
import spatialdata as sd
from sainsc import LazyKDE  # assuming LazyKDE is available
from scipy.sparse import coo_matrix
from sklearn.neighbors import KDTree
from spatialdata import SpatialData


def compute_extracellular_counts(data_extracell):  # would be good to change the name of this function
    """
    Compute observed, expected, and fold ratio for extracellular transcript counts.

    Parameters
    ----------
    data_extracell (pd.DataFrame)
        Data with extracellular transcripts.

    Returns
    -------
    pd.DataFrame
        Dataframe with observed, expected counts, fold ratios, and gene categories.
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


def define_extracellular(
    sdata: SpatialData,
    layer: str = "transcripts",
    method: str = "segmentation_free",
    min_prop_of_extracellular: float = 0.8,
    unassigned_tag: str = "UNASSIGNED",
    copy: bool = False,
    percentile_threshold: float = 10,
):
    """
    Identifies extracellular transcripts based on the specified method and updates the spatial data object accordingly.

    Parameters
    ----------
    sdata : SpatialData
        A spatial data object containing transcriptomic information.
    layer : str
        The layer in `sdata.points` containing the transcript data to process.
    method : str
        The method to define extracellular transcripts. Options:
            - 'spots2regions': Uses segmentation-free clustering results.
            - 'sainsc': Uses sainsc-derived signature matching.
            - 'nuclei': Uses overlap with nuclear annotations to classify extracellular transcripts.
            - 'cells': Classifies transcripts not assigned to a cell as extracellular.
    min_prop_of_extracellular : float, optional
        Minimum proportion of transcripts in a cluster required to be extracellular for it to be classified as such
        (used only with the 'spots2regions' method).
    unassigned_tag : str, optional
        Tag indicating transcripts not assigned to any cell.
    copy : bool
        If True, returns a copy of the updated spatial data. If False, updates the `sdata` object in-place.

    Returns
    -------
    Optional[SpatialData]
        If `copy` is True, returns a copy of the updated `sdata` object.
        Otherwise, updates the `sdata` object in-place and returns None.

    Notes
    -----
    - For the 'spots2regions' method, clustering results are used to determine extracellular transcripts.
    - The 'sainsc' method defines extracellular transcripts as those that are not assigned to any cell
      and whose local signature (match_cell_signature) is not False.
    - The 'nuclei' method classifies transcripts outside nuclei as extracellular.
    - The 'cells' method classifies transcripts unassigned to cells as extracellular.
    """
    import pandas as pd  # Ensure pandas is imported, if not already available in sdata environment

    # Compute the data layer from the spatial data object
    data = sdata.points[layer].compute()

    # Method: Segmentation-free clustering (spots2regions)
    if method == "spots2regions":
        data["overlaps_cell"] = (data["cell_id"] != unassigned_tag).astype(int)
        overlapping_cell = pd.crosstab(data["segmentation_free_clusters"], data["overlaps_cell"])
        # Compute proportions and define extracellular clusters
        cluster_totals = overlapping_cell.sum(axis=1)
        cluster_proportions = overlapping_cell.div(cluster_totals, axis=0)
        extracellular_clusters = cluster_proportions[cluster_proportions.loc[:, 0] >= min_prop_of_extracellular].index
        data["extracellular"] = ~data["segmentation_free_clusters"].isin(extracellular_clusters)

    # Method: Using sainsc analysis results
    elif method == "sainsc":
        # Here we define extracellular transcripts as those that:
        # 1. Are not assigned to any cell (cell_id equals unassigned_tag), and
        # 2. Have a match_cell_signature that is not False (i.e. have a cell-like signature locally).
        cosine_sim_threshold = np.nanpercentile(data.loc[data["overlaps_cell"] == True, "cosine_similarity"], percentile_threshold)
        data["match_cell_signature"] = data["cosine_similarity"] > cosine_sim_threshold
        data["extracellular"] = (data["overlaps_cell"] == False) & (data["match_cell_signature"] == False)
        print(f"Cosine similarity threshold for extracellular definition: {cosine_sim_threshold}")
    # Method: Based on nuclei overlap
    elif method == "nuclei":
        data["extracellular"] = data["overlaps_nucleus"] != 1

    # Method: Based on cell assignment
    elif method == "cells":
        data["extracellular"] = data["cell_id"] == unassigned_tag

    # Unsupported method
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'spots2regions', 'sainsc', 'nuclei', and 'cells'.")

    # Update the spatial data object with the new extracellular column
    sdata.points[layer] = sd.models.PointsModel.parse(data)

    return sdata if copy else None


def filter_xrna(
    sdata,
    min_counts=None,
    min_extracellular_proportion=None,
    control_probe=None,
    min_logfoldratio_over_noise=1,
    min_morani=None,
    gene_key="feature_name",
    filter_cellular=False,
    copy=False,
):
    """
    Filters xRNA based on specified criteria and updates the sdata object.

    Parameters
    ----------
        sdata: dict-like
            Spatial data object containing xRNA metadata and transcript information.
        min_counts: int, optional
            Minimum count threshold for xRNA selection.
        min_extracellular_proportion: float, optional
            Minimum extracellular proportion threshold for xRNA selection.
        control_probe: bool, optional
            If False, filters out control probes.
        min_logfoldratio_over_noise: float, default=1
            Minimum log fold-change over noise threshold for xRNA selection.
        min_morani: float, optional
            Minimum Moran's I threshold for spatial autocorrelation.
        gene_key: str, default='feature_name'
            Key for accessing gene names in transcript tables.
        filter_cellular: bool, default=False
            If True, also filters the cellular table.
        copy: bool, default=False
            If True, returns a filtered copy of sdata; otherwise, modifies in place.

    Returns
    -------
        dict-like or None
            Filtered sdata if copy=True, else modifies sdata in place and returns None.
    """
    if copy:
        sdata = sdata.copy()

    # Select genes based on the first provided criterion
    if min_counts is not None:
        selected_genes = sdata["xrna_metadata"].var[sdata["xrna_metadata"].var["count"] > min_counts].index
    elif min_extracellular_proportion is not None:
        selected_genes = sdata["xrna_metadata"].var[sdata["xrna_metadata"].var["extracellular_proportion"] > min_extracellular_proportion].index
    elif control_probe is False:
        selected_genes = sdata["xrna_metadata"].var[sdata["xrna_metadata"].var["control_probe"] == False].index
    elif min_logfoldratio_over_noise is not None:
        selected_genes = sdata["xrna_metadata"].var[sdata["xrna_metadata"].var["logfoldratio_over_noise"] > min_logfoldratio_over_noise].index
    elif min_morani is not None:
        selected_genes = sdata["xrna_metadata"].var[sdata["xrna_metadata"].var["moran_I"] > min_morani].index
    else:
        return sdata if copy else None

    # Filter transcripts
    sdata["transcripts"] = sdata["transcripts"][sdata["transcripts"][gene_key].compute().isin(selected_genes)]

    # Filter other related tables safely
    for key in ["segmentation_free_table", "xrna_metadata"]:
        if key in sdata:
            try:
                sdata[key] = sdata[key][:, sdata[key].var.index.isin(selected_genes)]
            except Exception:
                pass

    # Filter source_score and target_score by obs
    for key in ["source_score", "target_score"]:
        if key in sdata:
            try:
                sdata[key] = sdata[key][sdata[key].obs[gene_key].isin(selected_genes), :]
            except Exception:
                pass

    # Filter cellular table if requested
    if filter_cellular and "table" in sdata:
        try:
            sdata["table"] = sdata["table"][:, sdata["table"].var.index.isin(selected_genes)]
        except Exception:
            pass

    return sdata if copy else None


def process_dataframe(df: pl.DataFrame, binsize: float, n_threads: int = 4):
    """
    Process a Polars DataFrame by binning spatial coordinates,
    converting the gene column to categorical, and constructing sparse matrices for each gene.
    """
    # Compute bin coordinates.
    df = df.with_columns((df["x"] / binsize).floor().alias("bin_x"), (df["y"] / binsize).floor().alias("bin_y"))
    # Shift coordinates so that bins start at zero.
    df = df.with_columns(
        (df["bin_x"] - df["bin_x"].min()).cast(pl.Int32).alias("bin_x"), (df["bin_y"] - df["bin_y"].min()).cast(pl.Int32).alias("bin_y")
    )
    # Ensure count column exists; if not, add one.
    if "count" not in df.columns:
        df = df.with_columns(pl.lit(1, dtype=pl.UInt32).alias("count"))
    else:
        df = df.with_columns(df["count"].cast(pl.UInt32))
    # Make gene column categorical.
    df = df.with_columns(df["gene"].cast(pl.Categorical))

    # Determine grid shape from binned coordinates.
    shape = (df["bin_x"].max() + 1, df["bin_y"].max() + 1)

    # Partition by gene for parallel processing.
    gene_groups = df.partition_by("gene", maintain_order=False)
    process_gene_partial = partial(process_gene, shape=shape)

    with Pool(n_threads) as pool:
        gene_results = pool.map(process_gene_partial, gene_groups)

    results = {gene: matrix for gene, matrix in gene_results}
    return results, shape, df


def process_gene(group, shape):
    """
    Process a group corresponding to one gene to build a sparse matrix.
    """
    gene_name = group["gene"][0]
    x = group["bin_x"].to_numpy()
    y = group["bin_y"].to_numpy()
    counts = group["count"].to_numpy()
    sparse_matrix = coo_matrix((counts, (x, y)), shape=shape).tocsr()
    return gene_name, sparse_matrix


def segmentation_free_sainsc(
    sdata,
    binsize=3,
    celltype_key="leiden",
    background_filter=0.4,
    gaussian_kernel_key=2.5,
    n_threads=16,
    resolution=1000,
    codeword_category="predesigned_gene",
    return_sainsc=False,
    copy=False,
    # Defaults for missing bin info
    default_cell_type="unknown",
    default_numeric=np.nan,
):
    """
    Process an sdata object by running an integrated sainsc analysis pipeline with configurable parameters.

    Steps:
      1. Extract and filter transcript data.
      2. Compute cell type signatures.
      3. For KDE/cell type assignment, use only transcripts with overlapping genes.
      4. Bin all transcripts.
      5. Merge the bin-level metadata (computed from successful bins) back to the full transcript set.
      6. For transcripts that do NOT obtain metadata via the merge, assign them the data
         from their spatially closest neighbor (based on "x", "y").
    """
    # --- 1. Prepare Transcript Data ---
    transcripts_all = sdata.points["transcripts"][["gene", "x", "y", "transcript_id"]].compute().reset_index(drop=True)
    # transcripts_all = transcripts_all[transcripts_all["codeword_category"] == codeword_category]
    transcripts_all = transcripts_all[transcripts_all["gene"].astype(str) != "nan"]

    # Create a full DataFrame for binning (all transcripts are retained)
    transcripts_full = pl.from_pandas(transcripts_all[["gene", "x", "y"]].copy(), schema_overrides={"gene": pl.Categorical})

    # --- 2. Retrieve Cell Expression Data & Compute Signatures ---
    adata = sdata["table"]
    expr = adata.to_df()
    expr["cell type"] = adata.obs[celltype_key]
    signatures = expr.groupby("cell type").mean().transpose()

    # --- 3. Determine Overlapping Genes for Analysis ---
    common_genes = set(transcripts_all["gene"].unique()).intersection(set(signatures.index))
    transcripts_analysis = transcripts_full.filter(pl.col("gene").is_in(list(common_genes)))

    # --- 4. Create Brain Object Using LazyKDE (Analysis on overlapping genes) ---
    brain = LazyKDE.from_dataframe(transcripts_analysis, resolution=resolution, binsize=binsize, n_threads=n_threads)

    # --- 5. Process All Transcripts into Binned Data ---
    _, shape, transcript2bin_info = process_dataframe(transcripts_full, binsize=binsize, n_threads=n_threads)
    transcript2bin_info = transcript2bin_info.to_pandas()
    transcript2bin_info["bin_x_y_id"] = transcript2bin_info["bin_x"].astype(str) + "_" + transcript2bin_info["bin_y"].astype(str)

    # --- 6. Prepare Unique Identifiers for Merging ---
    # Round coordinates to create matching keys.
    transcripts_all["x"] = transcripts_all["x"].round(4)
    transcripts_all["y"] = transcripts_all["y"].round(4)
    transcript2bin_info["x"] = transcript2bin_info["x"].round(4)
    transcript2bin_info["y"] = transcript2bin_info["y"].round(4)
    transcripts_all["xy"] = transcripts_all["x"].astype(str) + "_" + transcripts_all["y"].astype(str)
    transcript2bin_info["xy"] = transcript2bin_info["x"].astype(str) + "_" + transcript2bin_info["y"].astype(str)

    # --- 7. Run KDE, Background Filtering and Cell Type Assignment ---
    brain.calculate_total_mRNA()
    brain.gaussian_kernel(gaussian_kernel_key, unit="um")
    brain.calculate_total_mRNA_KDE()
    brain.filter_background(background_filter)
    brain.assign_celltype(signatures, log=True)

    # --- 8. Compute Bin-level Features from Brain ---
    celltype_flat = brain.celltype_map.flatten()
    assignment_score_flat = brain.assignment_score.flatten()
    cosine_similarity_flat = brain.cosine_similarity.flatten()

    n_rows, n_cols = brain.celltype_map.shape
    x_coords, y_coords = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()

    output_df = pd.DataFrame(
        {
            "bin_x": x_coords,
            "bin_y": y_coords,
            "cell type": celltype_flat,
            "assignment_score": assignment_score_flat,
            "cosine_similarity": cosine_similarity_flat,
        }
    )
    num2ct = dict(zip(range(len(brain.celltypes)), brain.celltypes, strict=False))
    output_df["cell type"] = output_df["cell type"].map(num2ct)
    output_df["bin_x_y_id"] = output_df["bin_y"].astype(str) + "_" + output_df["bin_x"].astype(str)

    # Build mapping dictionaries.
    bin2celltype = dict(zip(output_df["bin_x_y_id"], output_df["cell type"], strict=False))
    bin2cosine = dict(zip(output_df["bin_x_y_id"], output_df["cosine_similarity"], strict=False))
    bin2assign = dict(zip(output_df["bin_x_y_id"], output_df["assignment_score"], strict=False))

    # --- 9. Merge Bin-level Features for Transcripts That Were Successfully Binned ---
    # Merge on the unique "xy" identifier.
    bin_features = transcript2bin_info[["xy", "bin_x_y_id"]].drop_duplicates()
    bin_features["closest_cell_type"] = bin_features["bin_x_y_id"].map(bin2celltype)
    bin_features["cosine_similarity"] = bin_features["bin_x_y_id"].map(bin2cosine)
    bin_features["assignment_score"] = bin_features["bin_x_y_id"].map(bin2assign)

    # Perform a left merge so all transcripts are retained.
    merged = pd.merge(transcripts_all, bin_features, on="xy", how="left")

    # --- 10. Apply Nearest-Neighbor Assignment Only When Merge Failed ---
    missing_mask = merged["closest_cell_type"].isna()
    if missing_mask.sum() > 0:
        # Use only transcripts that have valid bin metadata.
        valid = merged[~missing_mask]
        missing = merged[missing_mask]
        if not valid.empty:
            # Build KDTree using the positions (x,y) of valid transcripts.
            tree = KDTree(valid[["x", "y"]].values)
            distances, indices = tree.query(missing[["x", "y"]].values, k=1)
            nn_metadata = valid.iloc[indices.flatten()].reset_index(drop=True)
            for col in ["closest_cell_type", "cosine_similarity", "assignment_score"]:
                missing[col] = nn_metadata[col].values
            merged.loc[missing_mask, ["closest_cell_type", "cosine_similarity", "assignment_score", "match_cell_signature"]] = missing[
                ["closest_cell_type", "cosine_similarity", "assignment_score"]
            ]

    # --- 11. Fill Any Remaining Missing Values with Defaults ---
    merged["closest_cell_type"] = merged["closest_cell_type"].fillna(default_cell_type)
    merged["cosine_similarity"] = merged["cosine_similarity"].fillna(default_numeric)
    merged["assignment_score"] = merged["assignment_score"].fillna(default_numeric)

    # Optionlly, force the correct data types
    merged["closest_cell_type"] = merged["closest_cell_type"].astype(str)
    merged["cosine_similarity"] = merged["cosine_similarity"].astype(float)
    merged["assignment_score"] = merged["assignment_score"].astype(float)

    # --- 12. Update the sdata Object ---
    transi = sdata.points["transcripts"].compute().copy()
    transi["xy"] = transi["x"].round(4).astype(str) + "_" + transi["y"].round(4).astype(str)
    # Drop duplicated columns if needed
    # columns_to_remove = ["closest_cell_type", "cosine_similarity", "assignment_score"]
    try:
        transi = transi.drop(columns=[col for col in columns_to_remove if col in transi.columns])
    except Exception:
        pass
    # Then merge safely
    transi = pd.merge(
        transi,
        merged[["xy", "closest_cell_type", "cosine_similarity", "assignment_score"]],
        on="xy",
        how="left",
    )
    # Fill any remaining missing entries as a safeguard (should be none now)
    transi["closest_cell_type"] = transi["closest_cell_type"].fillna(default_cell_type).astype(str)
    transi["cosine_similarity"] = transi["cosine_similarity"].fillna(default_numeric).astype(float)
    transi["assignment_score"] = transi["assignment_score"].fillna(default_numeric).astype(float)
    # Update the transcripts layer
    sdata.points["transcripts"] = sd.models.PointsModel.parse(transi)

    # --- 13. Return Results ---
    sainsc_output = {"brain": brain, "transcript2bin_info": transcript2bin_info, "output_df": output_df}
    if return_sainsc:
        return sdata, sainsc_output
    else:
        return sdata if copy else None
