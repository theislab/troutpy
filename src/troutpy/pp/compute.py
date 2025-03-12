import numpy as np
import pandas as pd
import spatialdata as sd
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
):
    """
    Identifies extracellular transcripts based on the specified method and updates the spatial data object accordingly.

    Parameters
    ----------
    sdata (SpatialData)
        A spatial data object containing transcriptomic information.
    layer (str)
        The layer in `sdata.points` containing the transcript data to process.
    method (str)
        The method to define extracellular transcripts. Options:
            - 'segmentation_free': Uses segmentation-free clustering results.
            - 'nuclei': Uses overlap with nuclear annotations to classify extracellular transcripts.
            - 'cells': Classifies transcripts not assigned to a cell as extracellular.
    min_prop_of_extracellular (float, optional)
        Minimum proportion of transcripts in a cluster required to be extracellular for it to be classified as such (used only with 'segmentation_free' method).
    unassigned_tag (str, optional)
        Tag indicating transcripts not assigned to any cell.
    copy (bool)
        If True, returns a copy of the updated spatial data. If False, updates the `sdata` object in-place.

    Returns
    -------
    Optional[SpatialData]
        If `copy` is True, returns a copy of the updated `sdata` object.Otherwise, updates the `sdata` object in-place and returns None.

    Notes
    -----
    - The 'segmentation_free' method uses clustering results to determine extracellular transcripts.
    - The 'nuclei' method assumes transcripts outside nuclei are extracellular.
    - The 'cells' method classifies transcripts unassigned to cells as extracellular.
    """
    # Compute the data layer
    data = sdata.points[layer].compute()

    # Method: Segmentation-free clustering
    if method == "segmentation_free":
        data["overlaps_cell"] = (data["cell_id"] != unassigned_tag).astype(int)
        overlapping_cell = pd.crosstab(data["segmentation_free_clusters"], data["overlaps_cell"])

        # Compute proportions and define extracellular clusters
        cluster_totals = overlapping_cell.sum(axis=1)
        cluster_proportions = overlapping_cell.div(cluster_totals, axis=0)
        extracellular_clusters = cluster_proportions[cluster_proportions.loc[:, 0] >= min_prop_of_extracellular].index
        data["extracellular"] = ~data["segmentation_free_clusters"].isin(extracellular_clusters)

    # Method: Based on nuclei overlap
    elif method == "nuclei":
        data["extracellular"] = data["overlaps_nucleus"] != 1

    # Method: Based on cell assignment
    elif method == "cells":
        data["extracellular"] = data["cell_id"] == unassigned_tag

    # Unsupported method
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'segmentation_free', 'nuclei', and 'cells'.")

    # Update the spatial data object
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
