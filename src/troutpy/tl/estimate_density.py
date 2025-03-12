import os

import numpy as np
import pandas as pd
from spatialdata import SpatialData


## deprecated
def colocalization_proportion(
    sdata: SpatialData, outpath: str, threshold_colocalized: int = 1, filename: str = "proportion_of_grouped_exRNA.parquet", save: bool = True
):
    """
    Calculate the proportion of colocalized transcripts for each gene in the provided AnnData object.

    Parameters
    ----------
    sdata
        AnnData object with `.X` matrix containing the density of transcripts per gene.
    outpath
        The directory path where the output file should be saved.
    threshold_colocalized
        The threshold for considering a transcript colocalized (default is 1).
    filename
        The name of the output file (default is 'proportion_of_grouped_exRNA.parquet').

    Returns
    -------
    coloc
        DataFrame containing the proportion of colocalized transcripts for each gene.
    """
    # Load relevant data
    df = sdata.points["extracellular_transcripts_enriched"][["feature_name", "bin_id"]].compute()
    adata_density_raw = sdata["segmentation_free_table"]

    # Filter adata_density to include only bin_ids present in df
    filtered_bin_ids = df["bin_id"].astype(str).unique()
    filtered_feature_name_ids = df["feature_name"].astype(str).unique()
    adata_density = adata_density_raw[adata_density_raw.obs.index.isin(filtered_bin_ids)]
    adata_density = adata_density[:, adata_density.var.index.isin(filtered_feature_name_ids)]

    # Convert the sparse matrix to dense format (assuming the matrix is large, sparse ops can be done here)
    dense_matrix = adata_density.X.todense()

    # Calculate positive and colocalized counts for each gene
    positive_counts = np.sum(dense_matrix > 0, axis=0)  # Count non-zero (positive) values per gene
    colocalized_counts = np.sum(dense_matrix > threshold_colocalized, axis=0)  # Colocalized counts per gene

    # Calculate the proportion of colocalized transcripts
    proportions = np.divide(colocalized_counts, positive_counts, where=(positive_counts > 0))  # Avoid div by zero

    # Create the result DataFrame
    coloc = pd.DataFrame(
        data=proportions.A1,  # Convert to a 1D array
        index=adata_density.var.index,
        columns=["proportion_of_colocalized"],
    )

    # Ensure the output directory exists
    os.makedirs(outpath, exist_ok=True)

    # Save the DataFrame as a Parquet file
    if save:
        filepath = os.path.join(outpath, filename)
        coloc.to_parquet(filepath)

    return coloc
