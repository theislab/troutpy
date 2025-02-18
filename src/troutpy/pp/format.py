import os

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


## this should be deprecated
def format_adata(input_path: str, outpath_dummy: str, xlimits: list, ylimits: list):
    """
    Processes and formats AnnData and transcripts by loading data, merging cell information,applying spatial filters, and saving the processed data to a dummy output directory.

    Parameters
    ----------
    input_path
        Path to the input directory containing:'cell_feature_matrix.h5','cells.parquet','transcripts.parquet'
    outpath_dummy
        Path to the output directory where processed files will be saved.
    xlimits
        Spatial limits for the x-coordinate filtering [min_x, max_x].
    ylimits
        Spatial limits for the y-coordinate filtering [min_y, max_y].

    Raises
    ------
    FileNotFoundError
        If required input files are not found in the input_path.
    ValueError
        If xlimits or ylimits are not properly defined.
    """
    # Validate input limits
    if not (isinstance(xlimits, (list | tuple)) and len(xlimits) == 2):
        raise ValueError("xlimits must be a list or tuple with two elements: [min_x, max_x].")
    if not (isinstance(ylimits, (list | tuple)) and len(ylimits) == 2):
        raise ValueError("ylimits must be a list or tuple with two elements: [min_y, max_y].")

    # Ensure output directory exists
    os.makedirs(outpath_dummy, exist_ok=True)

    # Define input file paths
    cell_feature_matrix_path = os.path.join(input_path, "cell_feature_matrix.h5")
    cells_parquet_path = os.path.join(input_path, "cells.parquet")
    transcripts_parquet_path = os.path.join(input_path, "transcripts.parquet")

    # Check if input files exist
    for file_path in [cell_feature_matrix_path, cells_parquet_path, transcripts_parquet_path]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    # Load AnnData from 10x h5 file
    print("Loading AnnData from 10x h5 file...")
    adata = sc.read_10x_h5(cell_feature_matrix_path)

    # Load cells data
    print("Loading cells data from parquet file...")
    cells = pd.read_parquet(cells_parquet_path)

    # Merge cell information into adata.obs
    print("Merging cell information into AnnData object...")
    adata.obs["cell_id"] = adata.obs.index.astype(str)
    adata.obs = pd.merge(adata.obs, cells, on="cell_id", how="left")

    # Add spatial coordinates to adata.obsm
    print("Adding spatial coordinates to AnnData object...")
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].values

    # Load transcripts data
    print("Loading transcripts data from parquet file...")
    transcripts = pd.read_parquet(transcripts_parquet_path)

    # Apply spatial filters to AnnData
    print("Applying spatial filters to AnnData object...")
    spatial_filter = (
        (adata.obs["x_centroid"] > xlimits[0])
        & (adata.obs["x_centroid"] < xlimits[1])
        & (adata.obs["y_centroid"] > ylimits[0])
        & (adata.obs["y_centroid"] < ylimits[1])
    )
    adata.obs.index = adata.obs.index.astype(str)
    adata_filtered = adata[spatial_filter].copy()

    # Apply spatial filters to transcripts
    print("Applying spatial filters to transcripts data...")
    transcripts_filtered = transcripts[
        (transcripts["x_location"] > xlimits[0])
        & (transcripts["x_location"] < xlimits[1])
        & (transcripts["y_location"] > ylimits[0])
        & (transcripts["y_location"] < ylimits[1])
    ].copy()

    # Save the processed AnnData object
    adata_output_path = os.path.join(outpath_dummy, "adata_raw.h5ad")
    print(f"Saving processed AnnData to {adata_output_path}...")
    adata_filtered.write(adata_output_path)

    # Save the filtered transcripts
    transcripts_output_path = os.path.join(outpath_dummy, "transcripts.parquet")
    print(f"Saving filtered transcripts to {transcripts_output_path}...")
    transcripts_filtered.to_parquet(transcripts_output_path)

    # Optional: Plot spatial data
    print("Generating spatial plot...")
    sc.pl.spatial(adata_filtered, color="transcript_counts", spot_size=50)
    print("Processing complete.")
    # selected roi
    # selected roi
    plt.figure()
    plt.scatter(adata_filtered.obs["x_centroid"], adata_filtered.obs["y_centroid"], s=4, c="red")
    plt.scatter(transcripts_filtered["x_location"], transcripts_filtered["y_location"], s=0.0001)
    return adata_filtered, transcripts_filtered
