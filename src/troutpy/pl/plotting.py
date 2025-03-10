import os
from collections.abc import Sequence
from pathlib import Path

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.colors import Colormap
from spatialdata import SpatialData

# from troutpy.pp.compute import compute_crosstab


def sorted_heatmap(
    celltype_by_feature,
    output_path: str = "",
    filename: str = "Heatmap_target_cells_by_gene",
    format="pdf",
    cmap="viridis",
    vmax=None,
    save=False,
    figsize=(10, 10),
):
    """
    Plots the heatmap of target cells by gene.

    Parameters
    ----------
    celltype_by_feature (pd.DataFrame)
        DataFrame showing the fraction of each feature by cell type.
    outpath_dummy (str)
        Path to save the output plots.
    """
    figures_path = os.path.join(output_path, "figures")
    os.makedirs(figures_path, exist_ok=True)

    # Sort by maximum feature in cell types
    max_indices = np.argmax(celltype_by_feature.values, axis=1)
    celltype_by_feature = celltype_by_feature.iloc[np.argsort(max_indices)]
    celltype_by_feature.index = celltype_by_feature.index[np.argsort(max_indices)]

    # Heatmap plot
    plt.figure(figsize=figsize)
    sns.heatmap(celltype_by_feature, cmap=cmap, vmax=vmax)
    plt.ylabel(f"{celltype_by_feature.index.name}")
    plt.xlabel(f"{celltype_by_feature.columns.name}")
    plt.title(filename)
    if save:
        plt.savefig(os.path.join(figures_path, f"{filename}.{format}"))


def coupled_scatter(
    sdata,
    layer="extracellular_transcripts",
    output_path: str = "",
    transcript_group="distance_to_source_cell",
    save=True,
    format="pdf",
    xcoord="x",
    ycoord="y",
    xcellcoord="x_centroid",
    ycellcoord="y_centroid",
    colormap="Blues",
    size=2,
    color_cells="red",
    figsize=(10, 7),
    vmax=None,
):
    """
    Plots a scatter plot of transcript locations and cell centroids, coloring the transcripts by a specific feature (e.g., distance to the closest cell) and optionally saving the plot to a file.

    Parameters
    ----------
    sdata (sdata)
        A spatial data object that contains transcript and cell information.
    layer (str, optional)
        The key for the layer in `sdata.points` that contains transcript data (default: 'extracellular_transcripts').
    output_path (str, optional)
        The directory path where the plot will be saved. If not provided, the plot will not be saved (default: '').
    transcript_group (str, optional)
        The key in the transcript data (e.g., distance to the source cell) to be used for coloring the scatter plot (default: 'distance_to_source_cell').
    save (bool, optional)
        Whether to save the plot to a file. If `True`, the plot is saved to `output_path` (default: True).
    format (str, optional)
        The format for saving the plot (e.g., 'pdf', 'png'). This is only used if `save=True` (default: 'pdf').
    xcoord (str, optional)
        The column name in the transcript data representing the x-coordinate (default: 'x').
    ycoord (str, optional)
        The column name in the transcript data representing the y-coordinate (default: 'y').
    xcellcoord (str, optional)
        The column name in the cell data representing the x-coordinate of cell centroids (default: 'x_centroid').
    ycellcoord (str, optional)
        The column name in the cell data representing the y-coordinate of cell centroids (default: 'y_centroid').
    colormap (str, optional)
        The colormap to use for coloring the transcripts based on the `transcript_group` values (default: 'Blues').
    size (float, optional)
        The size of the scatter points for cells and transcripts. Transcripts are scaled down by 0.1 (default: 2).
    color_cells (str, optional)
        The color to use for the cell centroids (default: 'red').
    figsize (tuple, optional)
        The size of the figure in inches (width, height). This controls the dimensions of the plot (default: (10, 7)).
    vmax (float, optional)
        The upper limit for the colormap. If provided, this limits the color scale to values below `vmax` (default: None).

    Returns
    -------
    None

    Notes
    -----
    - The transcript data and cell centroid data are extracted from `sdata`.
    - The `vmax` parameter allows control over the maximum value of the color scale for better visualization control.
    - The plot is saved in the specified format and at the specified output path if `save=True`.
    """
    # Copy the AnnData object for cell data
    adata = sdata["table"].copy()

    # Use raw layer for transcript data
    adata.X = sdata["table"].layers["raw"]

    # Extract x, y centroid coordinates from the cell data
    adata.obs["x_centroid"] = [sp[0] for sp in adata.obsm["spatial"]]
    adata.obs["y_centroid"] = [sp[1] for sp in adata.obsm["spatial"]]

    # Extract transcript data from the specified layer
    transcripts = sdata.points[layer].compute()

    # Create output directory if it doesn't exist
    figures_path = os.path.join(output_path, "figures")
    os.makedirs(figures_path, exist_ok=True)

    # Create the scatter plot
    plt.figure(figsize=figsize)

    # Plot transcript locations, colored by the selected feature (transcript_group)
    plt.scatter(transcripts[xcoord], transcripts[ycoord], c=transcripts[transcript_group], s=size * 0.1, cmap=colormap, vmax=vmax)

    # Plot cell centroids
    plt.scatter(adata.obs[xcellcoord], adata.obs[ycellcoord], s=size, color=color_cells)

    # Set plot title
    plt.title(f"{transcript_group}")

    # Save the plot if specified
    if save:
        plt.savefig(os.path.join(figures_path, f"Scatter_{transcript_group}_{colormap}.{format}"))


def heatmap(
    data,
    output_path: str = "",
    save: bool = False,
    figsize=None,
    tag: str = "",
    title: str = None,
    cmap: str = "RdBu_r",
    annot: bool = False,
    cbar: bool = True,
    vmax=None,
    vmin=0,
    row_cluster: bool = True,
    col_cluster: bool = True,
):
    """
    Generate a clustered heatmap from the given data and optionally save it to a file.

    Parameters
    ----------
    data (pandas.DataFrame, numpy.ndarray)
        The data to visualize as a heatmap. Rows and columns will be clustered if specified.
    output_path (str, optional)
        Directory where the heatmap should be saved if `save` is True. Defaults to an empty string.
    save (bool, optional)
        Whether to save the generated heatmap to a file. Defaults to False.
    figsize (tuple, optional)
        Size of the figure as (width, height). If None, the size is calculated based on the data dimensions. Defaults to None.
    tag (str, optional)
        A tag to append to the saved file name. Defaults to an empty string.
    title (str, optional)
        Title of the heatmap. Defaults to None.
    cmap (str, optional)
        Colormap to use for the heatmap. Defaults to "RdBu_r".
    annot (bool, optional)
        Whether to annotate the heatmap cells with their values. Defaults to False.
    cbar (bool, optional)
        Whether to display a color bar in the heatmap. Defaults to True.
    vmax (float, optional)
        Maximum value for the colormap. Defaults to None.
    vmin (float, optional)
        Minimum value for the colormap. Defaults to 0.
    row_cluster (bool, optional)
        Whether to perform hierarchical clustering on rows. Defaults to True.
    col_cluster (bool, optional)
        Whether to perform hierarchical clustering on columns. Defaults to True.

    Returns
    -------
    None

    Notes
    -----
    - If `save` is True, the heatmap will be saved as a PDF file in the `output_path/figures` directory.
    - Clustering is performed using seaborn's `clustermap` function.
    """
    if figsize is None:
        figsize = (data.shape[1] / 3, (data.shape[0] / 7) + 2)
    g = sns.clustermap(data, cmap=cmap, annot=annot, figsize=figsize, vmax=vmax, vmin=vmin, col_cluster=col_cluster, row_cluster=row_cluster)
    g.fig.suptitle(title)
    if save:
        figures_path = os.path.join(output_path, "figures")
        os.makedirs(figures_path, exist_ok=True)
        plt.savefig(os.path.join(figures_path, "heatmap_" + tag + ".pdf"))
    plt.show()


def plot_crosstab(
    data,
    xvar: str = "",
    yvar: str = "",
    normalize=True,
    axis=1,
    kind="barh",
    save=True,
    figures_path: str = "",
    stacked=True,
    figsize=(6, 10),
    cmap="viridis",
    saving_format="pdf",
    sortby=None,
):
    """
    Plot a cross-tabulation between two variables in a dataset and visualize it as either a bar plot, horizontal bar plot, or heatmap.

    Parameters
    ----------
    data (pd.DataFrame)
        Input dataset containing the variables for the cross-tabulation.
    xvar (str, optional)
        The variable to use on the x-axis for the cross-tabulation.
    yvar (str, optional)
        The variable to use on the y-axis for the cross-tabulation.
    normalize (bool, optional)
        Whether to normalize the cross-tabulated data (percentages). If True, the data will be normalized.
    axis (int)
        The axis to normalize across. Use `1` for row normalization and `0` for column normalization.
    kind (str, optional)
        The kind of plot to generate. Options include:
            - 'barh': Horizontal bar plot
            - 'bar': Vertical bar plot
            - 'heatmap': Heatmap visualization
            - 'clustermap': Clustermap visualization
    save (bool)
        If True, the plot will be saved to a file.
    figures_path (str, optional)
        The directory path where the figure should be saved. If not specified, the plot will be saved in the current directory.
    stacked (bool, optional)
        If True, the bar plots will be stacked. Only applicable for 'barh' and 'bar' plot kinds.
    figsize (tuple, optional)
        The size of the figure for the plot (width, height).
    cmap (str, optional)
        The colormap to use for the plot, especially for heatmap and clustermap visualizations.
    saving_format (str, optional)
        The format to save the plot in. Options include 'png', 'pdf', etc.
    sortby (str, optional)
        The column or row to sort the cross-tabulated data by before plotting.

    Returns
    -------
    None
    """
    # Compute the crosstab data
    crosstab_data = compute_crosstab(data, xvar=xvar, yvar=yvar)

    # Normalize the data if required
    if normalize:
        crosstab_data = crosstab_data.div(crosstab_data.sum(axis=axis), axis=0)
        normtag = "normalize"
    else:
        normtag = "raw"

    # Sort the data if needed
    if sortby is not None:
        crosstab_data = crosstab_data.sort_values(by=sortby)

    # Generate the plot filename
    plot_filename = f"{kind}_{xvar}_{yvar}_{normtag}_{cmap}.{saving_format}"

    # Plot based on the selected kind
    if kind == "barh":
        plt.figure()
        crosstab_data.plot(kind="barh", stacked=stacked, figsize=figsize, width=0.99, colormap=cmap)
        plt.title(f"{xvar}_vs_{yvar}")
        if save:
            plt.savefig(os.path.join(figures_path, plot_filename))
        plt.show()

    elif kind == "bar":
        plt.figure()
        crosstab_data.plot(kind="bar", stacked=stacked, figsize=figsize, width=0.99, colormap=cmap)
        plt.title(f"{xvar}_vs_{yvar}")
        if save:
            plt.savefig(os.path.join(figures_path, plot_filename))
        plt.show()

    elif kind == "heatmap":
        plt.figure()
        sns.heatmap(crosstab_data, figsize=figsize, cmap=cmap)
        plt.title(f"{xvar}_vs_{yvar}")
        if save:
            plt.savefig(os.path.join(figures_path, plot_filename))
        plt.show()

    elif kind == "clustermap":
        plt.figure()
        sns.clustermap(crosstab_data, figsize=figsize, cmap=cmap)
        plt.title(f"{xvar}_vs_{yvar}")
        if save:
            plt.savefig(os.path.join(figures_path, plot_filename))
        plt.show()


def genes_over_noise(sdata, scores_by_genes, layer="extracellular_transcripts", output_path: str = "", save=True, format: str = "pdf"):
    """
    Function that plots log fold change per gene over noise using a boxplot.

    Parameters
    ----------
    data_quantified
        DataFrame containing the extracellular transcript data, including feature names and codeword categories.
    scores_by_genes
        DataFrame containing gene scores with feature names and log fold ratios.
    output_path
        Path to save the figure.
    """
    data_quantified = sdata.points[layer].compute()
    # Create the output directory for figures if it doesn't exist
    PATH_FIGURES = os.path.join(output_path, "figures")
    os.makedirs(PATH_FIGURES, exist_ok=True)

    # Map feature names to codeword categories
    feature2codeword = dict(zip(data_quantified["feature_name"], data_quantified["codeword_category"], strict=False))
    scores_by_genes["codeword_category"] = scores_by_genes["feature_name"].map(feature2codeword)

    # Plot the boxplot
    sns.boxplot(
        data=scores_by_genes,
        y="codeword_category",
        x="log_fold_ratio",
        hue="codeword_category",
    )
    # Plot the reference line at x = 0
    plt.plot([0, 0], [*plt.gca().get_ylim()], "r--")
    if save:
        # Save the figure
        plt.savefig(os.path.join(PATH_FIGURES, f"boxplot_log_fold_change_per_gene{format}"), bbox_inches="tight", pad_inches=0)
    # Show the plot
    plt.show()


def moranI_histogram(svg_df, save=True, figures_path: str = "", bins: int = 200, format: str = "pdf"):
    """
    Plots the distribution of Moran's I scores from a DataFrame.

    Parameters
    ----------
    svg_df (pandas.DataFrame)
        DataFrame containing a column 'I' with Moran's I scores.
    save (bool, optional)
        Whether to save the plot as a file.
    figures_path (str, optional)
        Path to save the figure. Only used if `save=True`.
    bins (int, optional)
        Number of bins to use in the histogram.
    format (str, optional)
        Format in which to save the figure (e.g., 'pdf', 'png').

    Returns
    -------
    None
    """
    # Check if figures_path exists if saving the figure
    if save and figures_path:
        if not os.path.exists(figures_path):
            raise ValueError(f"The provided path '{figures_path}' does not exist.")

    # Plot the distribution
    plt.figure(figsize=(8, 6))
    plt.hist(svg_df.sort_values(by="I", ascending=False)["I"], bins=bins)
    plt.xlabel("Moran's I")
    plt.ylabel("Frequency")
    plt.title("Distribution of Moran's I Scores")

    # Save the plot if requested
    if save:
        file_name = os.path.join(figures_path, f"barplot_moranI_by_gene.{format}")
        plt.savefig(file_name, format=format)
        print(f"Plot saved to: {file_name}")

    plt.show()


def proportion_above_threshold(
    df,
    threshold_col="proportion_above_threshold",
    feature_col="feature_name",
    top_percentile=0.05,
    bottom_percentile=0.05,
    specific_transcripts=None,
    figsize=(4, 10),
    orientation="h",
    bar_color="black",
    title="Proportion of distant exRNa (>30um) from source",
    xlabel="Proportion above threshold",
    ylabel="Feature",
    save=False,
    output_path: str = "",
    format="pdf",
):
    """
    Plots the top and bottom percentiles of features with the highest and lowest proportions above a threshold, or visualizes a specific list of transcripts.

    Parameters
    ----------
    - df
        DataFrame containing feature proportions.
    - threshold_col
        Column name for proportions above the threshold (default: 'proportion_above_threshold').
    - feature_col
        Column name for feature names (default: 'feature_name').
    - top_percentile
        Proportion (0-1) of features with the highest proportions to display (default: 0.05 for top 5%).
    - bottom_percentile
        Proportion (0-1) of features with the lowest proportions to display (default: 0.05 for bottom 5%).
    - specific_transcripts
        List of specific transcript names to plot (optional).
    - figsize
        Tuple specifying the size of the plot (default: (4, 10)).
    - orientation
        Orientation of the bars ('h' for horizontal, 'v' for vertical, default: 'h').
    - bar_color
        Color of the bars (default: 'black').
    - title
        Title of the plot (default: 'Proportion of distant exRNa (>30um) from source').
    - xlabel
        Label for the x-axis (default: 'Proportion above threshold').
    - ylabel
        Label for the y-axis (default: 'Feature').
    """
    df = df[~df[threshold_col].isna()]
    print(df.shape)
    # Filter for top and bottom percentiles if no specific transcripts are provided
    if specific_transcripts is None:
        top_cutoff = df[threshold_col].quantile(1 - top_percentile)
        bottom_cutoff = df[threshold_col].quantile(bottom_percentile)
        plot_data = pd.concat(
            [
                df[df[threshold_col] >= top_cutoff],  # Top percentile
                df[df[threshold_col] <= bottom_cutoff],  # Bottom percentile
            ]
        )
    else:
        plot_data = df[df[feature_col].isin(specific_transcripts)]

    # Plot
    plt.figure(figsize=figsize)
    if orientation == "h":
        plt.barh(plot_data["feature_name"], plot_data[threshold_col], color=bar_color)
    if orientation == "v":
        plt.bar(plot_data["feature_name"], plot_data[threshold_col], color=bar_color)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    figures_path = os.path.join(output_path, "figures")
    os.makedirs(figures_path, exist_ok=True)
    filename = f"barplot_distant_from_source_min{bottom_percentile}_max{top_percentile}_{bar_color}"
    if save:
        plt.savefig(os.path.join(figures_path, f"{filename}.{format}"))
    plt.show()


def nmf_factors_exrna_cells_W(
    sdata, nmf_adata_key: str = "nmf_data", save: bool = True, saving_path: str = "", spot_size: int = 30, cmap: str = "viridis"
):
    """
    Extracts the NMF (Non-negative Matrix Factorization) factors from the specified AnnData object within the spatial data (`sdata`) and creates spatial plots for each factor. The plots can be displayed or saved to disk.

    Parameters
    ----------
    sdata (SpatialData object)
        A spatial transcriptomics dataset that contains the NMF factors in the specified key.
    nmf_adata_key (str, optional)
        The key in `sdata` that contains the AnnData object with NMF results. Defaults to 'nmf_data'.
    save (bool, optional)
        Whether to save the spatial factor plots to disk. Defaults to True.
    saving_path (str, optional)
        Path where the plots should be saved if `save` is True. The plots are saved in a `figures` subdirectory.
    spot_size (int, optional)
        Size of the spots in the spatial plot. Defaults to 30.
    cmap (str, optional)
        Colormap to use for the spatial plots. Defaults to 'viridis'.

    Returns
    -------
    None
    """
    # Plot the factors for each cell in a spatial plot
    adata = sdata[nmf_adata_key]
    W = adata.obsm["W_nmf"]
    for factor in range(20):
        # Add the factor values to adata.obs for plotting
        adata.obs[f"NMF_factor_{factor + 1}"] = W[:, factor]
        # Plot spatial map of cells colored by this factor
        if save:
            sc.pl.spatial(adata, color=f"NMF_factor_{factor + 1}", cmap=cmap, title=f"NMF Factor {factor + 1}", spot_size=30, show=False)
            plt.savefig(saving_path + "/figures/" + f"spatialnmf{factor}.png")
            plt.show()
        else:
            sc.pl.spatial(adata, color=f"NMF_factor_{factor + 1}", cmap=cmap, title=f"NMF Factor {factor + 1}", spot_size=spot_size)


def nmf_gene_contributions(
    sdata,
    nmf_adata_key: str = "nmf_data",
    save: bool = True,
    vmin: float = 0.0,
    vmax: float = 0.02,
    saving_path: str = "",
    cmap: str = "viridis",
    figsize: tuple = (5, 5),
):
    """
    Extracts the NMF (Non-negative Matrix Factorization) gene loadings matrix from the specified AnnData object within the spatial data (`sdata`), filters genes based on their maximum loading value, and plots a heatmap of the filtered loadings.

    Parameters
    ----------
    sdata (SpatialData object)
        A spatial transcriptomics dataset that contains the NMF factors in the specified key.
    nmf_adata_key (str, optional)
        The key in `sdata` that contains the AnnData object with NMF results. Defaults to 'nmf_data'.
    save (bool, optional)
        Whether to save the heatmap plot to disk. Defaults to True.
    vmin (float, optional)
        Minimum value for the colormap scale. Defaults to 0.0.
    vmax (float, optional)
        Maximum value for the colormap scale. Defaults to 0.02.
    saving_path (str, optional)
        Path where the plot should be saved if `save` is True. The plot is saved in a `figures` subdirectory.
    cmap (str, optional)
        Colormap to use for the heatmap. Defaults to 'viridis'.
    figsize (tuple, optional)
        Size of the heatmap figure. Defaults to (5, 5).

    Returns
    -------
    None
    """
    adata = sdata[nmf_adata_key]
    loadings = pd.DataFrame(adata.uns["H_nmf"], columns=adata.var.index)
    loadings_filtered = loadings.loc[:, np.max(loadings, axis=0) > 0.05].transpose()
    figures_path = os.path.join(saving_path, "figures")
    os.makedirs(figures_path, exist_ok=True)

    # Sort by maximum feature in cell types
    max_indices = np.argmax(loadings_filtered.values, axis=1)
    loadings_filtered = loadings_filtered.iloc[np.argsort(max_indices)]
    loadings_filtered.index = loadings_filtered.index[np.argsort(max_indices)]

    # Heatmap plot
    plt.figure(figsize=figsize)
    sns.heatmap(loadings_filtered, cmap=cmap, vmax=1)
    if save:
        plt.savefig(os.path.join(figures_path, "loadings_NMF.pdf"))
    plt.show()
    plt.close()  # Close the figure to avoid memory issues


def apply_exrnaH_to_cellular_to_create_cellularW(adata_extracellular_with_nmf, adata_annotated_cellular):
    """
    Transfers the gene loadings (H matrix) derived from extracellular RNA analysis to a cellular dataset. It calculates the new W matrix for cellular data by multiplying the gene expression values of the cellular dataset with the filtered H matrix.

    Parameters
    ----------
    adata_extracellular_with_nmf (AnnData)
        An AnnData object containing the extracellular RNA data with the NMF results. The H matrix is expected to be stored in `adata.uns['H_nmf']`.
    adata_annotated_cellular (AnnData)
        An AnnData object containing the cellular RNA data with annotated gene expression values.

    Returns
    -------
    AnnData. The input `adata_annotated_cellular` object with the following updates. Adds the calculated NMF factors (W matrix) as a DataFrame to `adata.obsm['factors']`.Adds each NMF factor as individual columns in `adata.obs` with names `NMF_factor_1`, `NMF_factor_2`, etc.
    """
    # Extract the H matrix (NMF gene loadings) from the extracellular data
    H = adata_extracellular_with_nmf.uns["H_nmf"]

    # Check the genes in both datasets
    genes_spots2region = adata_extracellular_with_nmf.var_names
    genes_annotated = adata_annotated_cellular.var_names

    # Get the intersection of genes between the two datasets
    common_genes = genes_annotated.intersection(genes_spots2region)

    # Filter both datasets to retain only common genes
    adata_annotated_cellular = adata_annotated_cellular[:, common_genes]
    H_filtered = H[:, np.isin(genes_spots2region, common_genes)]  # Filter H matrix to include only common genes

    # Compute the new W matrix for the cellular dataset
    W_annotated = adata_annotated_cellular.X @ H_filtered.T

    # Store the W matrix in the obsm attribute as a DataFrame
    adata_annotated_cellular.obsm["factors"] = pd.DataFrame(W_annotated, index=adata_annotated_cellular.obs.index)

    # Add individual NMF factors to adata.obs
    for factor in range(W_annotated.shape[1]):
        adata_annotated_cellular.obs[f"NMF_factor_{factor + 1}"] = W_annotated[:, factor]

    return adata_annotated_cellular


def paired_nmf_factors(
    sdata,
    layer="nmf_data",
    n_factors=5,  # Number of NMF factors to plot
    figsize=(12, 6),  # Size of the figure
    spot_size_exrna=5,  # Spot size for extracellular transcripts
    spot_size_cells=10,  # Spot size for cell map
    cmap_exrna="YlGnBu",  # Colormap for extracellular transcripts
    cmap_cells="Reds",  # Colormap for cells
    vmax_exrna="p99",  # Maximum value for color scale (extracellular)
    vmax_cells=None,  # Maximum value for color scale (cells)
    save=False,
    output_path: str = "",
    format="pdf",
):
    """
    Plots the spatial distribution of NMF factors for extracellular transcripts and cells.

    Parameters
    ----------
    sdata (SpatialData object)
        spatial data object containing both extracellular and cell data.
    layer (str, optional)
        Layer in sdata to extract the NMF data from (default: 'nmf_data').
    n_factors (int, optional)
        Number of NMF factors to plot (default: 5).
    figsize (tuple, optional)
        Size of the figure for each subplot (default: (12, 6)).
    spot_size_exrna (float, optional)
        Size of the spots for extracellular transcript scatter plot (default: 5).
    spot_size_cells (float, optional)
        Size of the spots for cell scatter plot (default: 10).
    cmap_exrna (str, optional)
        Colormap for the extracellular transcript NMF factors (default: 'YlGnBu').
    cmap_cells (str, optional)
        Colormap for the cell NMF factors (default: 'Reds').
    vmax_exrna (str or float)
        Maximum value for extracellular transcript color scale (default: 'p99').
    vmax_cells (str or float)
        Maximum value for cell color scale (default: None).
    """
    # Extract NMF data from sdata
    adata = sdata[layer]
    adata_annotated = sdata["table"]

    # Get the factors from the obsm attribute (NMF results)
    factors = pd.DataFrame(adata.obsm["cell_loadings"], index=adata.obs.index)
    factors.columns = [f"Factor_{fact + 1}" for fact in factors.columns]

    # Add each NMF factor to adata.obs
    for f in factors.columns:
        adata.obs[f] = factors[f]

    # Add to each annotated one
    factors = pd.DataFrame(adata_annotated.obsm["factors_cell_loadings"], index=adata_annotated.obs.index)
    factors.columns = [f"Factor_{fact + 1}" for fact in factors.columns]
    # Add each NMF factor to adata.obs
    for f in factors.columns:
        adata_annotated.obs[f] = factors[f]

    # Loop over the specified number of NMF factors and plot
    for factor in range(n_factors):
        factor_name = f"Factor_{factor + 1}"

        # Create a figure with a single subplot for each factor
        fig, axs = plt.subplots(1, 1, figsize=figsize)

        # Plot the spatial distribution for extracellular transcripts
        sc.pl.spatial(
            adata,
            color=factor_name,
            cmap=cmap_exrna,
            title=f"Factor {factor + 1} (Extracellular)",
            ax=axs,
            show=False,
            spot_size=spot_size_exrna,
            vmax=vmax_exrna,
        )

        # Overlay the cell spatial distribution
        sc.pl.spatial(
            adata_annotated,
            color=factor_name,
            cmap=cmap_cells,
            title=f"Factor cell-red/exRNa-blue {factor + 1}",
            ax=axs,
            show=False,
            spot_size=spot_size_cells,
            vmax=vmax_cells,
        )
        if save:
            figures_path = os.path.join(output_path, "figures")
            os.makedirs(figures_path, exist_ok=True)
            file_name = os.path.join(figures_path, f"Spatial_NMF Factor {factor + 1}.{format}")
            plt.savefig(file_name)

        # Adjust layout and show the combined plot
        plt.tight_layout()
        plt.show()


def plot_nmf_factors_spatial(adata, n_factors, save=True):
    """
    Visualizes the spatial distribution of cells, colored by their corresponding NMF factor values, stored in `adata.obs`. It iterates over all specified NMF factors and generates spatial plots for each factor.

    Parameters
    ----------
    adata (AnnData): An AnnData object containing the dataset with NMF factors already added as columns in `adata.obs`.Each factor should be named `NMF_factor_1`, `NMF_factor_2`, ..., `NMF_factor_n`.
    n_factors (int): The number of NMF factors to plot.
    save (bool): If `True`, saves the plots to files with filenames `exo_to_cell_spatial_<factor>.png`.

    Returns
    -------
    None

    Notes
    -----
    - The plots are colored using the 'plasma' colormap.
    - The spot size for the spatial plots is set to 15 by default.
    - Files are saved in the current working directory unless specified otherwise using `sc.settings.figdir`.
    """
    for factor in range(n_factors):
        sc.pl.spatial(
            adata,
            color=f"NMF_factor_{factor + 1}",
            cmap="plasma",
            title=f"NMF Factor {factor + 1}",
            spot_size=15,
            save=f"exo_to_cell_spatial_{factor}.png" if save else None,
        )


def spatial_interactions(
    sdata: SpatialData,
    layer: str = "extracellular_transcripts_enriched",
    gene: str = "Arc",
    gene_key: str = "feature_name",
    cell_id_key: str = "cell_id",
    color_target: str = "blue",
    color_source: str = "red",
    color_transcript: str = "green",
    spatial_key: str = "spatial",
    img: bool | Sequence | None = None,
    img_alpha: float | None = None,
    image_cmap: Colormap | None = None,
    size: float | Sequence[float] | None = 8,
    alpha: float = 0.6,
    title: str | Sequence[str] | None = None,
    legend_loc: str | None = "best",
    figsize: tuple[float, float] = (10, 10),
    dpi: int | None = 100,
    save: str | Path | None = None,
    **kwargs,
):
    """
    Generates a scatter plot showing the positions of target cells, source cells, and extracellular RNA transcripts within a spatial omics dataset. The target and source cells are highlighted in different colors, while the RNA transcripts are shown as points at their respective positions. Optionally, a background image (e.g., tissue section) can be displayed.

    Parameters
    ----------
    sdata (AnnData)
        An AnnData object containing the spatial omics data, including transcript expression and cell positions.
    layer (str)
        The layer in the AnnData object that contains the extracellular RNA transcript data.
    gene (str)
        The gene of interest to be visualized in terms of its spatial interaction with source and target cells.
    gene_key (str)
        The column name in the AnnData object used to identify the gene.
    cell_id_key (str)
        The column name in the AnnData object used to identify individual cells.
    color_target (str)
        The color to be used for target cells in the plot.
    color_source (str)
        The color to be used for source cells in the plot.
    color_transcript (str)
        The color to be used for the RNA transcripts in the plot.
    spatial_key (str)
        The key in the AnnData object that stores the spatial coordinates of the cells.
    img (Optional[Union[bool, Sequence]])
        A background image to overlay on the plot, such as a tissue section. Can be set to `None` to omit.
    img_alpha (Optional[float])
        The transparency level of the background image. Ignored if `img` is `None`.
    image_cmap (Optional[Colormap])
        The colormap to be used for the background image, if applicable.
    size (Optional[Union[float, Sequence[float]]])
        The size of the scatter plot points for the cells and transcripts.
    alpha (float)
        The transparency level for the scatter plot points.
    title (Optional[Union[str, Sequence[str]]])
        The title of the plot. If `None`, the gene name is used.
    legend_loc (Optional[str])
        The location of the legend in the plot.
    figsize
        The dimensions of the plot in inches.
    dpi (Optional[int])
        The resolution (dots per inch) for the plot.
    save (Optional[Union[str, Path]])
        The path to save the plot image. If `None`, the plot is displayed but not saved.
    kwargs
        Any additional arguments passed to the `scatter` or `imshow` functions for customizing plot appearance.

    Returns
    -------
    None
    """
    # Extract relevant data
    transcripts = sdata.points[layer]
    trans_filt = transcripts[transcripts[gene_key] == gene]
    target_cells = trans_filt["closest_target_cell"].compute()
    source_cells = trans_filt["closest_source_cell"].compute()
    cell_positions = pd.DataFrame(sdata["table"].obsm[spatial_key], index=sdata.table.obs[cell_id_key], columns=["x", "y"])

    # Plotting
    plt.figure(figsize=figsize, dpi=dpi)
    if img is not None:
        plt.imshow(img, alpha=img_alpha, cmap=image_cmap, **kwargs)
    plt.scatter(cell_positions["x"], cell_positions["y"], c="grey", s=0.6, alpha=alpha, **kwargs)
    plt.scatter(cell_positions.loc[target_cells, "x"], cell_positions.loc[target_cells, "y"], c=color_target, s=size, label="Target Cells", **kwargs)
    plt.scatter(cell_positions.loc[source_cells, "x"], cell_positions.loc[source_cells, "y"], c=color_source, s=size, label="Source Cells", **kwargs)
    plt.scatter(trans_filt["x"], trans_filt["y"], c=color_transcript, s=size * 0.4, label="Transcripts", **kwargs)

    # Titles and Legends
    plt.title(title or gene)
    plt.legend(loc=legend_loc)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")

    # Save the plot if path provided
    if save:
        plt.savefig(save)
    plt.show()


def interactions_with_arrows(
    sdata: SpatialData,
    layer: str = "extracellular_transcripts_enriched",
    gene: str = "Arc",
    gene_key: str = "feature_name",
    cell_id_key: str = "cell_id",
    color_target: str = "blue",
    color_source: str = "red",
    color_transcript: str = "green",
    spatial_key: str = "spatial",
    img: bool | Sequence | None = None,
    img_alpha: float | None = None,
    image_cmap: Colormap | None = None,
    size: float | Sequence[float] | None = 8,
    alpha: float = 0.6,
    title: str | Sequence[str] | None = None,
    legend_loc: str | None = "best",
    figsize: tuple[float, float] = (10, 10),
    dpi: int | None = 100,
    save: str | Path | None = None,
    **kwargs,
):
    """
    The function plots arrows from source to target cells based on transcript proximity, color-coding source and target cells, and transcript locations. An optional image layer can be overlaid behind the plot.

    Parameters
    ----------
    - sdata (AnnData)
        The AnnData object containing the spatial omics data.
    - layer (str, optional)
        The key in `sdata` for the extracellular transcript layer to analyze. Default is 'extracellular_transcripts_enriched'.
    - gene (str, optional)
        The gene of interest. Default is 'Arc'.
    - gene_key (str, optional)
        The key for gene names in the data. Default is 'feature_name'.
    - cell_id_key (str, optional)
        The key for cell IDs. Default is 'cell_id'.
    - color_target (str, optional)
        Color for the target cells. Default is 'blue'.
    - color_source (str, optional)
        Color for the source cells. Default is 'red'.
    - color_transcript (str, optional)
        Color for the transcript locations. Default is 'green'.
    - spatial_key (str, optional)
        The key for spatial coordinates in `sdata`. Default is 'spatial'.
    - img (Optional[Union[bool, Sequence]], optional)
        Optional background image (e.g., tissue section) to display behind the plot.
    - img_alpha (Optional[float], optional)
        Transparency level for the background image. Default is None (no image).
    - image_cmap (Optional[Colormap], optional)
        Colormap for the image. Default is None.
    - size (Optional[Union[float, Sequence[float]]], optional)
        Size of the plotted points (cells and transcripts). Default is 8.
    - alpha (float, optional)
        Transparency level for plotted points. Default is 0.6.
    - title (Optional[Union[str, Sequence[str]]], optional)
        Title of the plot. Default is the gene name.
    - legend_loc (Optional[str], optional)
        Location of the legend on the plot. Default is 'best'.
    - figsize (Tuple[float, float], optional)
        Size of the plot. Default is (10, 10).
    - dpi (Optional[int], optional)
        Resolution of the plot. Default is 100.
    - save (Optional[Union[str, Path]], optional)
        If provided, the path where the plot will be saved.
    - kwargs
        Additional arguments passed to the `scatter` and `imshow` functions for customization.

    Returns
    -------
    - None
        The function displays or saves a plot of interactions between cells and transcripts.

    """
    # Extract relevant data
    transcripts = sdata.points[layer]
    trans_filt = transcripts[transcripts[gene_key] == gene]
    target_cells = trans_filt["closest_target_cell"].compute()
    source_cells = trans_filt["closest_source_cell"].compute()
    cell_positions = pd.DataFrame(sdata["table"].obsm[spatial_key], index=sdata.table.obs[cell_id_key], columns=["x", "y"])

    # Plotting
    plt.figure(figsize=figsize, dpi=dpi)
    if img is not None:
        plt.imshow(img, alpha=img_alpha, cmap=image_cmap, **kwargs)

    # Plot arrows between each paired source and target cell
    for source, target in zip(source_cells, target_cells, strict=False):
        if source in cell_positions.index and target in cell_positions.index:
            if source != target:
                x_start, y_start = cell_positions.loc[source, "x"], cell_positions.loc[source, "y"]
                x_end, y_end = cell_positions.loc[target, "x"], cell_positions.loc[target, "y"]
                plt.arrow(x_start, y_start, x_end - x_start, y_end - y_start, color="black", alpha=0.8, head_width=8, head_length=8)

    # Plot source and target cells
    plt.scatter(cell_positions["x"], cell_positions["y"], c="grey", s=0.6, alpha=alpha, **kwargs)
    plt.scatter(cell_positions.loc[target_cells, "x"], cell_positions.loc[target_cells, "y"], c=color_target, s=size, label="Target Cells", **kwargs)
    plt.scatter(cell_positions.loc[source_cells, "x"], cell_positions.loc[source_cells, "y"], c=color_source, s=size, label="Source Cells", **kwargs)
    plt.scatter(trans_filt["x"], trans_filt["y"], c=color_transcript, s=size * 0.4, label="Transcripts", **kwargs)

    # Titles and Legends
    plt.title(title or gene)
    plt.legend(loc=legend_loc)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")

    # Save the plot if path provided
    if save:
        plt.savefig(save)
    plt.show()


def intra_extra_density(
    sdata, genes, layer="transcripts", gene_key="feature_name", coord_keys=["x", "y"], intra_kde_kwargs=None, extra_kde_kwargs=None, figsize=None
):
    """
    Plots kernel density estimates (KDE) for the spatial distribution of intracellular and extracellular
    transcripts for a list of genes. Each gene is displayed in a separate row with intracellular and
    extracellular KDEs in side-by-side subplots.

    Parameters
    ----------
    - sdata: SpatialData object containing transcript locations and metadata.
    - genes: list of str, gene names to plot.
    - layer: str, layer within sdata.points where transcripts are stored (default: "transcripts").
    - gene_key: str, column name where the gene name is stored (default: "feature_name").
    - coord_keys: list of str, column names for spatial coordinates (default: ["x", "y"]).
    - intra_kde_kwargs: dict, optional arguments for seaborn's kdeplot for intracellular data.
    - extra_kde_kwargs: dict, optional arguments for seaborn's kdeplot for extracellular data.
    """
    if intra_kde_kwargs is None:
        intra_kde_kwargs = {"fill": True, "cmap": "Blues", "thresh": 0.05}
    if extra_kde_kwargs is None:
        extra_kde_kwargs = {"fill": True, "cmap": "Reds", "thresh": 0.05}

    # Convert Dask DataFrame to Pandas if necessary
    transcripts_df = sdata[layer]
    if isinstance(transcripts_df, dd.DataFrame):
        transcripts_df = transcripts_df.compute()

    if figsize == None:
        figsize = (12, 5 * len(genes))
    # Create subplots
    fig, axes = plt.subplots(len(genes), 2, figsize=figsize)
    if len(genes) == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one gene

    for i, gene in enumerate(genes):
        gene_df = transcripts_df[transcripts_df[gene_key] == gene]
        intracellular = gene_df[~gene_df["extracellular"]]
        extracellular = gene_df[gene_df["extracellular"]]

        # Intracellular KDE
        sns.kdeplot(data=intracellular, x=coord_keys[0], y=coord_keys[1], ax=axes[i][0], **intra_kde_kwargs)
        axes[i][0].set_title(f"{gene} - Intracellular")

        # Extracellular KDE
        sns.kdeplot(data=extracellular, x=coord_keys[0], y=coord_keys[1], ax=axes[i][1], **extra_kde_kwargs)
        axes[i][1].set_title(f"{gene} - Extracellular")

    plt.tight_layout()
    plt.show()
