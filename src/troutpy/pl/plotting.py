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


def sorted_heatmap(
    celltype_by_feature: pd.DataFrame,
    output_path: str = "",
    filename: str = "Heatmap_target_cells_by_gene",
    format: str = "pdf",
    cmap: str = "viridis",
    vmax: float | None = None,
    save: bool = False,
    figsize: tuple[float, float] = (10, 10),
) -> None:
    """Plot a heatmap of features by cell type, sorted by each feature's dominant cell type.

    Parameters
    ----------
    celltype_by_feature
        DataFrame showing the value of each feature by cell type.
    output_path
        Directory in which to save the figure. A ``figures`` subdirectory is created within it.
    filename
        Name used for the plot title and, if `save` is `True`, the saved file.
    format
        File format used when saving the figure.
    cmap
        Colormap for the heatmap.
    vmax
        Maximum value for the colormap.
    save
        Whether to save the figure to `output_path`.
    figsize
        Size of the figure.

    Returns
    -------
    None
    """
    figures_path = os.path.join(output_path, "figures")
    os.makedirs(figures_path, exist_ok=True)

    max_indices = np.argmax(celltype_by_feature.values, axis=1)
    celltype_by_feature = celltype_by_feature.iloc[np.argsort(max_indices)]
    celltype_by_feature.index = celltype_by_feature.index[np.argsort(max_indices)]

    plt.figure(figsize=figsize)
    sns.heatmap(celltype_by_feature, cmap=cmap, vmax=vmax)
    plt.ylabel(f"{celltype_by_feature.index.name}")
    plt.xlabel(f"{celltype_by_feature.columns.name}")
    plt.title(filename)
    if save:
        plt.savefig(os.path.join(figures_path, f"{filename}.{format}"))


def coupled_scatter(
    sdata: SpatialData,
    layer: str = "extracellular_transcripts",
    output_path: str = "",
    transcript_group: str = "distance_to_source_cell",
    save: bool = True,
    format: str = "pdf",
    xcoord: str = "x",
    ycoord: str = "y",
    xcellcoord: str = "x_centroid",
    ycellcoord: str = "y_centroid",
    colormap: str = "Blues",
    size: float = 2,
    color_cells: str = "red",
    figsize: tuple[float, float] = (10, 7),
    vmax: float | None = None,
) -> None:
    """Plot transcript locations and cell centroids, coloring transcripts by a chosen feature.

    Parameters
    ----------
    sdata
        A spatial data object that contains transcript and cell information.
    layer
        The key for the layer in `sdata.points` that contains transcript data.
    output_path
        The directory path where the plot will be saved. If not provided, the plot will not be saved.
    transcript_group
        The column in the transcript data (e.g., distance to the source cell) used for coloring the scatter plot.
    save
        Whether to save the plot to a file. If `True`, the plot is saved to `output_path`.
    format
        The format for saving the plot (e.g., 'pdf', 'png'). Only used if `save=True`.
    xcoord
        The column name in the transcript data representing the x-coordinate.
    ycoord
        The column name in the transcript data representing the y-coordinate.
    xcellcoord
        The column name in the cell data representing the x-coordinate of cell centroids.
    ycellcoord
        The column name in the cell data representing the y-coordinate of cell centroids.
    colormap
        The colormap to use for coloring the transcripts based on the `transcript_group` values.
    size
        The size of the scatter points for cells and transcripts. Transcripts are scaled down by 0.1.
    color_cells
        The color to use for the cell centroids.
    figsize
        The size of the figure in inches (width, height).
    vmax
        The upper limit for the colormap.

    Returns
    -------
    None
    """
    adata = sdata["table"].copy()
    adata.X = sdata["table"].layers["raw"]
    adata.obs["x_centroid"] = [sp[0] for sp in adata.obsm["spatial"]]
    adata.obs["y_centroid"] = [sp[1] for sp in adata.obsm["spatial"]]

    transcripts = sdata.points[layer].compute()

    figures_path = os.path.join(output_path, "figures")
    os.makedirs(figures_path, exist_ok=True)

    plt.figure(figsize=figsize)
    plt.scatter(transcripts[xcoord], transcripts[ycoord], c=transcripts[transcript_group], s=size * 0.1, cmap=colormap, vmax=vmax)
    plt.scatter(adata.obs[xcellcoord], adata.obs[ycellcoord], s=size, color=color_cells)
    plt.title(f"{transcript_group}")

    if save:
        plt.savefig(os.path.join(figures_path, f"Scatter_{transcript_group}_{colormap}.{format}"))


def heatmap(
    data: pd.DataFrame,
    output_path: str = "",
    save: bool = False,
    figsize: tuple[float, float] | None = None,
    tag: str = "",
    title: str | None = None,
    cmap: str = "RdBu_r",
    annot: bool = False,
    cbar: bool = True,
    vmax: float | None = None,
    vmin: float = 0,
    row_cluster: bool = True,
    col_cluster: bool = True,
) -> None:
    """Generate a clustered heatmap from the given data and optionally save it to a file.

    Parameters
    ----------
    data
        The data to visualize as a heatmap. Rows and columns will be clustered if specified.
    output_path
        Directory where the heatmap should be saved if `save` is True.
    save
        Whether to save the generated heatmap to a file.
    figsize
        Size of the figure as (width, height). If `None`, the size is calculated based on the data dimensions.
    tag
        A tag to append to the saved file name.
    title
        Title of the heatmap.
    cmap
        Colormap to use for the heatmap.
    annot
        Whether to annotate the heatmap cells with their values.
    cbar
        Whether to display a color bar in the heatmap.
    vmax
        Maximum value for the colormap.
    vmin
        Minimum value for the colormap.
    row_cluster
        Whether to perform hierarchical clustering on rows.
    col_cluster
        Whether to perform hierarchical clustering on columns.

    Returns
    -------
    None

    Notes
    -----
    Clustering is performed using :func:`seaborn.clustermap`.
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
    data: pd.DataFrame,
    xvar: str = "",
    yvar: str = "",
    normalize: bool = True,
    axis: int = 1,
    kind: str = "barh",
    save: bool = True,
    figures_path: str = "",
    stacked: bool = True,
    figsize: tuple[float, float] = (6, 10),
    cmap: str = "viridis",
    saving_format: str = "pdf",
    sortby: str | None = None,
) -> None:
    """Plot a cross-tabulation between two variables as a bar plot, horizontal bar plot, heatmap, or clustermap.

    Parameters
    ----------
    data
        Input dataset containing the variables for the cross-tabulation.
    xvar
        The variable to use on the x-axis for the cross-tabulation.
    yvar
        The variable to use on the y-axis for the cross-tabulation.
    normalize
        Whether to normalize the cross-tabulated data (percentages).
    axis
        The axis to normalize across. Use `1` for row normalization and `0` for column normalization.
    kind
        The kind of plot to generate: ``"barh"``, ``"bar"``, ``"heatmap"``, or ``"clustermap"``.
    save
        If `True`, the plot will be saved to a file.
    figures_path
        The directory path where the figure should be saved. If not specified, the plot will be saved in the current directory.
    stacked
        If `True`, the bar plots will be stacked. Only applicable for ``"barh"`` and ``"bar"`` plot kinds.
    figsize
        The size of the figure for the plot (width, height).
    cmap
        The colormap to use for the plot, especially for heatmap and clustermap visualizations.
    saving_format
        The format to save the plot in. Options include 'png', 'pdf', etc.
    sortby
        The column or row to sort the cross-tabulated data by before plotting.

    Returns
    -------
    None
    """
    crosstab_data = pd.crosstab(data[xvar], data[yvar])

    if normalize:
        crosstab_data = crosstab_data.div(crosstab_data.sum(axis=axis), axis=0)
        normtag = "normalize"
    else:
        normtag = "raw"

    if sortby is not None:
        crosstab_data = crosstab_data.sort_values(by=sortby)

    plot_filename = f"{kind}_{xvar}_{yvar}_{normtag}_{cmap}.{saving_format}"

    if kind == "barh":
        plt.figure()
        crosstab_data.plot(kind="barh", stacked=stacked, figsize=figsize, width=0.99, colormap=cmap)
        plt.xlabel("Total number of transcripts")
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


def genes_over_noise(
    sdata: SpatialData,
    scores_by_genes: pd.DataFrame,
    layer: str = "extracellular_transcripts",
    output_path: str = "",
    save: bool = True,
    format: str = "pdf",
) -> None:
    """Plot the log fold change per gene over noise as a boxplot grouped by codeword category.

    Parameters
    ----------
    sdata
        Spatial data object containing the extracellular transcript layer.
    scores_by_genes
        DataFrame containing gene scores with a ``feature_name`` and a ``log_fold_ratio`` column.
    layer
        Key of the points layer in `sdata` with feature names and codeword categories.
    output_path
        Directory in which to save the figure. A ``figures`` subdirectory is created within it.
    save
        Whether to save the figure to `output_path`.
    format
        File format used when saving the figure.

    Returns
    -------
    None
    """
    data_quantified = sdata.points[layer].compute()
    figures_path = os.path.join(output_path, "figures")
    os.makedirs(figures_path, exist_ok=True)

    feature2codeword = dict(zip(data_quantified["feature_name"], data_quantified["codeword_category"], strict=False))
    scores_by_genes["codeword_category"] = scores_by_genes["feature_name"].map(feature2codeword)

    sns.boxplot(
        data=scores_by_genes,
        y="codeword_category",
        x="log_fold_ratio",
        hue="codeword_category",
    )
    plt.plot([0, 0], [*plt.gca().get_ylim()], "r--")
    if save:
        plt.savefig(os.path.join(figures_path, f"boxplot_log_fold_change_per_gene.{format}"), bbox_inches="tight", pad_inches=0)
    plt.show()


def moranI_histogram(svg_df: pd.DataFrame, save: bool = True, figures_path: str = "", bins: int = 200, format: str = "pdf") -> None:
    """Plot the distribution of Moran's I scores from a DataFrame.

    Parameters
    ----------
    svg_df
        DataFrame containing a column 'I' with Moran's I scores.
    save
        Whether to save the plot as a file.
    figures_path
        Path to save the figure. Only used if `save=True`.
    bins
        Number of bins to use in the histogram.
    format
        Format in which to save the figure (e.g., 'pdf', 'png').

    Returns
    -------
    None
    """
    if save and figures_path:
        if not os.path.exists(figures_path):
            raise ValueError(f"The provided path '{figures_path}' does not exist.")

    plt.figure(figsize=(8, 6))
    plt.hist(svg_df.sort_values(by="I", ascending=False)["I"], bins=bins)
    plt.xlabel("Moran's I")
    plt.ylabel("Frequency")
    plt.title("Distribution of Moran's I Scores")

    if save:
        file_name = os.path.join(figures_path, f"barplot_moranI_by_gene.{format}")
        plt.savefig(file_name, format=format)
        print(f"Plot saved to: {file_name}")

    plt.show()


def proportion_above_threshold(
    df: pd.DataFrame,
    threshold_col: str = "proportion_above_threshold",
    feature_col: str = "feature_name",
    top_percentile: float = 0.05,
    bottom_percentile: float = 0.05,
    specific_transcripts: list[str] | None = None,
    figsize: tuple[float, float] = (4, 10),
    orientation: str = "h",
    bar_color: str = "black",
    title: str = "Proportion of distant exRNa (>30um) from source",
    xlabel: str = "Proportion above threshold",
    ylabel: str = "Feature",
    save: bool = False,
    output_path: str = "",
    format: str = "pdf",
) -> None:
    """Plot the top and bottom percentiles of features by proportion above a threshold, or a specific list of transcripts.

    Parameters
    ----------
    df
        DataFrame containing feature proportions.
    threshold_col
        Column name for proportions above the threshold.
    feature_col
        Column name for feature names.
    top_percentile
        Proportion (0-1) of features with the highest proportions to display.
    bottom_percentile
        Proportion (0-1) of features with the lowest proportions to display.
    specific_transcripts
        List of specific transcript names to plot. If provided, `top_percentile` and
        `bottom_percentile` are ignored.
    figsize
        Size of the figure.
    orientation
        Orientation of the bars: ``"h"`` for horizontal, ``"v"`` for vertical.
    bar_color
        Color of the bars.
    title
        Title of the plot.
    xlabel
        Label for the x-axis.
    ylabel
        Label for the y-axis.
    save
        Whether to save the figure to `output_path`.
    output_path
        Directory in which to save the figure. A ``figures`` subdirectory is created within it.
    format
        File format used when saving the figure.

    Returns
    -------
    None
    """
    df = df[~df[threshold_col].isna()]
    if specific_transcripts is None:
        top_cutoff = df[threshold_col].quantile(1 - top_percentile)
        bottom_cutoff = df[threshold_col].quantile(bottom_percentile)
        plot_data = pd.concat(
            [
                df[df[threshold_col] >= top_cutoff],
                df[df[threshold_col] <= bottom_cutoff],
            ]
        )
    else:
        plot_data = df[df[feature_col].isin(specific_transcripts)]

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
    sdata: SpatialData, nmf_adata_key: str = "nmf_data", save: bool = True, saving_path: str = "", spot_size: int = 30, cmap: str = "viridis"
) -> None:
    """Plot the spatial distribution of each NMF factor for cells.

    Extracts the NMF (Non-negative Matrix Factorization) cell loadings (``W`` matrix) from
    the AnnData object stored at `nmf_adata_key` within `sdata` and creates a spatial plot
    for each factor.

    Parameters
    ----------
    sdata
        A spatial transcriptomics dataset that contains the NMF factors in the specified key.
    nmf_adata_key
        The key in `sdata` that contains the AnnData object with NMF results.
    save
        Whether to save the spatial factor plots to disk. The plots are saved in a ``figures``
        subdirectory of `saving_path`.
    saving_path
        Path where the plots should be saved if `save` is `True`.
    spot_size
        Size of the spots in the spatial plot. Only used if `save` is `False`.
    cmap
        Colormap to use for the spatial plots.

    Returns
    -------
    None
    """
    adata = sdata[nmf_adata_key]
    W = adata.obsm["W_nmf"]
    for factor in range(20):
        adata.obs[f"NMF_factor_{factor + 1}"] = W[:, factor]
        if save:
            sc.pl.spatial(adata, color=f"NMF_factor_{factor + 1}", cmap=cmap, title=f"NMF Factor {factor + 1}", spot_size=30, show=False)
            plt.savefig(saving_path + "/figures/" + f"spatialnmf{factor}.png")
            plt.show()
        else:
            sc.pl.spatial(adata, color=f"NMF_factor_{factor + 1}", cmap=cmap, title=f"NMF Factor {factor + 1}", spot_size=spot_size)


def nmf_gene_contributions(
    sdata: SpatialData,
    nmf_adata_key: str = "nmf_data",
    save: bool = True,
    vmin: float = 0.0,
    vmax: float = 0.02,
    saving_path: str = "",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (5, 5),
) -> None:
    """Plot a heatmap of NMF gene loadings, filtered to genes with a high maximum loading.

    Extracts the NMF gene loadings matrix (``H`` matrix) from the AnnData object stored at
    `nmf_adata_key` within `sdata`, keeps only genes whose maximum loading exceeds 0.05, and
    plots a heatmap of the filtered loadings sorted by their dominant factor.

    Parameters
    ----------
    sdata
        A spatial transcriptomics dataset that contains the NMF factors in the specified key.
    nmf_adata_key
        The key in `sdata` that contains the AnnData object with NMF results.
    save
        Whether to save the heatmap plot to disk. The plot is saved in a ``figures``
        subdirectory of `saving_path`.
    vmin
        Minimum value for the colormap scale.
    vmax
        Maximum value for the colormap scale.
    saving_path
        Path where the plot should be saved if `save` is `True`.
    cmap
        Colormap to use for the heatmap.
    figsize
        Size of the heatmap figure.

    Returns
    -------
    None
    """
    adata = sdata[nmf_adata_key]
    loadings = pd.DataFrame(adata.uns["H_nmf"], columns=adata.var.index)
    loadings_filtered = loadings.loc[:, np.max(loadings, axis=0) > 0.05].transpose()
    figures_path = os.path.join(saving_path, "figures")
    os.makedirs(figures_path, exist_ok=True)

    max_indices = np.argmax(loadings_filtered.values, axis=1)
    loadings_filtered = loadings_filtered.iloc[np.argsort(max_indices)]
    loadings_filtered.index = loadings_filtered.index[np.argsort(max_indices)]

    plt.figure(figsize=figsize)
    sns.heatmap(loadings_filtered, cmap=cmap, vmax=1)
    if save:
        plt.savefig(os.path.join(figures_path, "loadings_NMF.pdf"))
    plt.show()
    plt.close()


def apply_exrnaH_to_cellular_to_create_cellularW(adata_extracellular_with_nmf, adata_annotated_cellular):
    """Transfer NMF gene loadings from an extracellular RNA dataset to a cellular dataset.

    Computes the cellular ``W`` matrix by multiplying the cellular gene expression values
    with the filtered ``H`` matrix (gene loadings) from the extracellular NMF results,
    restricted to genes shared between both datasets.

    Parameters
    ----------
    adata_extracellular_with_nmf: anndata.AnnData
        An AnnData object containing the extracellular RNA data with the NMF results. The H matrix is expected to be stored in `adata.uns['H_nmf']`.
    adata_annotated_cellular: anndata.AnnData
        An AnnData object containing the cellular RNA data with annotated gene expression values.

    Returns
    -------
    anndata.AnnData
        The input `adata_annotated_cellular` object with the calculated NMF factors (``W``
        matrix) added as a DataFrame to ``adata.obsm["factors"]``, and each factor also added
        as an individual column ``NMF_factor_1``, ``NMF_factor_2``, ... in ``adata.obs``.
    """
    H = adata_extracellular_with_nmf.uns["H_nmf"]

    genes_spots2region = adata_extracellular_with_nmf.var_names
    genes_annotated = adata_annotated_cellular.var_names

    common_genes = genes_annotated.intersection(genes_spots2region)

    adata_annotated_cellular = adata_annotated_cellular[:, common_genes]
    H_filtered = H[:, np.isin(genes_spots2region, common_genes)]

    W_annotated = adata_annotated_cellular.X @ H_filtered.T

    adata_annotated_cellular.obsm["factors"] = pd.DataFrame(W_annotated, index=adata_annotated_cellular.obs.index)

    for factor in range(W_annotated.shape[1]):
        adata_annotated_cellular.obs[f"NMF_factor_{factor + 1}"] = W_annotated[:, factor]

    return adata_annotated_cellular


def paired_nmf_factors(
    sdata: SpatialData,
    layer: str = "nmf_data",
    n_factors: int = 5,
    figsize: tuple[float, float] = (12, 6),
    spot_size_exrna: float = 5,
    spot_size_cells: float = 10,
    cmap_exrna: str = "YlGnBu",
    cmap_cells: str = "Reds",
    vmax_exrna: str | float | None = "p99",
    vmax_cells: str | float | None = None,
    save: bool = False,
    output_path: str = "",
    format: str = "pdf",
) -> None:
    """Plot the spatial distribution of NMF factors for extracellular transcripts and cells.

    For each factor, the extracellular spot loadings and the cellular loadings are overlaid
    on the same spatial axes.

    Parameters
    ----------
    sdata
        Spatial data object containing both extracellular and cell data.
    layer
        Layer in `sdata` to extract the extracellular NMF data from.
    n_factors
        Number of NMF factors to plot.
    figsize
        Size of the figure for each subplot.
    spot_size_exrna
        Size of the spots for the extracellular transcript scatter plot.
    spot_size_cells
        Size of the spots for the cell scatter plot.
    cmap_exrna
        Colormap for the extracellular transcript NMF factors.
    cmap_cells
        Colormap for the cell NMF factors.
    vmax_exrna
        Maximum value for the extracellular transcript color scale.
    vmax_cells
        Maximum value for the cell color scale.
    save
        Whether to save each factor's figure to `output_path`.
    output_path
        Directory in which to save the figures. A ``figures`` subdirectory is created within it.
    format
        File format used when saving the figures.

    Returns
    -------
    None
    """
    adata = sdata[layer]
    adata_annotated = sdata["table"]

    factors = pd.DataFrame(adata.obsm["cell_loadings"], index=adata.obs.index)
    factors.columns = [f"Factor_{fact + 1}" for fact in factors.columns]
    for f in factors.columns:
        adata.obs[f] = factors[f]

    factors = pd.DataFrame(adata_annotated.obsm["factors_cell_loadings"], index=adata_annotated.obs.index)
    factors.columns = [f"Factor_{fact + 1}" for fact in factors.columns]
    for f in factors.columns:
        adata_annotated.obs[f] = factors[f]

    for factor in range(n_factors):
        factor_name = f"Factor_{factor + 1}"

        _, axs = plt.subplots(1, 1, figsize=figsize)

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

        plt.tight_layout()
        plt.show()


def plot_nmf_factors_spatial(adata, n_factors: int, save: bool = True) -> None:
    """Plot the spatial distribution of cells colored by each NMF factor.

    Parameters
    ----------
    adata: anndata.AnnData
        An AnnData object containing the dataset with NMF factors already added as columns in `adata.obs`. Each factor should be named `NMF_factor_1`, `NMF_factor_2`, ..., `NMF_factor_n`.
    n_factors
        The number of NMF factors to plot.
    save
        If `True`, saves the plots to files with filenames `exo_to_cell_spatial_<factor>.png`.

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
) -> None:
    """Plot the positions of target cells, source cells, and extracellular RNA transcripts for a gene.

    Target and source cells are highlighted in different colors, while the RNA transcripts
    are shown as points at their respective positions. Optionally, a background image (e.g.
    a tissue section) can be displayed.

    Parameters
    ----------
    sdata
        Spatial data object containing the transcript and cell position data.
    layer
        The layer in `sdata.points` that contains the extracellular RNA transcript data, with
        ``closest_target_cell`` and ``closest_source_cell`` columns.
    gene
        The gene of interest to be visualized in terms of its spatial interaction with source
        and target cells.
    gene_key
        The column name used to identify the gene in `layer`.
    cell_id_key
        The column name in ``sdata["table"].obs`` used to identify individual cells.
    color_target
        The color to be used for target cells in the plot.
    color_source
        The color to be used for source cells in the plot.
    color_transcript
        The color to be used for the RNA transcripts in the plot.
    spatial_key
        The key in ``sdata["table"].obsm`` that stores the spatial coordinates of the cells.
    img
        A background image to overlay on the plot, such as a tissue section.
    img_alpha
        The transparency level of the background image. Ignored if `img` is `None`.
    image_cmap
        The colormap to be used for the background image, if applicable.
    size
        The size of the scatter plot points for the cells and transcripts.
    alpha
        The transparency level for the scatter plot points.
    title
        The title of the plot. If `None`, the gene name is used.
    legend_loc
        The location of the legend in the plot.
    figsize
        The dimensions of the plot in inches.
    dpi
        The resolution (dots per inch) for the plot.
    save
        The path to save the plot image. If `None`, the plot is displayed but not saved.
    kwargs
        Additional arguments passed to :func:`matplotlib.pyplot.scatter` and
        :func:`matplotlib.pyplot.imshow`.

    Returns
    -------
    None
    """
    transcripts = sdata.points[layer]
    trans_filt = transcripts[transcripts[gene_key] == gene]
    target_cells = trans_filt["closest_target_cell"].compute()
    source_cells = trans_filt["closest_source_cell"].compute()
    cell_positions = pd.DataFrame(sdata["table"].obsm[spatial_key], index=sdata.table.obs[cell_id_key], columns=["x", "y"])

    plt.figure(figsize=figsize, dpi=dpi)
    if img is not None:
        plt.imshow(img, alpha=img_alpha, cmap=image_cmap, **kwargs)
    plt.scatter(cell_positions["x"], cell_positions["y"], c="grey", s=0.6, alpha=alpha, **kwargs)
    plt.scatter(cell_positions.loc[target_cells, "x"], cell_positions.loc[target_cells, "y"], c=color_target, s=size, label="Target Cells", **kwargs)
    plt.scatter(cell_positions.loc[source_cells, "x"], cell_positions.loc[source_cells, "y"], c=color_source, s=size, label="Source Cells", **kwargs)
    plt.scatter(trans_filt["x"], trans_filt["y"], c=color_transcript, s=size * 0.4, label="Transcripts", **kwargs)

    plt.title(title or gene)
    plt.legend(loc=legend_loc)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")

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
) -> None:
    """Plot arrows from source to target cells based on transcript proximity for a gene.

    Source and target cells, and the transcript locations, are color-coded. An optional
    background image (e.g. a tissue section) can be overlaid behind the plot.

    Parameters
    ----------
    sdata
        Spatial data object containing the transcript and cell position data.
    layer
        The layer in `sdata.points` that contains the extracellular RNA transcript data, with
        ``closest_target_cell`` and ``closest_source_cell`` columns.
    gene
        The gene of interest.
    gene_key
        The column name used to identify the gene in `layer`.
    cell_id_key
        The column name in ``sdata["table"].obs`` used to identify individual cells.
    color_target
        Color for the target cells.
    color_source
        Color for the source cells.
    color_transcript
        Color for the transcript locations.
    spatial_key
        The key in ``sdata["table"].obsm`` that stores the spatial coordinates of the cells.
    img
        Optional background image (e.g., tissue section) to display behind the plot.
    img_alpha
        Transparency level for the background image. Ignored if `img` is `None`.
    image_cmap
        Colormap for the background image.
    size
        Size of the plotted points (cells and transcripts).
    alpha
        Transparency level for the plotted points.
    title
        Title of the plot. If `None`, the gene name is used.
    legend_loc
        Location of the legend on the plot.
    figsize
        Size of the plot.
    dpi
        Resolution of the plot.
    save
        If provided, the path where the plot will be saved.
    kwargs
        Additional arguments passed to :func:`matplotlib.pyplot.scatter` and
        :func:`matplotlib.pyplot.imshow`.

    Returns
    -------
    None
    """
    transcripts = sdata.points[layer]
    trans_filt = transcripts[transcripts[gene_key] == gene]
    target_cells = trans_filt["closest_target_cell"].compute()
    source_cells = trans_filt["closest_source_cell"].compute()
    cell_positions = pd.DataFrame(sdata["table"].obsm[spatial_key], index=sdata.table.obs[cell_id_key], columns=["x", "y"])

    plt.figure(figsize=figsize, dpi=dpi)
    if img is not None:
        plt.imshow(img, alpha=img_alpha, cmap=image_cmap, **kwargs)

    for source, target in zip(source_cells, target_cells, strict=False):
        if source in cell_positions.index and target in cell_positions.index:
            if source != target:
                x_start, y_start = cell_positions.loc[source, "x"], cell_positions.loc[source, "y"]
                x_end, y_end = cell_positions.loc[target, "x"], cell_positions.loc[target, "y"]
                plt.arrow(x_start, y_start, x_end - x_start, y_end - y_start, color="black", alpha=0.8, head_width=8, head_length=8)

    plt.scatter(cell_positions["x"], cell_positions["y"], c="grey", s=0.6, alpha=alpha, **kwargs)
    plt.scatter(cell_positions.loc[target_cells, "x"], cell_positions.loc[target_cells, "y"], c=color_target, s=size, label="Target Cells", **kwargs)
    plt.scatter(cell_positions.loc[source_cells, "x"], cell_positions.loc[source_cells, "y"], c=color_source, s=size, label="Source Cells", **kwargs)
    plt.scatter(trans_filt["x"], trans_filt["y"], c=color_transcript, s=size * 0.4, label="Transcripts", **kwargs)

    plt.title(title or gene)
    plt.legend(loc=legend_loc)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")

    if save:
        plt.savefig(save)
    plt.show()


def intra_extra_density(
    sdata: SpatialData,
    genes: list[str],
    layer: str = "transcripts",
    gene_key: str = "feature_name",
    coord_keys: list[str] | None = None,
    intra_kde_kwargs: dict | None = None,
    extra_kde_kwargs: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot KDEs of the spatial distribution of intracellular and extracellular transcripts for a list of genes.

    Each gene is displayed in a separate row with intracellular and extracellular KDEs in
    side-by-side subplots.

    Parameters
    ----------
    sdata
        Spatial data object containing transcript locations and metadata.
    genes
        Gene names to plot.
    layer
        Layer within `sdata` where transcripts are stored.
    gene_key
        Column name where the gene name is stored.
    coord_keys
        Column names for spatial coordinates. If `None`, defaults to ``["x", "y"]``.
    intra_kde_kwargs
        Optional arguments for :func:`seaborn.kdeplot` for intracellular data.
    extra_kde_kwargs
        Optional arguments for :func:`seaborn.kdeplot` for extracellular data.
    figsize
        Size of the figure. If `None`, computed automatically based on the number of genes.

    Returns
    -------
    None
    """
    if coord_keys is None:
        coord_keys = ["x", "y"]
    if intra_kde_kwargs is None:
        intra_kde_kwargs = {"fill": True, "cmap": "Blues", "thresh": 0.05}
    if extra_kde_kwargs is None:
        extra_kde_kwargs = {"fill": True, "cmap": "Reds", "thresh": 0.05}

    transcripts_df = sdata[layer]
    if isinstance(transcripts_df, dd.DataFrame):
        transcripts_df = transcripts_df.compute()

    if figsize is None:
        figsize = (12, 5 * len(genes))
    _, axes = plt.subplots(len(genes), 2, figsize=figsize)
    if len(genes) == 1:
        axes = [axes]

    for i, gene in enumerate(genes):
        gene_df = transcripts_df[transcripts_df[gene_key] == gene]
        intracellular = gene_df[~gene_df["extracellular"]]
        extracellular = gene_df[gene_df["extracellular"]]

        sns.kdeplot(data=intracellular, x=coord_keys[0], y=coord_keys[1], ax=axes[i][0], **intra_kde_kwargs)
        axes[i][0].set_title(f"{gene} - Intracellular")

        sns.kdeplot(data=extracellular, x=coord_keys[0], y=coord_keys[1], ax=axes[i][1], **extra_kde_kwargs)
        axes[i][1].set_title(f"{gene} - Extracellular")

    plt.tight_layout()
    plt.show()
