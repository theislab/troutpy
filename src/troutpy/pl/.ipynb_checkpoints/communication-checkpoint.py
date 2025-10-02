import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spatialdata as sd
from mpl_chord_diagram import chord_diagram
from spatialdata import SpatialData

def cell_type_contacts(
    sdata,
    kind="heatmap",
    celltype_key="cell type",
    vmax=None,
    vmin=None,
    cmap="BuPu",table_key='table',
    dendrogram_ratio=0.1,key='cell_contact_combined',
    **kwargs
):
    """Plot cell type-cell type interaction strength as a heatmap or chord diagram."""
    interaction_strength = sdata["table"].uns[key]
    source_table = sdata["source_score"]
    target_table = sdata["target_score"]

    # Sum over genes → interaction strength from source to target
    celltype_ints = interaction_strength
    celltype_ints_table = pd.DataFrame(celltype_ints, index=source_table.var.index, columns=target_table.var.index)

    # Assign colors to cell types
    try:
        cols = dict(zip(
            np.unique(sdata["table"].obs[celltype_key]),
            sdata["table"].uns[celltype_key + "_colors"], strict=False
        ))
        colors = [cols[c] for c in source_table.var.index]
    except KeyError:
        colpalette = plt.get_cmap("tab20")
        colors = [colpalette(i) for i in range(len(np.unique(source_table.var.index)))]

    # Plotting
    if kind == "heatmap":
        g = sns.clustermap(
            celltype_ints_table,
            vmax=vmax,
            vmin=vmin,
            cmap=cmap,
            row_colors=colors,
            col_colors=colors,
            dendrogram_ratio=dendrogram_ratio,
            cbar_pos=(0.02, 0.8, 0.02, 0.15),  # Adjust colorbar position: (x, y, width, height)
            **kwargs
        )
        g.fig.suptitle(key, y=1.05)
        g.ax_heatmap.grid(False)  # remove grid from heatmap

    elif kind == "chord":
        chord_diagram(
            celltype_ints,
            source_table.var.index,
            directed=True,
            fontsize=6,
            colors=colors,
            **kwargs
        )
        plt.title(key, fontweight="bold")
        plt.grid(False)

    else:
        raise ValueError("Invalid plot type. Choose 'heatmap' or 'chord'.")


def gene_communication(
    sdata: SpatialData, kind="heatmap", gene: str = "", celltype_key="cell type", vmax=None, vmin=None, cmap="BuPu", dendrogram_ratio=0.1, **kwargs
):
    """
    Plot cell type-cell type interaction strength as a heatmap or chord diagram.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        The spatial data object containing interaction scores.
    kind : str
        Type of plot, either 'heatmap' or 'chord'. Default is 'heatmap'.
    celltype_key : str
        Key for cell type colors in `sdata['table'].uns`. Default is 'cell type'.
    gene: str
        Name of the gene to be plotted
    vmax : float
        Max value for colormap scaling. Default is None.
    vmin : float
        Min value for colormap scaling. Default is None.
    cmap : str
        Colormap for heatmap or chord diagram. Default is 'BuPu'.
    kwargs : dict
        Additional arguments passed to the plotting functions.
    """
    gene_interaction_strength = sdata["source_score"].uns["gene_interaction_strength"]
    source_table = sdata["source_score"]
    target_table = sdata["target_score"]
    unique_cats = sdata.tables["source_score"].uns["gene_interaction_names"]
    if str(gene) not in unique_cats:
        raise KeyError("Gene name not found in the dataset")
    celltype_ints_table = pd.DataFrame(
        gene_interaction_strength[(unique_cats == gene), :, :].squeeze(), index=source_table.var.index, columns=target_table.var.index
    )
    try:
        colors = sdata["table"].uns[celltype_key + "_colors"]
    except KeyError:
        colpalette = plt.get_cmap("tab20")
        colors = [colpalette(i) for i in range(len(np.unique(source_table.var.index)))]

    if kind == "heatmap":
        sns.clustermap(
            celltype_ints_table, vmax=vmax, vmin=vmin, cmap=cmap, row_colors=colors, col_colors=colors, dendrogram_ratio=dendrogram_ratio, **kwargs
        ).fig.suptitle("Interaction strenght")
    elif kind == "chord":
        chord_diagram(
            gene_interaction_strength[(unique_cats == gene), :, :].squeeze(),
            source_table.var.index,
            directed=True,
            fontsize=5,
            colors=colors,
            **kwargs,
        )
        plt.title(str(gene) + " interaction Strength", fontweight="bold")
    else:
        raise ValueError("Invalid plot type. Choose 'heatmap' or 'chord'.")

    plt.show()


def target_score_by_celltype(
    sdata: sd.SpatialData,
    gene_key: str = "gene",
    min_counts: int = 100,
    min_value: float | None = None,
    max_value: float | None = None,
    title: str | None = "Target Score by Cell Type",
    cluster_axis: str = "both",
    cmap: str = "coolwarm",
    figsize: tuple = None,
) -> None:
    """
    Plots a heatmap or clustered heatmap of target scores by cell type.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        A SpatialData object containing `target_score` data.
    gene_key : str
        The key in `obs` that contains gene names.
    min_counts : int
        Minimum count threshold for genes to be included.
    min_value : float
        Genes presenting the highest target score below this will be filtered out in visualization
    max_value : float
        Genes presenting the highest target score above this will be filtered out in visualization
    title : str
        Custom title for the plot.
    cluster_axis : str
        Determines clustering:
        - "none" (no clustering)
        - "x" (cluster columns only)
        - "y" (cluster rows only)
        - "both" (cluster rows and columns)
    cmap : str
        Colormap for the heatmap.
    figsize : tuple
        Size of the figure.

    Returns
    -------
    None
        Displays the heatmap.
    """
    # Extract target scores
    target_score = sdata["target_score"].to_df()
    target_score["gene"] = sdata["target_score"].obs[gene_key]
    gene_by_celltype_score = target_score.groupby("gene").mean()

    if figsize is None:
        figsize = (10, 8)
    # Filter genes based on count threshold
    genes = sdata["xrna_metadata"].var.index[sdata["xrna_metadata"].var["count"] > min_counts]
    filtered_gene_by_celltype_score = gene_by_celltype_score.loc[gene_by_celltype_score.index.isin(genes), :]

    # Apply additional filtering based on min/max values
    if min_value is not None:
        filtered_gene_by_celltype_score = filtered_gene_by_celltype_score[np.max(filtered_gene_by_celltype_score, axis=1) >= min_value].dropna()
    if max_value is not None:
        filtered_gene_by_celltype_score = filtered_gene_by_celltype_score[np.max(filtered_gene_by_celltype_score, axis=1) <= max_value].dropna()

    if filtered_gene_by_celltype_score.empty:
        print("No data available to plot after filtering.")
        return

    # Determine clustering options
    valid_axes = ["none", "x", "y", "both"]
    cluster_axis = cluster_axis.lower()
    if cluster_axis not in valid_axes:
        raise ValueError(f"Invalid cluster_axis: {cluster_axis}. Must be one of {', '.join(valid_axes)}.")

    # Plot heatmap with clustering
    if cluster_axis == "none":
        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            filtered_gene_by_celltype_score,
            cmap=cmap,
            linewidths=0.001,
            linecolor="gray",
            annot=False,
            fmt=".2f",
            cbar=True,
            yticklabels=True,
        )
    else:
        g = sns.clustermap(
            filtered_gene_by_celltype_score,
            cmap=cmap,
            linewidths=0.001,
            linecolor="gray",
            annot=False,
            fmt=".2f",
            cbar=True,
            figsize=figsize,
            row_cluster=(cluster_axis in ["y", "both"]),
            col_cluster=(cluster_axis in ["x", "both"]),
        )
        g.ax_heatmap.set_title(title, fontsize=14, fontweight="bold", pad=15)
        plt.show()
        return

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.show()
