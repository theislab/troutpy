import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from spatialdata import SpatialData


def cell_type_contacts(
    sdata: SpatialData,
    kind: str = "heatmap",
    celltype_key: str = "cell type",
    vmax: float | None = None,
    vmin: float | None = None,
    cmap: str = "BuPu",
    table_key: str = "table",
    dendrogram_ratio: float = 0.1,
    key: str = "cell_contact_combined",
    **kwargs,
) -> None:
    """Plot cell type-cell type interaction strength as a heatmap or chord diagram.

    Parameters
    ----------
    sdata
        The spatial data object containing interaction scores.
    kind
        Type of plot, either ``"heatmap"`` or ``"chord"``.
    celltype_key
        Key in ``sdata[table_key].obs``/``.uns`` used to resolve per-cell-type colors.
    vmax
        Max value for colormap scaling.
    vmin
        Min value for colormap scaling.
    cmap
        Colormap for the heatmap.
    table_key
        Key of the table in `sdata` holding the cell-type metadata and colors.
    dendrogram_ratio
        Passed to :func:`seaborn.clustermap`.
    key
        Key in ``sdata[table_key].uns`` holding the cell type x cell type interaction matrix.
    kwargs
        Additional arguments passed to :func:`seaborn.clustermap` or :func:`mpl_chord_diagram.chord_diagram`.

    Returns
    -------
    None
    """
    interaction_strength = sdata[table_key].uns[key]
    source_table = sdata["source_score"]
    target_table = sdata["target_score"]

    celltype_ints_table = pd.DataFrame(interaction_strength, index=source_table.var.index, columns=target_table.var.index)

    try:
        cols = dict(
            zip(
                np.unique(sdata[table_key].obs[celltype_key]),
                sdata[table_key].uns[celltype_key + "_colors"],
                strict=False,
            )
        )
        colors = [cols[c] for c in source_table.var.index]
    except KeyError:
        colpalette = plt.get_cmap("tab20")
        colors = [colpalette(i) for i in range(len(np.unique(source_table.var.index)))]

    if kind == "heatmap":
        g = sns.clustermap(
            celltype_ints_table,
            vmax=vmax,
            vmin=vmin,
            cmap=cmap,
            row_colors=colors,
            col_colors=colors,
            dendrogram_ratio=dendrogram_ratio,
            cbar_pos=(0.02, 0.8, 0.02, 0.15),
            **kwargs,
        )
        g.fig.suptitle(key, y=1.05)
        g.ax_heatmap.grid(False)

    elif kind == "chord":
        try:
            from mpl_chord_diagram import chord_diagram
        except ImportError as err:
            raise ImportError("The 'mpl-chord-diagram' package is required for kind='chord'. Please install it with: pip install troutpy[chord]") from err

        chord_diagram(
            interaction_strength,
            source_table.var.index,
            directed=True,
            fontsize=6,
            colors=colors,
            **kwargs,
        )
        plt.title(key, fontweight="bold")
        plt.grid(False)

    else:
        raise ValueError("Invalid plot type. Choose 'heatmap' or 'chord'.")


def gene_communication(
    sdata: SpatialData,
    kind: str = "heatmap",
    gene: str = "",
    celltype_key: str = "cell type",
    vmax: float | None = None,
    vmin: float | None = None,
    cmap: str = "BuPu",
    dendrogram_ratio: float = 0.1,
    **kwargs,
) -> None:
    """Plot gene-level cell type-cell type interaction strength as a heatmap or chord diagram.

    Parameters
    ----------
    sdata
        The spatial data object containing interaction scores.
    kind
        Type of plot, either ``"heatmap"`` or ``"chord"``.
    gene
        Name of the gene to plot. Must be present in
        ``sdata["source_score"].uns["gene_interaction_names"]``.
    celltype_key
        Key for cell type colors in ``sdata["table"].uns``.
    vmax
        Max value for colormap scaling.
    vmin
        Min value for colormap scaling.
    cmap
        Colormap for heatmap or chord diagram.
    dendrogram_ratio
        Passed to :func:`seaborn.clustermap`.
    kwargs
        Additional arguments passed to :func:`seaborn.clustermap` or :func:`mpl_chord_diagram.chord_diagram`.

    Returns
    -------
    None
    """
    gene_interaction_strength = sdata["source_score"].uns["gene_interaction_strength"]
    source_table = sdata["source_score"]
    target_table = sdata["target_score"]
    unique_cats = source_table.uns["gene_interaction_names"]
    if gene not in unique_cats:
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
        ).fig.suptitle("Interaction strength")
    elif kind == "chord":
        try:
            from mpl_chord_diagram import chord_diagram
        except ImportError as err:
            raise ImportError("The 'mpl-chord-diagram' package is required for kind='chord'. Please install it with: pip install troutpy[chord]") from err

        chord_diagram(
            gene_interaction_strength[(unique_cats == gene), :, :].squeeze(),
            source_table.var.index,
            directed=True,
            fontsize=5,
            colors=colors,
            **kwargs,
        )
        plt.title(gene + " interaction strength", fontweight="bold")
    else:
        raise ValueError("Invalid plot type. Choose 'heatmap' or 'chord'.")

    plt.show()


def target_score_by_celltype(
    sdata: SpatialData,
    gene_key: str = "gene",
    min_counts: int = 100,
    min_value: float | None = None,
    max_value: float | None = None,
    title: str | None = "Target Score by Cell Type",
    cluster_axis: str = "both",
    cmap: str = "coolwarm",
    figsize: tuple | None = None,
) -> None:
    """Plot a heatmap or clustered heatmap of target scores by cell type.

    Parameters
    ----------
    sdata
        A SpatialData object containing ``target_score`` and ``xrna_metadata`` tables.
    gene_key
        The key in ``sdata["target_score"].obs`` that contains gene names.
    min_counts
        Minimum count threshold for genes to be included.
    min_value
        Genes whose highest target score is below this value are filtered out.
    max_value
        Genes whose highest target score is above this value are filtered out.
    title
        Custom title for the plot.
    cluster_axis
        Determines clustering: ``"none"`` (no clustering), ``"x"`` (cluster columns
        only), ``"y"`` (cluster rows only), or ``"both"`` (cluster rows and columns).
    cmap
        Colormap for the heatmap.
    figsize
        Size of the figure. If `None`, computed automatically.

    Returns
    -------
    None
    """
    target_score = sdata["target_score"].to_df()
    target_score["gene"] = sdata["target_score"].obs[gene_key]
    gene_by_celltype_score = target_score.groupby("gene").mean()

    if figsize is None:
        figsize = (10, 8)

    genes = sdata["xrna_metadata"].var.index[sdata["xrna_metadata"].var["count"] > min_counts]
    filtered_gene_by_celltype_score = gene_by_celltype_score.loc[gene_by_celltype_score.index.isin(genes), :]

    if min_value is not None:
        filtered_gene_by_celltype_score = filtered_gene_by_celltype_score[np.max(filtered_gene_by_celltype_score, axis=1) >= min_value].dropna()
    if max_value is not None:
        filtered_gene_by_celltype_score = filtered_gene_by_celltype_score[np.max(filtered_gene_by_celltype_score, axis=1) <= max_value].dropna()

    if filtered_gene_by_celltype_score.empty:
        print("No data available to plot after filtering.")
        return

    valid_axes = ["none", "x", "y", "both"]
    cluster_axis = cluster_axis.lower()
    if cluster_axis not in valid_axes:
        raise ValueError(f"Invalid cluster_axis: {cluster_axis}. Must be one of {', '.join(valid_axes)}.")

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


def celltype_contact_matrix(
    mat: pd.DataFrame,
    sdata: SpatialData | None = None,
    kind: str = "heatmap",
    cell_type_key: str = "leiden",
    cmap: str = "BuPu",
    vmin: float | None = None,
    vmax: float | None = None,
    dendrogram_ratio: float = 0.1,
    title: str = "Cell-type contact matrix",
    palette: str = "tab20",
    **kwargs,
):
    """Plot a cell type x cell type contact matrix as a heatmap or chord diagram.

    Parameters
    ----------
    mat
        Square contact matrix, as returned by :func:`troutpy.tl.celltype_contact_matrix`,
        with rows as source cell types and columns as neighbour cell types.
    sdata
        If provided, cell-type colors are taken from
        ``sdata["table"].uns[cell_type_key + "_colors"]``. Otherwise, `palette` is used
        as a fallback.
    kind
        Type of plot, either ``"heatmap"`` (:func:`seaborn.clustermap`) or ``"chord"``
        (:func:`mpl_chord_diagram.chord_diagram`).
    cell_type_key
        Key in ``sdata["table"].obs``/``.uns`` used to resolve per-cell-type colors.
    cmap
        Colormap for the heatmap.
    vmin
        Min value for colormap scaling.
    vmax
        Max value for colormap scaling.
    dendrogram_ratio
        Passed to :func:`seaborn.clustermap`.
    title
        Figure title.
    palette
        Matplotlib colormap name used as a fallback when `sdata` is `None` or has no
        stored colors for `cell_type_key`.
    kwargs
        Additional arguments passed to :func:`seaborn.clustermap` or :func:`mpl_chord_diagram.chord_diagram`.

    Returns
    -------
    For ``kind="heatmap"``, the :class:`seaborn.matrix.ClusterGrid`. For
    ``kind="chord"``, the :class:`matplotlib.figure.Figure`.
    """
    cell_types = list(mat.index)

    colors = None
    if sdata is not None:
        try:
            uns_colors = sdata["table"].uns[cell_type_key + "_colors"]
            cat_order = list(sdata["table"].obs[cell_type_key].astype("category").cat.categories)
            color_map = dict(zip(cat_order, uns_colors, strict=False))
            colors = [color_map[ct] for ct in cell_types]
        except (KeyError, TypeError):
            colors = None

    if colors is None:
        cmap_fb = plt.get_cmap(palette)
        colors = [cmap_fb(i / max(1, len(cell_types) - 1)) for i in range(len(cell_types))]

    if kind == "heatmap":
        g = sns.clustermap(
            mat,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            row_colors=colors,
            col_colors=colors,
            dendrogram_ratio=dendrogram_ratio,
            cbar_pos=(0.02, 0.8, 0.02, 0.15),
            **kwargs,
        )
        g.fig.suptitle(title, y=1.02, fontweight="bold")
        g.ax_heatmap.grid(False)
        g.ax_heatmap.set_xlabel("Neighbour cell type", labelpad=8)
        g.ax_heatmap.set_ylabel("Source cell type", labelpad=8)
        return g

    elif kind == "chord":
        try:
            from mpl_chord_diagram import chord_diagram
        except ImportError as err:
            raise ImportError("The 'mpl-chord-diagram' package is required for kind='chord'. Please install it with: pip install troutpy[chord]") from err

        matrix_values = mat.values.astype(float)
        fig = plt.figure(figsize=(10, 10))
        chord_diagram(
            matrix_values,
            cell_types,
            directed=True,
            fontsize=9,
            colors=colors,
            **kwargs,
        )
        plt.title(title, fontweight="bold", pad=20)
        plt.grid(False)
        return fig

    else:
        raise ValueError("Invalid kind. Choose 'heatmap' or 'chord'.")
