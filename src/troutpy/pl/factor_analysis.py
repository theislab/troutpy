from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib import gridspec, rcParams
from spatialdata import SpatialData

from troutpy.pl.colors import get_colormap


def rank_factor_genes_loadings(
    sdata: SpatialData,
    layer: str,
    n_genes: int = 20,
    fontsize: int = 15,
    ncols: int = 4,
    sharey: bool = True,
    show: bool = True,
    save: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> list[plt.Axes] | None:
    """Plot top scoring genes for each factor from NMF/LDA.

    Parameters
    ----------
    sdata
        The SpatialData object containing the factorized data.
    layer
        The layer name in `sdata` where the factorized results are stored.
    n_genes
        Number of top genes to display per factor.
    fontsize
        Font size for gene names.
    ncols
        Number of panels per row.
    sharey
        Whether to share the y-axis scale across subplots.
    show
        Whether to show the plot.
    save
        Path to save the figure. If `None`, the figure is not saved.
    ax
        Existing axes to plot into. If `None`, a new figure and grid of axes are created.
    kwargs
        Additional keyword arguments (currently unused).

    Returns
    -------
    If `show=True`, `None`. Otherwise, the list of :class:`matplotlib.axes.Axes` used for plotting.
    """
    adata = sdata[layer]
    gene_loadings = adata.varm["gene_loadings"].T  # Factors x Genes
    factor_names = [f"Factor {i + 1}" for i in range(gene_loadings.shape[0])]

    n_panels_x = min(ncols, len(factor_names))
    n_panels_y = int(np.ceil(len(factor_names) / n_panels_x))

    if ax is None:
        fig = plt.figure(figsize=(n_panels_x * rcParams["figure.figsize"][0], n_panels_y * rcParams["figure.figsize"][1]))
        gs = gridspec.GridSpec(n_panels_y, n_panels_x, fig, wspace=0.1, hspace=0.1)
    else:
        fig = ax.get_figure()
        gs = ax.get_subplotspec().subgridspec(n_panels_y, n_panels_x)

    axs = []
    ymin, ymax = np.inf, -np.inf

    for i, factor in enumerate(factor_names):
        top_genes_idx = np.argsort(gene_loadings[i])[::-1][:n_genes]
        top_gene_names = adata.var_names[top_genes_idx]
        top_gene_scores = gene_loadings[i, top_genes_idx]

        if sharey:
            ymin, ymax = min(ymin, np.min(top_gene_scores)), max(ymax, np.max(top_gene_scores))
            axs.append(fig.add_subplot(gs[i], sharey=axs[0] if axs else None))
        else:
            axs.append(fig.add_subplot(gs[i]))

        axs[-1].set_xlim(-0.9, n_genes - 0.1)
        axs[-1].set_title(factor)
        axs[-1].set_xticks([])

        for j, (gene, score) in enumerate(zip(top_gene_names, top_gene_scores, strict=False)):
            axs[-1].text(j, score, gene, rotation=90, fontsize=fontsize, verticalalignment="bottom", horizontalalignment="center")

        if i % n_panels_x == 0:
            axs[-1].set_ylabel("Score")

    if sharey:
        axs[0].set_ylim(ymin, ymax + 0.3 * (ymax - ymin))

    if save:
        fig.savefig(save, bbox_inches="tight")

    if show:
        plt.show()
        return None

    return axs


def rank_factor_genes_loadings_matrixplot(
    sdata: SpatialData,
    layer: str,
    n_genes: int = 5,
    cmap: str = "Pinks",
    vmin: float | None = None,
    vmax: float | None = None,
    show: bool = True,
    save: str | None = None,
    figsize: tuple | None = None,
):
    """Plot ranking of genes using a matrixplot based on factor loadings.

    Parameters
    ----------
    sdata
        The spatial data object containing gene expression information.
    layer
        The layer in `sdata` that contains the AnnData object.
    n_genes
        Number of top genes to plot per factor.
    cmap
        Colormap for visualization. If not a built-in Matplotlib colormap, it is
        resolved via :func:`troutpy.pl.colors.get_colormap`.
    vmin
        Lower color scaling limit.
    vmax
        Upper color scaling limit.
    show
        Whether to display the plot.
    save
        Path to save the figure. If `None`, the figure is not saved.
    figsize
        Figure size. If `None`, it is derived from the number of factors and genes.

    Returns
    -------
    If `show=True`, `None`. Otherwise, the current :class:`matplotlib.axes.Axes`.
    """
    adata: AnnData = sdata[layer]
    gene_loadings = adata.varm["gene_loadings"]  # Shape: genes x factors
    factor_names = [f"Factor {i + 1}" for i in range(gene_loadings.shape[1])]

    top_genes = {}
    for i, factor in enumerate(factor_names):
        top_indices = np.argsort(-np.abs(gene_loadings[:, i]))[:n_genes]
        top_genes[factor] = adata.var_names[top_indices].tolist()

    gene_list = [gene for gl in top_genes.values() for gene in gl]
    factor_matrix = np.array([gene_loadings[adata.var_names.get_indexer(gene_list), i] for i in range(gene_loadings.shape[1])]).T

    if cmap not in plt.colormaps():
        try:
            cmap = get_colormap(cmap)
        except ValueError:
            pass

    if figsize is None:
        plt.figure(figsize=(len(factor_names) * 0.3, len(gene_list) * 0.1))
    else:
        plt.figure(figsize=figsize)
    sns.heatmap(factor_matrix, xticklabels=factor_names, yticklabels=gene_list, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={"label": "Gene loading"})
    plt.xlabel("Factors")
    plt.ylabel("Genes")
    plt.title("Top Genes per Factor")
    plt.grid(False)

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        return plt.gca()


def factors_in_cells(
    sdata: SpatialData,
    layer: str = "table",
    method: str = "matrixplot",
    celltype_key: str = "cell_type",
    cmap: str = "troutpy",
    **kwargs: dict[str, Any],
):
    """Plot per-cell factor loadings from a specified table, grouped by cell type.

    Parameters
    ----------
    sdata
        The SpatialData object containing the data.
    layer
        The table in `sdata` from which to extract the factor loadings
        (``.obsm["factors_cell_loadings"]``).
    method
        The plotting method: ``"matrixplot"``, ``"dotplot"``, ``"violin"``, or ``"heatmap"``.
    celltype_key
        The key in `.obs` to group by.
    cmap
        Colormap for visualization. If not a built-in Matplotlib colormap, it is
        resolved via :func:`troutpy.pl.colors.get_colormap`.
    kwargs
        Additional keyword arguments passed to the underlying `scanpy.pl` plotting function.

    Returns
    -------
    None
    """
    loadata = sc.AnnData(sdata[layer].obsm["factors_cell_loadings"])
    loadata.var.index = [f"Factor {i + 1}" for i in range(loadata.shape[1])]

    loadata.obs = sdata[layer].obs.copy()
    if cmap not in plt.colormaps():
        try:
            cmap = get_colormap(cmap)
        except ValueError:
            pass
    if method == "matrixplot":
        sc.pl.matrixplot(loadata, loadata.var.index, groupby=celltype_key, cmap=cmap, **kwargs)
    elif method == "dotplot":
        sc.pl.dotplot(loadata, loadata.var.index, groupby=celltype_key, cmap=cmap, **kwargs)
    elif method == "violin":
        sc.pl.violin(loadata, loadata.var.index, groupby=celltype_key, cmap=cmap, **kwargs)
    elif method == "heatmap":
        sc.pl.heatmap(loadata, loadata.var.index, groupby=celltype_key, cmap=cmap, **kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'matrixplot', 'dotplot', or 'violin'.")
