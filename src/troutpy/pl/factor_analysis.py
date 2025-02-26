# Function 1- plot latent composition [DEG like plot]
from typing import Any

import anndata
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib import gridspec, rcParams
from spatialdata import SpatialData  # Assuming SpatialData is from scverse

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
    """
    Plot top scoring genes for each factor from NMF/LDA.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing the factorized data.
    layer : str
        The layer name in `sdata` where the factorized results are stored.
    n_genes : int, optional
        Number of top genes to display per factor, by default 20.
    fontsize : int, optional
        Font size for gene names, by default 8.
    ncols : int, optional
        Number of panels per row, by default 4.
    sharey : bool, optional
        Whether to share y-axis scale across subplots, by default True.
    show : bool, optional
        Whether to show the plot, by default True.
    save : str | None, optional
        Path to save the figure, by default None.
    ax : plt.Axes | None, optional
        Axes object to use for plotting, by default None.

    Returns
    -------
    list[plt.Axes] | None
        List of matplotlib Axes objects or None if show=True.
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
):
    """
    Plot ranking of genes using a matrixplot based on factor loadings.

    Parameters
    ----------
    sdata : SpatialData
        The spatial data object containing gene expression information.
    layer : str
        The layer in `sdata` that contains the AnnData object.
    n_genes : int, optional
        Number of top genes to plot per factor, by default 20.
    cmap : str, optional
        Colormap for visualization, by default "bwr".
    vmin, vmax : float, optional
        Color scaling limits.
    show : bool, optional
        Whether to display the plot, by default True.
    save : str, optional
        Path to save the figure, by default None.
    """
    adata: AnnData = sdata[layer]
    gene_loadings = adata.varm["gene_loadings"]  # Shape: genes x factors
    factor_names = [f"Factor {i + 1}" for i in range(gene_loadings.shape[1])]

    # Identify top genes per factor
    top_genes = {}
    for i, factor in enumerate(factor_names):
        top_indices = np.argsort(-np.abs(gene_loadings[:, i]))[:n_genes]
        top_genes[factor] = adata.var_names[top_indices].tolist()

    # Convert to heatmap-friendly format
    gene_list = [gene for gl in top_genes.values() for gene in gl]  # Unique genes across all factors
    factor_matrix = np.array([gene_loadings[adata.var_names.get_indexer(gene_list), i] for i in range(gene_loadings.shape[1])]).T
    print(top_genes)
    # Ensure colormap is valid
    if cmap not in plt.colormaps():
        try:
            cmap = get_colormap(cmap)
        except:
            pass

    # Plot heatmap
    plt.figure(figsize=(len(factor_names) * 0.3, len(gene_list) * 0.3))
    sns.heatmap(factor_matrix, xticklabels=factor_names, yticklabels=gene_list, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={"label": "Gene loading"})
    plt.xlabel("Factors")
    plt.ylabel("Genes")
    plt.title("Top Genes per Factor")

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        return plt.gca()


def factors_in_cells(
    sdata: anndata.AnnData,
    layer: str = "table",
    method: str = "matrixplot",
    celltype_key: str = "cell_type",
    cmap: str = "troutpy",
    **kwargs: dict[str, Any],
):
    """
    Plot factors from a specified layer in a Scanpy AnnData object.

    Parameters
    ----------
    - sdata (anndata.AnnData): The SpatialData object containing the data.
    - layer (str): The layer from which to extract the factors (default: 'table').
    - method (str): The plotting method ('matrixplot', 'dotplot', 'violin').
    - celltype_key (str): The key in `.obs` to group by (default: 'cell_type').
    - title (str): Title for the plot (default: '').
    - **kwargs: Additional keyword arguments passed to the plotting function.
    """
    loadata = sc.AnnData(sdata[layer].obsm["factors_cell_loadings"])
    loadata.var.index = [f"Factor {i + 1}" for i in range(loadata.shape[1])]

    loadata.obs = sdata[layer].obs.copy()
    if cmap not in plt.colormaps():
        try:
            cmap = get_colormap(cmap)
        except:
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
