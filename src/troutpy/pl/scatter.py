import os

import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns

import troutpy as tp


def spatial_inout_expression(
    sdata,
    gene,
    layer_cells="table",
    layer_extra="segmentation_free_table",
    spot_size_cells=7,
    spot_size_extra=12,
    extra_cmap: str = "Aquas_contrast",
    cells_cmap: str = "Pinks_contrast",
):
    """Plots intracellular and extracellular expression of a selected gene.
    Args:
        sdata
            SpatialData object containing cell and extra-cellular data.
        gene
            Name of the gene to plot.
        layer_cells
            Key in sdata where *cell* expression is stored.
        layer_extra
            Key in sdata where *extra-cellular* expression is stored.
        spot_size_cells
            Size of the marker for cells.
        spot_size_extra
            Size of the marker for extra-cellular spots.
    """
    if gene not in sdata[layer_cells].var_names or gene not in sdata[layer_extra].var_names:
        raise ValueError(f"Gene '{gene}' not found in both intracellular and extracellular data.")

    # Extract expression data
    cell_expression = sc.AnnData(sdata[layer_cells][:, gene].X, obs=sdata[layer_cells].obs, var=sdata[layer_cells].var.loc[[gene]])
    cell_expression.obsm["spatial"] = sdata[layer_cells].obsm["spatial"]

    extra_expression = sc.AnnData(sdata[layer_extra][:, gene].X, obs=sdata[layer_extra].obs, var=sdata[layer_extra].var.loc[[gene]])
    extra_expression.obsm["spatial"] = sdata[layer_extra].obsm["spatial"]

    # Define colormaps
    cold_palette = tp.pl.get_colormap(cells_cmap)  # For intracellular expression
    warm_palette = tp.pl.get_colormap(extra_cmap)  # For extracellular expression

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot extracellular expression (background)
    sc.pl.spatial(
        extra_expression, color=gene, spot_size=spot_size_extra, legend_loc="none", show=False, ax=ax, cmap=warm_palette, marker="*", vmax="p99.7"
    )

    # Plot intracellular expression (foreground with black outline)
    sc.pl.spatial(
        cell_expression,
        color=gene,
        spot_size=spot_size_cells,
        show=False,
        ax=ax,
        cmap=cold_palette,
        edgecolor="black",
        linewidth=0.05,
        legend_loc="none",
        vmax="p99.7",
    )

    ax.set_title(f"Expression of {gene}")

    plt.tight_layout()
    plt.show()


def diffusion_results(
    sdata,
    x_col="mean_displacement",
    y_col="-log_ks_pval",
    y_logscale=False,
    save: bool = False,
    figures_path: str = "",
    custom_plot_filename: str | None = None,
):
    """
    Plots a scatter plot of genes with their estimated diffusion coefficient (D_estimated) on the x-axis
    and a statistical metric (e.g., KS statistic) on the y-axis. Each point is labeled with the gene name.

    Parameters
    ----------
    sdata : AnnData
        The AnnData object containing the data.
    x_col (str):
        Column to use for x-axis (default: "mean_displacement").
    y_col (str):
        Column to use for y-axis (default: "-log_ks_pval").
    y_logscale (bool):
        Whether to use log scale for y-axis (default: False).
    save : bool, default=False
        Whether to save the figure.
    figures_path : str, default=""
        Directory path to save the figure.
    custom_plot_filename : str, optional
        Custom filename for saving the plot.

    Returns
    -------
    None
    """
    # Extract the diffusion results from sdata
    diffusion_results = sdata["xrna_metadata"].var

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=diffusion_results, x=x_col, y=y_col)

    # Add gene names as labels
    for gene, (x, y) in diffusion_results[[x_col, y_col]].dropna().iterrows():
        plt.text(x, y, gene, fontsize=10, ha="right", va="bottom")

    # Customize plot
    plt.xlabel(x_col.replace("_", " ").capitalize())
    plt.ylabel(y_col.replace("_", " ").capitalize())
    if y_logscale:
        plt.yscale("log")
    plt.title("Diffusion Pattern Analysis of Extracellular RNA")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axhline(y=3, color="r", linestyle="--", label="y=10")

    # Save the plot if requested
    if save:
        os.makedirs(figures_path, exist_ok=True)
        plot_filename = custom_plot_filename or f"diffusion_results_{x_col}_{y_col}.pdf"
        plt.savefig(os.path.join(figures_path, plot_filename), bbox_inches="tight")

    # Show the plot
    plt.show()
