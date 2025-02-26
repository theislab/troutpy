import matplotlib.pyplot as plt
import scanpy as sc

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
