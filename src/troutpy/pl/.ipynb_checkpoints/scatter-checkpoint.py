import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from adjustText import adjust_text
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from shapely.affinity import affine_transform
from shapely.geometry import box
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation

import troutpy as tp
from troutpy.pl import get_palette


def spatial_inout_expression(
    sdata,
    gene,
    layer_cells="table",
    layer_extra="segmentation_free_table",
    spot_size_cells=7,
    spot_size_extra=12,
    extra_cmap: str = "Aquas_contrast",
    cells_cmap: str = "Pinks_contrast",
    title: str = None,
    save: bool = False,
    figures_path: str = "",
    custom_plot_filename: str = None,
):
    """Plots intracellular and extracellular expression of a selected gene.

    Parameters
    ----------
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
        raise ValueError(f"Gene '{gene}' not found in both intracellular and extracellular layers.")

    def make_adata(layer):
        ad = sc.AnnData(X=sdata[layer][:, gene].X.copy(), obs=sdata[layer].obs.copy(), var=sdata[layer].var.loc[[gene]].copy())
        ad.obsm["spatial"] = sdata[layer].obsm["spatial"]
        return ad

    cell_expr = make_adata(layer_cells)
    extra_expr = make_adata(layer_extra)

    # Densify for min/max
    def to_dense(x):
        return x.toarray().flatten() if hasattr(x, "toarray") else x.flatten()

    cell_vals = to_dense(cell_expr.X)
    extra_vals = to_dense(extra_expr.X)

    # --- 2) Choose colormaps ---
    cmap_cells = tp.pl.get_colormap(cells_cmap)
    cmap_extra = tp.pl.get_colormap(extra_cmap)

    # --- 3) Main plotting ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # suppress Scanpy's colorbars via colorbar_loc=None
    sc.pl.spatial(
        extra_expr,
        color=gene,
        spot_size=spot_size_extra,
        cmap=cmap_extra,
        vmax="p99.7",
        legend_loc="none",
        colorbar_loc=None,
        show=False,
        ax=ax,
        marker="*",
    )
    sc.pl.spatial(
        cell_expr,
        color=gene,
        spot_size=spot_size_cells,
        cmap=cmap_cells,
        vmax="p99.7",
        edgecolor="black",
        linewidth=0.05,
        legend_loc="none",
        colorbar_loc=None,
        show=False,
        ax=ax,
    )

    ax.set_title(title or f"Expression of {gene}", fontsize=14, fontweight="bold")

    # --- 4) Build two ScalarMappables ---
    sm_extra = ScalarMappable(cmap=cmap_extra, norm=Normalize(vmin=extra_vals.min(), vmax=np.percentile(extra_vals, 99.7)))
    sm_cells = ScalarMappable(cmap=cmap_cells, norm=Normalize(vmin=cell_vals.min(), vmax=np.percentile(cell_vals, 99.7)))

    # --- 5) Add two small vertical colorbars, stacked on the right ---
    pos = ax.get_position()  # Bbox of main plot: [x0, y0, width, height]
    width = 0.03  # colorbar width as fraction of figure
    pad = 0.07  # horizontal gap between plot and colorbars
    height_frac = 0.48  # each colorbar is 40% of plot height

    # Top colorbar (extracellular)
    cax_ext = fig.add_axes(
        [
            pos.x1 + pad,  # x0
            pos.y0 + pos.height - height_frac * pos.height,  # y0 (top of main minus bar height)
            width,  # width
            height_frac * pos.height,  # height
        ]
    )
    cbar_ext = fig.colorbar(sm_extra, cax=cax_ext)
    cbar_ext.set_label("Extracellular Expression", fontsize=10)

    # Bottom colorbar (intracellular)
    cax_cel = fig.add_axes(
        [
            pos.x1 + pad,
            pos.y0,  # aligned with bottom of main
            width,
            height_frac * pos.height,
        ]
    )
    cbar_cel = fig.colorbar(sm_cells, cax=cax_cel)
    cbar_cel.set_label("Intracellular Expression", fontsize=10)

    plt.tight_layout()

    # --- 6) Optional saving ---
    if save:
        os.makedirs(figures_path, exist_ok=True)
        fname = custom_plot_filename or f"spatial_inout_{gene}.pdf"
        plt.savefig(os.path.join(figures_path, fname), bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def diffusion_results(
    sdata,
    x_col="mean_displacement",
    y_col="-log_ks_pval",
    non_control_probes: list[str] | None = None,
    label_top_n_x: int = 0,
    label_top_n_y: int = 0,
    label_bottom_n_x: int = 0,
    label_bottom_n_y: int = 0,
    y_logscale: bool = False,
    title: str | None = None,
    palette: str = "troutpy",
    save: bool = False,
    figures_path: str = "",
    custom_plot_filename: str | None = None,
) -> None:
    """Styled diffusion scatter plot matching metric_scatter aesthetics."""
    var_df = sdata["xrna_metadata"].var.copy()
    var_df[x_col] = var_df[x_col].replace([np.inf, -np.inf], np.nan)
    var_df[y_col] = var_df[y_col].replace([np.inf, -np.inf], np.nan)
    var_df.dropna(subset=[x_col, y_col], inplace=True)

    # Filter by non-control probes if provided
    if non_control_probes is not None:
        var_df = var_df[(var_df["control_probe"]) | (var_df.index.isin(non_control_probes))]

    # Determine palette
    DEFAULT_COLORS = ["#DC143C", "#1E90FF"]
    if palette == "default":
        plot_colors = DEFAULT_COLORS
    else:
        try:
            plot_colors = get_palette(palette)
        except KeyError:
            # Try matplotlib colormap fallback
            try:
                cmap = plt.get_cmap(palette)
                plot_colors = [cmap(0), cmap(1)]
            except ValueError:
                print(f"Palette '{palette}' not recognized. Falling back to default colors.")
                plot_colors = DEFAULT_COLORS

    color_map = {True: plot_colors[0], False: plot_colors[1]}
    var_df["plot_color"] = var_df["control_probe"].map(color_map)

    # Base scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(var_df[x_col], var_df[y_col], c=var_df["plot_color"], edgecolor="black", linewidth=0.2, alpha=0.8)

    # Select genes to label
    label_genes = set()
    if label_top_n_x > 0:
        label_genes |= set(var_df.nlargest(label_top_n_x, x_col).index)
    if label_top_n_y > 0:
        label_genes |= set(var_df.nlargest(label_top_n_y, y_col).index)
    if label_bottom_n_x > 0:
        label_genes |= set(var_df.nsmallest(label_bottom_n_x, x_col).index)
    if label_bottom_n_y > 0:
        label_genes |= set(var_df.nsmallest(label_bottom_n_y, y_col).index)

    # Add text labels
    texts = []
    for gene in label_genes:
        texts.append(plt.text(var_df.at[gene, x_col], var_df.at[gene, y_col], gene, fontsize=10, fontweight="bold", ha="right"))

    # Adjust text to avoid overlap
    adjust_text(
        texts,
        arrowprops={"arrowstyle": "-", "color": "black", "lw": 1},
        only_move={"points": "xy", "text": "xy"},
        expand_text=(5, 5),
        expand_points=(4, 4),
        force_text=(5, 5),
        force_points=(6, 6),
        force_explode=(4, 4),
    )

    # Axis limits with padding
    x_min, x_max = var_df[x_col].min(), var_df[x_col].max()
    y_min, y_max = var_df[y_col].min(), var_df[y_col].max()
    x_pad = (x_max - x_min) * 0.2
    y_pad = (y_max - y_min) * 0.2

    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(10e-2, y_max + y_pad)

    # Labels and title
    plt.xlabel(x_col.replace("_", " ").title(), fontsize=12, fontweight="bold")
    plt.ylabel(y_col.replace("_", " ").title(), fontsize=12, fontweight="bold")
    if y_logscale:
        plt.yscale("log")
    plt.title(title or "Diffusion Pattern Analysis of Extracellular RNA", fontsize=14, fontweight="bold")

    # Reference line
    plt.axhline(y=3, color="r", linestyle="--", label="y=3")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Control Probe", markerfacecolor=plot_colors[0], markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Non-Control Probe", markerfacecolor=plot_colors[1], markeredgecolor="black", markersize=8),
    ]
    plt.legend(handles=legend_elements, loc="best", frameon=True)

    # Save if requested
    if save:
        os.makedirs(figures_path, exist_ok=True)
        fname = custom_plot_filename or f"diffusion_results_{x_col}_vs_{y_col}.pdf"
        plt.savefig(os.path.join(figures_path, fname), bbox_inches="tight")

    plt.show()


def spatial_transcripts(
    sdata: SpatialData,
    gene_key: str = "gene",
    gene_list: list[str] | None = None,
    color_key: str = "transcript_type",
    shapes_key: str = "cell_boundaries",
    colormap: str = "30colors",
    boundary_linewidth: float = 0.5,
    scatter_size: float = 1.0,
    alpha: float = 0.8,
    use_roi: bool = False,
    roi: tuple[float, float, float, float] = (0, 20000, 0, 20000),
    figsize: tuple[float, float] = (8, 8),
    title: str = "Cell Boundaries + Transcripts",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots a scatter plot of genes with their estimated diffusion coefficient (D_estimated) on the x-axis and a statistical metric (e.g., KS statistic) on the y-axis. Each point is labeled with the gene name.

    Parameters
    ----------
        diffusion_results: pandas.DataFrame
            DataFrame containing diffusion results with gene names as index.
        x_col: str
            Column to use for x-axis (default: "D_estimated").
        y_col: str
            Column to use for y-axis (default: "ks_stat").
    """
    # 1) Filter transcripts by gene_list
    trans_ad = sdata["transcripts"].compute()
    if gene_list is not None:
        trans_ad = trans_ad[trans_ad[gene_key].isin(gene_list)]

    # annotate transcript types
    trans_ad["segstatus"] = trans_ad["overlaps_cell"].astype(str) + "_" + trans_ad["extracellular"].astype(str)
    stat2lab = {"True_False": "cell", "False_True": "extracellular", "False_False": "cell-like"}
    trans_ad["transcript_type"] = trans_ad["segstatus"].map(stat2lab)

    # 2) Prepare color values based on type (categorical or numeric)
    trans_df = trans_ad.copy()
    if pd.api.types.is_numeric_dtype(trans_df[color_key]):
        is_numeric = True
        color_values = trans_df[color_key].astype(float)
        cmap = plt.get_cmap(colormap)
        norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
    else:
        is_numeric = False
        cats = trans_df[color_key].astype("category")
        color_values = cats.cat.codes.values
        n_cats = len(cats.cat.categories)
        try:
            import troutpy

            pal = troutpy.pl.get_palette(colormap, n_cats)
            cmap = ListedColormap(pal)
        except KeyError:
            cmap = plt.get_cmap(colormap, n_cats)

    # 3) Obtain affine transformation and apply to geometries and points
    cb = sdata[shapes_key]
    df = cb.copy()
    trans = get_transformation(cb)

    if hasattr(trans, "matrix") and trans.matrix is not None:
        try:
            matrix = trans.matrix  # expected 3x3 or 2x3 affine
            a, b, xoff = matrix[0]
            d, e, yoff = matrix[1]
            params = [a, b, d, e, xoff, yoff]
            df["geometry"] = df["geometry"].apply(lambda geom: affine_transform(geom, params))
        except KeyError:
            print(f"Transformation could not be applied: {e}")
    else:
        print("No transformation matrix found. Proceeding without transformation.")

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=None)

    # 4) Optionally clip to ROI
    if use_roi:
        xmin, xmax, ymin, ymax = roi
        roi_poly = box(xmin, ymin, xmax, ymax)
        gdf = gdf[gdf.geometry.intersects(roi_poly)]
        mask = (trans_df["x"] >= xmin) & (trans_df["x"] <= xmax) & (trans_df["y"] >= ymin) & (trans_df["y"] <= ymax)
        trans_df = trans_df[mask]
        color_values = color_values[mask]  # mask color values too

    # 5) Plot
    fig, ax = plt.subplots(figsize=figsize)
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=boundary_linewidth)
    scatter = ax.scatter(
        trans_df["x"],
        trans_df["y"],
        c=color_values,
        cmap=cmap,
        s=scatter_size,
        edgecolor="none",
        alpha=alpha,
        norm=norm if is_numeric else None,
    )

    # 6) Legend or colorbar
    if is_numeric:
        fig.colorbar(scatter, ax=ax, label=color_key, fraction=0.046, pad=0.04)
    else:
        for idx, cat in enumerate(cats.cat.categories):
            ax.scatter([], [], color=cmap(idx), label=str(cat), s=20)
        ax.legend(title=color_key, markerscale=1.5, bbox_to_anchor=(1.02, 1), loc="upper left")

    ax.set_aspect("equal")
    ax.set_title(title)
    if use_roi:
        ax.set_xlim(roi[0], roi[1])
        ax.set_ylim(roi[2], roi[3])
    ax.axis("off")
    plt.tight_layout()

    return fig, ax
