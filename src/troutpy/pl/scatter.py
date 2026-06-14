import os

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from adjustText import adjust_text
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from shapely.affinity import affine_transform
from shapely.geometry import box
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation

from troutpy.pl.colors import get_colormap, get_palette


def spatial_inout_expression(
    sdata: SpatialData,
    gene: str,
    layer_cells: str = "table",
    layer_extra: str = "segmentation_free_table",
    spot_size_cells: float = 7,
    spot_size_extra: float = 12,
    extra_cmap: str = "Aquas_contrast",
    cells_cmap: str = "Pinks_contrast",
    title: str | None = None,
    save: bool = False,
    figures_path: str = "",
    custom_plot_filename: str | None = None,
) -> None:
    """Plot intracellular and extracellular expression of a selected gene side by side.

    Parameters
    ----------
    sdata
        SpatialData object containing cell and extracellular data.
    gene
        Name of the gene to plot.
    layer_cells
        Key in `sdata` where *cell* expression is stored.
    layer_extra
        Key in `sdata` where *extracellular* expression is stored.
    spot_size_cells
        Size of the marker for cells.
    spot_size_extra
        Size of the marker for extracellular spots.
    extra_cmap
        Colormap for extracellular expression, resolved via :func:`troutpy.pl.colors.get_colormap`.
    cells_cmap
        Colormap for cellular expression, resolved via :func:`troutpy.pl.colors.get_colormap`.
    title
        Custom title for the plot. If `None`, defaults to ``f"Expression of {gene}"``.
    save
        Whether to save the figure to `figures_path` instead of showing it.
    figures_path
        Directory to save the figure in, if `save` is `True`.
    custom_plot_filename
        Custom filename for the saved figure. If `None`, defaults to ``f"spatial_inout_{gene}.pdf"``.

    Returns
    -------
    None
    """
    if gene not in sdata[layer_cells].var_names or gene not in sdata[layer_extra].var_names:
        raise ValueError(f"Gene '{gene}' not found in both intracellular and extracellular layers.")

    def make_adata(layer):
        ad = sc.AnnData(X=sdata[layer][:, gene].X.copy(), obs=sdata[layer].obs.copy(), var=sdata[layer].var.loc[[gene]].copy())
        ad.obsm["spatial"] = sdata[layer].obsm["spatial"]
        return ad

    cell_expr = make_adata(layer_cells)
    extra_expr = make_adata(layer_extra)

    def to_dense(x):
        return x.toarray().flatten() if hasattr(x, "toarray") else x.flatten()

    cell_vals = to_dense(cell_expr.X)
    extra_vals = to_dense(extra_expr.X)

    cmap_cells = get_colormap(cells_cmap)
    cmap_extra = get_colormap(extra_cmap)

    fig, ax = plt.subplots(figsize=(8, 8))

    # colorbar_loc=None suppresses Scanpy's default colorbars so the custom ones below can be used instead
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

    sm_extra = ScalarMappable(cmap=cmap_extra, norm=Normalize(vmin=extra_vals.min(), vmax=np.percentile(extra_vals, 99.7)))
    sm_cells = ScalarMappable(cmap=cmap_cells, norm=Normalize(vmin=cell_vals.min(), vmax=np.percentile(cell_vals, 99.7)))

    # Add two small vertical colorbars, stacked on the right of the main plot
    pos = ax.get_position()
    width = 0.03
    pad = 0.07
    height_frac = 0.48

    cax_ext = fig.add_axes(
        [
            pos.x1 + pad,
            pos.y0 + pos.height - height_frac * pos.height,
            width,
            height_frac * pos.height,
        ]
    )
    cbar_ext = fig.colorbar(sm_extra, cax=cax_ext)
    cbar_ext.set_label("Extracellular Expression", fontsize=10)

    cax_cel = fig.add_axes(
        [
            pos.x1 + pad,
            pos.y0,
            width,
            height_frac * pos.height,
        ]
    )
    cbar_cel = fig.colorbar(sm_cells, cax=cax_cel)
    cbar_cel.set_label("Intracellular Expression", fontsize=10)

    plt.tight_layout()

    if save:
        os.makedirs(figures_path, exist_ok=True)
        fname = custom_plot_filename or f"spatial_inout_{gene}.pdf"
        plt.savefig(os.path.join(figures_path, fname), bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def diffusion_results(
    sdata: SpatialData,
    x_col: str = "mean_displacement",
    y_col: str = "-log_ks_pval",
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
    """Scatter plot of two diffusion metrics per probe, colored by control-probe status.

    Parameters
    ----------
    sdata
        SpatialData object containing an ``xrna_metadata`` table whose `.var` holds
        the diffusion metrics and a boolean ``control_probe`` column.
    x_col
        Column in ``sdata["xrna_metadata"].var`` to plot on the x-axis.
    y_col
        Column in ``sdata["xrna_metadata"].var`` to plot on the y-axis.
    non_control_probes
        If provided, probes are kept only if they are flagged as control probes or
        their name is in this list.
    label_top_n_x
        Number of probes with the highest `x_col` values to label.
    label_top_n_y
        Number of probes with the highest `y_col` values to label.
    label_bottom_n_x
        Number of probes with the lowest `x_col` values to label.
    label_bottom_n_y
        Number of probes with the lowest `y_col` values to label.
    y_logscale
        Whether to plot the y-axis on a log scale.
    title
        Custom title for the plot.
    palette
        Two-color palette name, resolved via :func:`troutpy.pl.colors.get_palette`, used for
        control vs. non-control probes. ``"default"`` uses a hardcoded crimson/blue pair.
    save
        Whether to save the figure to `figures_path` instead of showing it.
    figures_path
        Directory to save the figure in, if `save` is `True`.
    custom_plot_filename
        Custom filename for the saved figure. If `None`, defaults to
        ``f"diffusion_results_{x_col}_vs_{y_col}.pdf"``.

    Returns
    -------
    None
    """
    var_df = sdata["xrna_metadata"].var.copy()
    var_df[x_col] = var_df[x_col].replace([np.inf, -np.inf], np.nan)
    var_df[y_col] = var_df[y_col].replace([np.inf, -np.inf], np.nan)
    var_df.dropna(subset=[x_col, y_col], inplace=True)

    if non_control_probes is not None:
        var_df = var_df[(var_df["control_probe"]) | (var_df.index.isin(non_control_probes))]

    default_colors = ["#DC143C", "#1E90FF"]
    if palette == "default":
        plot_colors = default_colors
    else:
        try:
            plot_colors = get_palette(palette)
        except ValueError:
            try:
                cmap = plt.get_cmap(palette)
                plot_colors = [cmap(0), cmap(1)]
            except ValueError:
                print(f"Palette '{palette}' not recognized. Falling back to default colors.")
                plot_colors = default_colors

    color_map = {True: plot_colors[0], False: plot_colors[1]}
    var_df["plot_color"] = var_df["control_probe"].map(color_map)

    plt.figure(figsize=(8, 6))
    plt.scatter(var_df[x_col], var_df[y_col], c=var_df["plot_color"], edgecolor="black", linewidth=0.2, alpha=0.8)

    label_genes = set()
    if label_top_n_x > 0:
        label_genes |= set(var_df.nlargest(label_top_n_x, x_col).index)
    if label_top_n_y > 0:
        label_genes |= set(var_df.nlargest(label_top_n_y, y_col).index)
    if label_bottom_n_x > 0:
        label_genes |= set(var_df.nsmallest(label_bottom_n_x, x_col).index)
    if label_bottom_n_y > 0:
        label_genes |= set(var_df.nsmallest(label_bottom_n_y, y_col).index)

    texts = [plt.text(var_df.at[gene, x_col], var_df.at[gene, y_col], gene, fontsize=10, fontweight="bold", ha="right") for gene in label_genes]
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

    x_min, x_max = var_df[x_col].min(), var_df[x_col].max()
    y_min, y_max = var_df[y_col].min(), var_df[y_col].max()
    x_pad = (x_max - x_min) * 0.2
    y_pad = (y_max - y_min) * 0.2

    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(10e-2, y_max + y_pad)

    plt.xlabel(x_col.replace("_", " ").title(), fontsize=12, fontweight="bold")
    plt.ylabel(y_col.replace("_", " ").title(), fontsize=12, fontweight="bold")
    if y_logscale:
        plt.yscale("log")
    plt.title(title or "Diffusion Pattern Analysis of Extracellular RNA", fontsize=14, fontweight="bold")

    plt.axhline(y=3, color="r", linestyle="--", label="y=3")
    plt.grid(True, linestyle="--", alpha=0.5)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Control Probe", markerfacecolor=plot_colors[0], markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Non-Control Probe", markerfacecolor=plot_colors[1], markeredgecolor="black", markersize=8),
    ]
    plt.legend(handles=legend_elements, loc="best", frameon=True)

    if save:
        os.makedirs(figures_path, exist_ok=True)
        fname = custom_plot_filename or f"diffusion_results_{x_col}_vs_{y_col}.pdf"
        plt.savefig(os.path.join(figures_path, fname), bbox_inches="tight")

    plt.show()


def spatial_transcripts(
    sdata: SpatialData,
    gene_key: str = "gene",
    gene_list: list[str] | None = None,
    color_key: str = "point_source_cellular",
    shapes_key: str = "cell_boundaries",
    table_key: str = "table",
    structure_table_key: str = "structure_table",
    source_table_key: str = "source_score",
    colormap: str = "30colors",
    continuous_colormap: str = "viridis",
    boundary_linewidth: float = 0.5,
    boundary_zorder: int = 4,
    scatter_size: float = 1.0,
    alpha: float = 0.8,
    use_roi: bool = False,
    roi: tuple[float, float, float, float] = (0, 20000, 0, 20000),
    figsize: tuple[float, float] = (8, 8),
    title: str = "Cell Boundaries + Transcripts",
    vmin: float | None = None,
    vmax: float | None = None,
    rasterize: bool = True,
    missing_val_sentinel: int | str | float = -1,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot spatial transcripts colored by a grouping variable, overlaid on cell boundaries.

    For ``color_key="point_source_cellular"`` (the default), transcripts are colored by
    whether they overlap a cell (``"cellular transcripts"``) or, for extracellular
    transcripts, by their predicted source category from ``sdata[source_table_key]``.
    For any other `color_key`, the column is taken directly from the transcripts table
    if present, otherwise looked up from ``sdata[source_table_key]``,
    ``sdata[structure_table_key]``, or ``sdata[table_key]`` (in that order) via
    ``structure_id``/``cell_id`` mappings.

    Parameters
    ----------
    sdata
        SpatialData object containing the ``"transcripts"`` points element, the cell
        boundary shapes, and the tables referenced by `table_key`, `structure_table_key`,
        and `source_table_key`.
    gene_key
        Column in the transcripts table containing gene names.
    gene_list
        If provided, restrict the plot to transcripts of these genes.
    color_key
        Column (or derived category) used to color transcripts. See above for the
        special handling of ``"point_source_cellular"``.
    shapes_key
        Key of the cell boundary shapes element in `sdata`.
    table_key
        Key of the cell-level table in `sdata`, used as a fallback source for `color_key`.
    structure_table_key
        Key of the segmentation-free structure table in `sdata`, used as a fallback
        source for `color_key` via ``structure_id``.
    source_table_key
        Key of the source-score table in `sdata`, used as a fallback source for `color_key`.
    colormap
        Colormap for categorical `color_key` values, resolved via
        :func:`troutpy.pl.colors.get_palette`, or a continuous colormap name or
        :class:`~matplotlib.colors.ListedColormap` if `color_key` is numeric.
    continuous_colormap
        Fallback continuous colormap used when `colormap` is not a valid matplotlib
        colormap and `color_key` is numeric.
    boundary_linewidth
        Line width of the cell boundary outlines.
    boundary_zorder
        Z-order of the cell boundary outlines.
    scatter_size
        Marker size for background (``"Unmapped"``) transcripts; foreground transcripts
        are plotted at 1.5x this size.
    alpha
        Transparency of background (``"Unmapped"``) transcripts.
    use_roi
        Whether to restrict the plot to the region defined by `roi`.
    roi
        Region of interest as ``(xmin, xmax, ymin, ymax)``, used if `use_roi` is `True`.
    figsize
        Size of the figure.
    title
        Title of the plot.
    vmin
        Minimum value for color scaling, if `color_key` is numeric.
    vmax
        Maximum value for color scaling, if `color_key` is numeric.
    rasterize
        Whether to rasterize the scatter points (recommended for large point counts).
    missing_val_sentinel
        Value in `color_key` to treat as missing, in addition to actual `NaN` values.

    Returns
    -------
    The figure and axes containing the plot.
    """
    # Determine the minimal set of columns to pull from the (lazy, dask-backed) transcripts table
    target_col = "point_source" if color_key == "point_source_cellular" else color_key
    required_cols = ["x", "y", gene_key, "overlaps_cell"]

    transcripts_lazy = sdata.points["transcripts"]
    is_direct_column = target_col in transcripts_lazy.columns

    if is_direct_column:
        required_cols.append(target_col)

    # Tables that may provide target_col via an id-column join with the transcripts table
    search_configs = [
        (source_table_key, None),
        (structure_table_key, "structure_id"),
        (table_key, "cell_id"),
    ]

    for t_key, trans_id_col in search_configs:
        if t_key in sdata.tables and target_col in sdata[t_key].obs.columns:
            if trans_id_col is not None:
                required_cols.append(trans_id_col)
            break

    required_cols = list(set(required_cols))
    available_cols = [c for c in required_cols if c in transcripts_lazy.columns]
    transcripts_lazy_subset = transcripts_lazy[available_cols]

    if gene_list is not None:
        transcripts_lazy_subset = transcripts_lazy_subset[transcripts_lazy_subset[gene_key].isin(gene_list)]

    trans_ad = transcripts_lazy_subset.compute().copy()

    if color_key not in trans_ad.columns:
        trans_ad[color_key] = np.nan

    # Map color_key (or, for "point_source_cellular", the underlying "point_source")
    # from the relevant table's obs onto the transcripts, via the id columns in search_configs
    if color_key == "point_source_cellular":
        if target_col not in trans_ad.columns:
            trans_ad[target_col] = np.nan

            for t_key, trans_id_col in search_configs:
                if t_key in sdata.tables and target_col in sdata[t_key].obs.columns:
                    table_obs = sdata[t_key].obs
                    if trans_id_col is None:
                        table_series = table_obs[target_col]
                        table_series.index = table_series.index.astype(str)
                        trans_ad[target_col] = trans_ad.index.astype(str).map(table_series)
                    else:
                        if "instance_id" in table_obs.columns:
                            map_source = table_obs.set_index("instance_id")[target_col]
                        elif "cell_id" in table_obs.columns:
                            map_source = table_obs.set_index("cell_id")[target_col]
                        else:
                            map_source = table_obs[target_col]

                        map_source.index = map_source.index.astype(str)
                        trans_ad[target_col] = trans_ad[trans_id_col].astype(str).map(map_source)

                    if not trans_ad[target_col].isna().all():
                        break

        is_intra_bool = trans_ad["overlaps_cell"].astype(str).str.lower() == "true"
        trans_ad["point_source_cellular"] = np.where(is_intra_bool, "cellular transcripts", trans_ad[target_col].astype(str))
        trans_ad.loc[trans_ad[target_col].isna(), "point_source_cellular"] = np.nan

    elif is_direct_column is False:
        for t_key, trans_id_col in search_configs:
            if t_key in sdata.tables and color_key in sdata[t_key].obs.columns:
                table_obs = sdata[t_key].obs
                if trans_id_col is None:
                    table_series = table_obs[color_key]
                    table_series.index = table_series.index.astype(str)
                    trans_ad[color_key] = trans_ad.index.astype(str).map(table_series)
                else:
                    if "instance_id" in table_obs.columns:
                        map_source = table_obs.set_index("instance_id")[color_key]
                    elif "cell_id" in table_obs.columns:
                        map_source = table_obs.set_index("cell_id")[color_key]
                    else:
                        map_source = table_obs[color_key]

                    map_source.index = map_source.index.astype(str)
                    trans_ad[color_key] = trans_ad[trans_id_col].astype(str).map(map_source)

                if not trans_ad[color_key].isna().all():
                    break

    # Treat string "nan" and the missing-value sentinel as NaN
    trans_ad[color_key] = trans_ad[color_key].replace("nan", np.nan)
    if missing_val_sentinel is not None:
        sentinel_str = str(missing_val_sentinel)
        trans_ad[color_key] = trans_ad[color_key].replace([missing_val_sentinel, sentinel_str], np.nan)

    is_nan_mask = trans_ad[color_key].isna()
    trans_ad[color_key] = trans_ad[color_key].fillna("Unmapped")
    is_bg = trans_ad[color_key] == "Unmapped"
    trans_df = trans_ad.copy()

    # Cell boundary geometries, transformed into the same coordinate space as the transcripts
    cb = sdata[shapes_key]
    df = cb.copy()
    try:
        trans = get_transformation(cb)
        if hasattr(trans, "matrix") and trans.matrix is not None:
            m = trans.matrix
            params = [m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[0, 2], m[1, 2]]
            df["geometry"] = df["geometry"].apply(lambda g: affine_transform(g, params))
    except Exception:  # noqa: BLE001 - best-effort: plot untransformed shapes if transform lookup/application fails
        pass
    gdf = gpd.GeoDataFrame(df, geometry="geometry")

    if use_roi:
        xmin, xmax, ymin, ymax = roi
        roi_poly = box(xmin, ymin, xmax, ymax)
        gdf = gdf[gdf.geometry.intersects(roi_poly)]
        mask = (trans_df["x"] >= xmin) & (trans_df["x"] <= xmax) & (trans_df["y"] >= ymin) & (trans_df["y"] <= ymax)
        trans_df = trans_df[mask]
        is_bg = is_bg[mask]
        is_nan_mask = is_nan_mask[mask]

    fg_df = trans_df[~is_bg].copy()
    unmapped_df = trans_df[is_bg & ~is_nan_mask].copy()

    cmap, norm, is_numeric, cats = None, None, False, None

    if not fg_df.empty:
        numeric_check = pd.to_numeric(fg_df[color_key], errors="coerce")
        if not numeric_check.isna().all() and pd.api.types.is_numeric_dtype(numeric_check):
            is_numeric = True
            vals = fg_df[color_key].astype(float)

            try:
                cmap = plt.get_cmap(colormap)
            except ValueError:
                cmap = plt.get_cmap(continuous_colormap)

            plot_vmin = vmin if vmin is not None else vals.min()
            plot_vmax = vmax if vmax is not None else (1.0 if vals.max() <= 1.0 else vals.max())
            norm = Normalize(vmin=plot_vmin, vmax=plot_vmax)
        else:
            is_numeric = False

            if color_key == "point_source_cellular":
                defined_order = ["cellular transcripts", "cell_like_source", "other", "high_density_source"]
                present_cats = [c for c in defined_order if c in fg_df[color_key].unique()]
                extra_cats = [c for c in fg_df[color_key].unique() if c not in defined_order]
                final_order = present_cats + extra_cats
                fg_df[color_key] = pd.Categorical(fg_df[color_key], categories=final_order, ordered=True)
            else:
                fg_df[color_key] = fg_df[color_key].astype("category")

            cats = fg_df[color_key].cat.categories

            if isinstance(colormap, ListedColormap):
                cmap = colormap
            else:
                try:
                    pal = get_palette(colormap, len(cats))
                    cmap = ListedColormap(pal)
                except ValueError:
                    cmap = plt.get_cmap(colormap if colormap in plt.colormaps() else "tab20", len(cats))

    fig, ax = plt.subplots(figsize=figsize)

    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=boundary_linewidth, alpha=0.4, zorder=boundary_zorder)

    if not unmapped_df.empty:
        ax.scatter(unmapped_df["x"], unmapped_df["y"], c="#A0A0A0", s=scatter_size, edgecolor="none", alpha=alpha, zorder=2, rasterized=rasterize)

    if not fg_df.empty:
        c_vals = fg_df[color_key] if is_numeric else fg_df[color_key].cat.codes
        scatter = ax.scatter(
            fg_df["x"],
            fg_df["y"],
            c=c_vals,
            cmap=cmap,
            norm=norm if is_numeric else None,
            s=scatter_size * 1.5,
            edgecolor="none",
            alpha=1.0,
            zorder=3,
            rasterized=rasterize,
        )
        if is_numeric:
            fig.colorbar(scatter, ax=ax, label=color_key, fraction=0.046, pad=0.04)
        else:
            if len(cats) <= 30:
                for idx, cat in enumerate(cats):
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


def point_metric_comparison(
    sdata: SpatialData,
    points_key: str = "transcripts",
    x_col: str = "cosine_similarity",
    y_col: str = "cosine_sim_stainings",
    hue_col: str = "enrichment_class",
    gene_key: str = "gene",
    gene_list: list[str] | None = None,
    subsample_fraction: float | None = None,
    table_key: str = "table",
    structure_table_key: str = "structure_table",
    source_table_key: str = "source_score",
    palette_name: str = "30colors",
    continuous_palette: str = "viridis",
    s: float = 2.0,
    alpha: float = 0.3,
    figsize: tuple[float, float] = (7, 6),
    title: str | None = None,
    show_consensus: bool = True,
    seed: int = 42,
) -> sns.JointGrid:
    """Plot a joint scatter/KDE comparison of two point-level metrics, colored by a third variable.

    Builds a :class:`seaborn.JointGrid` comparing `x_col` and `y_col` from
    ``sdata[points_key]``, colored or grouped by `hue_col`. Any of `x_col`, `y_col`,
    or `hue_col` not already present in the points table is looked up from
    ``sdata[source_table_key]``, ``sdata[structure_table_key]``, or ``sdata[table_key]``
    (in that order) via ``structure_id``/``cell_id`` mappings.

    Parameters
    ----------
    sdata
        SpatialData object containing the points element and the tables referenced by
        `table_key`, `structure_table_key`, and `source_table_key`.
    points_key
        Key of the points element in `sdata`.
    x_col
        Column to plot on the x-axis.
    y_col
        Column to plot on the y-axis.
    hue_col
        Column used to color/group points. If numeric, points are colored on a
        continuous scale; otherwise treated as categorical.
    gene_key
        Column in the points table containing gene names.
    gene_list
        If provided, restrict the plot to points of these genes.
    subsample_fraction
        If provided and less than 1, randomly subsample the points to this fraction
        before plotting.
    table_key
        Key of the cell-level table in `sdata`, used as a fallback source for
        `x_col`/`y_col`/`hue_col`.
    structure_table_key
        Key of the segmentation-free structure table in `sdata`, used as a fallback
        source for `x_col`/`y_col`/`hue_col` via ``structure_id``.
    source_table_key
        Key of the source-score table in `sdata`, used as a fallback source for
        `x_col`/`y_col`/`hue_col`.
    palette_name
        Palette for categorical `hue_col` values, resolved via
        :func:`troutpy.pl.colors.get_palette`, or colormap name for numeric `hue_col`.
    continuous_palette
        Fallback continuous colormap used when `palette_name` is not a valid matplotlib
        colormap and `hue_col` is numeric.
    s
        Marker size for the scatter plot.
    alpha
        Transparency of the scatter points.
    figsize
        Size of the figure.
    title
        Custom title for the plot. If `None`, a title is generated from `x_col`, `y_col`,
        and `hue_col`.
    show_consensus
        If `True` and both axes range over roughly ``[0, 1]``, draw a diagonal
        consensus (``y = x``) reference line.
    seed
        Random seed used for subsampling, if `subsample_fraction` is set.

    Returns
    -------
    The joint grid containing the plot.
    """
    if points_key not in sdata:
        raise KeyError(f"The key '{points_key}' was not found in the provided SpatialData object.")

    points_element = sdata[points_key]

    potential_needed_cols = {x_col, y_col, hue_col, gene_key, "overlaps_cell", "point_source", "structure_id", "cell_id"}
    load_cols = [col for col in potential_needed_cols if col in points_element.columns]
    trans_df = points_element[load_cols].compute()

    if gene_list is not None:
        if gene_key not in trans_df.columns:
            raise KeyError(f"Gene key '{gene_key}' not found in {points_key} columns.")
        trans_df = trans_df[trans_df[gene_key].isin(gene_list)]

    if trans_df.empty:
        raise ValueError("Points dataframe is empty after filtering for specified genes.")

    if subsample_fraction is not None:
        if not (0.0 < subsample_fraction <= 1.0):
            raise ValueError("subsample_fraction must be a float between 0.0 and 1.0")
        if subsample_fraction < 1.0:
            trans_df = trans_df.sample(frac=subsample_fraction, random_state=seed)

    # Map any of x_col/y_col/hue_col not already on the points table from the relevant table's obs
    for target_col in [x_col, y_col, hue_col]:
        if target_col not in trans_df.columns:
            search_configs = [
                (source_table_key, None),
                (structure_table_key, "structure_id"),
                (table_key, "cell_id"),
            ]

            for t_key, trans_id_col in search_configs:
                if t_key in sdata.tables and target_col in sdata[t_key].obs.columns:
                    table_obs = sdata[t_key].obs

                    if trans_id_col and trans_id_col not in trans_df.columns:
                        continue

                    if trans_id_col is None:
                        table_series = table_obs[target_col]
                        table_series.index = table_series.index.astype(str)
                        trans_df[target_col] = trans_df.index.astype(str).map(table_series)
                    else:
                        if "instance_id" in table_obs.columns:
                            map_source = table_obs.set_index("instance_id")[target_col]
                        elif "cell_id" in table_obs.columns:
                            map_source = table_obs.set_index("cell_id")[target_col]
                        else:
                            map_source = table_obs[target_col]

                        map_source.index = map_source.index.astype(str)
                        trans_df[target_col] = trans_df[trans_id_col].astype(str).map(map_source)

                    if not trans_df[target_col].isna().all():
                        break

    for col in [x_col, y_col, hue_col]:
        if col in trans_df.columns:
            trans_df[col] = trans_df[col].replace("nan", np.nan)

    trans_df = trans_df.dropna(subset=[x_col, y_col, hue_col])
    if trans_df.empty:
        raise ValueError(f"No matching data left after processing columns: {x_col}, {y_col}, {hue_col}")

    is_numeric = pd.api.types.is_numeric_dtype(trans_df[hue_col])
    palette = None
    norm = None

    if is_numeric:
        try:
            palette = plt.get_cmap(palette_name)
        except ValueError:
            palette = plt.get_cmap(continuous_palette)

        vals = trans_df[hue_col].astype(float)
        vmax = 1.0 if vals.max() <= 1.0 else vals.max()
        norm = Normalize(vmin=vals.min(), vmax=vmax)
    else:
        binary_targets = ["overlaps_cell", "extracellular", "extracellular_stainings"]
        if hue_col in binary_targets:
            palette = {True: "#d62728", False: "#1f77b4", "True": "#d62728", "False": "#1f77b4"}
        else:
            if hue_col == "point_source_cellular" and "overlaps_cell" in trans_df.columns:
                is_intra_bool = trans_df["overlaps_cell"].astype(str).str.lower() == "true"
                alt_source = "point_source" if "point_source" in trans_df.columns else hue_col
                trans_df["point_source_cellular"] = np.where(is_intra_bool, "cellular transcripts", trans_df[alt_source].astype(str))

            trans_df[hue_col] = trans_df[hue_col].astype("category")
            cats = trans_df[hue_col].cat.categories

            try:
                pal_colors = get_palette(palette_name, len(cats))
                palette = list(pal_colors)
            except ValueError:
                if palette_name in plt.colormaps():
                    cmap_obj = plt.get_cmap(palette_name)
                    palette = [cmap_obj(i / max(1, len(cats) - 1)) for i in range(len(cats))]
                else:
                    palette = sns.color_palette("deep" if len(cats) <= 10 else "husl", len(cats))

    grid_palette = palette if not is_numeric else None

    g = sns.JointGrid(data=trans_df, x=x_col, y=y_col, hue=hue_col if not is_numeric else None, palette=grid_palette)
    g.fig.set_size_inches(figsize)

    if is_numeric:
        scatter = g.ax_joint.scatter(trans_df[x_col], trans_df[y_col], c=trans_df[hue_col], cmap=palette, norm=norm, s=s, alpha=alpha, edgecolor=None)
        g.fig.colorbar(scatter, ax=g.ax_joint, label=hue_col.replace("_", " ").title(), fraction=0.046, pad=0.04)
    else:
        g.plot_joint(sns.scatterplot, s=s, alpha=alpha, edgecolor=None)

    marginal_hue = None if is_numeric else hue_col
    g.plot_marginals(sns.kdeplot, fill=True, alpha=0.2, common_norm=False, hue=marginal_hue, palette=grid_palette)

    if show_consensus:
        x_min, x_max = trans_df[x_col].min(), trans_df[x_col].max()
        y_min, y_max = trans_df[y_col].min(), trans_df[y_col].max()
        if (0 <= x_min <= 0.15 and 0.85 <= x_max <= 1.05) and (0 <= y_min <= 0.15 and 0.85 <= y_max <= 1.05):
            g.ax_joint.plot([0, 1], [0, 1], color="#2B2B2B", linestyle="--", alpha=0.5, label="Consensus", zorder=1)

    clean_x = x_col.replace("_", " ").title()
    clean_y = y_col.replace("_", " ").title()
    clean_hue = hue_col.replace("_", " ").title()

    g.ax_joint.set_xlabel(clean_x, fontweight="medium")
    g.ax_joint.set_ylabel(clean_y, fontweight="medium")

    if title is None:
        sub_info = f" (Downsampled to {subsample_fraction * 100:.1f}%)" if subsample_fraction else ""
        title = f"{clean_x} vs {clean_y}{sub_info}\nStratified by: {clean_hue}"
    g.fig.suptitle(title, fontweight="bold", y=1.02, fontsize=11)

    if not is_numeric:
        handles, labels = g.ax_joint.get_legend_handles_labels()
        if handles:
            unique_labels = dict(zip(labels, handles, strict=False))
            g.ax_joint.legend(unique_labels.values(), unique_labels.keys(), title=clean_hue, bbox_to_anchor=(1.22, 1), loc="upper left", frameon=True)
    elif g.ax_joint.get_legend() is not None:
        g.ax_joint.get_legend().remove()

    return g


def spatial_transcripts_source(
    sdata: SpatialData,
    cell_color_key: str = "leiden",
    extra_color_key: str = "leiden",
    gene_list: str | list[str] | None = None,
    gene_key: str = "gene",
    separate_compartments: bool = False,
    use_roi: bool = False,
    roi: tuple[float, float, float, float] | None = None,
    padding: float = 0,
    point_size: float = 0.1,
    min_score: float = 0.1,
    show_edges: bool = False,
    edge_alpha: float = 0.1,
    fig_bg_color: str = "white",
    ambient_color: str = "grey",
    palette: str = "tab20",
    shapes_key: str = "cell_boundaries",
    boundary_linewidth: float = 0.5,
    boundary_zorder: int = 4,
) -> tuple[plt.Figure, plt.Axes] | tuple[None, None]:
    """Plot transcripts colored by intracellular and extracellular source cell type, overlaid on cell boundaries.

    Intracellular ("cell body") transcripts are colored by `cell_color_key` from
    ``sdata["table"].obs``. Extracellular transcripts with an ``assignment_score`` of
    at least `min_score` (in ``sdata["source_score"].obs``) are considered "halo"
    transcripts and colored by `extra_color_key`, taken either directly from
    ``sdata["source_score"].obs`` or, if not present there, via each transcript's
    predicted parent cell in ``sdata["table"].obs``. Remaining "ambient" transcripts
    are plotted in `ambient_color`.

    Parameters
    ----------
    sdata
        SpatialData object containing the ``"table"`` and ``"source_score"`` tables,
        the ``"transcripts"`` points element, and the cell boundary shapes.
    cell_color_key
        Column in ``sdata["table"].obs`` used to color intracellular transcripts.
    extra_color_key
        Column used to color extracellular ("halo") transcripts, looked up in
        ``sdata["source_score"].obs`` or, as a fallback, in ``sdata["table"].obs``.
    gene_list
        Specific gene(s) to isolate. If provided, only transcripts of these genes are plotted.
    gene_key
        Column in the transcripts table containing gene names.
    separate_compartments
        If `True`, intracellular and extracellular transcripts get separate legend
        entries/colors (``"Cell: X"`` / ``"Extra: X"``); if `False`, they share a
        single ``"Cluster: X"``-style legend.
    use_roi
        Whether to restrict the plot to the region defined by `roi`.
    roi
        Region of interest as ``(xmin, xmax, ymin, ymax)``, used if `use_roi` is `True`.
    padding
        If `roi` is not used, crop this many units from each side of the full
        transcript extent.
    point_size
        Base marker size for transcripts; foreground (labeled) transcripts are plotted
        at twice this size.
    min_score
        Minimum ``assignment_score`` for an extracellular transcript to be considered
        assigned ("halo") to a cell.
    show_edges
        Whether to draw edges connecting halo transcripts to their assigned cell's centroid.
    edge_alpha
        Transparency of the edges, if `show_edges` is `True`.
    fig_bg_color
        Background color of the figure and axes.
    ambient_color
        Color for unassigned ("ambient") transcripts.
    palette
        Matplotlib colormap name used to assign colors to categories.
    shapes_key
        Key of the cell boundary shapes element in `sdata`.
    boundary_linewidth
        Line width of the cell boundary outlines.
    boundary_zorder
        Z-order of the cell boundary outlines.

    Returns
    -------
    The figure and axes containing the plot, or ``(None, None)`` if `gene_list` is
    provided and no transcripts match.
    """
    res_obs = sdata.tables["source_score"].obs
    cells = sdata["table"]
    all_transcripts = sdata.points["transcripts"].compute()

    all_transcripts.index = all_transcripts.index.astype(int)
    res_obs.index = res_obs.index.astype(int)

    if gene_list is not None:
        if isinstance(gene_list, str):
            gene_list = [gene_list]
        all_transcripts = all_transcripts[all_transcripts[gene_key].isin(gene_list)].copy()

        if all_transcripts.empty:
            print(f"No transcripts found for gene(s): {gene_list}")
            return None, None

    cell_map = dict(zip(cells.obs["cell_id"].astype(str).str.strip(), cells.obs[cell_color_key].astype(str), strict=False))

    if extra_color_key in res_obs.columns:
        extra_map = dict(zip(res_obs.index.astype(str).str.strip(), res_obs[extra_color_key].astype(str), strict=False))
        direct_extra_mapping = True
    elif extra_color_key in cells.obs.columns:
        parent_to_extra_trait_map = dict(zip(cells.obs["cell_id"].astype(str).str.strip(), cells.obs[extra_color_key].astype(str), strict=False))
        direct_extra_mapping = False
    else:
        raise KeyError(f"'{extra_color_key}' not found in source_score or table metadata.")

    centroid_map = dict(zip(cells.obs["cell_id"].astype(str).str.strip(), cells.obsm["spatial"], strict=False))

    if use_roi and roi is not None:
        x_min, x_max, y_min, y_max = roi
        df_roi = all_transcripts[(all_transcripts.x.between(x_min, x_max)) & (all_transcripts.y.between(y_min, y_max))].copy()
    elif padding > 0:
        x_min = all_transcripts["x"].min() + padding
        x_max = all_transcripts["x"].max() - padding
        y_min = all_transcripts["y"].min() + padding
        y_max = all_transcripts["y"].max() - padding
        df_roi = all_transcripts[(all_transcripts.x.between(x_min, x_max)) & (all_transcripts.y.between(y_min, y_max))].copy()
    else:
        x_min, x_max = all_transcripts["x"].min(), all_transcripts["x"].max()
        y_min, y_max = all_transcripts["y"].min(), all_transcripts["y"].max()
        df_roi = all_transcripts.copy()

    # Assign each transcript to a "cell body" (intracellular) and/or "halo" (extracellular) label
    df_roi = df_roi.join(res_obs[["predicted_parent", "assignment_score"]], how="left")

    df_roi["parent_id"] = df_roi["predicted_parent"].astype(str).str.strip()
    df_roi["clean_cell_id"] = df_roi["cell_id"].astype(str).str.strip()
    df_roi["transcript_id_str"] = df_roi.index.astype(str).str.strip()

    if direct_extra_mapping:
        is_halo = (df_roi["assignment_score"] >= min_score) & (df_roi["transcript_id_str"].isin(extra_map))
    else:
        is_halo = (df_roi["assignment_score"] >= min_score) & (df_roi["parent_id"].isin(parent_to_extra_trait_map))

    is_body = (df_roi["overlaps_cell"].astype(str).str.lower() == "true") & (df_roi["clean_cell_id"].isin(cell_map))

    df_roi["cell_label"] = np.nan
    df_roi["extra_label"] = np.nan

    df_roi.loc[is_body, "cell_label"] = df_roi.loc[is_body, "clean_cell_id"].map(cell_map)
    if direct_extra_mapping:
        df_roi.loc[is_halo, "extra_label"] = df_roi.loc[is_halo, "transcript_id_str"].map(extra_map)
    else:
        df_roi.loc[is_halo, "extra_label"] = df_roi.loc[is_halo, "parent_id"].map(parent_to_extra_trait_map)

    if not separate_compartments:
        df_roi["unified_label"] = df_roi["cell_label"].fillna(df_roi["extra_label"])

    def extract_uns_colors(uns_dict, key):
        for possible_key in [f"{key}_colors", "colors"]:
            if possible_key in uns_dict:
                return list(uns_dict[possible_key])
        return None

    if not separate_compartments:
        # Shared categories: a single legend entry per cluster, covering both compartments
        unique_labels = sorted(df_roi["unified_label"].dropna().unique())
        uns_colors = None
        if hasattr(cells, "uns") and cell_color_key in cells.uns:
            uns_colors = extract_uns_colors(cells.uns, cell_color_key)

        if uns_colors and len(uns_colors) >= len(unique_labels):
            cat_order = list(cells.obs[cell_color_key].astype("category").cat.categories)
            unified_colors = {lbl: uns_colors[cat_order.index(lbl)] for lbl in unique_labels if lbl in cat_order}
        else:
            cmap = plt.get_cmap(palette)
            unified_colors = {lbl: cmap(i / max(1, len(unique_labels) - 1)) for i, lbl in enumerate(unique_labels)}
    else:
        # Separate legend entries/colors for intracellular ("Cell: X") and extracellular ("Extra: X")
        unique_cells = set(df_roi["cell_label"].dropna().unique())
        unique_extra = set(df_roi["extra_label"].dropna().unique())
        matching_elements = (unique_cells == unique_extra) and (cell_color_key == extra_color_key)

        if matching_elements:
            uns_colors = None
            if hasattr(cells, "uns") and cell_color_key in cells.uns:
                uns_colors = extract_uns_colors(cells.uns, cell_color_key)

            all_labels = sorted(unique_cells.union(unique_extra))
            if uns_colors and len(uns_colors) >= len(all_labels):
                cat_order = list(cells.obs[cell_color_key].astype("category").cat.categories)
                colors = {lbl: uns_colors[cat_order.index(lbl)] for lbl in all_labels if lbl in cat_order}
            else:
                cmap = plt.get_cmap(palette)
                colors = {lbl: cmap(i / max(1, len(all_labels) - 1)) for i, lbl in enumerate(all_labels)}
            cell_colors = extra_colors = colors
        else:
            sorted_cells = sorted(unique_cells)
            cmap_cells = plt.get_cmap(palette)
            cell_colors = {lbl: cmap_cells(i / max(1, len(sorted_cells) - 1)) for i, lbl in enumerate(sorted_cells)}

            sorted_extra = sorted(unique_extra)
            cmap_extra = plt.get_cmap("Set3" if palette != "Set3" else "tab20b")
            extra_colors = {lbl: cmap_extra(i / max(1, len(sorted_extra) - 1)) for i, lbl in enumerate(sorted_extra)}

    # Cell boundary geometries, transformed into the same coordinate space as the transcripts
    cb = sdata[shapes_key]
    df_shapes = cb.copy()
    try:
        trans = get_transformation(cb)
        if hasattr(trans, "matrix") and trans.matrix is not None:
            m = trans.matrix
            params = [m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[0, 2], m[1, 2]]
            df_shapes["geometry"] = df_shapes["geometry"].apply(lambda g: affine_transform(g, params))
    except Exception:  # noqa: BLE001 - best-effort: plot untransformed shapes if transform lookup/application fails
        pass

    gdf = gpd.GeoDataFrame(df_shapes, geometry="geometry")
    if (use_roi and roi is not None) or padding > 0:
        roi_poly = box(x_min, y_min, x_max, y_max)
        gdf = gdf[gdf.geometry.intersects(roi_poly)]

    text_color = "white" if fig_bg_color in ["black", "#000000", "#252525"] else "black"
    fig, ax = plt.subplots(figsize=(25, 25), facecolor=fig_bg_color)
    ax.set_facecolor(fig_bg_color)

    ambient_mask = ~(is_halo | is_body)
    ax.scatter(df_roi.loc[ambient_mask, "x"], df_roi.loc[ambient_mask, "y"], c=ambient_color, s=point_size, alpha=0.1, edgecolors="none", zorder=0)

    if show_edges:
        halo_data = df_roi[is_halo]
        lines = [[(row["x"], row["y"]), centroid_map.get(row["parent_id"])] for _, row in halo_data.iterrows() if row["parent_id"] in centroid_map]
        lc = LineCollection(lines, colors=text_color, linewidths=0.2, alpha=edge_alpha, zorder=1)
        ax.add_collection(lc)

    if not separate_compartments:
        for lbl in unique_labels:
            subset = df_roi[df_roi["unified_label"] == lbl]
            ax.scatter(subset.x, subset.y, color=unified_colors[lbl], s=point_size * 2, label=f"Cluster {lbl}", edgecolors="none", zorder=2)
    else:
        for lbl in sorted(unique_cells):
            subset = df_roi[df_roi["cell_label"] == lbl]
            ax.scatter(subset.x, subset.y, color=cell_colors[lbl], s=point_size * 2, label=f"Cell: {lbl}", edgecolors="none", zorder=2)
        for lbl in sorted(unique_extra):
            subset = df_roi[df_roi["extra_label"] == lbl]
            lbl_legend = f"Extra: {lbl}" if not matching_elements else f"Extra Halo: {lbl}"
            ax.scatter(subset.x, subset.y, color=extra_colors[lbl], s=point_size * 2, label=lbl_legend, edgecolors="none", zorder=3)

    border_color = "black" if fig_bg_color == "white" else "white"
    gdf.plot(ax=ax, facecolor="none", edgecolor=border_color, linewidth=boundary_linewidth, alpha=0.4, zorder=boundary_zorder)

    title_suffix = f" | Subsetting for {', '.join(gene_list)}" if gene_list is not None else ""
    plt.title(f"Global Connectivity | Boundaries + Expression Channels{title_suffix}", color=text_color, fontsize=16)

    box_pos = ax.get_position()
    ax.set_position([box_pos.x0, box_pos.y0, box_pos.width * 0.75, box_pos.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), facecolor=fig_bg_color, edgecolor=text_color, labelcolor=text_color, markerscale=5, ncol=2)

    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.axis("off")

    return fig, ax


def urna_vs_source_score(
    sdata: SpatialData,
    y_var: str = "extracellular_proportion",
    y_label: str | None = None,
    title: str | None = None,
    grid: bool = False,
    saving_path: str | None = None,
    lfc_thresh: float = 2,
    x_thresh: float | None = None,
    y_thresh: float | None = None,
    top_x_genes: int | None = None,
    bottom_x_genes: int | None = None,
    top_y_genes: int | None = None,
    bottom_y_genes: int | None = None,
    color: str = "#1ad6b4",
    point_size: float = 2,
    figsize: tuple[float, float] = (6, 6),
    xlim: tuple[float, float] | None = (0.5, 1.1),
    ylim: tuple[float, float] | None = None,
    filename: str | None = None,
) -> None:
    """Plot a per-gene Y-axis variable from ``xrna_metadata.var`` against the mean source score.

    Genes are first filtered by ``logfoldchange_over_noise`` and can optionally be
    labeled based on coordinate thresholds and/or rank (top/bottom genes along
    either axis).

    Parameters
    ----------
    sdata
        Spatial data object containing ``source_score`` and ``xrna_metadata`` tables.
    y_var
        Column name from ``sdata["xrna_metadata"].var`` to plot on the y-axis.
    y_label
        Custom label for the y-axis. If `None`, defaults to ``"uRNA proportion"``
        for ``"extracellular_proportion"``, otherwise defaults to `y_var`.
    title
        Title to display at the top of the plot.
    grid
        Whether to show background grid lines.
    saving_path
        Directory to save the plot in. If `None`, the plot is not saved.
    lfc_thresh
        Minimum ``logfoldchange_over_noise`` for a gene to be included.
    x_thresh
        If set, genes with a mean source score below this value are labeled.
    y_thresh
        If set, genes with `y_var` above this value are labeled.
    top_x_genes
        Number of genes with the highest mean source score to label.
    bottom_x_genes
        Number of genes with the lowest mean source score to label.
    top_y_genes
        Number of genes with the highest `y_var` to label.
    bottom_y_genes
        Number of genes with the lowest `y_var` to label.
    color
        Color of the scatter points.
    point_size
        Size of the scatter points.
    figsize
        Size of the figure.
    xlim
        x-axis limits.
    ylim
        y-axis limits.
    filename
        Filename to save the plot as. Defaults to ``f"scatter_{y_var}_vs_source_score.pdf"``.

    Returns
    -------
    None
    """
    x_matrix = sdata["source_score"].X
    if hasattr(x_matrix, "toarray"):
        sdata["source_score"].obs["sum_source_score"] = np.asarray(x_matrix.sum(axis=1)).flatten()
    else:
        sdata["source_score"].obs["sum_source_score"] = np.sum(x_matrix, axis=1)

    source_score_by_gene = sdata["source_score"].obs.groupby("gene").mean("sum_source_score")
    count_by_gene = sdata["source_score"].obs.groupby("gene").count()
    source_score_by_gene["total_counts"] = count_by_gene["distance_to_source"]

    combined_urna_metadata = pd.concat([source_score_by_gene, sdata["xrna_metadata"].var], axis=1)

    if y_var not in combined_urna_metadata.columns:
        raise ValueError(f"'{y_var}' not found in the combined metadata columns.")

    df_filtered = combined_urna_metadata[combined_urna_metadata["logfoldchange_over_noise"] > lfc_thresh].copy()

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    sc.set_figure_params(
        scanpy=True,
        dpi=200,
        dpi_save=200,
        frameon=True,
        vector_friendly=True,
        fontsize=14,
        figsize=(10, 4),
        color_map=None,
        format="pdf",
        facecolor=None,
        transparent=False,
        ipython_format="retina",
    )

    plt.figure(figsize=figsize)
    plt.grid(grid)

    plt.scatter(df_filtered["sum_source_score"], df_filtered[y_var], s=point_size, c=color)
    plt.xlabel("Mean. source score")

    if y_label is None:
        y_label = "uRNA proportion" if y_var == "extracellular_proportion" else y_var
    plt.ylabel(y_label)

    if title:
        plt.title(title, pad=12)

    genes_to_label = set()

    if x_thresh is not None or y_thresh is not None:
        thresh_df = df_filtered.copy()
        if x_thresh is not None:
            thresh_df = thresh_df[thresh_df["sum_source_score"] < x_thresh]
        if y_thresh is not None:
            thresh_df = thresh_df[thresh_df[y_var] > y_thresh]
        genes_to_label.update(thresh_df.index)

    if top_x_genes:
        genes_to_label.update(df_filtered.nlargest(top_x_genes, "sum_source_score").index)
    if bottom_x_genes:
        genes_to_label.update(df_filtered.nsmallest(bottom_x_genes, "sum_source_score").index)
    if top_y_genes:
        genes_to_label.update(df_filtered.nlargest(top_y_genes, y_var).index)
    if bottom_y_genes:
        genes_to_label.update(df_filtered.nsmallest(bottom_y_genes, y_var).index)

    for gene in genes_to_label:
        if gene in df_filtered.index:
            row = df_filtered.loc[gene]
            plt.annotate(gene, (row["sum_source_score"], row[y_var]), fontsize=10, xytext=(3, 3), textcoords="offset points")

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()

    if saving_path:
        os.makedirs(saving_path, exist_ok=True)
        if filename is None:
            filename = f"scatter_{y_var}_vs_source_score.pdf"
        plt.savefig(os.path.join(saving_path, filename))

    plt.show()
