import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, shapiro, ttest_ind
from spatialdata import SpatialData

from troutpy.pl.colors import get_colormap, get_palette


def top_bottom_probes(
    sdata: SpatialData,
    metric: str,
    top_n: int = 10,
    bottom_n: int = 10,
    title: str | None = None,
    figures_path: str = "",
    save: bool = False,
    custom_plot_filename: str = "",
    palette: str = "Blues",
) -> None:
    """Plot a horizontal bar chart of the top and bottom genes for a metric.

    Bars are colored based on whether the gene is a control probe or not.

    Parameters
    ----------
    sdata
        SpatialData object that contains an ``xrna_metadata`` table with a `var`
        DataFrame.
    metric
        The metric to sort genes by. Must be a column in ``sdata["xrna_metadata"].var``.
    top_n
        Number of top genes to display.
    bottom_n
        Number of bottom genes to display.
    title
        Title for the plot.
    figures_path
        Directory path to save the figure.
    save
        Whether to save the figure.
    custom_plot_filename
        Custom filename for saving the plot.
    palette
        Color palette name, resolved via :func:`troutpy.pl.colors.get_palette`. Falls
        back to a Matplotlib colormap, then to grey/black, if neither is found.

    Returns
    -------
    None
    """
    var_df = sdata["xrna_metadata"].var.copy()

    valid_metrics = list(var_df.columns)
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric '{metric}'. Available metrics in the data are: {', '.join(valid_metrics)}")

    var_df[metric] = var_df[metric].replace([float("inf"), -float("inf")], pd.NA)

    top_genes = var_df.nlargest(top_n, metric) if top_n > 0 else pd.DataFrame(columns=[metric])
    bottom_genes = var_df.nsmallest(bottom_n, metric) if bottom_n > 0 else pd.DataFrame(columns=[metric])

    plot_df = pd.concat([bottom_genes, top_genes])
    if plot_df.empty:
        raise ValueError("No valid genes found for the selected metric.")

    plot_df = plot_df.sort_values(by=metric)

    control_color = "lightgrey"
    gene_color = "black"

    try:
        pal = get_palette(palette)
        control_color = pal[0] if len(pal) > 0 else control_color
        gene_color = pal[1] if len(pal) > 1 else gene_color
    except ValueError:
        try:
            cmap = plt.get_cmap(palette)
            control_color = cmap(0.1)
            gene_color = cmap(0.8)
        except ValueError:
            pass

    colors = plot_df["control_probe"].map({True: control_color, False: gene_color}).tolist()

    plt.figure(figsize=(4, max(5, len(plot_df) * 0.4)))
    plt.barh(
        y=plot_df.index,
        width=plot_df[metric],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    plt.xlabel(metric.replace("_", " ").title(), fontsize=12, fontweight="bold", labelpad=12)
    plt.ylabel("")
    plt.title(title if title else "", fontsize=14, fontweight="bold", pad=12)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    if save:
        os.makedirs(figures_path, exist_ok=True)
        plot_filename = custom_plot_filename or f"top_bottom_probes_{metric}.pdf"
        plt.savefig(os.path.join(figures_path, plot_filename), bbox_inches="tight")

    plt.show()


def metric_scatter(
    sdata: SpatialData,
    x: str,
    y: str,
    size: int = 1,
    non_control_probes: list[str] | None = None,
    label_top_n_x: int = 0,
    label_top_n_y: int = 0,
    label_bottom_n_x: int = 0,
    label_bottom_n_y: int = 0,
    title: str | None = None,
    linewidth: float = 0.5,
    figures_path: str = "",
    save: bool = False,
    custom_plot_filename: str | None = None,
    palette: str = "troutpy",
    group_col: str = "is_control",
    control_label: str = "Control probes",
    gene_label: str = "Gene probes",
    show_control_probes: bool = True,
) -> None:
    """Plot a scatter plot of two metrics from ``sdata["xrna_metadata"].var``.

    Control and non-control probes are colored differently, and the top/bottom `n`
    genes along each axis can be labeled. The figure is saved with Illustrator-editable
    text (Type 42 fonts) and open x/y axes.

    Parameters
    ----------
    sdata
        SpatialData object that contains an ``xrna_metadata`` table with a `var`
        DataFrame.
    x
        Column in ``sdata["xrna_metadata"].var`` to plot on the x-axis.
    y
        Column in ``sdata["xrna_metadata"].var`` to plot on the y-axis.
    size
        Marker size.
    non_control_probes
        If `show_control_probes` is `True`, also include these probes even if
        `group_col` marks them as controls.
    label_top_n_x
        Number of genes with the highest `x` values to label.
    label_top_n_y
        Number of genes with the highest `y` values to label.
    label_bottom_n_x
        Number of genes with the lowest `x` values to label.
    label_bottom_n_y
        Number of genes with the lowest `y` values to label.
    title
        Title for the plot.
    linewidth
        Marker edge line width.
    figures_path
        Directory path to save the figure.
    save
        Whether to save the figure.
    custom_plot_filename
        Custom filename for saving the plot.
    palette
        Two-color palette name, resolved via :func:`troutpy.pl.colors.get_palette`.
        Falls back to a Matplotlib colormap, then to red/blue, if neither is found.
    group_col
        Boolean column in ``sdata["xrna_metadata"].var`` distinguishing control probes
        (`True`) from gene probes (`False`).
    control_label
        Legend label for control probes.
    gene_label
        Legend label for gene probes.
    show_control_probes
        Whether to include control probes in the plot.

    Returns
    -------
    None
    """
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    var_df = sdata["xrna_metadata"].var.copy()

    var_df[x] = var_df[x].replace([np.inf, -np.inf], np.nan)
    var_df[y] = var_df[y].replace([np.inf, -np.inf], np.nan)
    var_df.dropna(subset=[x, y], inplace=True)

    if not show_control_probes:
        var_df = var_df[~var_df[group_col]]

    if show_control_probes and non_control_probes is not None:
        var_df = var_df[var_df[group_col] | var_df.index.isin(non_control_probes)]

    if palette == "default":
        plot_colors = ["#DC143C", "#1E90FF"]
    else:
        try:
            plot_colors = get_palette(palette)
        except ValueError:
            try:
                cmap = plt.get_cmap(palette)
                plot_colors = [cmap(0), cmap(1)]
            except ValueError:
                print(f"Palette '{palette}' not recognized. Falling back to default colors.")
                plot_colors = ["#DC143C", "#1E90FF"]

    color_map = {True: plot_colors[0], False: plot_colors[1]}
    var_df["plot_color"] = var_df[group_col].map(color_map)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        var_df[x],
        var_df[y],
        c=var_df["plot_color"],
        edgecolor="black",
        s=size,
        linewidth=linewidth,
        alpha=0.9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    genes_only_df = var_df[~var_df[group_col]]

    label_genes = set()
    if label_top_n_x > 0:
        label_genes.update(genes_only_df.nlargest(label_top_n_x, x).index)
    if label_top_n_y > 0:
        label_genes.update(genes_only_df.nlargest(label_top_n_y, y).index)
    if label_bottom_n_x > 0:
        label_genes.update(genes_only_df.nsmallest(label_bottom_n_x, x).index)
    if label_bottom_n_y > 0:
        label_genes.update(genes_only_df.nsmallest(label_bottom_n_y, y).index)

    texts = [ax.text(var_df.loc[gene, x], var_df.loc[gene, y], gene, fontsize=11, ha="right", fontweight="normal") for gene in label_genes]

    x_min, x_max = var_df[x].min(), var_df[x].max()
    y_min, y_max = var_df[y].min(), var_df[y].max()
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1

    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    if texts:
        adjust_text(
            texts,
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 1},
            only_move={"points": "xy", "text": "xy"},
            expand_points=(4, 4),
            expand_text=(5, 5),
            force_text=(5, 5),
            force_points=(6, 6),
            force_explode=(4, 4),
        )

    ax.set_xlabel(x.replace("_", " ").title(), fontsize=12, fontweight="normal")
    ax.set_ylabel(y.replace("_", " ").title(), fontsize=12, fontweight="normal")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    if show_control_probes:
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=control_label, markerfacecolor=plot_colors[0], markeredgecolor="black", markersize=8),
            Line2D([0], [0], marker="o", color="w", label=gene_label, markerfacecolor=plot_colors[1], markeredgecolor="black", markersize=8),
        ]
    else:
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=gene_label, markerfacecolor=plot_colors[1], markeredgecolor="black", markersize=8),
        ]
    ax.legend(handles=legend_elements, loc="best", frameon=False)

    ax.tick_params(labelsize=10)

    if save:
        fig_dir = Path(figures_path)
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = custom_plot_filename or f"{x}_vs_{y}_scatter.pdf"
        plt.savefig(fig_dir / plot_filename, bbox_inches="tight")

    plt.show()


def logfoldratio_over_noise(
    sdata: SpatialData,
    control_key: str = "control_probe",
    test_method: str = "auto",
    figures_path: str = "",
    save: bool = False,
    custom_plot_filename: str | None = None,
    palette: str = "troutpy",
) -> None:
    """Plot a violin plot comparing ``logfoldratio_over_noise`` for control vs non-control probes.

    A statistical test is used to compare the two groups, and the result is
    annotated on the plot.

    Parameters
    ----------
    sdata
        SpatialData object that contains an ``xrna_metadata`` table with a `var`
        DataFrame.
    control_key
        Column in ``sdata["xrna_metadata"].var`` indicating which probes are controls.
    test_method
        Statistical test to use: ``"t-test"`` (Welch's t-test), ``"mannwhitney"``
        (Mann-Whitney U test), or ``"auto"`` to choose based on a Shapiro-Wilk
        normality test of both groups.
    figures_path
        Directory path to save the figure.
    save
        Whether to save the figure.
    custom_plot_filename
        Custom filename for saving the plot.
    palette
        Two-color palette name, resolved via :func:`troutpy.pl.colors.get_palette`.
        Falls back to a Matplotlib colormap, then to grey/black, if neither is found.

    Returns
    -------
    None
    """
    var_df = sdata["xrna_metadata"].var.copy()

    var_df["control_probe_cat"] = var_df[control_key].map({True: "Control Probes", False: "Non-Control Probes"})
    var_df["control_probe_cat"] = pd.Categorical(var_df["control_probe_cat"], categories=["Control Probes", "Non-Control Probes"], ordered=True)

    control_values = var_df[var_df["control_probe_cat"] == "Control Probes"]["logfoldratio_over_noise"]
    non_control_values = var_df[var_df["control_probe_cat"] == "Non-Control Probes"]["logfoldratio_over_noise"]

    var_df["logfoldratio_over_noise"] = var_df["logfoldratio_over_noise"].replace([np.inf, -np.inf], np.nan)

    if test_method == "auto":
        control_normal = shapiro(control_values).pvalue > 0.05
        non_control_normal = shapiro(non_control_values).pvalue > 0.05
        if control_normal and non_control_normal:
            test_method = "t-test"
        else:
            test_method = "mannwhitney"

    if test_method == "t-test":
        stat, p_value = ttest_ind(control_values, non_control_values, equal_var=False)
        test_used = "t-test"
    elif test_method == "mannwhitney":
        stat, p_value = mannwhitneyu(control_values, non_control_values, alternative="two-sided")
        test_used = "Mann-Whitney U"
    else:
        raise ValueError("Invalid test_method. Choose from 't-test', 'mannwhitney' or 'auto'.")

    if palette == "default":
        plot_colors = ["lightgrey", "black"]
    else:
        try:
            plot_colors = get_palette(palette)
        except ValueError:
            try:
                cmap = plt.get_cmap(palette)
                plot_colors = [cmap(i) for i in range(2)]
            except ValueError:
                print(f"Palette '{palette}' not recognized. Falling back to default colors.")
                plot_colors = ["lightgrey", "black"]

    plt.figure(figsize=(4, 5))
    sns.violinplot(
        x="control_probe_cat",
        y="logfoldratio_over_noise",
        data=var_df,
        palette=plot_colors,
        inner=None,
        order=["Control Probes", "Non-Control Probes"],
    )

    y_max = max(var_df["logfoldratio_over_noise"]) * 1.5
    plt.ylim(min(var_df["logfoldratio_over_noise"]) * 1.2, y_max * 1.35)

    x1, x2 = 0, 1
    plt.plot([x1, x1, x2, x2], [y_max, y_max * 1.07, y_max * 1.07, y_max], lw=1.5, color="black")

    sig_text = f"{test_used}, p = {p_value:.3e}" if p_value >= 0.001 else f"{test_used}, p < 0.001"
    plt.text((x1 + x2) / 2, y_max * 1.14, sig_text, ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.xlabel("")
    plt.ylabel("Log Fold Ratio Over Noise", fontsize=12, fontweight="bold")
    plt.gca().xaxis.set_tick_params(pad=10)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    if save:
        os.makedirs(figures_path, exist_ok=True)
        plot_filename = custom_plot_filename or "logfoldratio_over_noise_violin.pdf"
        plt.savefig(os.path.join(figures_path, plot_filename))
    plt.show()


def gene_metric_heatmap(
    sdata: SpatialData,
    metrics: list[str] | None = None,
    probes_to_plot: list[str] | None = None,
    title: str | None = None,
    cluster_axis: str = "both",
    save: bool = False,
    figures_path: str = "",
    custom_plot_filename: str | None = None,
    cmap: str = "coolwarm",
) -> None:
    """Plot a heatmap or clustered heatmap of per-gene metrics.

    Control probes are highlighted in the y-tick labels (when shown) by coloring
    them grey instead of black.

    Parameters
    ----------
    sdata
        SpatialData object that contains an ``xrna_metadata`` table with a `var`
        DataFrame.
    metrics
        Columns in ``sdata["xrna_metadata"].var`` to include as heatmap columns. If
        `None`, defaults to a fixed set of quality-control metrics.
    probes_to_plot
        If provided, restrict the heatmap to these probes/genes, and show y-tick
        labels if there are at most 20 of them.
    title
        Title for the plot.
    cluster_axis
        Determines clustering: ``"none"`` (no clustering), ``"x"`` (cluster columns
        only), ``"y"`` (cluster rows only), or ``"both"`` (cluster rows and columns).
    save
        Whether to save the figure.
    figures_path
        Directory path to save the figure.
    custom_plot_filename
        Custom filename for saving the plot.
    cmap
        Colormap for the heatmap, resolved via :func:`troutpy.pl.colors.get_colormap`
        if not a built-in Matplotlib colormap.

    Returns
    -------
    None
    """
    var_df = sdata["xrna_metadata"].var.copy()

    valid_metrics = {
        "logfoldratio_extracellular",
        "extracellular_proportion",
        "moran_I",
        "proportion_of_colocalized",
        "intracellular_proportion",
    }

    if metrics is None:
        metrics = list(valid_metrics)

    valid_axes = ["none", "x", "y", "both"]
    cluster_axis = cluster_axis.lower()
    if cluster_axis not in valid_axes:
        raise ValueError(f"Invalid cluster_axis: {cluster_axis}. Must be one of {', '.join(valid_axes)}.")

    if probes_to_plot is not None:
        var_df = var_df[var_df.index.isin(probes_to_plot)]

    heatmap_data = var_df[metrics].replace([np.inf, -np.inf], np.nan)
    heatmap_data.dropna(how="all", axis=0, inplace=True)
    heatmap_data.dropna(how="all", axis=1, inplace=True)

    if heatmap_data.empty:
        print("No finite data available to plot.")
        return

    num_genes = heatmap_data.shape[0]
    show_ylabels = (probes_to_plot is not None) and (num_genes <= 20)

    ytick_colors = ["gray" if (gene in var_df.index and var_df.loc[gene, "control_probe"]) else "black" for gene in heatmap_data.index]

    try:
        cmap = get_colormap(cmap)
    except ValueError:
        try:
            cmap = plt.get_cmap(cmap)
        except ValueError:
            print(f"Palette '{cmap}' not recognized. Falling back to 'coolwarm'.")
            cmap = "coolwarm"

    if cluster_axis == "none":
        plt.figure(figsize=(len(heatmap_data.columns) * 0.4, max(5, num_genes * 0.2)))

        ax = sns.heatmap(
            heatmap_data,
            cmap=cmap,
            linewidths=0.5,
            linecolor="gray",
            annot=False,
            fmt=".2f",
            cbar=True,
            yticklabels=heatmap_data.index if show_ylabels else False,
        )

        if show_ylabels:
            for tick_label, color in zip(ax.get_yticklabels(), ytick_colors, strict=False):
                tick_label.set_color(color)

        ax.set_xlabel("", fontsize=12, fontweight="bold")
        ax.set_ylabel("", fontsize=12, fontweight="bold")
        ax.set_title(title if title else "", fontsize=14, fontweight="bold", pad=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

        if save:
            os.makedirs(figures_path, exist_ok=True)
            plot_filename = custom_plot_filename or "gene_metric_heatmap.pdf"
            plt.savefig(os.path.join(figures_path, plot_filename), bbox_inches="tight")

        plt.show()

    else:
        row_cluster = cluster_axis in ["y", "both"]
        col_cluster = cluster_axis in ["x", "both"]

        g = sns.clustermap(
            data=heatmap_data,
            cmap=cmap,
            linewidths=0.5,
            linecolor="gray",
            annot=False,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            dendrogram_ratio=(0.15, 0.08),
            cbar_pos=(1.3, 0.4, 0.08, 0.3),
        )

        g.fig.set_size_inches((len(heatmap_data.columns) * 0.4, max(5, heatmap_data.shape[0] * 0.1)))

        if not show_ylabels:
            g.ax_heatmap.set_yticklabels([])
        elif row_cluster and g.dendrogram_row is not None:
            row_order = g.dendrogram_row.reordered_ind
            new_index = heatmap_data.index[row_order]
            for tick_label, gene in zip(g.ax_heatmap.get_yticklabels(), new_index, strict=False):
                color = "gray" if (gene in var_df.index and var_df.loc[gene, "control_probe"]) else "black"
                tick_label.set_color(color)
        else:
            for tick_label, color in zip(g.ax_heatmap.get_yticklabels(), ytick_colors, strict=False):
                tick_label.set_color(color)

        g.ax_heatmap.set_xlabel("", fontsize=7, fontweight="bold")
        g.ax_heatmap.set_ylabel("", fontsize=7, fontweight="bold")
        g.ax_heatmap.set_title(title if title else "", fontsize=7, fontweight="bold", pad=15 if cluster_axis == "y" else 35)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=7)

        if save:
            os.makedirs(figures_path, exist_ok=True)
            plot_filename = custom_plot_filename or "gene_metric_clustermap.pdf"
            g.savefig(os.path.join(figures_path, plot_filename), bbox_inches="tight")

        plt.show()
