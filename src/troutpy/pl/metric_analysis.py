import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spatialdata as sd  # Assuming SpatialData is from this package
from adjustText import adjust_text  # Helps prevent overlapping labels
from scipy.stats import mannwhitneyu, shapiro, ttest_ind

from troutpy.pl import get_palette


def top_bottom_probes_of_metric(
    sdata: sd.SpatialData,
    metric: str,
    top_n: int = 10,
    bottom_n: int = 10,
    title: str | None = None,
    save: bool = False,
    figures_path: str = "",
    custom_plot_filename: str | None = None,
    palette: str = "default",
) -> None:
    """
    Creates a horizontal bar plot showing the top and bottom genes based on a
    specified metric. Bars are colored based on whether the gene is a control
    probe or not.

    Parameters
    ----------
    sdata : sd.SpatialData
        SpatialData object that contains 'xrna_metadata' with 'var' DataFrame.

    metric : str
        The metric to sort genes by. Must exist as a column in sdata['xrna_metadata'].var.

    top_n : int, default=10
        Number of top genes to display.

    bottom_n : int, default=10
        Number of bottom genes to display.

    title : str, optional
        Title for the plot.

    save : bool, default=False
        Whether to save the figure.

    figures_path : str, default=""
        Directory path to save the figure.

    custom_plot_filename : str, optional
        Custom filename for saving the plot.

    palette : str, default="default"
        Color palette name. First tries troutpy palettes, then matplotlib colormaps, else fallback to defaults.
    """
    import os

    var_df = sdata["xrna_metadata"].var.copy()

    valid_metrics = list(var_df.columns)
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric '{metric}'. Available metrics in the data are: {', '.join(valid_metrics)}")

    var_df[metric] = var_df[metric].replace([float("inf"), -float("inf")], pd.NA).dropna()

    top_genes = var_df.nlargest(top_n, metric) if top_n > 0 else pd.DataFrame(columns=[metric])
    bottom_genes = var_df.nsmallest(bottom_n, metric) if bottom_n > 0 else pd.DataFrame(columns=[metric])

    plot_df = pd.concat([bottom_genes, top_genes])
    if plot_df.empty:
        raise ValueError("No valid genes found for the selected metric.")

    plot_df = plot_df.sort_values(by=metric)

    # Determine colors
    control_color = "lightgrey"
    gene_color = "black"

    try:
        pal = get_palette(palette)
        control_color = pal[0] if len(pal) > 0 else control_color
        gene_color = pal[1] if len(pal) > 1 else gene_color
    except Exception:
        try:
            cmap = plt.get_cmap(palette)
            control_color = cmap(0.1)
            gene_color = cmap(0.8)
        except Exception:
            pass  # fallback remains default

    colors = plot_df["control_probe"].map({True: control_color, False: gene_color}).tolist()

    # Plotting
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
    sdata: sd.SpatialData,
    x: str,
    y: str,
    non_control_probes: list[str] | None = None,
    label_top_n_x: int = 0,
    label_top_n_y: int = 0,
    label_bottom_n_x: int = 0,
    label_bottom_n_y: int = 0,
    title: str | None = None,
    figures_path: str = "",
    save: bool = False,
    custom_plot_filename: str = None,
    palette: str = "troutpy",
) -> None:
    """
    Creates a scatter plot of two specified metrics from a SpatialData object,
    highlighting control vs. non-control probes, with options to label top genes
    in each metric and save the figure.

    Parameters
    ----------
    sdata : sd.SpatialData
        A SpatialData object containing 'xrna_metadata' with 'var' DataFrame.
    x : str
        Metric for x-axis.
    y : str
        Metric for y-axis.
    non_control_probes : Optional[List[str]]
        List of specific non-control probes to include in the plot.
    label_top_n_x : int
        Number of top genes based on x to label.
    label_top_n_y : int
        Number of top genes based on y to label.
    label_bottom_n_x : int
        Number of bottom genes based on x to label.
    label_bottom_n_y : int
        Number of bottom genes based on y to label.
    title : Optional[str]
        Custom title for the plot.
    figures_path : str
        Directory to save the figure.
    save : bool
        Whether to save the figure.
    custom_plot_filename : str
        Custom filename for the saved plot.
    palette : str
        Color palette for the plot.
    """
    import os

    var_df = sdata["xrna_metadata"].var.copy()

    var_df[x] = var_df[x].replace([np.inf, -np.inf], np.nan)
    var_df[y] = var_df[y].replace([np.inf, -np.inf], np.nan)
    var_df.dropna(subset=[x, y], inplace=True)

    if non_control_probes is not None:
        var_df = var_df[(var_df["control_probe"] is True) | (var_df.index.isin(non_control_probes))]

    # Determine palette
    if palette == "default":
        plot_colors = ["#DC143C", "#1E90FF"]
    else:
        try:
            plot_colors = get_palette(palette)
        except:  # noqa: E722
            try:
                cmap = plt.get_cmap(palette)
                plot_colors = [cmap(0), cmap(1)]
            except ValueError:
                print(f"Palette '{palette}' not recognized. Falling back to default colors.")
                plot_colors = ["#DC143C", "#1E90FF"]

    color_map = {True: plot_colors[0], False: plot_colors[1]}
    var_df["plot_color"] = var_df["control_probe"].map(color_map)

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(var_df[x], var_df[y], c=var_df["plot_color"], edgecolor="black", linewidth=0.5, alpha=0.8)

    label_genes = set()
    if label_top_n_x > 0:
        label_genes.update(var_df.nlargest(label_top_n_x, x).index)
    if label_top_n_y > 0:
        label_genes.update(var_df.nlargest(label_top_n_y, y).index)
    if label_bottom_n_x > 0:
        label_genes.update(var_df.nsmallest(label_bottom_n_x, x).index)
    if label_bottom_n_y > 0:
        label_genes.update(var_df.nsmallest(label_bottom_n_y, y).index)

    texts = []
    for gene in label_genes:
        texts.append(plt.text(var_df.loc[gene, x], var_df.loc[gene, y], gene, fontsize=11, ha="right", fontweight="bold"))

    x_min, x_max = var_df[x].min(), var_df[x].max()
    y_min, y_max = var_df[y].min(), var_df[y].max()

    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1

    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.ylim(y_min - y_padding, y_max + y_padding)

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

    plt.xlabel(x.replace("_", " ").title(), fontsize=12, fontweight="bold")
    plt.ylabel(y.replace("_", " ").title(), fontsize=12, fontweight="bold")

    if title:
        plt.title(title, fontsize=14, fontweight="bold")

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Control Probe", markerfacecolor=plot_colors[0], markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Non-Control Probe", markerfacecolor=plot_colors[1], markeredgecolor="black", markersize=8),
    ]
    plt.legend(handles=legend_elements, loc="best", frameon=True)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    if save:
        os.makedirs(figures_path, exist_ok=True)
        plot_filename = custom_plot_filename or f"{x}_vs_{y}_scatter.pdf"
        plt.savefig(os.path.join(figures_path, plot_filename), bbox_inches="tight")

    plt.show()


def logfoldratio_over_noise(
    sdata: sd.SpatialData,
    control_key="control_probe",
    test_method: str = "auto",
    figures_path: str = "",
    save: bool = False,
    custom_plot_filename: str = None,
    palette: str = "troutpy",
) -> None:
    """
    Creates a violin plot comparing logfoldratio_over_noise values for control
    vs non-control probes, and tests for significance using the specified test.

    Parameters
    ----------
    sdata: SpatialData object that contains 'xrna_metadata' with 'var' DataFrame.
    control_key: The column in var indicating which probes are controls.
    test_method: Statistical test to use. Options:
                 - "t-test" → Welch's t-test
                 - "mannwhitney" → Mann-Whitney U-test
                 - "auto" (default) → Chooses test based on normality test.
    figures_path: Directory to save the figure.
    save: If True, saves the figure as a PDF in the specified path.
    custom_plot_filename: Custom filename for the saved plot.
    palette: Matplotlib palette name or list of two colors. Default is ["lightgrey", "black"].
    """
    import os

    import numpy as np
    import pandas as pd

    var_df = sdata["xrna_metadata"].var.copy()

    var_df["control_probe_cat"] = var_df[control_key].map({True: "Control Probes", False: "Non-Control Probes"})
    var_df["control_probe_cat"] = pd.Categorical(var_df["control_probe_cat"], categories=["Control Probes", "Non-Control Probes"], ordered=True)

    control_values = var_df[var_df["control_probe_cat"] == "Control Probes"]["logfoldratio_over_noise"]
    non_control_values = var_df[var_df["control_probe_cat"] == "Non-Control Probes"]["logfoldratio_over_noise"]

    var_df["logfoldratio_over_noise"] = var_df["logfoldratio_over_noise"].replace([np.inf, -np.inf], np.nan).dropna()

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

    # Determine palette
    if palette == "default":
        plot_colors = ["lightgrey", "black"]
    else:
        try:
            plot_colors = get_palette(palette)
        except:  # noqa: E722
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
    sdata: sd.SpatialData,
    metrics: list[str] | None = None,
    probes_to_plot: list[str] | None = None,
    title: str | None = None,
    cluster_axis: str = "both",
    save: bool = False,
    figures_path: str = "",
    custom_plot_filename: str = None,
    cmap: str = "coolwarm",
) -> None:
    """
    Creates a heatmap or clustered heatmap of gene metrics, with color
    differentiation for control probes and optional saving.
    """
    import os

    import numpy as np

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

    # Determine colormap
    try:
        cmap = get_colormap(cmap)  # your custom function
    except:
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

        # Rotate x-tick labels
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
        else:
            if row_cluster and g.dendrogram_row is not None:
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

        # Rotate x-tick labels
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=7)

        if save:
            os.makedirs(figures_path, exist_ok=True)
            plot_filename = custom_plot_filename or "gene_metric_clustermap.pdf"
            g.savefig(os.path.join(figures_path, plot_filename), bbox_inches="tight")

        plt.show()
