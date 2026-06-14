import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from spatialdata import SpatialData


def global_distribution_from_source(
    sdata: SpatialData,
    cluster_key: str = "kmeans_distribution",
    gene_key: str = "gene",
    distance_key: str = "distance",
    n_bins: int = 20,
    how: str = "full",
) -> None:
    """Plot gene distance distributions, optionally grouped by cluster.

    Depending on `how`, this shows either (1) a clustermap of all genes' distance
    distributions sorted by cluster, with row colors (``"full"``), (2) a collapsed
    clustermap showing the mean distribution per cluster (``"collapsed"``), or (3) a
    line plot of the mean distribution for each cluster (``"lineplot"``).

    Parameters
    ----------
    sdata
        Spatial data object containing ``source_score`` and ``xrna_metadata`` tables.
    cluster_key
        Column in ``sdata["xrna_metadata"].var`` containing cluster assignments.
    gene_key
        Column that contains gene names.
    distance_key
        Column that contains the distances.
    n_bins
        Number of bins for the histogram representation.
    how
        ``"full"`` shows a clustermap of all genes sorted by cluster with row colors,
        ``"collapsed"`` shows a clustermap of the mean distribution per cluster, and
        ``"lineplot"`` shows a line plot of the collapsed distributions per cluster.

    Returns
    -------
    None
    """
    obs_df = sdata["source_score"].obs

    meta_df = sdata["xrna_metadata"].var
    if cluster_key not in meta_df.columns:
        raise ValueError(f"Clustering key '{cluster_key}' not found in sdata['xrna_metadata'].var")

    merged_df = obs_df.merge(meta_df[[cluster_key]], left_on=gene_key, right_index=True)

    global_max = merged_df[distance_key].max()
    bin_edges = np.linspace(0, global_max, n_bins + 1)

    gene_histograms = {}
    for gene, group in merged_df.groupby(gene_key):
        distances = group[distance_key].dropna().values
        counts, _ = np.histogram(distances, bins=bin_edges)
        norm_counts = counts / counts.sum() if counts.sum() > 0 else counts
        gene_histograms[gene] = norm_counts

    hist_df = pd.DataFrame.from_dict(gene_histograms, orient="index")
    hist_df[cluster_key] = meta_df[cluster_key]

    hist_df_sorted = hist_df.sort_values(cluster_key)
    clusters = hist_df_sorted[cluster_key]
    hist_df_sorted = hist_df_sorted.drop(columns=[cluster_key])
    bin_intervals = [f"({bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}]" for i in range(len(bin_edges) - 1)]
    hist_df_sorted.columns = bin_intervals

    unique_clusters = sorted(clusters.unique())
    palette = sns.color_palette("tab10", n_colors=len(unique_clusters))
    cluster_colors = {cluster: palette[i] for i, cluster in enumerate(unique_clusters)}
    row_colors = clusters.map(cluster_colors)

    if how == "full":
        g = sns.clustermap(
            hist_df_sorted,
            cmap="viridis",
            figsize=(12, 8),
            row_colors=row_colors,
            cbar_pos=(0, 0.2, 0.03, 0.4),
            linewidths=0,
            col_cluster=False,
            row_cluster=False,
        )
        ax = g.ax_heatmap
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=cluster, markersize=10, markerfacecolor=color)
            for cluster, color in cluster_colors.items()
        ]
        ax.legend(handles=handles, title="Cluster", loc="upper right", bbox_to_anchor=(1, 1))
        g.fig.suptitle("Gene Distance Distributions Sorted by Cluster", fontsize=16)
        plt.show()

    elif how == "collapsed":
        cluster_means = hist_df.groupby(cluster_key).mean()

        g = sns.clustermap(
            cluster_means,
            cmap="viridis",
            figsize=(10, 6),
            row_cluster=False,
            col_cluster=False,
            cbar_pos=(0, 0.2, 0.03, 0.4),
            annot=False,
            fmt=".2f",
            row_colors=[cluster_colors[c] for c in cluster_means.index],
        )
        g.fig.suptitle("Collapsed Distribution Per Cluster", fontsize=16)
        ax = g.ax_heatmap
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=cluster, markersize=10, markerfacecolor=color)
            for cluster, color in cluster_colors.items()
        ]
        ax.legend(handles=handles, title="Cluster", loc="upper right", bbox_to_anchor=(1, 1))
        plt.show()

    elif how == "lineplot":
        cluster_means = hist_df.groupby(cluster_key).mean()

        plt.figure(figsize=(10, 6))
        for cluster in cluster_means.index:
            sns.lineplot(x=cluster_means.columns, y=cluster_means.loc[cluster], label=f"Cluster {cluster}", color=cluster_colors[cluster])

        plt.title("Mean Distribution of Distance per Cluster", fontsize=16)
        plt.xlabel("Distance Bin", fontsize=14)
        plt.ylabel("Normalized Frequency", fontsize=14)
        plt.legend(title="Cluster")
        plt.show()
    else:
        raise ValueError("Invalid value for 'how'. Choose 'full', 'collapsed', 'lineplot'")


def distributions_by_cluster(
    sdata: SpatialData,
    gene_key: str = "gene",
    cluster_key: str = "kmeans_distribution",
    groups: list | None = None,
    distance_key: str = "distance",
    n_bins: int = 20,
) -> None:
    """Plot the average normalized distance distribution for each cluster.

    Each cluster's empirical histogram is overlaid with the theoretical Rayleigh
    distribution expected from purely diffusive transport, fitted globally across
    all genes. The plot title reports goodness-of-fit statistics (KS test, MAE)
    comparing the cluster's distribution to this expected pattern.

    Parameters
    ----------
    sdata
        Spatial data object containing ``source_score`` and ``xrna_metadata`` tables.
    gene_key
        Column name that contains the gene names.
    cluster_key
        Column in ``sdata["xrna_metadata"].var`` containing cluster assignments.
    groups
        Subset of cluster labels to plot. If `None`, all clusters are plotted.
    distance_key
        Column name that contains the distance from the source cell.
    n_bins
        Number of bins to use for the histograms.

    Returns
    -------
    None
    """
    cluster_labels = sdata["xrna_metadata"].var[cluster_key]

    global_distances = sdata["source_score"].obs[distance_key].dropna().values
    global_distances = global_distances[global_distances > 0]
    global_param = stats.rayleigh.fit(global_distances, floc=0)
    global_sigma = global_param[1]

    global_max = global_distances.max()
    bin_edges = np.linspace(0, global_max, n_bins + 1)

    clusters = cluster_labels.unique() if groups is None else groups

    for cluster in clusters:
        genes_in_cluster = sdata["xrna_metadata"].var[cluster_labels == cluster].index

        obs_df = sdata["source_score"].obs
        cluster_distances = obs_df[obs_df[gene_key].isin(genes_in_cluster)][distance_key].dropna().values
        cluster_distances = cluster_distances[cluster_distances > 0]

        counts, _ = np.histogram(cluster_distances, bins=bin_edges)
        norm_counts = counts / counts.sum() if counts.sum() > 0 else counts
        avg_hist = norm_counts

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = np.diff(bin_edges)[0]

        mean_distance = np.mean(cluster_distances) if len(cluster_distances) > 0 else np.nan
        std_distance = np.std(cluster_distances) if len(cluster_distances) > 0 else np.nan

        expected_pdf = stats.rayleigh.pdf(bin_centers, loc=0, scale=global_sigma) * bin_width

        ks_stat, ks_pval = stats.kstest(cluster_distances, "rayleigh", args=(0, global_sigma))

        mae = np.mean(np.abs(avg_hist - expected_pdf))

        plt.figure(figsize=(6, 4))
        plt.bar(bin_centers, avg_hist, width=bin_width, align="center", alpha=0.7, label="Avg. Normalized Histogram")
        plt.plot(bin_centers, expected_pdf, "r-", lw=2, label="Expected Rayleigh")
        plt.title(
            f"Cluster {cluster} - Avg. Distance Distribution\n"
            f"# Genes: {len(genes_in_cluster)} | Mean: {mean_distance:.2f} | Std: {std_distance:.2f}\n"
            f"Global Sigma: {global_sigma:.2f} | KS Stat: {ks_stat:.3f}, KS p: {ks_pval:.3f} | MAE: {mae:.3f}"
        )
        plt.xlabel("Distance")
        plt.ylabel("Probability Mass per Bin")
        plt.legend()
        plt.show()


def gene_distribution_from_source(
    sdata: SpatialData,
    cool_pattern: list[str],
    gene_key: str = "gene",
    distance_key: str = "distance",
    bins: int = 30,
    bar_color: str = "lightblue",
    n_cols: int = 3,
) -> None:
    """Plot the diffusion distance distribution of specified genes as subplots in a grid.

    Each subplot shows the empirical histogram of distances for one gene, overlaid
    with the Rayleigh distribution fitted globally (across all genes) and the
    Rayleigh distribution fitted to that gene alone.

    Parameters
    ----------
    sdata
        The spatial dataset containing gene expression and diffusion data.
    cool_pattern
        List of gene names to analyze.
    gene_key
        Column name for gene features.
    distance_key
        Column name for distance values.
    bins
        Number of bins for the histogram.
    bar_color
        Color of the histogram bars.
    n_cols
        Number of columns in the subplot grid.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If no valid distance values, or none of `cool_pattern`, are found in the dataset.
    """
    diffusion_results = sdata["xrna_metadata"].var
    global_distances = sdata["source_score"].obs[distance_key].dropna().values

    if len(global_distances) == 0:
        raise ValueError("No valid distance values found in the dataset.")

    global_distances = global_distances[global_distances > 0]
    param = stats.rayleigh.fit(global_distances, floc=0)

    valid_genes = [gene for gene in cool_pattern if gene in diffusion_results.index]

    if not valid_genes:
        raise ValueError("No specified genes were found in the dataset.")

    n_genes = len(valid_genes)
    n_rows = math.ceil(n_genes / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = np.array(axes).reshape(-1)

    for i, gene in enumerate(valid_genes):
        ax = axes[i]

        try:
            distances = sdata["source_score"].obs.query(f"{gene_key} == @gene")[distance_key].dropna().values
            param_g = stats.rayleigh.fit(distances, floc=0)
            if len(distances) == 0:
                print(f"Warning: No valid distances found for gene {gene}, skipping.")
                continue

            x = np.linspace(0, max(global_distances), 100)
            y = stats.rayleigh.pdf(x, *param)

            x_g = np.linspace(0, max(distances), 100)
            y_g = stats.rayleigh.pdf(x_g, *param_g)

            sns.histplot(distances, bins=bins, stat="density", kde=False, color=bar_color, label="Empirical", ax=ax)
            ax.plot(x, y, "r-", label="Global Fitted Rayleigh")
            ax.plot(x_g, y_g, "b-", label="Gene Fitted Rayleigh")

            gene_stats = diffusion_results.loc[diffusion_results.index == gene].iloc[0]
            ax.set_title(f"{gene}\nKS p-value: {gene_stats['ks_pval']:.3f}, LR Stat: {gene_stats['lr_stat']:.2f}")
            ax.set_xlabel("Distance")
            ax.set_ylabel("Density")
            ax.legend()

        except KeyError:
            print(f"Error processing gene {gene}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def source_score_by_celltype(
    sdata: SpatialData,
    gene_key: str = "gene",
    min_counts: int = 100,
    min_value: float | None = None,
    max_value: float | None = None,
    title: str | None = "Source Score by Cell Type",
    cluster_axis: str = "both",
    cmap: str = "coolwarm",
    figsize: tuple = (10, 8),
) -> None:
    """Plot a heatmap or clustered heatmap of source scores by cell type.

    Parameters
    ----------
    sdata
        A SpatialData object containing ``source_score`` and ``xrna_metadata`` tables.
    gene_key
        The key in ``sdata["source_score"].obs`` that contains gene names.
    min_counts
        Minimum count threshold for genes to be included.
    min_value
        Genes whose highest source score is below this value are filtered out.
    max_value
        Genes whose highest source score is above this value are filtered out.
    title
        Custom title for the plot.
    cluster_axis
        Determines clustering: ``"none"`` (no clustering), ``"x"`` (cluster columns
        only), ``"y"`` (cluster rows only), or ``"both"`` (cluster rows and columns).
    cmap
        Colormap for the heatmap.
    figsize
        Size of the figure.

    Returns
    -------
    None

    Notes
    -----
    This is the ``source_score`` counterpart of :func:`troutpy.pl.target_score_by_celltype`,
    which implements the same plot for ``target_score``.
    """
    source_score = sdata["source_score"].to_df()
    source_score["gene"] = sdata["source_score"].obs[gene_key]
    gene_by_celltype_score = source_score.groupby("gene").mean()

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
