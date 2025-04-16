import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import spatialdata as sd  # Assuming this is the correct import
from spatialdata import SpatialData  # Assuming this is the correct import

# from troutpy.pp.compute import compute_crosstab


def global_distribution_from_source(
    sdata: SpatialData,
    cluster_key: str = "kmeans_from_distribution",
    gene_key: str = "gene",
    distance_key: str = "distance",
    n_bins: int = 20,
    how: str = "full",
):
    """
    Plots either:
    1. A clustermap of all genes' distance distributions, sorted by cluster, with row colors.
    2. A collapsed clustermap showing the mean distribution per cluster, with color-coded labels.
    3. A line plot showing the mean distribution for each cluster, if 'how' is 'lineplot'.

    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing 'source_score' and 'xrna_metadata' layers.
    cluster_key : str, default "kmeans_from_distribution"
        Column in sdata['xrna_metadata'].var containing cluster assignments.
    gene_key : str, default "gene"
        Column that contains gene names.
    distance_key : str, default "distance"
        Column that contains the distances.
    n_bins : int, default 20
        Number of bins for the histogram representation.
    how : str, default "full"
        - "full": Show clustermap of all genes, sorted by cluster, with row colors.
        - "collapsed": Show clustermap of mean distribution per cluster, with color-coded labels.
        - "lineplot": Show a line plot of the collapsed distributions per cluster.
    """
    # Extract transcript distance data
    obs_df = sdata["source_score"].obs

    # Ensure clustering key exists
    meta_df = sdata["xrna_metadata"].var
    if cluster_key not in meta_df.columns:
        raise ValueError(f"Clustering key '{cluster_key}' not found in sdata['xrna_metadata'].var")

    # Ensure all genes in metadata exist in observed data
    merged_df = obs_df.merge(meta_df[[cluster_key]], left_on=gene_key, right_index=True)

    # Set bin edges based on global max distance
    global_max = merged_df[distance_key].max()
    bin_edges = np.linspace(0, global_max, n_bins + 1)

    # Compute histograms for each gene
    gene_histograms = {}
    for gene, group in merged_df.groupby(gene_key):
        distances = group[distance_key].dropna().values
        counts, _ = np.histogram(distances, bins=bin_edges)
        norm_counts = counts / counts.sum() if counts.sum() > 0 else counts
        gene_histograms[gene] = norm_counts

    # Convert to DataFrame
    hist_df = pd.DataFrame.from_dict(gene_histograms, orient="index")

    # Merge with clusters
    hist_df[cluster_key] = meta_df[cluster_key]

    # Sorting by cluster
    hist_df_sorted = hist_df.sort_values(cluster_key)
    clusters = hist_df_sorted[cluster_key]
    clust_key = hist_df_sorted[cluster_key]
    hist_df_sorted = hist_df_sorted.drop(columns=[cluster_key])
    # Create column names as intervals
    bin_intervals = [f"({bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}]" for i in range(len(bin_edges) - 1)]
    # Assign the new column names to hist_df_sorted
    hist_df_sorted.columns = bin_intervals

    # Define a color palette for clusters
    unique_clusters = sorted(clusters.unique())
    palette = sns.color_palette("tab10", n_colors=len(unique_clusters))
    cluster_colors = {cluster: palette[i] for i, cluster in enumerate(unique_clusters)}

    # Create a color vector for the sorted genes (row colors)
    row_colors = clusters.map(cluster_colors)

    # Compute global expected Rayleigh parameter from all distances in the data.
    global_distances = sdata["source_score"].obs[distance_key].dropna().values
    global_distances = global_distances[global_distances > 0]  # Filter out non-positive distances
    global_param = stats.rayleigh.fit(global_distances, floc=0)
    global_sigma = global_param[1]

    # Create expected Rayleigh PDF for comparison
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = np.diff(bin_edges)[0]
    expected_pdf = stats.rayleigh.pdf(bin_centers, loc=0, scale=global_sigma) * bin_width

    # Create clustermap based on `how` argument
    if how == "full":
        # Clustermap for individual gene distributions
        g = sns.clustermap(
            hist_df_sorted,
            cmap="viridis",
            figsize=(12, 8),
            row_colors=row_colors,
            cbar_pos=(0, 0.2, 0.03, 0.4),
            linewidths=0,
            col_cluster=False,
            row_cluster=False,
        )  # Cluster only rows, not columns
        ax = g.ax_heatmap
        # Add a custom legend for cluster colors
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=cluster, markersize=10, markerfacecolor=color)
            for cluster, color in cluster_colors.items()
        ]
        ax.legend(handles=handles, title="Cluster", loc="upper right", bbox_to_anchor=(1, 1))
        g.fig.suptitle("Gene Distance Distributions Sorted by Cluster", fontsize=16)
        plt.show()

    elif how == "collapsed":
        # Compute collapsed means per cluster
        cluster_means = hist_df.groupby(cluster_key).mean()

        # Clustermap for collapsed distributions per cluster
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
        # Add a custom legend for cluster colors
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=cluster, markersize=10, markerfacecolor=color)
            for cluster, color in cluster_colors.items()
        ]
        ax.legend(handles=handles, title="Cluster", loc="upper right", bbox_to_anchor=(1, 1))
        plt.show()

    elif how == "lineplot":
        # Compute collapsed means per cluster
        cluster_means = hist_df.groupby(cluster_key).mean()

        # Line plot for collapsed distributions per cluster
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
    groups=None,
    distance_key: str = "distance",
    n_bins: int = 20,
):
    """
    Plots the average normalized distance distribution for each cluster, overlaid with the
    expected (theoretical) diffusion pattern from a Rayleigh distribution computed from the
    global data (purely diffusion-based). Also displays statistics on how well the genes in the
    cluster match the expected diffusion pattern.

    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing a 'source_score' layer with an obs DataFrame.
    gene_key : str, default "gene"
        Column name that contains the gene names.
    distance_key : str, default "distance"
        Column name that contains the distance from the source cell.
    n_bins : int, default 20
        Number of bins to use for the histograms.
    """
    # Extract the cluster labels from the SpatialData object
    cluster_labels = sdata["xrna_metadata"].var[cluster_key]

    # Compute global expected Rayleigh parameter from all distances in the data.
    global_distances = sdata["source_score"].obs[distance_key].dropna().values
    # Filter out non-positive values (Rayleigh is defined only for x > 0)
    global_distances = global_distances[global_distances > 0]
    global_param = stats.rayleigh.fit(global_distances, floc=0)
    global_sigma = global_param[1]

    # Set bin edges based on global max distance
    global_max = global_distances.max()
    bin_edges = np.linspace(0, global_max, n_bins + 1)

    if groups is None:
        clusters = cluster_labels.unique()
    else:
        clusters = groups
    # Get the unique clusters from the cluster labels

    for cluster in clusters:
        # Find the genes assigned to this cluster
        genes_in_cluster = sdata["xrna_metadata"].var[cluster_labels == cluster].index

        # Combine all distances for the genes in the cluster.
        obs_df = sdata["source_score"].obs
        cluster_distances = obs_df[obs_df[gene_key].isin(genes_in_cluster)][distance_key].dropna().values
        # Filter out non-positive distances as well.
        cluster_distances = cluster_distances[cluster_distances > 0]

        # Compute histograms for the cluster
        counts, _ = np.histogram(cluster_distances, bins=bin_edges)
        norm_counts = counts / counts.sum() if counts.sum() > 0 else counts
        avg_hist = norm_counts

        # Compute bin centers and bin width for plotting.
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = np.diff(bin_edges)[0]

        # Compute mean and standard deviation of distances in the cluster.
        mean_distance = np.mean(cluster_distances) if len(cluster_distances) > 0 else np.nan
        std_distance = np.std(cluster_distances) if len(cluster_distances) > 0 else np.nan

        # Compute the expected Rayleigh PDF using the global_sigma (expected diffusion pattern).
        expected_pdf = stats.rayleigh.pdf(bin_centers, loc=0, scale=global_sigma) * bin_width

        # Perform a KS test comparing the cluster distances to the expected Rayleigh distribution.
        ks_stat, ks_pval = stats.kstest(cluster_distances, "rayleigh", args=(0, global_sigma))

        # Compute the mean absolute error between the average histogram and expected probability mass.
        mae = np.mean(np.abs(avg_hist - expected_pdf))

        # Plotting
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
    cool_pattern,
    gene_key: str = "gene",
    distance_key: str = "distance",
    bins: int = 30,
    bar_color: str = "lightblue",
    n_cols: int = 3,
):
    """
    Plots the diffusion distribution of specified genes as subplots in a grid.

    Parameters
    ----------
    - sdata (SpatialData): The spatial dataset containing gene expression and diffusion data.
    - cool_pattern (list): List of gene names to analyze.
    - gene_key (str): Column name for gene features (default: 'gene').
    - distance_key (str): Column name for distance values (default: 'distance').
    - bins (int): Number of bins for histogram (default: 30).
    - bar_color (str): Color of histogram bars (default: 'lightblue').
    - n_cols (int): Number of columns in the subplot grid (default: 3).

    Raises
    ------
    - ValueError: If no valid genes are found in the dataset.
    """
    diffusion_results = sdata["xrna_metadata"].var
    global_distances = sdata["source_score"].obs[distance_key].dropna().values

    if len(global_distances) == 0:
        raise ValueError("No valid distance values found in the dataset.")

    # Filter out non-positive values (Rayleigh is defined only for x > 0)
    global_distances = global_distances[global_distances > 0]
    param = stats.rayleigh.fit(global_distances, floc=0)

    valid_genes = [gene for gene in cool_pattern if gene in diffusion_results.index]

    if not valid_genes:
        raise ValueError("No specified genes were found in the dataset.")

    n_genes = len(valid_genes)
    n_rows = math.ceil(n_genes / n_cols)  # Calculate required rows based on columns

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = np.array(axes).reshape(-1)  # Flatten to make indexing easier

    for i, gene in enumerate(valid_genes):
        ax = axes[i]  # Select subplot

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

        except Exception as e:
            print(f"Error processing gene {gene}: {e}")

    # Remove any unused subplots if the number of genes is less than n_rows * n_cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def source_score_by_celltype(
    sdata: sd.SpatialData,
    gene_key: str = "gene",
    min_counts: int = 100,
    min_value: float | None = None,
    max_value: float | None = None,
    title: str | None = "Source Score by Cell Type",
    cluster_axis: str = "both",
    cmap: str = "coolwarm",
    figsize: tuple = (10, 8),
) -> None:
    """
    Plots a heatmap or clustered heatmap of source scores by cell type.

    Parameters
    ----------
    sdata : sd.SpatialData
        A SpatialData object containing `source_score` data.
    gene_key : str, default="gene"
        The key in `obs` that contains gene names.
    min_counts : int, default=100
        Minimum count threshold for genes to be included.
    min_value : float, optional
        Genes presenting the highest source score below this will be filtered out in visualization
    max_value : float, optional
        Genes presenting the highest source score above this will be filtered out in visualization
    title : str, optional
        Custom title for the plot.
    cluster_axis : str, default="both"
        Determines clustering:
        - "none" (no clustering)
        - "x" (cluster columns only)
        - "y" (cluster rows only)
        - "both" (cluster rows and columns)
    cmap : str, default="coolwarm"
        Colormap for the heatmap.
    figsize : tuple, default=(10, 8)
        Size of the figure.

    Returns
    -------
    None
        Displays the heatmap.
    """
    # Extract source scores
    source_score = sdata["source_score"].to_df()
    source_score["gene"] = sdata["source_score"].obs[gene_key]
    gene_by_celltype_score = source_score.groupby("gene").mean()

    # Filter genes based on count threshold
    genes = sdata["xrna_metadata"].var.index[sdata["xrna_metadata"].var["count"] > min_counts]
    filtered_gene_by_celltype_score = gene_by_celltype_score.loc[gene_by_celltype_score.index.isin(genes), :]

    # Apply additional filtering based on min/max values
    if min_value is not None:
        filtered_gene_by_celltype_score = filtered_gene_by_celltype_score[np.max(filtered_gene_by_celltype_score, axis=1) >= min_value].dropna()
    if max_value is not None:
        filtered_gene_by_celltype_score = filtered_gene_by_celltype_score[np.max(filtered_gene_by_celltype_score, axis=1) <= max_value].dropna()

    if filtered_gene_by_celltype_score.empty:
        print("No data available to plot after filtering.")
        return

    # Determine clustering options
    valid_axes = ["none", "x", "y", "both"]
    cluster_axis = cluster_axis.lower()
    if cluster_axis not in valid_axes:
        raise ValueError(f"Invalid cluster_axis: {cluster_axis}. Must be one of {', '.join(valid_axes)}.")

    # Plot heatmap with clustering
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
