import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from spatialdata import SpatialData

from troutpy.pl.colors import get_colormap, get_palette


def pie(
    sdata: SpatialData,
    groupby: str,
    layer: str = "transcripts",
    group_key: str | None = None,
    figures_path: str = "",
    save: bool = True,
    title: str | None = None,
    custom_plot_filename: str | None = None,
    palette: str = "tab20",
) -> None:
    """Generate pie charts of the proportion of each category of `groupby`.

    If `group_key` is provided, separate pie charts are drawn for each category in
    `group_key`.

    Parameters
    ----------
    sdata
        The input spatial data object containing the categorical variable.
    groupby
        The column name in the data to group by.
    layer
        The layer in `sdata.points` to extract data from.
    group_key
        If provided, generates separate pie charts for each category in this column.
    figures_path
        Path where the pie chart will be saved if `save` is `True`.
    save
        Whether to save the figure as a PDF. If `False`, the chart is displayed.
    title
        Title for the pie chart.
    custom_plot_filename
        Custom filename for saving the pie chart.
    palette
        Name of the color palette to use.

    Returns
    -------
    None
    """
    data = sdata.points[layer][[groupby] + ([group_key] if group_key else [])].compute()
    all_categories = sorted(data[groupby].dropna().unique())

    if palette in plt.colormaps():
        cmap = plt.get_cmap(palette)
        color_mapping = {cat: cmap(i / len(all_categories)) for i, cat in enumerate(all_categories)}
    else:
        color_mapping = dict.fromkeys(all_categories)  # fallback

    if group_key:
        unique_groups = data[group_key].dropna().unique()
        num_groups = len(unique_groups)

        cols = min(3, num_groups)
        rows = math.ceil(num_groups / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.flatten() if num_groups > 1 else [axes]

        for i, group in enumerate(unique_groups):
            subset = data[data[group_key] == group]
            category_counts = subset[groupby].value_counts().reindex(all_categories, fill_value=0)
            colors = [color_mapping[cat] for cat in category_counts.index]

            axes[i].pie(category_counts, labels=category_counts.index, colors=colors, autopct="%1.1f%%")
            axes[i].set_title(f"{title if title else ''} {group_key}: {group}")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        if save:
            plot_filename = custom_plot_filename or f"pie_{groupby}_by_{group_key}.pdf"
            plt.savefig(os.path.join(figures_path, plot_filename))
            plt.close()
        else:
            plt.show()

    else:
        category_counts = data[groupby].value_counts().reindex(all_categories, fill_value=0)
        colors = [color_mapping[cat] for cat in category_counts.index]

        plt.figure(figsize=(5, 5))
        plt.pie(category_counts, labels=category_counts.index, colors=colors, autopct="%1.1f%%")
        plt.title(title if title else f"Proportion of {groupby}")

        if save:
            plot_filename = custom_plot_filename or f"pie_{groupby}.pdf"
            plt.savefig(os.path.join(figures_path, plot_filename))
            plt.close()
        else:
            plt.show()


def crosstab(
    sdata: SpatialData,
    xvar: str,
    yvar: str,
    layer: str = "transcripts",
    normalize: bool = True,
    axis: int = 1,
    kind: str = "barh",
    save: bool = True,
    figures_path: str = "",
    stacked: bool = True,
    figsize: tuple | None = None,
    cmap: str = "troutpy",
    saving_format: str = "pdf",
    sortby: str | None = None,
) -> None:
    """Generate a cross-tabulation plot between two categorical variables from `sdata.points`.

    Parameters
    ----------
    sdata
        The input spatial data object containing the categorical variables.
    xvar
        The categorical variable for the x-axis.
    yvar
        The categorical variable for the y-axis.
    layer
        The layer in `sdata.points` to extract data from.
    normalize
        Whether to normalize proportions.
    axis
        Axis to normalize across (1 = row, 0 = column).
    kind
        Plot type: ``"barh"``, ``"bar"``, ``"heatmap"``, or ``"clustermap"``.
    save
        Whether to save the plot.
    figures_path
        Path to save the plot.
    stacked
        Whether bar plots should be stacked.
    figsize
        Automatically computed based on the number of categories unless manually specified.
    cmap
        Custom color palette, resolved via :func:`troutpy.pl.get_colormap`.
    saving_format
        File format to save the plot (e.g. ``"png"``, ``"pdf"``).
    sortby
        Column/row to sort before plotting.

    Returns
    -------
    None
    """
    data = sdata.points[layer][[xvar, yvar]].compute()
    crosstab_data = pd.crosstab(data[yvar], data[xvar])

    if normalize:
        crosstab_data = crosstab_data.div(crosstab_data.sum(axis=axis), axis=0)
        normtag = "normalized"
    else:
        normtag = "raw"

    if sortby:
        crosstab_data = crosstab_data.sort_values(by=sortby)

    try:
        cmap = get_colormap(cmap)
    except ValueError:
        cmap = None

    if figsize is None:
        num_x = len(crosstab_data.columns)
        num_y = len(crosstab_data.index)
        width = max(5, num_x * 0.5)
        height = max(5, num_y * 0.4)
        figsize = (width, height)

    plot_filename = f"{kind}_{xvar}_{yvar}_{normtag}_{saving_format}.{saving_format}"

    plt.figure(figsize=figsize)

    if kind in ["barh", "bar"]:
        crosstab_data.T.plot(kind=kind, stacked=stacked, colormap=cmap, figsize=figsize)
        plt.title(f"{xvar} vs {yvar}")
        plt.xlabel("Total number of cells")

    elif kind == "heatmap":
        sns.heatmap(crosstab_data, cmap=cmap, annot=True, fmt=".2f")
        plt.title(f"{xvar} vs {yvar} (Heatmap)")

    elif kind == "clustermap":
        sns.clustermap(crosstab_data, cmap=cmap)

    if save:
        plt.savefig(os.path.join(figures_path, plot_filename))
    else:
        plt.show()


def histogram(
    sdata: SpatialData,
    x: str,
    hue: str | None = None,
    layer: str = "transcripts",
    group_key: str | None = None,
    figures_path: str = "",
    save: bool = True,
    title: str | None = None,
    custom_plot_filename: str | None = None,
    palette: str = "tab10",
    bins: int = 30,
    kde: bool = False,
) -> None:
    """Plot histograms of a numeric variable with optional grouping and faceting.

    Parameters
    ----------
    sdata
        The input spatial data object.
    x
        The name of the numeric column to plot on the x-axis.
    hue
        The column name used for color grouping.
    layer
        The layer in `sdata.points` to extract data from.
    group_key
        If provided, creates subplots by the unique values of this column.
    figures_path
        Path to save the plot if `save` is `True`.
    save
        Whether to save the figure as a PDF.
    title
        Overall title for the plot.
    custom_plot_filename
        Custom filename to use when saving the figure.
    palette
        Color palette name for seaborn, resolved via :func:`troutpy.pl.get_palette`
        if not a built-in Matplotlib colormap.
    bins
        Number of histogram bins.
    kde
        Whether to overlay a kernel density estimate.

    Returns
    -------
    None
    """
    data = sdata.points[layer][[x] + ([hue] if hue else []) + ([group_key] if group_key else [])].compute()

    if palette not in plt.colormaps():
        try:
            palette = get_palette(palette)
        except ValueError:
            palette = None

    if group_key:
        unique_groups = data[group_key].unique()
        num_groups = len(unique_groups)

        cols = min(3, num_groups)
        rows = math.ceil(num_groups / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=True, sharey=True)
        axes = axes.flatten() if num_groups > 1 else [axes]

        for i, group in enumerate(unique_groups):
            subset = data[data[group_key] == group]
            sns.histplot(data=subset, x=x, hue=hue, bins=bins, kde=kde, palette=palette, ax=axes[i], element="step")
            axes[i].set_title(f"{group_key}: {group}")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle(title if title else f"{x} histogram grouped by {group_key}", y=1.02)
        plt.tight_layout()

        if save:
            plot_filename = custom_plot_filename or f"histogram_{x}_by_{group_key}.pdf"
            plt.savefig(os.path.join(figures_path, plot_filename), bbox_inches="tight")
        else:
            plt.show()

    else:
        sns.displot(data=data, x=x, hue=hue, bins=bins, kde=kde, palette=palette, height=5, aspect=1.5, element="step").set(
            title=title if title else f"Distribution of {x}"
        )

        if save:
            plot_filename = custom_plot_filename or f"histogram_{x}.pdf"
            plt.savefig(os.path.join(figures_path, plot_filename), bbox_inches="tight")
        else:
            plt.show()
