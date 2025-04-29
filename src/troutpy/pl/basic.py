import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from troutpy.pl.colors import get_colormap, get_palette


def pie(
    sdata,
    groupby: str,
    layer: str = "transcripts",
    group_key: str = None,
    figures_path: str = "",
    save: bool = True,
    title: str = None,
    custom_plot_filename: str = None,
    palette: str = "tab20",
):
    """
    Generates pie charts showing the proportion of different categories for a specified categorical variable.
    If `group_key` is provided, it creates subplots with individual pie charts for each category in `group_key`.
    """
    import math
    import os

    import matplotlib.pyplot as plt

    data = sdata.points[layer][[groupby] + ([group_key] if group_key else [])].compute()

    # Determine all categories globally
    all_categories = sorted(data[groupby].dropna().unique())

    # Determine colormap
    if palette not in plt.colormaps():
        try:
            palette = get_palette(palette)
        except:
            palette = plt.cm.tab20.colors  # fallback
    else:
        cmap = plt.get_cmap(palette)
        palette = [cmap(i) for i in range(len(all_categories))]

    # Build category-color mapping
    color_mapping = dict(zip(all_categories, palette, strict=False))

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

            # Apply consistent colors
            colors = [color_mapping[cat] for cat in category_counts.index]

            axes[i].pie(category_counts, labels=category_counts.index, colors=colors, autopct="%1.1f%%")
            axes[i].set_title(f"{title if title else ''} {group_key}: {group}")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        if save:
            plot_filename = custom_plot_filename or f"pie_{groupby}_by_{group_key}.pdf"
            plt.savefig(os.path.join(figures_path, plot_filename))
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
        else:
            plt.show()


def crosstab(
    sdata,
    xvar: str,
    yvar: str,
    layer: str = "transcripts",
    normalize=True,
    axis=1,
    kind="barh",
    save=True,
    figures_path: str = "",
    stacked=True,
    figsize=None,  # Now auto-computed unless specified
    cmap: str = "troutpy",
    saving_format="pdf",
    sortby=None,
):
    """
    Generates a cross-tabulation plot between two categorical variables from `sdata.points`.

    Parameters
    ----------
    sdata : sd.SpatialData
        The input spatial data object containing the categorical variables.
    xvar : str
        The categorical variable for the x-axis.
    yvar : str
        The categorical variable for the y-axis.
    layer : str, optional
        The layer in `sdata.points` to extract data from (default is "transcripts").
    normalize : bool, optional
        Whether to normalize proportions (default is True).
    axis : int, optional
        Axis to normalize across (1 = row, 0 = column).
    kind : str, optional
        Plot type: 'barh', 'bar', 'heatmap', 'clustermap'.
    save : bool, optional
        Whether to save the plot (default is True).
    figures_path : str, optional
        Path to save the plot.
    stacked : bool, optional
        Whether bar plots should be stacked (default is True).
    figsize : tuple, optional
        Automatically computed based on the number of categories unless manually specified.
    cmap : str, optional
        Custom color palette.
    saving_format : str, optional
        File format to save the plot (e.g., 'png', 'pdf').
    sortby : str, optional
        Column/row to sort before plotting.

    Returns
    -------
    None
    """
    # Extract relevant data from sdata
    data = sdata.points[layer][[xvar, yvar]].compute()

    # Compute the cross-tabulation
    crosstab_data = pd.crosstab(data[yvar], data[xvar])

    # Normalize if required
    if normalize:
        crosstab_data = crosstab_data.div(crosstab_data.sum(axis=axis), axis=0)
        normtag = "normalized"
    else:
        normtag = "raw"

    # Sort if specified
    if sortby:
        crosstab_data = crosstab_data.sort_values(by=sortby)

    # Handle custom colormap
    try:
        cmap = get_colormap(cmap)
    except:
        cmap = None  # Use default colors if custom palette fails

    # **Compute automatic figsize based on category counts**
    if figsize is None:
        num_x = len(crosstab_data.columns)  # Number of x categories
        num_y = len(crosstab_data.index)  # Number of y categories

        # Adjust figure size dynamically
        width = max(5, num_x * 0.5)  # Scale width based on x categories
        height = max(5, num_y * 0.4)  # Scale height based on y categories

        figsize = (width, height)

    # Generate plot filename
    plot_filename = f"{kind}_{xvar}_{yvar}_{normtag}_{saving_format}.{saving_format}"

    # Create figure
    plt.figure(figsize=figsize)

    if kind in ["barh", "bar"]:
        crosstab_data.T.plot(kind=kind, stacked=stacked, colormap=cmap)
        plt.title(f"{xvar} vs {yvar}")

    elif kind == "heatmap":
        sns.heatmap(crosstab_data, cmap=cmap, annot=True, fmt=".2f")
        plt.title(f"{xvar} vs {yvar} (Heatmap)")

    elif kind == "clustermap":
        sns.clustermap(crosstab_data, cmap=cmap)

    # Save or show
    if save:
        plt.savefig(os.path.join(figures_path, plot_filename))
    else:
        plt.show()


def histogram(
    sdata,
    x: str,
    hue: str = None,
    layer: str = "transcripts",
    group_key: str = None,
    figures_path: str = "",
    save: bool = True,
    title: str = None,
    custom_plot_filename: str = None,
    palette: str = "tab10",
    bins: int = 30,
    kde: bool = False,
):
    """
    Plots histograms of a numeric variable with optional grouping and faceting.

    Parameters
    ----------
    sdata : sd.SpatialData
        The input spatial data object.
    x : str
        The name of the numeric column to plot on the x-axis.
    hue : str, optional
        The column name used for color grouping (optional).
    layer : str, optional
        The layer in sdata.points to extract data from (default is "transcripts").
    group_key : str, optional
        If provided, creates subplots by the unique values of this column.
    figures_path : str, optional
        Path to save the plot if `save` is True.
    save : bool, optional
        Whether to save the figure as a PDF.
    title : str, optional
        Overall title for the plot.
    custom_plot_filename : str, optional
        Custom filename to use when saving the figure.
    palette : str, optional
        Color palette name for seaborn (default is "tab10").
    bins : int, optional
        Number of histogram bins (default is 30).
    kde : bool, optional
        Whether to overlay a kernel density estimate (default is False).

    Returns
    -------
    None
    """
    data = sdata.points[layer][[x] + ([hue] if hue else []) + ([group_key] if group_key else [])].compute()

    # Load color palette
    if palette not in plt.colormaps():
        try:
            palette = get_palette(palette)
        except:
            palette = None  # Fallback if not found

    if group_key:
        unique_groups = data[group_key].unique()
        num_groups = len(unique_groups)

        cols = min(3, num_groups)  # Max 3 columns per row
        rows = math.ceil(num_groups / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=True, sharey=True)
        axes = axes.flatten() if num_groups > 1 else [axes]

        for i, group in enumerate(unique_groups):
            subset = data[data[group_key] == group]
            sns.histplot(data=subset, x=x, hue=hue, bins=bins, kde=kde, palette=palette, ax=axes[i], element="step")
            axes[i].set_title(f"{group_key}: {group}")

        # Hide unused subplots
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
        # Single plot
        sns.displot(data=data, x=x, hue=hue, bins=bins, kde=kde, palette=palette, height=5, aspect=1.5, element="step").set(
            title=title if title else f"Distribution of {x}"
        )

        if save:
            plot_filename = custom_plot_filename or f"histogram_{x}.pdf"
            plt.savefig(os.path.join(figures_path, plot_filename), bbox_inches="tight")
        else:
            plt.show()
