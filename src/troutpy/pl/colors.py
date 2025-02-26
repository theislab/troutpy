import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns


def get_palette(name="default", n_colors=None):
    """
    Returns a discrete color palette as a list of hex codes.

    Parameters
    ----------
    - name (str): Name of the palette ("default", "coolwarm", "viridis", etc.).
    - n_colors (int, optional): Number of colors to return. If None, return all.

    Returns
    -------
    - list of hex color codes.
    """
    palettes = {
        "30colors": [
            "#0FCCAC",
            "#ED2073",
            "#123755",
            "#BFCE00",
            "#DF5AFF",
            "#FFFF7B",
            "#FC911D",
            "#E891B4",
            "#C1F4C5",
            "#37BEE2",
            "#790DB7",
            "#9CCE00",
            "#43FBFF",
            "#F7C894",
            "#BCBCBC",
            "#444444",
            "#096858",
            "#3A8458",
            "#0000C9",
            "#FF5757",
            "#990E0E",
            "#754B19",
            "#CC882F",
            "#FF00F3",
            "#C4A7C2",
            "#9CFFFC",
            "#00E50B",
            "#000000",
            "#FF3600",
            "#FFC088",
        ],
        "calid_colors": [
            "#ED2073",
            "#ebd234",
            "#FC911D",
            "#E891B4",
            "#F7C894",
            "#FF5757",
            "#990E0E",
            "#754B19",
            "#CC882F",
            "#FF00F3",
            "#FF3600",
            "#FFC088",
            "#DF5AFF",
        ],
        "cold_colors": [
            "#0FCCAC",
            "#123755",
            "#C1F4C5",
            "#37BEE2",
            "#790DB7",
            "#9CCE00",
            "#43FBFF",
            "#BCBCBC",
            "#444444",
            "#096858",
            "#3A8458",
            "#0000C9",
            "#C4A7C2",
            "#9CFFFC",
            "#00E50B",
            "#000000",
            "#BFCE00",
        ],
        "BuPi": ["#123755", "#DBDBDB", "#ED2073"],
        "Pinks": ["white", "#ED2073"],
        "Aquas": ["white", "#0FCCAC"],
        "GnPi": ["#0FCCAC", "#ED2073"],
        "Aquas_contrast": ["#f9ffba", "#c9ffba", "#0FCCAC", "#066253"],
        "Pinks_contrast": ["#fbe7ff", "#ffc7dd", "#ED2073", "#94003c"],
        "coolwarm": sns.color_palette("coolwarm", as_cmap=False),
        "viridis": sns.color_palette("viridis", as_cmap=False),
        "muted": sns.color_palette("muted"),
        "troutpy": ["#123755", "#0FCCAC"],
        "troutpy_reversed": ["#0FCCAC", "#123755"],
    }

    if name not in palettes:
        raise ValueError(f"Palette '{name}' not found. Available palettes: {list(palettes.keys())}")

    palette = palettes[name]
    return palette[:n_colors] if n_colors else palette


def get_colormap(name="default"):
    """
    Returns a continuous colormap for Matplotlib.

    Parameters
    ----------
    - name (str): Name of the colormap ("default", "coolwarm", "viridis", etc.).

    Returns
    -------
    - matplotlib.colors.Colormap object.
    """
    colormaps = {
        "30colors": mcolors.LinearSegmentedColormap.from_list("custom_cmap", get_palette("30colors")),
        "cold_colors": mcolors.LinearSegmentedColormap.from_list("custom_cmap", get_palette("cold_colors")),
        "calid_colors": mcolors.LinearSegmentedColormap.from_list("custom_cmap", get_palette("calid_colors")),
        "Aquas_contrast": mcolors.LinearSegmentedColormap.from_list("custom_cmap", get_palette("Aquas_contrast")),
        "Pinks_contrast": mcolors.LinearSegmentedColormap.from_list("custom_cmap", get_palette("Pinks_contrast")),
        "BuPi": mcolors.LinearSegmentedColormap.from_list("custom_cmap", get_palette("BuPi")),
        "GnPi": mcolors.LinearSegmentedColormap.from_list("custom_cmap", get_palette("GnPi")),
        "Pinks": mcolors.LinearSegmentedColormap.from_list("custom_cmap", get_palette("Pinks")),
        "Aquas": mcolors.LinearSegmentedColormap.from_list("custom_cmap", get_palette("Aquas")),
        "troutpy": mcolors.LinearSegmentedColormap.from_list("custom_cmap", get_palette("troutpy")),
        "troutpy_reversed": mcolors.LinearSegmentedColormap.from_list("custom_cmap", get_palette("troutpy_reversed")),
        "coolwarm": plt.get_cmap("coolwarm"),
        "viridis": plt.get_cmap("viridis"),
    }

    if name not in colormaps:
        raise ValueError(f"Colormap '{name}' not found. Available colormaps: {list(colormaps.keys())}")

    return colormaps[name]
