import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns


def get_palette(name: str = "default", n_colors: int | None = None) -> list[str]:
    """Return a discrete color palette as a list of hex codes.

    Parameters
    ----------
    name
        Name of the palette (``"default"``, ``"coolwarm"``, ``"viridis"``, etc.).
    n_colors
        Number of colors to return. If `None`, return all.

    Returns
    -------
    List of hex color codes.
    """
    if name == "default":
        name = "30colors"

    palettes = {
        "30colors": [
            "#0FCCAC", "#ED2073", "#123755", "#E891B4", "#BFCE00", "#DF5AFF",
            "#FFFF7B", "#FC911D", "#E891B4", "#C1F4C5", "#37BEE2", "#790DB7",
            "#9CCE00", "#43FBFF", "#F7C894", "#BCBCBC", "#444444", "#096858",
            "#3A8458", "#0000C9", "#FF5757", "#990E0E", "#754B19", "#CC882F",
            "#FF00F3", "#C4A7C2", "#9CFFFC", "#00E50B", "#000000", "#FF3600",
            "#FFC088"
        ],
        "calid_colors": [
            "#ED2073", "#ebd234", "#FC911D", "#E891B4", "#F7C894", "#FF5757",
            "#990E0E", "#754B19", "#CC882F", "#FF00F3", "#FF3600", "#FFC088",
            "#DF5AFF"
        ],
        "cold_colors": [
            "#0FCCAC", "#123755", "#C1F4C5", "#37BEE2", "#790DB7", "#9CCE00",
            "#43FBFF", "#BCBCBC", "#444444", "#096858", "#3A8458", "#0000C9",
            "#C4A7C2", "#9CFFFC", "#00E50B", "#000000", "#BFCE00"
        ],
        "BuPi": ["#123755", "#DBDBDB", "#ED2073"],
        "Pinks": ["#FFFFFF", "#ED2073"],
        "Aquas": ["#FFFFFF", "#0FCCAC"],
        "GnPi": ["#0FCCAC", "#ED2073"],
        "Aquas_contrast": ["#f9ffba", "#c9ffba", "#0FCCAC", "#066253"],
        "Pinks_contrast": ["#fbe7ff", "#ffc7dd", "#ED2073", "#94003c"],
        "troutpy": ["#123755", "#0FCCAC"],
        "troutpy_reversed": ["#0FCCAC", "#123755"],
        "tech_corporate": ["#0A2540", "#635BFF", "#00D4B2", "#708090", "#EF98CF"],
        "retro_vintage": ["#D4A373", "#E3D5CA", "#F5ECE1", "#FAEDCD", "#CCD5AE"],
        "cyberpunk": ["#00F0FF", "#FF007F", "#9D00FF", "#FFFF00", "#00FF66"],
        "pastel_minimal": ["#FFB7B2", "#FFDAC1", "#E2F0CB", "#B5EAD7", "#C7CEEA"],
        "earth_warm": ["#4A3B32", "#8B5E3C", "#B8860B", "#BC8F8F", "#D2B48C"],
    }

    # Seaborn palettes are returned as RGB tuples; convert them to hex for consistency
    sns_palettes = ["coolwarm", "viridis", "muted", "rocket", "mako", "flare", "crest"]
    if name in sns_palettes:
        amt = n_colors if n_colors else 6
        raw_sns = sns.color_palette(name, n_colors=amt)
        return [mcolors.to_hex(c) for c in raw_sns]

    if name not in palettes:
        available = list(palettes.keys()) + sns_palettes
        raise ValueError(f"Palette '{name}' not found. Available palettes: {available}")

    palette = palettes[name]
    return palette[:n_colors] if n_colors else palette


def get_colormap(name: str = "default") -> mcolors.Colormap:
    """Return a continuous colormap for Matplotlib.

    Parameters
    ----------
    name
        Name of the colormap (``"default"``, ``"coolwarm"``, ``"viridis"``, etc.).

    Returns
    -------
    A matplotlib colormap object.
    """
    if name == "default":
        name = "30colors"

    mpl_native = ["coolwarm", "viridis", "rocket", "mako", "flare", "crest"]
    if name in mpl_native:
        return plt.get_cmap(name)

    try:
        hex_colors = get_palette(name=name)
        return mcolors.LinearSegmentedColormap.from_list(f"custom_{name}", hex_colors)
    except ValueError as e:
        raise ValueError(f"Colormap '{name}' could not be resolved from existing palettes.") from e
