import matplotlib.colors as mcolors
import pytest

import troutpy as tp


def test_get_palette_default():
    palette = tp.pl.get_palette()

    assert len(palette) == 31
    assert all(c.startswith("#") for c in palette)


def test_get_palette_named():
    assert tp.pl.get_palette("troutpy") == ["#123755", "#0FCCAC"]


def test_get_palette_n_colors():
    assert tp.pl.get_palette("troutpy", n_colors=1) == ["#123755"]


def test_get_palette_seaborn():
    palette = tp.pl.get_palette("viridis", n_colors=4)

    assert len(palette) == 4
    assert all(c.startswith("#") for c in palette)


def test_get_palette_unknown_raises():
    with pytest.raises(ValueError, match="not found"):
        tp.pl.get_palette("not-a-palette")


def test_get_colormap_default():
    cmap = tp.pl.get_colormap()

    assert isinstance(cmap, mcolors.Colormap)


def test_get_colormap_mpl_native():
    cmap = tp.pl.get_colormap("viridis")

    assert cmap.name == "viridis"


def test_get_colormap_unknown_raises():
    with pytest.raises(ValueError, match="could not be resolved"):
        tp.pl.get_colormap("not-a-colormap")
