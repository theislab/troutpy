import matplotlib.pyplot as plt

import troutpy as tp


def test_pie_no_group(raw_sdata):
    tp.pl.pie(raw_sdata, groupby="codeword_category", save=False)
    plt.close("all")


def test_pie_group_key_save(raw_sdata, tmp_path):
    tp.pl.pie(raw_sdata, groupby="codeword_category", group_key="control_probe", save=True, figures_path=str(tmp_path))

    assert (tmp_path / "pie_codeword_category_by_control_probe.pdf").exists()


def test_crosstab_barh(raw_sdata):
    tp.pl.crosstab(raw_sdata, xvar="codeword_category", yvar="control_probe", kind="barh", save=False)
    plt.close("all")


def test_crosstab_heatmap_save(raw_sdata, tmp_path):
    tp.pl.crosstab(raw_sdata, xvar="codeword_category", yvar="control_probe", kind="heatmap", save=True, figures_path=str(tmp_path))

    assert (tmp_path / "heatmap_codeword_category_control_probe_normalized_pdf.pdf").exists()


def test_histogram_no_group(raw_sdata):
    tp.pl.histogram(raw_sdata, x="x", hue="codeword_category", save=False)
    plt.close("all")


def test_histogram_group_key_save(raw_sdata, tmp_path):
    tp.pl.histogram(raw_sdata, x="x", group_key="codeword_category", save=True, figures_path=str(tmp_path))

    assert (tmp_path / "histogram_x_by_codeword_category.pdf").exists()
