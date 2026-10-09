import matplotlib.pyplot as plt
import pytest

import troutpy as tp


def test_metric_scatter(sdata):
    tp.pl.metric_scatter(sdata, x="extracellular_proportion", y="moran_I", save=False)
    plt.close("all")


def test_gene_metric_heatmap_invalid_cluster_axis_raises(sdata):
    with pytest.raises(ValueError, match="Invalid cluster_axis"):
        tp.pl.gene_metric_heatmap(sdata, cluster_axis="not-an-axis")


def test_top_bottom_probes(sdata):
    tp.pl.top_bottom_probes(sdata, metric="moran_I", top_n=3, bottom_n=3, save=False)


def test_gene_metric_heatmap(sdata):
    tp.pl.gene_metric_heatmap(sdata, cluster_axis="none", save=False)


def test_logfoldratio_over_noise(sdata):
    tp.pl.logfoldratio_over_noise(sdata, save=False)
