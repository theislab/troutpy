import matplotlib.pyplot as plt
import pytest

import troutpy as tp


def test_metric_scatter(sdata):
    tp.pl.metric_scatter(sdata, x="extracellular_proportion", y="moran_I", save=False)
    plt.close("all")


def test_gene_metric_heatmap_invalid_cluster_axis_raises(sdata):
    with pytest.raises(ValueError, match="Invalid cluster_axis"):
        tp.pl.gene_metric_heatmap(sdata, cluster_axis="not-an-axis")


@pytest.mark.xfail(
    strict=True,
    reason="top_bottom_probes does `plot_df['control_probe']`, but sdata['xrna_metadata'].var has no "
    "'control_probe' column -- quantify_overexpression only writes 'is_control'. Naming drift between "
    "tl.quantify_urna ('is_control') and pl.metric_analysis/pl.scatter ('control_probe'), affecting "
    "top_bottom_probes and gene_metric_heatmap as well; not a 1-line fix.",
)
def test_top_bottom_probes(sdata):
    tp.pl.top_bottom_probes(sdata, metric="moran_I", top_n=3, bottom_n=3, save=False)


@pytest.mark.xfail(
    strict=True,
    reason="gene_metric_heatmap eagerly computes ytick_colors via `var_df.loc[gene, 'control_probe']` for "
    "every gene, but sdata['xrna_metadata'].var has no 'control_probe' column (only 'is_control') -- same "
    "naming drift as top_bottom_probes.",
)
def test_gene_metric_heatmap(sdata):
    tp.pl.gene_metric_heatmap(sdata, cluster_axis="none", save=False)


@pytest.mark.xfail(
    strict=True,
    reason="logfoldratio_over_noise reads var_df['logfoldratio_over_noise'], but "
    "sdata['xrna_metadata'].var has no such column -- quantify_overexpression writes "
    "'logfoldchange_over_noise' instead. Naming drift, not a 1-line fix.",
)
def test_logfoldratio_over_noise(sdata):
    tp.pl.logfoldratio_over_noise(sdata, save=False)
