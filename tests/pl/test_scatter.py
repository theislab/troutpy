import matplotlib.pyplot as plt
import pytest

import troutpy as tp


def test_spatial_inout_expression(sdata):
    tp.pl.spatial_inout_expression(sdata, gene="Gene05", save=False)
    plt.close("all")


def test_spatial_inout_expression_unknown_gene_raises(sdata):
    with pytest.raises(ValueError, match="not found in both"):
        tp.pl.spatial_inout_expression(sdata, gene="not-a-gene")


def test_point_metric_comparison(sdata):
    g = tp.pl.point_metric_comparison(sdata, x_col="x", y_col="y", hue_col="enrichment_class")
    plt.close("all")

    assert g is not None


def test_point_metric_comparison_unknown_points_key_raises(sdata):
    with pytest.raises(KeyError):
        tp.pl.point_metric_comparison(sdata, points_key="not-a-key")


def test_point_metric_comparison_empty_gene_list_raises(sdata):
    with pytest.raises(ValueError, match="empty after filtering"):
        tp.pl.point_metric_comparison(sdata, x_col="x", y_col="y", hue_col="enrichment_class", gene_list=["not-a-gene"])


def test_urna_vs_source_score(sdata):
    tp.pl.urna_vs_source_score(sdata, lfc_thresh=-100)
    plt.close("all")


def test_urna_vs_source_score_unknown_yvar_raises(sdata):
    with pytest.raises(ValueError, match="not found in the combined metadata"):
        tp.pl.urna_vs_source_score(sdata, y_var="not-a-column")


@pytest.mark.xfail(
    strict=True,
    reason="diffusion_results does `var_df['control_probe'].map(...)`, but sdata['xrna_metadata'].var has no "
    "'control_probe' column -- quantify_overexpression only writes 'is_control'. Same naming drift as "
    "pl.metric_analysis.top_bottom_probes/gene_metric_heatmap.",
)
def test_diffusion_results(sdata):
    tp.pl.diffusion_results(sdata, save=False)


@pytest.mark.skip(reason="requires sdata.shapes['cell_boundaries'] (cell segmentation polygons), not present in the synthetic fixture")
def test_spatial_transcripts(sdata):
    tp.pl.spatial_transcripts(sdata)


@pytest.mark.skip(reason="requires sdata.shapes['cell_boundaries'] (cell segmentation polygons), not present in the synthetic fixture")
def test_spatial_transcripts_source(sdata):
    tp.pl.spatial_transcripts_source(sdata)
