import numpy as np
import pytest

import troutpy as tp


def test_create_urna_metadata(raw_sdata):
    tp.tl.create_urna_metadata(raw_sdata, layer="transcripts", gene_key="gene")

    assert "xrna_metadata" in raw_sdata.tables
    assert raw_sdata["xrna_metadata"].n_vars == 12


def test_extracellular_enrichment(sdata):
    tp.tl.extracellular_enrichment(sdata, gene_key="gene", layer="transcripts")

    var = sdata["xrna_metadata"].var
    assert {"intracellular_proportion", "extracellular_proportion", "logfoldratio_extracellular"} <= set(var.columns)
    assert var["intracellular_proportion"].between(0, 1).all()
    assert var["extracellular_proportion"].between(0, 1).all()


def test_quantify_overexpression(sdata):
    tp.tl.quantify_overexpression(
        sdata,
        codeword_key="codeword_category",
        control_codewords=["negative_control_probe"],
        gene_key="gene",
        layer="transcripts",
    )

    var = sdata["xrna_metadata"].var
    assert {"count", "logfoldchange_over_noise", "p_val_noise", "is_control", "fdr_noise"} <= set(var.columns)
    assert var.loc[["BLANK-001", "BLANK-002"], "is_control"].all()
    assert var["p_val_noise"].between(0, 1).all()


def test_in_out_correlation(sdata):
    tp.tl.in_out_correlation(sdata, extracellular_layer="segmentation_free_table", cellular_layer="table", n_neighbors=5)

    var = sdata["xrna_metadata"].var
    assert {"in_out_spearmanR", "in_out_pvalue"} <= set(var.columns)


def test_spatial_variability(sdata):
    tp.tl.spatial_variability(sdata, gene_key="gene", n_neighbors=5, kde_resolution=50, square_size=20)

    var = sdata["xrna_metadata"].var
    assert {"moran_I", "moran_pval_norm", "moran_var_norm", "moran_pval_norm_fdr_bh"} <= set(var.columns)
    assert var["moran_I"].between(-1, 1).all()


def test_spatial_colocalization(sdata):
    tp.tl.spatial_colocalization(sdata, gene_key="gene", resolution=50, square_size=20)

    var = sdata["xrna_metadata"].var
    assert "proportion_of_colocalized" in var.columns
    assert var["proportion_of_colocalized"].dropna().between(0, 1).all()


def test_compare_intra_extra_distribution(sdata):
    tp.tl.compare_intra_extra_distribution(sdata, layer="transcripts", gene_key="gene", n_bins=10)

    var = sdata["xrna_metadata"].var
    assert {"centroid_shift_distance", "spatial_density_correlation", "spatial_js_divergence"} <= set(var.columns)


def test_assess_diffusion(sdata):
    tp.tl.assess_diffusion(sdata, gene_key="gene", distance_key="distance_to_source", min_transcripts=5)

    var = sdata["xrna_metadata"].var
    assert {"ks_stat", "ks_pval", "lr_stat", "mean_displacement", "n_transcripts", "sigma_est", "-log_ks_pval"} <= set(var.columns)


def test_cluster_distribution_from_source(sdata):
    tp.tl.cluster_distribution_from_source(sdata, gene_key="gene", distance_key="distance_to_source", n_clusters=3, n_bins=20)

    var = sdata["xrna_metadata"].var
    assert "kmeans_distribution" in var.columns


@pytest.mark.parametrize(
    ("p", "q", "expected"),
    [
        (np.array([0.2, 0.3, 0.5]), np.array([0.1, 0.4, 0.5]), 0.012078628383665548),
        (np.array([0.2, 0.3, 0.5]), np.array([0.2, 0.3, 0.5]), 0.0),
    ],
)
def test_compute_js_divergence(p, q, expected):
    assert tp.tl.compute_js_divergence(p, q) == pytest.approx(expected, abs=1e-6)
