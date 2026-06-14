import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import troutpy as tp


def test_sorted_heatmap(tmp_path):
    df = pd.DataFrame(np.random.rand(4, 3), index=[f"feature{i}" for i in range(4)], columns=[f"celltype{i}" for i in range(3)])

    tp.pl.sorted_heatmap(df, output_path=str(tmp_path), save=False)
    plt.close("all")


def test_heatmap():
    df = pd.DataFrame(np.random.rand(4, 4))

    tp.pl.heatmap(df, save=False)
    plt.close("all")


def test_moranI_histogram():
    df = pd.DataFrame({"I": np.random.rand(20)})

    tp.pl.moranI_histogram(df, save=False)
    plt.close("all")


def test_moranI_histogram_missing_path_raises():
    df = pd.DataFrame({"I": np.random.rand(20)})

    with pytest.raises(ValueError, match="does not exist"):
        tp.pl.moranI_histogram(df, save=True, figures_path="/not/a/real/path")


def test_apply_exrnaH_to_cellular_to_create_cellularW():
    genes = ["Gene01", "Gene02", "Gene03", "Gene04"]
    n_factors = 2

    adata_extracellular = ad.AnnData(X=np.zeros((5, len(genes))), var=pd.DataFrame(index=genes))
    adata_extracellular.uns["H_nmf"] = np.random.rand(n_factors, len(genes))

    adata_cellular = ad.AnnData(X=np.random.rand(10, len(genes)), var=pd.DataFrame(index=genes))

    result = tp.pl.apply_exrnaH_to_cellular_to_create_cellularW(adata_extracellular, adata_cellular)

    assert result.obsm["factors"].shape == (10, n_factors)
    assert "NMF_factor_1" in result.obs.columns
    assert "NMF_factor_2" in result.obs.columns


def test_paired_nmf_factors(sdata):
    tp.pl.paired_nmf_factors(sdata, layer="segmentation_free_table", n_factors=3, save=False)
    plt.close("all")


def test_intra_extra_density(sdata):
    tp.pl.intra_extra_density(sdata, genes=["Gene01"], layer="transcripts", gene_key="feature_name")
    plt.close("all")


@pytest.mark.skip(
    reason="requires sdata.points['extracellular_transcripts'] (legacy layer name) with a 'distance_to_source_cell' "
    "column -- the current pipeline only produces a 'transcripts' points layer and does not write "
    "'distance_to_source_cell' to any points layer."
)
def test_coupled_scatter(sdata):
    tp.pl.coupled_scatter(sdata)


@pytest.mark.skip(
    reason="requires sdata.points['extracellular_transcripts'] (legacy layer name), not produced by the current "
    "pipeline (only 'transcripts' is present)."
)
def test_genes_over_noise(sdata):
    tp.pl.genes_over_noise(sdata, scores_by_genes=pd.DataFrame({"feature_name": [], "log_fold_ratio": []}))


@pytest.mark.skip(
    reason="requires sdata['nmf_data'].obsm['W_nmf'] with >= 20 columns (hardcoded `range(20)`), but the current "
    "factor-analysis pipeline (tl.latent_factor) writes obsm['cell_loadings']/varm['gene_loadings'] with "
    "n_components=3, not obsm['W_nmf'] -- legacy NMF key naming."
)
def test_nmf_factors_exrna_cells_W(sdata):
    tp.pl.nmf_factors_exrna_cells_W(sdata)


@pytest.mark.skip(
    reason="requires sdata['nmf_data'].uns['H_nmf'], but tl.latent_factor writes "
    "varm['gene_loadings']/obsm['cell_loadings'] instead -- same legacy NMF key naming as nmf_factors_exrna_cells_W."
)
def test_nmf_gene_contributions(sdata):
    tp.pl.nmf_gene_contributions(sdata)


@pytest.mark.skip(
    reason="requires sdata.points['extracellular_transcripts_enriched'] with 'closest_target_cell'/"
    "'closest_source_cell' columns and a default gene='Arc' not present in the 12-gene synthetic fixture -- this "
    "legacy points layer and its columns are not produced by the current pipeline."
)
def test_spatial_interactions(sdata):
    tp.pl.spatial_interactions(sdata)


@pytest.mark.skip(
    reason="requires sdata.points['extracellular_transcripts_enriched'] with 'closest_target_cell'/"
    "'closest_source_cell' columns and a default gene='Arc' not present in the 12-gene synthetic fixture -- same "
    "legacy points layer/columns issue as spatial_interactions."
)
def test_interactions_with_arrows(sdata):
    tp.pl.interactions_with_arrows(sdata)
