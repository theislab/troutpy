import copy

import pytest

import troutpy as tp


@pytest.fixture
def segfree_sdata(raw_sdata):
    tp.pp.define_urna(raw_sdata, layer="transcripts", method="cells", copy=False)
    tp.pp.aggregate_urna(raw_sdata, layer="transcripts", gene_key="gene", method="knn", knn_k=5, extracellular_only=True, copy=False)
    return raw_sdata


@pytest.mark.parametrize("method", ["NMF", "LDA"])
def test_latent_factor(segfree_sdata, method):
    sdata_copy = copy.deepcopy(segfree_sdata)
    tp.tl.latent_factor(sdata_copy, method=method, layer="segmentation_free_table", n_components=3, random_state=0)

    adata = sdata_copy["segmentation_free_table"]
    assert adata.obsm["cell_loadings"].shape == (adata.n_obs, 3)
    assert adata.varm["gene_loadings"].shape == (adata.n_vars, 3)


def test_latent_factor_unsupported_method_raises(segfree_sdata):
    with pytest.raises(ValueError, match="Unsupported method"):
        tp.tl.latent_factor(segfree_sdata, method="not-a-method", layer="segmentation_free_table", n_components=3)


def test_factors_to_cells(sdata):
    tp.tl.factors_to_cells(sdata, extracellular_layer="segmentation_free_table", cellular_layer="table")

    adata = sdata["table"]
    assert "factors_cell_loadings" in adata.obsm
    assert adata.obsm["factors_cell_loadings"].shape[0] == adata.n_obs
