import pytest

import troutpy as tp


def test_create_grid_squares(raw_sdata):
    shapes, centroids = tp.pp.create_grid_squares(raw_sdata, layer="transcripts", square_size=50)

    assert len(shapes) == len(centroids)
    assert centroids.shape[1] == 2


def test_aggregate_urna_knn(raw_sdata):
    tp.pp.define_urna(raw_sdata, layer="transcripts", method="cells", copy=False)

    tp.pp.aggregate_urna(raw_sdata, layer="transcripts", gene_key="gene", method="knn", knn_k=5, extracellular_only=True, copy=False)

    assert "segmentation_free_table" in raw_sdata.tables
    adata = raw_sdata["segmentation_free_table"]
    assert adata.n_obs > 0
    assert "spatial" in adata.obsm


@pytest.mark.xfail(
    strict=True,
    reason="aggregate_urna(method='bin') calls sdata.aggregate(..., agg_func='count'), which raises "
    "KeyError: 'spatialdata_attrs' in spatialdata 0.4.0's relational_query._locate_value "
    "(version incompatibility, not a fixture issue)",
)
def test_aggregate_urna_bin(raw_sdata):
    tp.pp.define_urna(raw_sdata, layer="transcripts", method="cells", copy=False)

    tp.pp.aggregate_urna(raw_sdata, layer="transcripts", gene_key="gene", method="bin", square_size=50, extracellular_only=True, copy=False)

    assert "segmentation_free_table" in raw_sdata.tables
