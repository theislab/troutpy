import troutpy as tp


def test_compute_target_score(sdata):
    tp.tl.compute_target_score(sdata, layer="transcripts", gene_key="gene", celltype_key="leiden", k_neighbors=10, copy=False)

    assert "target_score" in sdata.tables
    obs = sdata["target_score"].obs
    assert {"gene", "distance", "closest_cell", "closest_cell_type"} <= set(obs.columns)


def test_calculate_target_cells(sdata):
    tp.tl.calculate_target_cells(sdata, celltype_key="leiden", copy=False)

    transcripts = sdata["transcripts"].compute()
    assert {"closest_target_cell", "closest_target_cell_type", "distance_to_target_cell"} <= set(transcripts.columns)
    assert "target" in sdata["xrna_metadata"].varm
    assert sdata["xrna_metadata"].varm["target"].shape[0] == sdata["xrna_metadata"].n_vars


def test_define_target_by_celltype(sdata):
    tp.tl.calculate_target_cells(sdata, celltype_key="leiden", copy=False)

    result = tp.tl.define_target_by_celltype(sdata, closest_celltype_key="closest_target_cell_type", feature_key="gene")

    assert set(result.columns) <= {"TypeA", "TypeB", "TypeC"}
    assert result.shape[0] == sdata["xrna_metadata"].n_vars
    assert result.sum(axis=1).dropna().between(0.99, 1.01).all()
