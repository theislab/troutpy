import troutpy as tp


def test_get_proportion_expressed_per_cell_type(sdata):
    proportions = tp.tl.get_proportion_expressed_per_cell_type(sdata["table"], cell_type_key="leiden")

    assert set(proportions.columns) == set(sdata["table"].obs["leiden"].unique())
    assert list(proportions.index) == list(sdata["table"].var_names)


def test_compute_contribution_score(sdata):
    tp.tl.compute_contribution_score(sdata)

    obs = sdata["table"].obs
    assert {"urna_contribution_score", "normalized_urna_contribution_score"} <= set(obs.columns)


def test_adaptative_source_score(sdata):
    tp.tl.adaptative_source_score(sdata, cell_type_col="leiden")

    assert "source_score" in sdata.tables
    source_obs = sdata["source_score"].obs
    assert {"gene", "predicted_parent", "distance_to_source", "assignment_score"} <= set(source_obs.columns)
    assert {"urna_source_score", "normalized_urna_source_score"} <= set(sdata["table"].obs.columns)
    assert {"aggregated_source_score", "mean_assignment_score"} <= set(sdata["xrna_metadata"].var.columns)


def test_adaptative_source_score_optimized(sdata):
    tp.tl.adaptative_source_score_optimized(sdata, cell_type_col="leiden", max_k=5)

    assert "source_score" in sdata.tables
    source_obs = sdata["source_score"].obs
    assert {"gene", "predicted_parent", "distance_to_source", "assignment_score"} <= set(source_obs.columns)
    assert "urna_source_score" in sdata["table"].obs.columns
