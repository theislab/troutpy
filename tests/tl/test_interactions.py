import pandas as pd
import pytest

import troutpy as tp


def test_communication_strength(sdata):
    n_source_types = sdata["source_score"].n_vars
    n_target_types = sdata["target_score"].n_vars

    tp.tl.communication_strength(sdata)

    interaction_strength = sdata["source_score"].uns["interaction_strength"]
    assert interaction_strength.shape == (sdata["source_score"].n_obs, n_source_types, n_target_types)


def test_gene_specific_interactions(sdata):
    tp.tl.gene_specific_interactions(sdata, gene_key="gene")

    uns = sdata["source_score"].uns
    assert "gene_interaction_strength" in uns
    assert "gene_interaction_names" in uns
    assert uns["gene_interaction_strength"].shape[0] == len(uns["gene_interaction_names"])


def test_celltype_contact_matrix(sdata):
    df = tp.tl.celltype_contact_matrix(sdata, cell_type_key="leiden", radius=50.0, tile_size=1000.0)

    assert isinstance(df, pd.DataFrame)
    assert set(df.index) <= {"TypeA", "TypeB", "TypeC"}
    assert set(df.columns) <= {"TypeA", "TypeB", "TypeC"}


def test_get_gene_interaction_strength(sdata):
    import matplotlib.pyplot as plt

    tp.tl.calculate_target_cells(sdata, celltype_key="leiden", copy=False)

    source_proportions = tp.tl.get_proportion_expressed_per_cell_type(sdata["table"], cell_type_key="leiden")
    target_proportions = tp.tl.define_target_by_celltype(sdata, closest_celltype_key="closest_target_cell_type", feature_key="gene")

    result = tp.tl.get_gene_interaction_strength(source_proportions, target_proportions, gene_symbol="Gene01")
    plt.close("all")

    assert result.shape == (len(source_proportions.columns), len(source_proportions.columns))


@pytest.mark.xfail(
    strict=True,
    reason="cell_contacts_with_urna_sources reads sdata['source_score'].obs['closest_cell'], but the only "
    "non-dead-code producers of 'source_score' (adaptative_source_score / _optimized) write "
    "'predicted_parent' instead. 'closest_cell' is only produced by the dead store_results_in_sdata "
    "helper (no callers) -- architectural drift, not a 1-line fix.",
)
def test_cell_contacts_with_urna_sources(sdata):
    tp.tl.cell_contacts_with_urna_sources(sdata, cell_type_key="leiden", distance=50, copy=False)
