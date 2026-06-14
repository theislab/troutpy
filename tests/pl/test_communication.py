import matplotlib.pyplot as plt
import pytest

import troutpy as tp


def test_gene_communication_heatmap(sdata):
    gene = sdata["source_score"].uns["gene_interaction_names"][0]

    tp.pl.gene_communication(sdata, kind="heatmap", gene=gene, celltype_key="leiden")
    plt.close("all")


def test_gene_communication_unknown_gene_raises(sdata):
    with pytest.raises(KeyError, match="Gene name not found"):
        tp.pl.gene_communication(sdata, gene="not-a-gene", celltype_key="leiden")


def test_gene_communication_invalid_kind_raises(sdata):
    gene = sdata["source_score"].uns["gene_interaction_names"][0]

    with pytest.raises(ValueError, match="Invalid plot type"):
        tp.pl.gene_communication(sdata, kind="not-a-kind", gene=gene, celltype_key="leiden")


def test_target_score_by_celltype(sdata):
    tp.pl.target_score_by_celltype(sdata, gene_key="gene", min_counts=0, cluster_axis="none")
    plt.close("all")


def test_target_score_by_celltype_invalid_axis_raises(sdata):
    with pytest.raises(ValueError, match="Invalid cluster_axis"):
        tp.pl.target_score_by_celltype(sdata, gene_key="gene", min_counts=0, cluster_axis="not-an-axis")


def test_pl_celltype_contact_matrix(sdata):
    mat = tp.tl.celltype_contact_matrix(sdata, cell_type_key="leiden", radius=50.0, tile_size=1000.0)

    g = tp.pl.celltype_contact_matrix(mat, sdata=None, cell_type_key="leiden")
    plt.close("all")

    assert g is not None


def test_pl_celltype_contact_matrix_invalid_kind_raises(sdata):
    mat = tp.tl.celltype_contact_matrix(sdata, cell_type_key="leiden", radius=50.0, tile_size=1000.0)

    with pytest.raises(ValueError, match="Invalid kind"):
        tp.pl.celltype_contact_matrix(mat, sdata=None, kind="not-a-kind", cell_type_key="leiden")


@pytest.mark.skip(
    reason="cell_type_contacts reads sdata[table_key].uns['cell_contact_combined'], which is only produced by "
    "cell_contacts_with_urna_sources -- xfail'd in tests/tl/test_interactions.py due to architectural drift, so "
    "it is never present in the synthetic fixture"
)
def test_cell_type_contacts(sdata):
    tp.pl.cell_type_contacts(sdata, celltype_key="leiden", table_key="table")
