import matplotlib.pyplot as plt
import pytest

import troutpy as tp


def test_rank_factor_genes_loadings(sdata):
    n_factors = sdata["segmentation_free_table"].varm["gene_loadings"].shape[1]

    axs = tp.pl.rank_factor_genes_loadings(sdata, layer="segmentation_free_table", n_genes=5, show=False)
    plt.close("all")

    assert len(axs) == n_factors


def test_rank_factor_genes_loadings_matrixplot(sdata):
    ax = tp.pl.rank_factor_genes_loadings_matrixplot(sdata, layer="segmentation_free_table", n_genes=3, cmap="viridis", show=False)
    plt.close("all")

    assert ax is not None


def test_factors_in_cells_matrixplot(sdata):
    tp.pl.factors_in_cells(sdata, layer="table", method="matrixplot", celltype_key="leiden", show=False)
    plt.close("all")


def test_factors_in_cells_unsupported_method_raises(sdata):
    with pytest.raises(ValueError, match="Unsupported method"):
        tp.pl.factors_in_cells(sdata, layer="table", method="not-a-method", celltype_key="leiden")
