import matplotlib.pyplot as plt
import pytest

import troutpy as tp


@pytest.mark.parametrize("how", ["full", "collapsed", "lineplot"])
def test_global_distribution_from_source(sdata, how):
    tp.pl.global_distribution_from_source(sdata, distance_key="distance_to_source", how=how)
    plt.close("all")


def test_global_distribution_from_source_unknown_cluster_key_raises(sdata):
    with pytest.raises(ValueError, match="not found in sdata"):
        tp.pl.global_distribution_from_source(sdata, cluster_key="not-a-column", distance_key="distance_to_source")


def test_global_distribution_from_source_invalid_how_raises(sdata):
    with pytest.raises(ValueError, match="Invalid value for 'how'"):
        tp.pl.global_distribution_from_source(sdata, distance_key="distance_to_source", how="not-a-how")


def test_distributions_by_cluster(sdata):
    tp.pl.distributions_by_cluster(sdata, distance_key="distance_to_source", n_bins=5)
    plt.close("all")


def test_gene_distribution_from_source(sdata):
    tp.pl.gene_distribution_from_source(sdata, cool_pattern=["Gene01", "Gene02"], distance_key="distance_to_source")
    plt.close("all")


def test_gene_distribution_from_source_no_valid_genes_raises(sdata):
    with pytest.raises(ValueError, match="No specified genes were found"):
        tp.pl.gene_distribution_from_source(sdata, cool_pattern=["not-a-gene"], distance_key="distance_to_source")


def test_source_score_by_celltype(sdata):
    tp.pl.source_score_by_celltype(sdata, gene_key="gene", min_counts=0, cluster_axis="none")
    plt.close("all")


def test_source_score_by_celltype_invalid_axis_raises(sdata):
    with pytest.raises(ValueError, match="Invalid cluster_axis"):
        tp.pl.source_score_by_celltype(sdata, gene_key="gene", min_counts=0, cluster_axis="not-an-axis")
