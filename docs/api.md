# API

## Preprocessing

```{eval-rst}
.. module:: troutpy.pp
.. currentmodule:: troutpy

.. autosummary::
    :toctree: generated

    pp.compute_extracellular_counts
    pp.define_extracellular
    pp.compute_crosstab
    pp.format_adata
```

## Tools

```{eval-rst}
.. module:: troutpy.tl
.. currentmodule:: troutpy

.. autosummary::
    :toctree: generated

    tl.compute_source_cells
    tl.distance_to_source_cell
    tl.compute_distant_cells_prop
    tl.get_proportion_expressed_per_cell_type
    tl.calculate_target_cells
    tl.define_target_by_celltype
    tl.colocalization_proportion
    tl.spatial_variability
    tl.create_xrna_metadata
    tl.quantify_overexpression
    tl.extracellular_enrichment
    tl.spatial_colocalization
    tl.get_number_of_communication_genes
    tl.get_gene_interaction_strength
    tl.apply_nmf_to_adata
    tl.nmf
    tl.apply_exrna_factors_to_cells
    tl.segmentation_free_clustering
```

## Plotting

```{eval-rst}
.. module:: troutpy.pl
.. currentmodule:: troutpy

.. autosummary::
    :toctree: generated

    pl.sorted_heatmap
    pl.coupled_scatter
    pl.heatmap
    pl.plot_crosstab
    pl.pie_of_positive
    pl.genes_over_noise
    pl.moranI_histogram
    pl.proportion_above_threshold
    pl.nmf_factors_exrna_cells_W
    pl.nmf_gene_contributions
    pl.apply_exrnaH_to_cellular_to_create_cellularW
    pl.paired_nmf_factors
    pl.spatial_interactions
    pl.interactions_with_arrows
```
