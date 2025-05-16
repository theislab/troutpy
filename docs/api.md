# API

## Preprocessing

```{eval-rst}
.. module:: troutpy.pp
.. currentmodule:: troutpy

.. autosummary::
    :toctree: generated

    pp.create_grid_squares
    pp.aggregate_extracellular_transcripts
    pp.compute_extracellular_counts
    pp.define_extracellular
    pp.filter_xrna
    pp.process_dataframe
    pp.segmentation_free_sainsc
    pp.xenium_converter
    pp.cosmx_converter
    pp.format_adata
```

## Tools

```{eval-rst}
.. module:: troutpy.tl
.. currentmodule:: troutpy

.. autosummary::
    :toctree: generated

    tl.colocalization_proportion
    tl.latent_factor
    tl.apply_exrna_factors_to_cells
    tl.image_intensities_per_transcript
    tl.get_gene_interaction_strength
    tl.compute_communication_strength
    tl.gene_specific_interactions
    tl.spatial_variability
    tl.create_xrna_metadata
    tl.quantify_overexpression
    tl.extracellular_enrichment
    tl.spatial_colocalization
    tl.in_out_correlation
    tl.assess_diffussion
    tl.cluster_distribution_from_source
    tl.compute_js_divergence
    tl.compare_intra_extra_distribution
    tl.segmentation_free_clustering
    tl.get_proportion_expressed_per_cell_type
    tl.compute_source_score
    tl.compute_target_score
```

## Plotting

```{eval-rst}
.. module:: troutpy.pl
.. currentmodule:: troutpy

.. autosummary::
    :toctree: generated

    pl.pie
    pl.crosstab
    pl.histogram
    pl.get_palette
    pl.get_colormap
    pl.celltype_communication
    pl.gene_communication
    pl.target_score_by_celltype
    pl.rank_factor_genes_loadings
    pl.rank_factor_genes_loadings_matrixplot
    pl.factors_in_cells
    pl.top_bottom_probes_of_metric
    pl.metric_scatter
    pl.logfoldratio_over_noise
    pl.gene_metric_heatmap
    pl.sorted_heatmap
    pl.coupled_scatter
    pl.heatmap
    pl.genes_over_noise
    pl.moranI_histogram
    pl.spatial_interactions
    pl.interactions_with_arrows
    pl.intra_extra_density
    pl.spatial_inout_expression
    pl.diffusion_results
    pl.spatial_transcripts
    pl.global_distribution_from_source
    pl.distributions_by_cluster
    pl.gene_distribution_from_source
    pl.source_score_by_celltype

```
