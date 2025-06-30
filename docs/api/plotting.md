# API

## Basic plots

```{eval-rst}
.. module:: troutpy.pl
.. currentmodule:: troutpy

.. autosummary::
    :toctree: generated/basic_plots
    
    pl.crosstab
    pl.histogram
    pl.pie
    pl.coupled_scatter
    pl.heatmap
    pl.sorted_heatmap
    pl.spatial_transcripts
```

## uRNA quantification

```{eval-rst}
.. autosummary::
    :toctree: generated/quantification
    pl.gene_metric_heatmap
    pl.logfoldratio_over_noise
    pl.metric_scatter
    pl.top_bottom_probes
    pl.genes_over_noise
    pl.intra_extra_density
    pl.moranI_histogram
    pl.diffusion_results
    pl.spatial_inout_expression
    
```




## Source, target and communication

```{eval-rst}
.. autosummary::
    :toctree: generated/communication
    pl.celltype_communication
    pl.gene_communication
    pl.global_distribution_from_source
    pl.distributions_by_cluster
    pl.gene_distribution_from_source
    pl.source_score_by_celltype
    pl.target_score_by_celltype
    pl.interactions_with_arrows
    pl.spatial_interactions
    
```

## Factor analysis

```{eval-rst}
.. autosummary::
    :toctree: generated/factor_analysis
    pl.factors_in_cells
    pl.rank_factor_genes_loadings
    pl.rank_factor_genes_loadings_matrixplot
    pl.nmf_factors_exrna_cells_W
    pl.nmf_gene_contributions
    pl.paired_nmf_factors
    pl.apply_exrnaH_to_cellular_to_create_cellularW

    
```




## Colormaps & palettes

```{eval-rst}
.. autosummary::
    :toctree: generated/colormaps

    pl.get_colormap
    pl.get_palette
```



