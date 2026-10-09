# Tools: `tl`

```{eval-rst}
.. module:: troutpy.tl
```

```{eval-rst}
.. currentmodule:: troutpy
```

## Local density

```{eval-rst}
.. autosummary::
    :toctree: generated/density

    tl.density_similarity
    tl.identify_density_k_neighbors
    tl.calculate_heuristic_radius_by_cells
    tl.colocalization_proportion
    tl.segment_protrusions
```

## uRNA quantification

```{eval-rst}
.. autosummary::
    :toctree: generated/quantification

    tl.spatial_variability
    tl.create_urna_metadata
    tl.quantify_overexpression
    tl.extracellular_enrichment
    tl.spatial_colocalization
    tl.in_out_correlation
    tl.compare_intra_extra_distribution
    tl.segmentation_free_clustering
    tl.get_proportion_expressed_per_cell_type

```

## Source, target and communication

```{eval-rst}
.. autosummary::
    :toctree: generated/communication

    tl.adaptative_source_score
    tl.adaptative_source_score_optimized
    tl.calculate_target_cells
    tl.compute_target_score
    tl.define_target_by_celltype
    tl.cluster_distribution_from_source
    tl.cell_contacts_with_urna_sources
    tl.celltype_contact_matrix
    tl.get_gene_interaction_strength
    tl.communication_strength
    tl.gene_specific_interactions
```

## Spillover and decontamination

Correct segmented-cell expression for transcripts that spilled over from
neighbouring cells (e.g. via protrusions) into another cell's mask.

```{eval-rst}
.. autosummary::
    :toctree: generated/decontamination

    tl.flag_intracellular_spillover
    tl.decontaminate_cell_expression
    tl.credit_spillover_to_source
    tl.subtract_local_urna_background
    tl.generalized_source_score
    tl.apply_soft_decontamination
    tl.iterative_soft_decontamination
```

## Cell scores

```{eval-rst}
.. autosummary::
    :toctree: generated/cell_scores

    tl.compute_contribution_score
```

## Factor analysis

```{eval-rst}
.. autosummary::
    :toctree: generated/factors

    tl.factors_to_cells
    tl.latent_factor
```

## Diffusion

```{eval-rst}
.. autosummary::
    :toctree: generated/diffusion

    tl.assess_diffusion
    tl.compute_js_divergence

```

## Multimodal quantification

```{eval-rst}
.. autosummary::
    :toctree: generated/multimodal

    tl.image_intensities_per_transcript
```
