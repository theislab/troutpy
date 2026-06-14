import pytest

import troutpy as tp


def test_density_similarity(raw_sdata):
    tp.tl.density_similarity(raw_sdata, radius=10.0)

    transcripts = raw_sdata["transcripts"].compute()
    assert {"enrichment_class", "density_similarity_score", "enrichment_over_random", "local_count_in_radius"} <= set(transcripts.columns)
    assert set(transcripts["enrichment_class"].unique()) <= {"Intracellular", "Background", "Moderate Density", "High Density"}


def test_calculate_heuristic_radius_by_cells(raw_sdata):
    radius = tp.tl.calculate_heuristic_radius_by_cells(raw_sdata, k=5, n_cells=20)

    assert radius > 0


def test_identify_density_k_neighbors(raw_sdata):
    import matplotlib.pyplot as plt

    selected_k = tp.tl.identify_density_k_neighbors(raw_sdata, k_range=range(3, 9, 2), crop_size=200, d_threshold=0.5)
    plt.close("all")

    assert selected_k in range(3, 9, 2)


def test_segment_protrusions(sdata):
    transcripts = sdata["transcripts"].compute()
    assert "structure_id" in transcripts.columns

    obs = sdata["structure_table"].obs
    assert {"predicted_parent", "parent_score", "neighborhood_score", "assignment_confidence", "is_ambiguous"} <= set(obs.columns)


@pytest.mark.skip(
    reason="deprecated; requires an 'extracellular_transcripts_enriched' points layer with "
    "feature_name/bin_id columns, not produced by the synthetic fixture"
)
def test_colocalization_proportion(sdata, tmp_path):
    tp.tl.colocalization_proportion(sdata, outpath=str(tmp_path), save=False)
