import pytest

import troutpy as tp


def test_define_urna_method_cells(raw_sdata):
    tp.pp.define_urna(raw_sdata, layer="transcripts", method="cells", copy=False)
    transcripts = raw_sdata["transcripts"].compute()

    assert "extracellular" in transcripts.columns
    assert (transcripts["extracellular"] == (transcripts["cell_id"] == "UNASSIGNED")).all()


def test_define_urna_method_nuclei(raw_sdata):
    tp.pp.define_urna(raw_sdata, layer="transcripts", method="nuclei", copy=False)
    transcripts = raw_sdata["transcripts"].compute()

    assert "extracellular" in transcripts.columns
    assert (transcripts["extracellular"] == (transcripts["overlaps_nucleus"] != 1)).all()


def test_define_urna_unknown_method_raises(raw_sdata):
    with pytest.raises(ValueError, match="Unknown method"):
        tp.pp.define_urna(raw_sdata, layer="transcripts", method="not-a-method", copy=False)


def test_segmentation_free_sainsc(raw_sdata):
    tp.pp.define_urna(raw_sdata, layer="transcripts", method="cells", copy=False)
    tp.tl.density_similarity(raw_sdata, radius=10.0)

    tp.pp.segmentation_free_sainsc(raw_sdata, binsize=5, celltype_key="leiden", n_threads=1, resolution=1000, copy=False)

    transcripts = raw_sdata["transcripts"].compute()
    assert {"closest_cell_type", "cosine_similarity", "prob_is_urna", "extracellular"} <= set(transcripts.columns)


def test_define_urna_probability(raw_sdata):
    tp.pp.define_urna(raw_sdata, layer="transcripts", method="cells", copy=False)
    tp.tl.density_similarity(raw_sdata, radius=10.0)
    tp.pp.segmentation_free_sainsc(raw_sdata, binsize=5, celltype_key="leiden", n_threads=1, resolution=1000, copy=False)

    tp.pp.define_urna_probability(raw_sdata, copy=False)

    transcripts = raw_sdata["transcripts"].compute()
    assert "prob_is_urna" in transcripts.columns
    assert "extracellular" in transcripts.columns
    assert transcripts["prob_is_urna"].dropna().between(0, 1).all()


def test_find_optimal_segmentation_free_bin_size(raw_sdata):
    import matplotlib.pyplot as plt

    tp.pp.define_urna(raw_sdata, layer="transcripts", method="cells", copy=False)

    results_df, metrics_df, optimal_bin = tp.pp.find_optimal_segmentation_free_bin_size(
        raw_sdata, bin_sizes=(4, 8), cell_type_key="leiden", roi_size=120
    )
    plt.close("all")

    assert {"bin_x", "bin_y", "cosine_sim", "is_cellular", "bin_size"} <= set(results_df.columns)
    assert set(metrics_df["bin_size"]) == {4, 8}
    assert optimal_bin in (4, 8)


def test_filter_urna_drops_control_probes(sdata):
    n_genes_before = sdata["xrna_metadata"].var.shape[0]
    n_transcripts_before = len(sdata["transcripts"])

    tp.pp.filter_urna(sdata, min_counts=1, max_p_val_noise=0.05, gene_key="gene", copy=False)

    var = sdata["xrna_metadata"].var
    assert var.shape[0] < n_genes_before
    assert "BLANK-001" not in var.index
    assert "BLANK-002" not in var.index
    assert len(sdata["transcripts"]) < n_transcripts_before


def test_filter_urna_copy_returns_new_sdata(sdata):
    import copy as copy_module

    original = copy_module.deepcopy(sdata)
    result = tp.pp.filter_urna(sdata, min_counts=1, gene_key="gene", copy=True)

    assert result is not None
    assert result is not sdata
    # original sdata is untouched
    assert sdata["xrna_metadata"].var.shape[0] == original["xrna_metadata"].var.shape[0]
    assert result["xrna_metadata"].var.shape[0] <= original["xrna_metadata"].var.shape[0]


def test_get_transcript_categories(sdata):
    cats = tp.pp.get_transcript_categories(sdata)

    expected_categories = {
        "Intracellular",
        "Cell-Like Connected",
        "Cell-Like Unconnected",
        "High-Density Connected",
        "High-Density Unconnected",
        "Noise Spectrum",
        "Diffusion Compatible",
        "Diffusion Incompatible",
    }
    assert set(cats.index) == expected_categories
    assert cats.sum() == len(sdata["transcripts"])


def test_compute_extracellular_counts(sdata):
    transcripts = sdata["transcripts"].compute()
    extracellular = transcripts[transcripts["extracellular"]]

    counts = tp.pp.compute_extracellular_counts(extracellular)

    assert {"observed", "expected", "fold_ratio", "codeword_category"} <= set(counts.columns)
    assert counts["observed"].sum() == len(extracellular)


@pytest.mark.skip(reason="requires a sdata.labels[...] segmentation-mask image, not present in the synthetic fixture")
def test_add_morphological_metrics(sdata):
    tp.pp.add_morphological_metrics(sdata, labels_key="cell_labels", copy=False)


@pytest.mark.skip(
    reason="requires an 'image_intensity_per_transcript' table (from tl.image_intensities_per_transcript) "
    "and a 'closest_cell_type' transcript column, neither present in the synthetic fixture"
)
def test_define_urna_probability_stainings(sdata):
    tp.pp.define_urna_probability_stainings(sdata, copy=False)
