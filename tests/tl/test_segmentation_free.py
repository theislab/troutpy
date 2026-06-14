import pytest

import troutpy as tp

pytest.importorskip("points2regions")


def test_segmentation_free_clustering(raw_sdata):
    params = {"num_clusters": 3, "pixel_width": 5, "pixel_smoothing": 5}

    tp.tl.segmentation_free_clustering(
        raw_sdata, params=params, layer="transcripts", gene_key="feature_name", transcript_id_key="transcript_id", copy=False
    )

    assert "segmentation_free_table" in raw_sdata.tables
    transcripts = raw_sdata["transcripts"].compute()
    assert {"segmentation_free_clusters", "bin_id"} <= set(transcripts.columns)


def test_segmentation_free_clustering_missing_param_raises(raw_sdata):
    with pytest.raises(ValueError, match="Missing required parameter"):
        tp.tl.segmentation_free_clustering(
            raw_sdata,
            params={"num_clusters": 3},
            layer="transcripts",
            gene_key="feature_name",
            transcript_id_key="transcript_id",
            copy=False,
        )


def test_segmentation_free_clustering_unknown_method_raises(raw_sdata):
    with pytest.raises(ValueError, match="Unknown method"):
        tp.tl.segmentation_free_clustering(
            raw_sdata,
            params={"num_clusters": 3, "pixel_width": 5, "pixel_smoothing": 5},
            layer="transcripts",
            gene_key="feature_name",
            transcript_id_key="transcript_id",
            method="not-a-method",
            copy=False,
        )
