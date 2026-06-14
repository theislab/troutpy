import troutpy


def test_package_has_version():
    assert troutpy.__version__ is not None


def test_processed_sdata_fixture(sdata):
    """Smoke-test the shared pipeline fixture used across the test suite."""
    assert "table" in sdata.tables
    assert "xrna_metadata" in sdata.tables
    assert "segmentation_free_table" in sdata.tables
    assert "source_score" in sdata.tables
    assert "target_score" in sdata.tables
    assert "structure_table" in sdata.tables

    transcripts = sdata["transcripts"].compute()
    assert "extracellular" in transcripts.columns
    assert "enrichment_class" in transcripts.columns
