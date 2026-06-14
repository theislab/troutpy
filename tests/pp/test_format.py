import pytest

import troutpy as tp


@pytest.mark.skip(reason="requires real Xenium output files on disk (cell_feature_matrix.h5, cells.parquet, transcripts.parquet)")
def test_format_adata():
    tp.pp.format_adata("/path/to/xenium/output", "/tmp/troutpy_format_adata_out", xlimits=[0, 100], ylimits=[0, 100])
