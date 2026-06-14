import pytest

import troutpy as tp


@pytest.mark.skip(reason="requires a raw 10x Xenium-format SpatialData object (specific points/table column layout), not produced by the synthetic fixture")
def test_xenium_converter(sdata):
    tp.pp.xenium_converter(sdata, copy=False)


@pytest.mark.skip(
    reason="requires a raw multi-FOV CosMx-format SpatialData object (per-FOV images/labels/points and "
    "table.obs['fov']/['cell_ID']), not produced by the synthetic fixture"
)
def test_cosmx_converter(sdata):
    tp.pp.cosmx_converter(sdata, copy_data=False)
