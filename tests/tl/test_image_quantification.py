import pytest

import troutpy as tp


@pytest.mark.skip(reason="requires an sdata.images[...] image layer (e.g. Xenium morphology image), not present in the synthetic fixture")
def test_image_intensities_per_transcript(sdata):
    tp.tl.image_intensities_per_transcript(sdata, image_key="morphology", scale="scale0", transcript_key="transcripts")
