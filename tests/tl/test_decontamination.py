import numpy as np
import pandas as pd
from spatialdata.models import PointsModel

import troutpy as tp


def _add_spillover_columns(sdata, n_spill=5):
    """Mimic segmentation_free_sainsc output: mark `n_spill` intracellular transcripts as foreign-looking."""
    transcripts = sdata.points["transcripts"].compute()
    own_type = dict(zip(sdata["table"].obs["cell_id"].astype(str), sdata["table"].obs["leiden"].astype(str), strict=False))
    types = sorted(set(own_type.values()))

    transcripts["prob_is_urna"] = 0.0
    transcripts["closest_cell_type"] = transcripts["cell_id"].astype(str).map(own_type).fillna(types[0])
    spill_idx = transcripts.index[transcripts["overlaps_cell"]][:n_spill]
    transcripts.loc[spill_idx, "prob_is_urna"] = 0.9
    transcripts.loc[spill_idx, "closest_cell_type"] = [
        next(t for t in types if t != own_type[c]) for c in transcripts.loc[spill_idx, "cell_id"].astype(str)
    ]

    sdata.points["transcripts"] = PointsModel.parse(transcripts, coordinates={"x": "x", "y": "y"})
    return n_spill


def test_flag_intracellular_spillover(sdata):
    n_spill = _add_spillover_columns(sdata)

    spillover = tp.tl.flag_intracellular_spillover(sdata, cell_type_col="leiden")

    assert {"gene", "own_cell_id", "own_cell_type", "predicted_true_type", "prob_is_urna", "is_spillover"} <= set(spillover.columns)
    assert spillover["is_spillover"].sum() == n_spill


def test_decontaminate_cell_expression(sdata):
    n_spill = _add_spillover_columns(sdata)
    spillover = tp.tl.flag_intracellular_spillover(sdata, cell_type_col="leiden")

    tp.tl.decontaminate_cell_expression(sdata, spillover, layer_key="raw", output_layer="decontaminated")

    adata = sdata["table"]
    assert "decontaminated" in adata.layers
    assert np.asarray(adata.layers["raw"]).sum() - adata.layers["decontaminated"].sum() == n_spill


def test_credit_spillover_to_source():
    extra = pd.DataFrame(0, index=["g1", "g2"], columns=["A", "B"])
    spillover = pd.DataFrame(
        {
            "gene": pd.Categorical(["g1", "g1", "g2"]),
            "predicted_true_type": pd.Categorical(["0", "0", "1"]),
            "is_spillover": [True, True, False],
        }
    )

    corrected = tp.tl.credit_spillover_to_source(extra, spillover, leiden_to_celltype={"0": "A", "1": "B"})

    assert corrected.loc["g1", "A"] == 2
    assert corrected.to_numpy().sum() == 2
    assert extra.to_numpy().sum() == 0


def test_subtract_local_urna_background(sdata):
    tp.tl.subtract_local_urna_background(sdata, radius=50, inner_radius=5, cell_area_col=None)

    adata = sdata["table"]
    assert "decontaminated_density" in adata.layers
    corrected = adata.layers["decontaminated_density"]
    assert corrected.min() >= 0
    assert corrected.sum() <= np.asarray(adata.layers["raw"]).sum()


def test_generalized_source_score_and_apply(sdata):
    tp.tl.generalized_source_score(sdata, cell_type_col="leiden")

    contamination = sdata["intracellular_contamination"]
    assert {"cell_id", "gene", "best_other_cell_id"} <= set(contamination.obs.columns)
    assert {"own_weight", "contamination_weight", "best_other_weight"} <= set(contamination.var_names)

    tp.tl.apply_soft_decontamination(sdata, raw_layer="raw", output_layer="decontaminated")
    assert sdata["table"].layers["decontaminated"].min() >= 0


def test_iterative_soft_decontamination(sdata):
    history = tp.tl.iterative_soft_decontamination(
        sdata, n_iter=2, raw_layer="raw", score_kwargs={"cell_type_col": "leiden"}, extracellular_score_kwargs={"cell_type_col": "leiden", "max_k": 5}
    )

    assert list(history.columns) == ["iteration", "relative_change", "rank_correlation"]
    assert 1 <= len(history) <= 2
    assert "decontaminated" in sdata["table"].layers
