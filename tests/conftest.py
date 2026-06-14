"""Shared fixtures: a small synthetic SpatialData object exercising the troutpy core pipeline.

The fixture is a 8x5 grid of 40 cells (3 cell types) with ~785 transcripts covering
12 genes (10 "real" genes + 2 negative-control probes), plus a tight extracellular
transcript cluster designed to be picked up by ``tl.segment_protrusions``.

``processed_sdata``/``sdata`` run the object through (most of) the core troutpy
pipeline (pp.define_urna -> tl.density_similarity -> uRNA metadata/quantification ->
pp.aggregate_urna -> factor analysis -> source/target scoring -> diffusion/structure
analysis -> spatial variability/colocalization -> interactions), so that downstream
tests can exercise functions against realistic, already-populated SpatialData objects
without re-running the whole pipeline themselves.
"""

import copy
import warnings

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from spatialdata.models import PointsModel, TableModel

import troutpy as tp

matplotlib.use("Agg")

N_COLS, N_ROWS = 8, 5
N_CELLS = N_COLS * N_ROWS
GENES = [f"Gene{i:02d}" for i in range(1, 11)]
CONTROL_PROBES = ["BLANK-001", "BLANK-002"]
ALL_GENES = GENES + CONTROL_PROBES


def _build_base_sdata():
    rng = np.random.default_rng(0)

    cell_xs, cell_ys = np.meshgrid(np.arange(N_COLS) * 30.0, np.arange(N_ROWS) * 30.0)
    cell_xs = cell_xs.ravel()
    cell_ys = cell_ys.ravel()
    cell_ids = [f"cell_{i}" for i in range(N_CELLS)]
    cell_types = np.array(["TypeA", "TypeB", "TypeC"])[np.arange(N_CELLS) % 3]

    # per-cell-type gene expression signature (probabilities over GENES)
    sig = {
        "TypeA": np.array([3, 3, 2, 2, 1, 1, 1, 1, 1, 1], dtype=float),
        "TypeB": np.array([1, 1, 1, 2, 3, 3, 2, 1, 1, 1], dtype=float),
        "TypeC": np.array([1, 1, 1, 1, 1, 1, 2, 2, 3, 3], dtype=float),
    }
    for k in sig:
        sig[k] = sig[k] / sig[k].sum()

    records = []
    tid = 0

    # intracellular transcripts, drawn per-cell from its cell-type signature
    n_intra_per_cell = 15
    for ci, (cx, cy, ct) in enumerate(zip(cell_xs, cell_ys, cell_types)):
        g = rng.choice(GENES, size=n_intra_per_cell, p=sig[ct])
        xs = cx + rng.normal(0, 1.5, size=n_intra_per_cell)
        ys = cy + rng.normal(0, 1.5, size=n_intra_per_cell)
        for j in range(n_intra_per_cell):
            records.append(
                dict(
                    transcript_id=tid,
                    x=xs[j],
                    y=ys[j],
                    gene=g[j],
                    feature_name=g[j],
                    cell_id=cell_ids[ci],
                    overlaps_cell=True,
                    overlaps_nucleus=bool(rng.random() < 0.5),
                    codeword_category="predesigned_gene",
                    control_probe=False,
                )
            )
            tid += 1

    # scattered extracellular transcripts across the whole field
    n_extra = 150
    xmin, xmax = cell_xs.min() - 5, cell_xs.max() + 5
    ymin, ymax = cell_ys.min() - 5, cell_ys.max() + 5
    g = rng.choice(GENES, size=n_extra)
    xs = rng.uniform(xmin, xmax, size=n_extra)
    ys = rng.uniform(ymin, ymax, size=n_extra)
    for j in range(n_extra):
        records.append(
            dict(
                transcript_id=tid,
                x=xs[j],
                y=ys[j],
                gene=g[j],
                feature_name=g[j],
                cell_id="UNASSIGNED",
                overlaps_cell=False,
                overlaps_nucleus=False,
                codeword_category="predesigned_gene",
                control_probe=False,
            )
        )
        tid += 1

    # negative-control probes, scattered extracellular
    n_ctrl = 20
    g = rng.choice(CONTROL_PROBES, size=n_ctrl)
    xs = rng.uniform(xmin, xmax, size=n_ctrl)
    ys = rng.uniform(ymin, ymax, size=n_ctrl)
    for j in range(n_ctrl):
        records.append(
            dict(
                transcript_id=tid,
                x=xs[j],
                y=ys[j],
                gene=g[j],
                feature_name=g[j],
                cell_id="UNASSIGNED",
                overlaps_cell=False,
                overlaps_nucleus=False,
                codeword_category="negative_control_probe",
                control_probe=True,
            )
        )
        tid += 1

    # tight extracellular "protrusion" cluster (for tl.segment_protrusions / DBSCAN)
    n_cluster = 15
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    xs = cx + rng.uniform(0, 0.5, size=n_cluster)
    ys = cy + rng.uniform(0, 0.5, size=n_cluster)
    for j in range(n_cluster):
        records.append(
            dict(
                transcript_id=tid,
                x=xs[j],
                y=ys[j],
                gene="Gene05",
                feature_name="Gene05",
                cell_id="UNASSIGNED",
                overlaps_cell=False,
                overlaps_nucleus=False,
                codeword_category="predesigned_gene",
                control_probe=False,
            )
        )
        tid += 1

    transcripts_df = pd.DataFrame.from_records(records)
    for col in ("gene", "feature_name", "cell_id", "codeword_category"):
        transcripts_df[col] = transcripts_df[col].astype("category")

    points = PointsModel.parse(transcripts_df, coordinates={"x": "x", "y": "y"})

    # cell x gene count matrix, built by summing intracellular transcripts per cell
    counts = np.zeros((N_CELLS, len(ALL_GENES)), dtype=np.float32)
    intra = transcripts_df[transcripts_df["cell_id"] != "UNASSIGNED"]
    gene_idx = {g: i for i, g in enumerate(ALL_GENES)}
    cell_idx = {c: i for i, c in enumerate(cell_ids)}
    for _, row in intra.iterrows():
        counts[cell_idx[row["cell_id"]], gene_idx[row["gene"]]] += 1

    # NOTE: deliberately no "cell type" (with a space) column -- spatialdata 0.4.0's
    # obs-name validation (validate_table_attr_keys) rejects names containing spaces
    # whenever a table is (re-)assigned via `sdata[key] = adata`, which several
    # troutpy functions do (e.g. factors_to_cells, adaptative_source_score). Several
    # functions default `celltype_key`/`cell_type_col`/`cell_type_key` to "cell type";
    # pass `"leiden"` explicitly for those in tests using this fixture.
    obs = pd.DataFrame({"cell_id": cell_ids, "leiden": pd.Categorical(cell_types)}, index=cell_ids)
    var = pd.DataFrame(index=ALL_GENES)
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.obsm["spatial"] = np.column_stack([cell_xs, cell_ys])
    adata.layers["raw"] = adata.X.copy()

    table = TableModel.parse(adata)

    return sd.SpatialData(points={"transcripts": points}, tables={"table": table})


def _build_processed_sdata(base):
    sdata = copy.deepcopy(base)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        tp.pp.define_urna(sdata, layer="transcripts", method="cells", copy=False)
        tp.tl.density_similarity(sdata, radius=10.0)

        tp.tl.create_urna_metadata(sdata, layer="transcripts", gene_key="gene")
        tp.tl.extracellular_enrichment(sdata, gene_key="gene", layer="transcripts")
        tp.tl.quantify_overexpression(
            sdata,
            codeword_key="codeword_category",
            control_codewords=["negative_control_probe"],
            gene_key="gene",
            layer="transcripts",
        )

        # method="bin" raises KeyError: 'spatialdata_attrs' under spatialdata 0.4.0
        # (sdata.aggregate(...) -> relational_query._locate_value); "knn" works.
        tp.pp.aggregate_urna(sdata, layer="transcripts", gene_key="gene", method="knn", knn_k=5, extracellular_only=True, copy=False)

        tp.tl.latent_factor(sdata, method="NMF", layer="segmentation_free_table", n_components=3, random_state=0)
        tp.tl.factors_to_cells(sdata, extracellular_layer="segmentation_free_table", cellular_layer="table")

        tp.tl.adaptative_source_score(sdata, cell_type_col="leiden")
        # default k_neighbors=50 > n_cells=40 -> "k must be <= number of training points"
        tp.tl.compute_target_score(sdata, layer="transcripts", gene_key="gene", celltype_key="leiden", k_neighbors=10)

        tp.tl.assess_diffusion(sdata, gene_key="gene", distance_key="distance_to_source", min_transcripts=5)
        tp.tl.cluster_distribution_from_source(sdata, gene_key="gene", distance_key="distance_to_source", n_clusters=3, n_bins=20)

        tp.tl.segment_protrusions(sdata, layer="transcripts")

        tp.tl.in_out_correlation(sdata, extracellular_layer="segmentation_free_table", cellular_layer="table", n_neighbors=5)
        tp.tl.spatial_variability(sdata, gene_key="gene", n_neighbors=5, kde_resolution=50, square_size=20)
        tp.tl.spatial_colocalization(sdata, gene_key="gene", resolution=50, square_size=20)
        tp.tl.compare_intra_extra_distribution(sdata, layer="transcripts", gene_key="gene", n_bins=10)

        tp.tl.communication_strength(sdata)
        tp.tl.gene_specific_interactions(sdata, gene_key="gene")

    return sdata


@pytest.fixture(scope="session")
def base_sdata():
    """A small synthetic, unprocessed SpatialData object (40 cells, ~785 transcripts, 12 genes)."""
    return _build_base_sdata()


@pytest.fixture
def raw_sdata(base_sdata):
    """Per-test deep copy of the raw (unprocessed) fixture, safe to mutate."""
    return copy.deepcopy(base_sdata)


@pytest.fixture(scope="session")
def processed_sdata(base_sdata):
    """Session-scoped SpatialData run through the troutpy core pipeline (built once)."""
    return _build_processed_sdata(base_sdata)


@pytest.fixture
def sdata(processed_sdata):
    """Per-test deep copy of the fully-processed fixture, safe to mutate."""
    return copy.deepcopy(processed_sdata)
