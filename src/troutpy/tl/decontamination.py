"""Iterative, per-transcript soft decontamination for uRNA source scoring.

Builds on :func:`troutpy.tl.density_similarity` and
:func:`troutpy.tl.adaptative_source_score_optimized`, extending per-transcript
source scoring (which only covers extracellular/unassigned transcripts) to
intracellular transcripts.

Background: protrusions are long and thin, so some protrusion-derived
transcripts land inside a *neighboring* cell's segmentation mask instead of
empty space. Standard segmentation-based counting then silently attributes
them to the wrong cell, inflating that neighbor's raw counts -- which in turn
makes the neighbor look like a *more plausible* uRNA source to
``adaptative_source_score_optimized``'s expression-context matching (since
that matching reads the cell's current, contaminated ``.X``). This is a
compounding bias, not a static double-count: fixing it requires correcting
expression *and* re-scoring against the corrected expression, iteratively.

This module gives every transcript -- including intracellular ones -- a
continuous contamination weight (no hard threshold), derived from the same
per-transcript (not spatial-bin) context-matching idea already used for uRNA,
then iterates: decontaminate -> re-score against the corrected expression ->
decontaminate again -> ... until the correction stabilizes.

A second, simpler and independent correction is also provided here,
:func:`subtract_local_urna_background`: rather than identifying which
specific neighboring cell a contaminating transcript might have come from, it
just asks whether a cell's own intracellular count for a gene is
distinguishable from the density of that gene's genuinely-unassigned (uRNA)
transcripts found within a fixed radius of the cell -- i.e. is this count any
different from what ambient background at the locally-observed uRNA density
would produce anyway. The two corrections answer different questions and can
be used independently or in sequence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import spearmanr
from spatialdata import SpatialData
from tqdm import tqdm

from .estimate_density import density_similarity
from .source_cell import adaptative_source_score_optimized

__all__ = [
    "generalized_source_score",
    "apply_soft_decontamination",
    "iterative_soft_decontamination",
    "subtract_local_urna_background",
]


def _dense_bool_expression(x) -> np.ndarray:
    """Dense boolean (n_cells, n_genes) gene-presence matrix from an AnnData `.X`."""
    return (x.toarray() if issparse(x) else np.asarray(x)) > 0


def _build_cell_shell(assigned_df: pd.DataFrame, obs: pd.DataFrame, cell_type_col: str):
    """Build a per-cell boundary shell from each cell's extreme (min/max x/y) assigned transcripts.

    Fresh reimplementation of the same shell idea used internally by
    :func:`troutpy.tl.adaptative_source_score_optimized` -- that function has
    no standalone helper to call into, so this is deliberately separate code,
    not a modification of that file.
    """
    shell_idx = pd.concat(
        [
            assigned_df.groupby("cell_id")["x"].idxmin(),
            assigned_df.groupby("cell_id")["x"].idxmax(),
            assigned_df.groupby("cell_id")["y"].idxmin(),
            assigned_df.groupby("cell_id")["y"].idxmax(),
        ]
    ).unique()
    shell_df = assigned_df.loc[shell_idx].copy()

    cell_id_to_row = {cid: i for i, cid in enumerate(obs["cell_id"])}
    shell_coords = shell_df[["x", "y"]].values.astype(np.float64)
    shell_row = np.array([cell_id_to_row[c] for c in shell_df["cell_id"]], dtype=np.int64)
    return shell_coords, shell_row, cell_id_to_row


def generalized_source_score(
    sdata: SpatialData,
    max_dist: float = 100.0,
    lmbda: float = 0.1,
    max_k: int = 10,
    ambient_floor: float = 1.0,
    signal_threshold: float = 10.0,
    residual: float = 0.01,
    magnitude_weight: float = 0.85,
    cell_type_col: str = "leiden",
    layer: str = "transcripts",
    density_kwargs: dict | None = None,
    run_density_similarity: bool = True,
    copy: bool = False,
) -> SpatialData | None:
    """Score every intracellular transcript against nearby candidate cells, including its own.

    Generalizes the context-gene / distance-decay matching that
    :func:`troutpy.tl.adaptative_source_score_optimized` already applies only
    to extracellular (unassigned) transcripts, so intracellular transcripts
    also get a continuous measure of how well they fit their own containing
    cell vs. some other nearby cell -- the basis for detecting contamination
    from cell-protrusion spillover.

    Each candidate cell is scored by a blend of two signals:

    1. **Own-gene magnitude (primary).** How much of this specific gene's
       locally-visible expression (summed over the container plus every
       candidate cell within `max_dist`) sits in this candidate, i.e. a
       "share" of that gene's nearby mass. This is the dominant signal because
       it directly answers the question that matters: is *this gene*, in
       *this transcript*, more consistent with the container or with a
       neighbor? An earlier version of this scorer relied solely on k-NN
       neighbor-transcript context (below) and was empirically found to fail
       exactly on the contamination scenario this module targets: a handful
       of stray transcripts of a foreign gene embedded in a neighbor cell's
       own, much larger, population of genuinely-expressed transcripts get
       their local context diluted by the host's own abundant genes, masking
       the very signal meant to flag them.
    2. **k-NN neighbor-transcript context (secondary).** A context gene set
       built from a transcript's `k_adaptive` nearest-neighbor *transcripts*
       (by position, over all transcripts -- not a spatial grid, and not
       restricted to other uRNA), where `k_adaptive` scales with local
       ``enrichment_over_random`` (computed for every transcript via
       :func:`troutpy.tl.density_similarity` with ``process_all=True`` if not
       already present). Kept as a secondary, blended-in signal -- useful when
       the own-gene magnitude signal is weak or ambiguous (e.g. very low
       counts) -- rather than the sole detector.

    Candidate cells come from a per-cell boundary "shell" search within
    `max_dist`; the transcript's own containing cell is *always* included as a
    candidate at distance 0 as well, regardless of whether the shell search
    finds it -- otherwise a transcript deep inside a large cell could have its
    own cell's shell points farther than `max_dist` away, making a nearby
    smaller cell look spuriously closer purely due to that boundary-distance
    proxy.

    Extracellular transcripts are not touched here -- they are already handled
    by :func:`troutpy.tl.adaptative_source_score_optimized`.

    Parameters
    ----------
    sdata
        SpatialData object with a ``"table"`` AnnData (``.obs["cell_id"]``,
        `cell_type_col`, `.X` current expression) and a `layer` points table
        with ``"gene"``, ``"x"``, ``"y"``, ``"cell_id"`` columns.
    max_dist
        Maximum distance to search for candidate cells.
    lmbda
        Exponential distance-decay rate.
    max_k
        Maximum number of context neighbor transcripts.
    ambient_floor, signal_threshold
        ``enrichment_over_random`` thresholds controlling adaptive context
        size, matching the convention in
        :func:`troutpy.tl.adaptative_source_score_optimized`.
    residual
        Constant added to the weight normalization denominator. Defaults to
        ``0.01`` here, deliberately much smaller than the ``0.1`` used by
        :func:`troutpy.tl.adaptative_source_score_optimized`: that default
        represents genuine "truly unassigned" probability mass for
        extracellular transcripts, but an intracellular transcript is always
        attributed to *some* cell, so the same constant would act as a
        uniform contamination-weight floor applied to every transcript --
        small per transcript, but summing to a large, wrongly-directed
        correction when multiplied across a cell's much larger population of
        genuinely native transcripts (confirmed empirically: with
        ``residual=0.1`` a synthetic test cell lost more native signal to this
        floor than it gained back from real contamination credited to it).
    magnitude_weight
        Blend factor in ``[0, 1]`` between the own-gene magnitude signal and
        the k-NN context signal (``combined = magnitude_weight * gene_share +
        (1 - magnitude_weight) * context_match``). Defaults to ``0.85``,
        making own-gene magnitude the primary driver.
    cell_type_col
        Column in ``sdata["table"].obs`` with cell-type annotations (kept for
        parity with the existing scorer's signature; not otherwise used here
        since candidates are compared directly by cell, not by type).
    layer
        Points layer holding transcripts.
    density_kwargs
        Extra keyword arguments passed to :func:`troutpy.tl.density_similarity`.
    run_density_similarity
        If `True` (default), calls ``density_similarity(sdata, process_all=True,
        **density_kwargs)`` to ensure ``enrichment_over_random`` is available
        for every transcript. Set `False` to skip -- e.g. on repeat calls
        within an iterative loop, since this quantity depends only on
        transcript positions and does not change when `.X` is corrected.
    copy
        If `True`, return `sdata`; otherwise modify in place and return `None`.

    Returns
    -------
    spatialdata.SpatialData or None
        `sdata` with a new ``"intracellular_contamination"`` table added: one
        row per intracellular transcript, ``.obs`` holding ``cell_id``,
        ``gene``, ``best_other_cell_id``, and ``.X`` columns
        ``own_weight``, ``contamination_weight``, ``best_other_weight``.
    """
    density_kwargs = dict(density_kwargs or {})
    if run_density_similarity:
        density_similarity(sdata, process_all=True, **density_kwargs)

    cells = sdata["table"]
    obs = cells.obs.copy()
    obs["cell_id"] = obs["cell_id"].astype(str).str.strip()

    transcripts = sdata.points[layer].compute().reset_index(drop=True)
    transcripts["cell_id"] = transcripts["cell_id"].astype(str).str.strip()

    if "enrichment_over_random" not in transcripts.columns:
        raise KeyError(
            "Column 'enrichment_over_random' missing from transcripts. Run "
            "troutpy.tl.density_similarity(sdata, process_all=True) first, or "
            "leave run_density_similarity=True."
        )

    valid_ids = set(obs["cell_id"])
    is_assigned = transcripts["cell_id"].isin(valid_ids)
    assigned_df = transcripts[is_assigned].copy()
    n_transcripts = len(transcripts)

    if len(assigned_df) == 0:
        raise ValueError("No transcripts assigned to a valid cell_id; cannot build the cell shell.")

    print("Building per-cell boundary shell...")
    shell_coords, shell_row, cell_id_to_row = _build_cell_shell(assigned_df, obs, cell_type_col)
    shell_tree = cKDTree(shell_coords)

    gene_to_idx = {g: i for i, g in enumerate(cells.var_names)}
    transcript_gene_idx = transcripts["gene"].astype(str).map(gene_to_idx).fillna(-1).astype(int).values

    all_coords = transcripts[["x", "y"]].values.astype(np.float64)
    all_tree = cKDTree(all_coords)

    enrich = transcripts["enrichment_over_random"].fillna(0.0).values
    k_vals = np.clip(1 + (max_k - 1) * (enrich - ambient_floor) / (signal_threshold - ambient_floor), 1, max_k).astype(int)

    print("Querying k-nearest-neighbor transcripts for context genes...")
    query_k = min(max_k + 1, n_transcripts)
    _, neighbor_idx = all_tree.query(all_coords, k=query_k)
    if neighbor_idx.ndim == 1:
        neighbor_idx = neighbor_idx[:, None]
    neighbor_gene_idx = transcript_gene_idx[neighbor_idx]  # (n_transcripts, query_k)

    cell_expr_bool = _dense_bool_expression(cells.X)  # (n_cells, n_genes) -- depends on current .X
    cell_counts = cells.X.toarray().astype(float) if issparse(cells.X) else np.asarray(cells.X, dtype=float)  # magnitude, same dependency

    own_row_of_transcript = np.full(n_transcripts, -1, dtype=np.int64)
    is_assigned_arr = is_assigned.values
    assigned_pos = np.where(is_assigned_arr)[0]
    own_row_of_transcript[assigned_pos] = [cell_id_to_row[c] for c in transcripts.loc[assigned_pos, "cell_id"]]

    own_weight = np.full(n_transcripts, np.nan)
    best_other_row = np.full(n_transcripts, -1, dtype=np.int64)
    best_other_weight = np.zeros(n_transcripts)

    print(f"Scoring {len(assigned_pos)} intracellular transcripts against candidate cells...")
    for i in tqdm(assigned_pos, desc="Generalized source scoring"):
        p_coord = all_coords[i]
        own_row = own_row_of_transcript[i]
        own_gene_col = transcript_gene_idx[i]

        k = max(int(k_vals[i]), 1)
        ctx_idx = neighbor_gene_idx[i, 1 : k + 1]  # column 0 is the transcript itself (distance 0)
        ctx_idx = np.unique(ctx_idx[ctx_idx >= 0])
        n_ctx = len(ctx_idx)

        shell_hits = shell_tree.query_ball_point(p_coord, r=max_dist)
        cand_dist: dict[int, float] = {}
        for s_idx in shell_hits:
            r = int(shell_row[s_idx])
            d = float(np.hypot(*(shell_coords[s_idx] - p_coord)))
            if r not in cand_dist or d < cand_dist[r]:
                cand_dist[r] = d
        cand_dist[int(own_row)] = 0.0  # explicit container injection, at distance 0

        # Primary signal: this transcript's own gene's share of its locally-visible
        # mass (container + every candidate within max_dist) -- directly answers
        # "is this specific gene more consistent with the container or a neighbor,"
        # robust to context dilution by the host's own abundant, unrelated genes.
        if own_gene_col >= 0:
            gene_counts = {r: cell_counts[r, own_gene_col] for r in cand_dist}
            total_gene_mass = sum(gene_counts.values()) + 1e-9
            gene_share = {r: v / total_gene_mass for r, v in gene_counts.items()}
        else:
            gene_share = dict.fromkeys(cand_dist, 0.0)

        # Secondary signal: k-NN neighbor-transcript context match (unchanged idea).
        if n_ctx > 0:
            context_match = {r: cell_expr_bool[r, ctx_idx].sum() / n_ctx for r in cand_dist}
        else:
            context_match = dict.fromkeys(cand_dist, 0.0)

        weights = {}
        for r, d in cand_dist.items():
            combined = magnitude_weight * gene_share[r] + (1 - magnitude_weight) * context_match[r]
            weights[r] = combined * np.exp(-lmbda * d)

        total = sum(weights.values()) + residual
        norm_weights = {r: w / total for r, w in weights.items()}

        own_weight[i] = norm_weights.get(int(own_row), 0.0)
        others = {r: w for r, w in norm_weights.items() if r != own_row}
        if others:
            best_r = max(others, key=others.get)
            best_other_row[i] = best_r
            best_other_weight[i] = others[best_r]

    contamination_weight = 1.0 - own_weight
    row_to_cell_id = obs["cell_id"].values

    result = pd.DataFrame(
        {
            "cell_id": transcripts.loc[assigned_pos, "cell_id"].values,
            "gene": transcripts.loc[assigned_pos, "gene"].values,
            "own_weight": own_weight[assigned_pos],
            "contamination_weight": contamination_weight[assigned_pos],
            "best_other_cell_id": [row_to_cell_id[r] if r >= 0 else None for r in best_other_row[assigned_pos]],
            "best_other_weight": best_other_weight[assigned_pos],
        },
        index=assigned_pos,
    )

    numeric_cols = ["own_weight", "contamination_weight", "best_other_weight"]
    contam_adata = sc.AnnData(X=result[numeric_cols].to_numpy(dtype=float))
    contam_adata.var_names = numeric_cols
    contam_adata.obs["cell_id"] = result["cell_id"].values
    contam_adata.obs["gene"] = result["gene"].values
    contam_adata.obs["best_other_cell_id"] = result["best_other_cell_id"].values

    sdata.tables["intracellular_contamination"] = contam_adata

    print(
        f"Done: mean contamination_weight={np.nanmean(contamination_weight[assigned_pos]):.4f}, "
        f"median={np.nanmedian(contamination_weight[assigned_pos]):.4f}."
    )

    return sdata if copy else None


def apply_soft_decontamination(
    sdata: SpatialData,
    contamination_key: str = "intracellular_contamination",
    expr_key: str = "table",
    raw_layer: str = "raw",
    output_layer: str = "decontaminated",
    update_X: bool = True,
) -> SpatialData:
    """Remove contamination-weighted counts from each transcript's container cell, crediting them to the predicted true source.

    Uses the continuous ``contamination_weight`` / ``best_other_cell_id``
    output of :func:`generalized_source_score`. For each intracellular
    transcript, subtracts its contamination weight from its own cell's raw
    count for that gene, and adds the same weight to the best-matching
    alternative cell's count for that gene, if one was found within range;
    otherwise the mass is dropped, analogous to an ambient/background sink in
    droplet-scRNA-seq decontamination methods (e.g. SoupX). The *decision* of
    how much to move is fully continuous -- only the write-back to a discrete
    count matrix requires a floor at 0.

    Parameters
    ----------
    sdata
        SpatialData object containing `contamination_key` (output of
        :func:`generalized_source_score`) and `expr_key` (an AnnData with
        `raw_layer`).
    contamination_key
        Key of the per-transcript contamination table in ``sdata.tables``.
    expr_key
        Key of the AnnData table to correct.
    raw_layer
        Layer to copy and correct from.
    output_layer
        Name of the new corrected layer written to ``sdata[expr_key].layers``.
    update_X
        If `True` (default), also copy the corrected matrix into
        ``sdata[expr_key].X`` -- required for a subsequent round of an
        iterative loop to see the correction, since candidate-cell gene
        matching in :func:`generalized_source_score` and uRNA-to-source
        matching in :func:`troutpy.tl.adaptative_source_score_optimized` both
        read `.X`. ``.layers[raw_layer]`` is never modified.

    Returns
    -------
    spatialdata.SpatialData
        `sdata`, modified in place, with `output_layer` added.
    """
    adata = sdata[expr_key]
    contam = sdata.tables[contamination_key]

    raw = adata.layers[raw_layer]
    corrected = raw.toarray().astype(float) if issparse(raw) else np.array(raw, dtype=float, copy=True)

    cell_id_to_row = {cid: i for i, cid in enumerate(adata.obs["cell_id"].astype(str))}
    gene_to_col = {g: i for i, g in enumerate(adata.var_names)}

    own_cell_ids = contam.obs["cell_id"].values
    genes = contam.obs["gene"].values
    weights = contam.X[:, contam.var_names.get_loc("contamination_weight")]
    best_other = contam.obs["best_other_cell_id"].values

    n_removed, n_credited, n_skipped = 0.0, 0.0, 0
    for own_id, gene, w, other_id in zip(own_cell_ids, genes, weights, best_other, strict=False):
        if w <= 0:
            continue
        row = cell_id_to_row.get(str(own_id))
        col = gene_to_col.get(gene)
        if row is None or col is None:
            n_skipped += 1
            continue
        corrected[row, col] = max(corrected[row, col] - w, 0.0)
        n_removed += w

        if other_id is not None:
            other_row = cell_id_to_row.get(str(other_id))
            if other_row is not None:
                corrected[other_row, col] += w
                n_credited += w

    adata.layers[output_layer] = csr_matrix(corrected)
    if update_X:
        adata.X = adata.layers[output_layer]

    print(
        f"Decontaminated '{output_layer}': removed {n_removed:.1f}, credited {n_credited:.1f} counts (skipped {n_skipped} unresolved id/gene pairs)."
    )
    return sdata


def iterative_soft_decontamination(
    sdata: SpatialData,
    n_iter: int = 5,
    tol: float = 1e-3,
    raw_layer: str = "raw",
    output_layer: str = "decontaminated",
    score_kwargs: dict | None = None,
    extracellular_score_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Iteratively decontaminate segmented-cell expression, closing the source-score feedback loop.

    Each round:

    1. :func:`generalized_source_score` -- recompute per-transcript
       contamination weights against the *current* expression (skipping the
       ``enrichment_over_random`` recomputation after round 0, since it
       depends only on transcript positions, not expression).
    2. :func:`troutpy.tl.adaptative_source_score_optimized` -- re-run uRNA-to-source assignment against the *current*
       expression. This is the step that actually closes the feedback loop:
       a contaminated cell's inflated counts make it look like a more
       plausible uRNA source, so re-scoring against progressively cleaner
       expression is what stops that compounding bias, rather than a single
       one-shot cleanup pass.
    3. :func:`apply_soft_decontamination` -- write the correction.

    Parameters
    ----------
    sdata
        SpatialData object to correct in place.
    n_iter
        Maximum number of rounds.
    tol
        Stop early once ``||X_t - X_(t-1)||_1 / ||X_0||_1`` falls below this.
    raw_layer, output_layer
        Passed through to :func:`apply_soft_decontamination`. Each round's
        result also overwrites ``sdata['table'].layers[output_layer]``; a
        per-iteration snapshot is additionally kept at
        ``f"{output_layer}_iter{i}"`` for inspection/rollback.
    score_kwargs
        Extra keyword arguments passed to :func:`generalized_source_score`.
    extracellular_score_kwargs
        Extra keyword arguments passed to
        :func:`troutpy.tl.adaptative_source_score_optimized`.

    Returns
    -------
    pandas.DataFrame
        One row per completed iteration: ``iteration``, ``relative_change``
        (``||X_t - X_(t-1)||_1 / ||X_0||_1``), and ``rank_correlation``
        (Spearman correlation of ``obs['urna_source_score']`` vs. the previous
        round).
    """
    score_kwargs = dict(score_kwargs or {})
    extracellular_score_kwargs = dict(extracellular_score_kwargs or {})

    adata = sdata["table"]
    x0 = adata.layers[raw_layer]
    x0_dense = x0.toarray() if issparse(x0) else np.asarray(x0)
    x0_norm = np.abs(x0_dense).sum() + 1e-9

    prev_x = x0_dense.copy()
    prev_rank = adata.obs["urna_source_score"].copy() if "urna_source_score" in adata.obs else None

    records = []
    for it in range(n_iter):
        print(f"\n=== Iteration {it} ===")
        generalized_source_score(sdata, run_density_similarity=(it == 0), **score_kwargs)
        adaptative_source_score_optimized(sdata, **extracellular_score_kwargs)

        iter_layer = f"{output_layer}_iter{it}"
        apply_soft_decontamination(sdata, raw_layer=raw_layer, output_layer=iter_layer, update_X=True)
        adata.layers[output_layer] = adata.layers[iter_layer]

        cur_x = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
        rel_change = float(np.abs(cur_x - prev_x).sum() / x0_norm)

        rank_corr = np.nan
        if "urna_source_score" in adata.obs and prev_rank is not None:
            rank_corr = spearmanr(adata.obs["urna_source_score"], prev_rank).correlation
        prev_rank = adata.obs["urna_source_score"].copy() if "urna_source_score" in adata.obs else prev_rank

        records.append({"iteration": it, "relative_change": rel_change, "rank_correlation": rank_corr})
        print(f"[iter {it}] relative_change={rel_change:.4g} rank_correlation={rank_corr}")

        prev_x = cur_x
        if rel_change < tol:
            print(f"Converged after {it + 1} iteration(s).")
            break

    return pd.DataFrame(records)


def subtract_local_urna_background(
    sdata: SpatialData,
    radius: float = 100.0,
    inner_radius: float = 10.0,
    expr_key: str = "table",
    raw_layer: str = "raw",
    output_layer: str = "decontaminated_density",
    layer: str = "transcripts",
    gene_key: str = "gene",
    xrna_key: str = "xrna_metadata",
    cell_area_col: str | None = "cell_area",
    credit_to_urna_pool: bool = True,
    update_X: bool = False,
    copy: bool = False,
) -> SpatialData | None:
    """Subtract each cell's locally-expected uRNA background from its own intracellular counts.

    A simpler, independent alternative to :func:`generalized_source_score` /
    :func:`apply_soft_decontamination`: rather than identifying which specific
    neighboring cell a contaminating transcript might have come from, this
    only asks whether a cell's own intracellular count for a gene is
    distinguishable from the density of that gene's genuinely-unassigned
    (uRNA) transcripts found within a fixed `radius` of the cell. If a cell's
    count for gene g is no larger than what would be expected purely from the
    locally-observed uRNA density of g (scaled to the cell's own footprint
    area), that count is not trustworthy as real expression -- regardless of
    which neighbor, if any, it might really belong to.

    For each cell c and gene g:

    1. Count g's uRNA transcripts in the annulus between `inner_radius` and
       `radius` of c's centroid -- the local ambient level for that gene in
       that neighborhood. The inner exclusion matters: right at a cell's own
       boundary, segmentation errors (a slightly too-small mask, a missed
       protrusion) can leak genuinely-native transcripts just outside the
       mask, which would otherwise get counted as "ambient" uRNA and bias the
       background estimate upward -- excluding that zone keeps the estimate
       from eating into real signal near imperfect boundaries.
    2. Scale that count by ``cell_area(c) / (pi * (radius**2 - inner_radius**2))``
       to get the number of g's transcripts expected to land inside a
       footprint the size of c's own, purely from ambient background at that
       density. ``cell_area(c)`` is read from ``sdata[expr_key].obs[cell_area_col]``
       when that column exists (real segmentation area); otherwise it falls
       back to the convex hull of c's own assigned transcripts (cells with
       fewer than 3 assigned transcripts are skipped in that fallback).
    3. Subtract that expected background from c's observed count for g,
       floored at 0 -- a continuous correction, not a hard discard: if the
       observed count is far above the local ambient level, little is
       removed; if it's comparable to or below it, most or all of it is.

    Parameters
    ----------
    sdata
        SpatialData object with `expr_key` (an AnnData with ``.obs["cell_id"]``,
        ``.obsm["spatial"]`` centroids, `raw_layer`) and a `layer` points table
        with `gene_key`, ``"x"``, ``"y"``, ``"cell_id"``, ``"overlaps_cell"``
        columns (and ``"extracellular"`` if available, from
        :func:`troutpy.pp.define_urna_probability` /
        :func:`troutpy.pp.segmentation_free_sainsc` -- falls back to
        ``~overlaps_cell`` otherwise).
    radius
        Outer radius around each cell's centroid used to estimate the local
        uRNA density for each gene.
    inner_radius
        Inner radius excluded from that search (an annulus from `inner_radius`
        to `radius` is used, not a full disk) -- keeps segmentation-boundary
        errors from biasing the background estimate with real, just-outside-
        the-mask signal. Pass ``0`` to use a full disk as before.
    expr_key, raw_layer, output_layer
        AnnData table/layer keys, matching :func:`apply_soft_decontamination`'s
        convention.
    layer, gene_key
        Points layer and gene-identifier column.
    xrna_key
        Key of the per-gene metadata table whose ``var["count"]`` column holds
        each gene's uRNA total (see :func:`troutpy.tl.quantify_overexpression`).
    cell_area_col
        Column in ``sdata[expr_key].obs`` holding each cell's real segmentation
        area, used in place of the convex-hull estimate when present (e.g.
        Xenium output typically has an ``obs["cell_area"]`` column already).
        Pass `None` to always use the convex-hull fallback.
    credit_to_urna_pool
        If `True` (default), adds the total mass removed for each gene to
        ``sdata[xrna_key].var["count"]`` -- so signal reclassified as
        background is not simply dropped, but counted toward that gene's
        extracellular total, feeding directly into any downstream
        extrasomatic-proportion computation that reads this column. Does
        *not* add rows to the transcripts table itself, so functions that
        recompute uRNA proportions directly from the transcripts table (e.g.
        :func:`troutpy.tl.extracellular_enrichment`) will not reflect this
        credit -- only ``xrna_metadata``-based consumers will. Requires
        `xrna_key` to already exist (see
        :func:`troutpy.tl.create_urna_metadata`); a ``"count"`` column is
        initialized to 0 if not already present (it is normally populated
        separately by :func:`troutpy.tl.quantify_overexpression`), then
        incremented by the removed mass.
    update_X
        If `True`, also copy the corrected matrix into ``sdata[expr_key].X``.
    copy
        If `True`, return `sdata`; otherwise modify in place and return `None`.

    Returns
    -------
    spatialdata.SpatialData or None
        `sdata` with `output_layer` added to ``sdata[expr_key].layers``, and
        (if `credit_to_urna_pool`) ``sdata[xrna_key].var["count"]`` updated.
    """
    cells = sdata[expr_key]
    obs = cells.obs.copy()
    obs["cell_id"] = obs["cell_id"].astype(str).str.strip()

    if credit_to_urna_pool and xrna_key not in sdata.tables:
        raise KeyError(
            f"sdata.tables[{xrna_key!r}] not found. Run troutpy.tl.create_xrna_metadata(sdata) " "first, or pass credit_to_urna_pool=False."
        )

    transcripts = sdata.points[layer].compute().reset_index(drop=True)
    transcripts["cell_id"] = transcripts["cell_id"].astype(str).str.strip()

    urna_mask = transcripts["extracellular"] if "extracellular" in transcripts.columns else ~transcripts["overlaps_cell"]
    urna_df = transcripts[urna_mask & transcripts[gene_key].notna()]
    if len(urna_df) == 0:
        raise ValueError("No unassigned (uRNA) transcripts found; cannot estimate local background.")

    urna_coords = urna_df[["x", "y"]].values.astype(np.float64)
    urna_genes = urna_df[gene_key].astype(str).values
    urna_tree = cKDTree(urna_coords)

    use_real_area = cell_area_col is not None and cell_area_col in obs.columns
    if use_real_area:
        cell_area_arr = obs[cell_area_col].astype(float).values
        grouped_by_cell = None
    else:
        valid_ids = set(obs["cell_id"])
        is_assigned = transcripts["cell_id"].isin(valid_ids)
        assigned_df = transcripts[is_assigned]
        grouped_by_cell = assigned_df.groupby("cell_id")[["x", "y"]]
        cell_area_arr = None

    gene_to_idx = {g: i for i, g in enumerate(cells.var_names)}

    raw = cells.layers[raw_layer]
    corrected = raw.toarray().astype(float) if issparse(raw) else np.array(raw, dtype=float, copy=True)
    removed_per_gene = np.zeros(len(cells.var_names))

    if inner_radius >= radius:
        raise ValueError(f"inner_radius ({inner_radius}) must be smaller than radius ({radius}).")
    disk_area = np.pi * (radius**2 - inner_radius**2)
    centroids = cells.obsm["spatial"]
    cell_ids = obs["cell_id"].values

    print(
        f"Estimating local uRNA background for {len(obs)} cells (annulus {inner_radius}-{radius}), "
        f"using {'real segmentation areas' if use_real_area else 'convex-hull area estimates'}..."
    )
    n_skipped_area = 0
    for row_i, cell_id in enumerate(tqdm(cell_ids, desc="Local background subtraction")):
        if use_real_area:
            cell_area = cell_area_arr[row_i]
        else:
            try:
                pts = grouped_by_cell.get_group(cell_id).values
                cell_area = ConvexHull(pts).volume if len(pts) >= 3 else np.nan
            except KeyError:
                cell_area = np.nan

        if not np.isfinite(cell_area) or cell_area <= 0:
            n_skipped_area += 1
            continue

        centroid = centroids[row_i]
        nearby_idx = urna_tree.query_ball_point(centroid, r=radius)
        if inner_radius > 0 and len(nearby_idx) > 0:
            dists = np.linalg.norm(urna_coords[nearby_idx] - centroid, axis=1)
            nearby_idx = [idx for idx, d in zip(nearby_idx, dists, strict=False) if d >= inner_radius]
        if len(nearby_idx) == 0:
            continue

        nearby_genes, nearby_counts = np.unique(urna_genes[nearby_idx], return_counts=True)
        for gene, urna_count in zip(nearby_genes, nearby_counts, strict=False):
            col = gene_to_idx.get(gene)
            if col is None:
                continue
            expected_background = (urna_count / disk_area) * cell_area
            observed = corrected[row_i, col]
            new_val = max(observed - expected_background, 0.0)
            removed_per_gene[col] += observed - new_val
            corrected[row_i, col] = new_val

    cells.layers[output_layer] = csr_matrix(corrected)
    if update_X:
        cells.X = cells.layers[output_layer]

    if credit_to_urna_pool:
        xrna = sdata[xrna_key]
        if "count" not in xrna.var.columns:
            xrna.var["count"] = 0.0
        removed_series = pd.Series(removed_per_gene, index=list(cells.var_names))
        xrna.var["count"] = xrna.var["count"].astype(float) + xrna.var.index.map(removed_series).fillna(0.0)

    print(
        f"Local background subtraction: removed {removed_per_gene.sum():.1f} counts total "
        f"across {len(obs)} cells ({n_skipped_area} cells skipped for insufficient shape data)."
    )
    return sdata if copy else None
