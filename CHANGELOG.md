# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- Basic tool, preprocessing and plotting functions
- Test suite (`tests/`) covering the public `pp`, `tl`, and `pl` APIs against a
  synthetic `SpatialData` fixture (`tests/conftest.py`), including
  `pytest.mark.xfail`/`skip` markers documenting known spatialdata-0.4.0
  incompatibilities and `pl/plotting.py` legacy-pipeline drift.

### Fixed

- `tl.adaptative_source_score_optimized`: `AttributeError:
  'numpy.ndarray' object has no attribute 'tocsr'` when `sdata["table"].X`
  is a dense array — now converts via `csr_matrix(...)` instead of calling
  `.tocsr()`.
- `pl.global_distribution_from_source`: fixed the default `cluster_key`
  (`"kmeans_from_distribution"` → `"kmeans_distribution"`) to match the
  column actually written by `tl.cluster_distribution_from_source` and used
  by `pl.distributions_by_cluster`.

## [0.1.1]

### Removed

- Deprecated functions `tl.compute_source_cells`, `tl.distance_to_source_cell`,
  `tl.compute_distant_cells_proportion`, `tl.compute_source_score`, and
  `tl.characterize_cell_like_structures` (superseded by
  `tl.adaptative_source_score` / `tl.adaptative_source_score_optimized`).

### Changed

- Trimmed `dependencies` to the minimal set required to `import troutpy`
  (added a missing `numba`, dropped unused `joblib`), and moved `sainsc`,
  `points2regions`, `squidpy`, `mpl-chord-diagram`, `scikit-image`, and `drvi`
  to lazily-imported optional extras (`spatial-stats`, `segmentation-free`,
  `chord`, `morphology`, `factor-analysis`; `pip install troutpy[all]` for
  everything).
- `docs/api/*.md` now reflects the full public API exported by `troutpy.pp`,
  `troutpy.tl`, and `troutpy.pl`.
