# API

Import troutpy as:

```python
import troutpy as tp
```

Following the [scverse](https://scverse.org) convention (as in scanpy and squidpy),
the API is split into three modules. Functions operate on a
{class}`~spatialdata.SpatialData` object holding a `"transcripts"` points layer and a
segmented-cell `"table"`, and write their results back into it (new points columns,
`sdata["xrna_metadata"].var` for per-gene uRNA metrics, or new tables such as
`sdata["source_score"]`), unless `copy=True` is passed.

```{toctree}
:maxdepth: 1

preprocessing
tools
plotting
```
