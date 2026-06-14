import spatialdata as sd
from spatialdata import SpatialData


def segmentation_free_clustering(
    sdata: SpatialData,
    params: dict | None = None,
    layer: str = "transcripts",
    coord_keys: list | None = None,
    gene_key: str = "feature_name",
    method: str = "points2regions",
    transcript_id_key: str = "transcript_id_key",
    copy: bool = False,
):
    """Cluster transcripts without relying on pre-defined cell or tissue segmentations.

    Supports multiple segmentation-free clustering methods, with Points2Regions being
    the default.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        A spatial data object containing transcriptomic information.
    params : dict, optional
        Parameters for the selected clustering method. For ``method="points2regions"``
        this must contain ``"num_clusters"``, ``"pixel_width"``, and ``"pixel_smoothing"``.
    layer : str, optional
        Key of the points layer in ``sdata`` containing the transcripts to cluster.
        Defaults to ``"transcripts"``.
    coord_keys : list of str, optional
        Names of the x- and y-coordinate columns in ``sdata.points[layer]``. Defaults to
        ``["x", "y"]``.
    gene_key : str, optional
        Column name holding the gene/feature assigned to each transcript.
    method : str, optional
        Clustering method to use. Options:

        - ``"points2regions"``: cluster using the Points2Regions algorithm (default).
        - ``"sainsc"``: not yet implemented.
    transcript_id_key : str, optional
        Column name holding the transcript IDs.
    copy : bool, optional
        If ``True``, return the resulting :class:`~anndata.AnnData` object. Otherwise
        update ``sdata`` in place and return ``None``.

    Returns
    -------
    anndata.AnnData or None
        Clustering-result AnnData (also stored as ``sdata["segmentation_free_table"]``)
        if ``copy=True``; otherwise ``None``.
    """
    if coord_keys is None:
        coord_keys = ["x", "y"]
    sdata.points[layer] = sdata.points[layer].reset_index(drop=True)
    data = sdata.points[layer][[coord_keys[0], coord_keys[1], gene_key, transcript_id_key]].compute()

    if method == "points2regions":
        try:
            from points2regions import Points2Regions
        except ImportError as err:
            raise ImportError(
                "The 'points2regions' package is required for method='points2regions'. Please install it with: pip install troutpy[segmentation-free]"
            ) from err

        # Validate required parameters for Points2Regions
        required_keys = ["num_clusters", "pixel_width", "pixel_smoothing"]
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Missing required parameter for 'points2regions': '{key}'")

        # Initialize and fit Points2Regions clustering model
        p2r = Points2Regions(
            data[[coord_keys[0], coord_keys[1]]],
            data[gene_key].astype(str),
            pixel_width=params["pixel_width"],
            pixel_smoothing=params["pixel_smoothing"],
        )
        p2r.fit(num_clusters=params["num_clusters"])

        # Retrieve clustering results
        adata = p2r._get_anndata()
        transcript_id_key_to_bin = dict(zip(adata.uns["reads"].index, adata.uns["reads"]["pixel_ind"], strict=False))
        data_all = sdata.points[layer].compute().reset_index(drop=True)
        data_all["segmentation_free_clusters"] = p2r.predict(output="marker")
        data_all["bin_id"] = data_all.index.map(transcript_id_key_to_bin)

    elif method == "sainsc":
        # Placeholder for another clustering method
        raise NotImplementedError("The 'sainsc' method is not yet implemented.")

    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'points2regions' and 'sainsc'.")

    # Update the sdata object
    sdata.points[layer] = sd.models.PointsModel.parse(data_all)
    sdata["segmentation_free_table"] = adata

    return adata if copy else None
