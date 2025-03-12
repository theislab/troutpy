import spatialdata as sd
from points2regions import Points2Regions
from spatialdata import SpatialData


def segmentation_free_clustering(
    sdata: SpatialData,
    params: dict = None,  # type: ignore
    layer: str = "transcripts",
    coord_keys: list = None,  # type: ignore
    gene_key: str = "feature_name",
    method: str = "points2regions",
    transcript_id_key: str = "transcript_id_key",
    copy: bool = False,
):
    """
    Clusters transcriptomic data without relying on pre-defined cell or tissue segmentations.It supports multiple clustering methods, with Points2Regions being the default.

    Parameters
    ----------
    sdata
        A spatial data object containing transcriptomic information.
    params
        A dictionary of parameters for the selected clustering method.For `points2regions`:
    coord_keys
       List of x and y gene columns within sdata[layer]
    y
        Column name for the y-coordinates of transcripts.
    gene_key
        Column name for the feature names.
    method (str, optional)
        Clustering method to use. Options:
            - 'points2regions': Uses the Points2Regions algorithm for clustering.
            - 'sainsc': Placeholder for another clustering method.
    transcript_id_key (str, optional)
        Column name for the transcript IDs.
    copy (bool)
        If True, returns a copy of the clustering results. If False, updates `sdata` in-place.

    Returns
    -------
    Optional[anndata.AnnData]
        If `copy` is True, returns an AnnData object containing the clustering results.Otherwise, updates the `sdata` object in-place and returns None.
    """
    # Reset transcript indexing if not unique
    if coord_keys is None:
        coord_keys = ["x", "y"]
    sdata.points[layer] = sdata.points[layer].reset_index(drop=True)
    # Prepare data for clustering
    data = sdata.points[layer][[coord_keys[0], coord_keys[1], gene_key, transcript_id_key]].compute()

    if method == "points2regions":
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
        data_all["segmentation_free_clusters"] = p2r.predict(output="marker")  # .astype('category')
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
