import spatialdata_io
import spatialdata as sd
from points2regions import Points2Regions
import pandas as pd
from spatialdata import SpatialData


def segmentation_free_clustering(
    sdata:SpatialData, 
    params: None, 
    x: str = 'x', 
    y: str = 'y', 
    feature_name: str = 'feature_name', 
    method: str = 'points2regions',
    transcript_id: str = 'transcript_id', 
    copy: bool = False
):
    """
    This function clusters transcriptomic data without relying on pre-defined cell or tissue segmentations.It supports multiple clustering methods, with Points2Regions being the default.

    Parameters:
    sdata (SpatialData): A spatial data object containing transcriptomic information.
    params (dict): A dictionary of parameters for the selected clustering method.For `points2regions`:
    - 'num_clusters' (int): Number of clusters (default: 300).
    - 'pixel_width' (float): Pixel width parameter (default: 0.4).
    - 'pixel_smoothing' (float): Pixel smoothing parameter (default: 3.5).
    x (str): Column name for the x-coordinates of transcripts.
    y (str): Column name for the y-coordinates of transcripts.
    feature_name (str): Column name for the feature names.
    method (str, optional): Clustering method to use. Options:
    - 'points2regions': Uses the Points2Regions algorithm for clustering.
    - 'sainsc': Placeholder for another clustering method.
    transcript_id (str, optional): Column name for the transcript IDs.
    copy (bool): If True, returns a copy of the clustering results. If False, updates `sdata` in-place.

    Returns:
    Optional[anndata.AnnData]: If `copy` is True, returns an AnnData object containing the clustering results.Otherwise, updates the `sdata` object in-place and returns None.
    """
    # Reset transcript indexing if not unique
    sdata.points['transcripts'] = sdata.points['transcripts'].reset_index(drop=True)

    # Prepare data for clustering
    data = sdata.points['transcripts'][[x, y, feature_name, transcript_id]].compute()

    if method == 'points2regions':
        # Validate required parameters for Points2Regions
        required_keys = ['num_clusters', 'pixel_width', 'pixel_smoothing']
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Missing required parameter for 'points2regions': '{key}'")

        # Initialize and fit Points2Regions clustering model
        p2r = Points2Regions(
            data[[x, y]], 
            data[feature_name].astype(str), 
            pixel_width=params['pixel_width'], 
            pixel_smoothing=params['pixel_smoothing']
        )
        p2r.fit(num_clusters=params['num_clusters'])

        # Retrieve clustering results
        adata = p2r._get_anndata()
        transcript_id_to_bin = dict(zip(adata.uns['reads'].index, adata.uns['reads']['pixel_ind']))
        data_all = sdata.points['transcripts'].compute().reset_index(drop=True)
        data_all['segmentation_free_clusters'] = p2r.predict(output='marker')#.astype('category')
        data_all['bin_id'] = data_all.index.map(transcript_id_to_bin)

    elif method == 'sainsc':
        # Placeholder for another clustering method
        raise NotImplementedError("The 'sainsc' method is not yet implemented.")

    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'points2regions' and 'sainsc'.")

    # Update the sdata object
    sdata.points['transcripts'] = sd.models.PointsModel.parse(data_all)
    sdata['segmentation_free_table'] = adata

    return adata if copy else None
