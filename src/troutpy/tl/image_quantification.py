import numpy as np
import scanpy as sc
import spatialdata as sd
import xarray as xr


def image_intensities_per_transcript(
    sdata: sd.SpatialData, image_key: str, scale: str, transcript_key: str, extracellular: bool = False, copy: bool = True
) -> sd.SpatialData:
    """
    Extracts image intensities at transcript locations and adds them as a new layer in the SpatialData object.

    Args:
        sdata
            The input SpatialData object.
        image_key
            The key for the image layer in sdata.images.
        scale
            The scale of the image to use.
        transcript_key
            The key for the transcript points in sdata.points.
        extracellular
            Whether to include only extracellular transcripts (default: False). If True, only transcripts where `extracellular` is True are used. If false, all transcripts are used.
        copy
            Whether to create a copy of the SpatialData object (default: True).

    Returns
    -------
        A SpatialData object with the added 'transcripts_image_intensities' layer.  If copy=True, a new SpatialData object is returned. Otherwise, the original sdata object is modified and returned.

    Raises
    ------
        KeyError: If the specified image or transcript key is not found in the SpatialData object.
    """
    try:
        image = sdata.images[image_key][scale]  # Image as xarray.DataArray
        transcripts = sdata.points[transcript_key].compute()  # Transcript coordinates
    except KeyError as e:
        raise KeyError(f"Key not found in sdata: {e}")  # noqa: B904

    imarray = image.image.compute()

    minx = transcripts.x.min()
    maxx = transcripts.x.max()
    transcripts_size_x = maxx - minx
    image_size_x = imarray.shape[2]
    multi_factor_x = image_size_x / transcripts_size_x

    miny = transcripts.y.min()
    maxy = transcripts.y.max()
    transcripts_size_y = maxy - miny
    image_size_y = imarray.shape[1]
    multi_factor_y = image_size_y / transcripts_size_y

    transcripts["x_scaled"] = (transcripts["x"] - minx) * multi_factor_x
    transcripts["y_scaled"] = (transcripts["y"] - miny) * multi_factor_y

    if extracellular:
        transcripts = transcripts[transcripts["extracellular"] == True]  # Select extracellular transcripts
    else:
        transcripts = transcripts

    # Get (x, y) coordinates of transcripts
    xy_positions = np.column_stack((transcripts["x_scaled"], transcripts["y_scaled"]))
    xy_positions_raw = np.column_stack((transcripts["x"], transcripts["y"]))

    # Interpolate or fetch nearest pixel values
    intensities = image.sel(
        x=xr.DataArray(xy_positions[:, 0], dims="points"), y=xr.DataArray(xy_positions[:, 1], dims="points"), method="nearest"
    )  # Use "linear" for interpolation

    intensity_tab = np.zeros([len(intensities.x.values), 2])
    intensity_tab[:, 0] = intensities.x.values
    intensity_tab[:, 1] = intensities.y.values

    ad_data = intensities.image.values.transpose()
    if ad_data.ndim == 2:  # Check if it is a 2D image
        ad = sc.AnnData(ad_data)
        ad.var.index = intensities.c.values
    else:  # If not, assume it is a 3D image and flatten the last dimension
        ad = sc.AnnData(ad_data.reshape(ad_data.shape[0], -1))
        ad.var.index = [f"{c}_{z}" for c in intensities.c.values for z in range(ad_data.shape[2])]  # Create combined channel and z-slice names.

    ad.obs["feature_name"] = list(transcripts["feature_name"])  # Add feature names
    ad.obsm["spatial"] = xy_positions_raw  # Add raw spatial coordinates

    # Add the AnnData object as a new spatial data object
    sdata["image_intensity_per_transcript"] = ad

    return sdata if copy else None
