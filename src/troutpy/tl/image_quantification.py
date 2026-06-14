import numpy as np
import scanpy as sc
import spatialdata as sd
import xarray as xr


def image_intensities_per_transcript(
    sdata: sd.SpatialData,
    image_key: str,
    scale: str,
    transcript_key: str,
    extracellular: bool = False,
    copy: bool = True,
    gene_key: str = "gene",
) -> sd.SpatialData | None:
    """Extract image intensities at transcript locations and store them as a new table.

    For each transcript, the pixel value of every channel (and z-slice, if present) of
    ``sdata.images[image_key][scale]`` nearest to its rescaled coordinates is extracted
    into an :class:`~anndata.AnnData` stored at ``sdata["image_intensity_per_transcript"]``.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        The input SpatialData object.
    image_key : str
        Key of the image layer in ``sdata.images``.
    scale : str
        Scale of the image to use (e.g. ``"scale0"``).
    transcript_key : str
        Key of the transcript points layer in ``sdata.points``.
    extracellular : bool, optional
        If ``True``, only transcripts where ``extracellular`` is ``True`` are used.
        Otherwise all transcripts are used. Defaults to ``False``.
    copy : bool, optional
        If ``True``, return the modified SpatialData object. Otherwise modify ``sdata``
        in place and return ``None``. Defaults to ``True``.
    gene_key : str, optional
        Column holding the gene assigned to each transcript. Defaults to ``"gene"``.

    Returns
    -------
    spatialdata.SpatialData or None
        SpatialData with the added ``"image_intensity_per_transcript"`` table if
        ``copy=True``; otherwise ``None``.

    Raises
    ------
    KeyError
        If ``image_key`` or ``transcript_key`` is not found in ``sdata``.
    """
    try:
        image = sdata.images[image_key][scale]
        transcripts = sdata.points[transcript_key].compute()
    except KeyError as e:
        raise KeyError(f"Key not found in sdata: {e}") from e

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
        transcripts = transcripts[transcripts["extracellular"]]

    # nearest-pixel lookup at the rescaled transcript coordinates
    xy_positions = np.column_stack((transcripts["x_scaled"], transcripts["y_scaled"]))
    xy_positions_raw = np.column_stack((transcripts["x"], transcripts["y"]))

    intensities = image.sel(x=xr.DataArray(xy_positions[:, 0], dims="points"), y=xr.DataArray(xy_positions[:, 1], dims="points"), method="nearest")

    ad_data = intensities.image.values.transpose()
    if ad_data.ndim == 2:
        ad = sc.AnnData(ad_data)
        ad.var.index = intensities.c.values
    else:
        # flatten the z-slice dimension into combined channel/z-slice names
        ad = sc.AnnData(ad_data.reshape(ad_data.shape[0], -1))
        ad.var.index = [f"{c}_{z}" for c in intensities.c.values for z in range(ad_data.shape[2])]

    ad.obs[gene_key] = list(transcripts[gene_key])
    ad.obsm["spatial"] = xy_positions_raw

    sdata["image_intensity_per_transcript"] = ad

    return sdata if copy else None
