import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from spatialdata.models import Image2DModel, Labels2DModel


def xenium_converter(sdata, copy=False, unassigned_tag="UNASSIGNED"):
    """
    Converts a SpatialData object (sdata) into the Xenium format.

    The modifications performed are as follows:

    Points / Transcripts:
      - In sdata.Points['transcripts'] (a pandas DataFrame):
          * Rename the 'feature_name' column to 'Gene' and convert it to a categorical variable.
          * Create a boolean column 'control_probe' based on 'codeword_category':
                if the value is 'predesigned_gene' -> False;
                otherwise -> True.
          * Cast 'transcript_id' to a string and ensure the values are unique.
          * Cast 'overlaps_nucleus' to boolean.
          * Create a new column 'overlaps_cell' where if 'cell_id' == 'UNASSIGNED' then False, else True.

    Tables:
      - In sdata.Tables['table'] (an AnnData object):
          * Check that the .obs DataFrame contains a 'cell_id' column.
          * Ensure that the .obs index is unique.
          * Create a new layer named 'raw' in table.layers containing a copy of table.X (the raw expression data).

    Images:
      - From sdata.Images['morphology_focus'] (a DataTree of images at multiple resolutions):
          * Extract the highest quality image (scale 'scale0') and store it in sdata.Stainings under the key 'default'.

    Parameters
    ----------
    sdata : SpatialData
        A SpatialData object with attributes such as Images, Points, and Tables,
        conforming to the structure detailed in the SpatialData documentation.

    Returns
    -------
    sdata : SpatialData
        The converted SpatialData object with the modifications applied.
    """
    # ----- 1. Modify the Points/transcripts DataFrame -----
    try:
        transcripts = sdata["transcripts"].compute().reset_index(drop=True)
    except (KeyError, AttributeError) as e:
        raise ValueError("SpatialData must contain Points['transcripts'] as a DataFrame.") from e

    # a. Rename 'feature_name' to 'Gene' and convert to categorical
    if "feature_name" in transcripts.columns:
        transcripts = transcripts.rename(columns={"feature_name": "gene"})
        transcripts["gene"] = transcripts["gene"].astype("category")
    else:
        print("Warning: 'feature_name' column not found in Points['transcripts'].")

    # b. Create new boolean column 'control_probe' based on 'codeword_category'
    if "codeword_category" in transcripts.columns:
        transcripts["control_probe"] = transcripts["codeword_category"].apply(lambda x: False if x == "predesigned_gene" else True)
    else:
        print("Warning: 'codeword_category' column not found in Points['transcripts']. Gene name-based control probe assignment will be used.")
        transcripts["control_probe"] = transcripts["gene"].str.startswith(("Neg", "BLANK"))

    # c. Ensure 'transcript_id' is a string and has unique values.
    if "transcript_id" in transcripts.columns:
        transcripts["transcript_id"] = transcripts["transcript_id"].astype(str)
        if transcripts["transcript_id"].duplicated().any():
            raise ValueError("Duplicate values found in 'transcript_id'. They must be unique.")
    else:
        print("Warning: 'transcript_id' column not found in Points['transcripts'].")

    # d. Convert 'overlaps_nucleus' to boolean.
    if "overlaps_nucleus" in transcripts.columns:
        transcripts["overlaps_nucleus"] = transcripts["overlaps_nucleus"].astype(bool)
    else:
        print("Warning: 'overlaps_nucleus' column not found in Points['transcripts'].")

    # e. Create new 'overlaps_cell' boolean column based on 'cell_id'.
    if "cell_id" in transcripts.columns:
        transcripts["overlaps_cell"] = transcripts["cell_id"].apply(lambda x: False if x == unassigned_tag else True)
    else:
        print("Warning: 'cell_id' column not found in Points['transcripts']. Cannot create 'overlaps_cell'.")

    # Update the transcripts in Points
    sdata["transcripts"] = sd.models.PointsModel.parse(transcripts)

    # ----- 2. Modify the Tables object -----
    try:
        table = sdata["table"]
    except (KeyError, AttributeError) as e:
        raise ValueError("SpatialData must contain Tables['table'] as an AnnData object.") from e

    # Verify that table.obs contains the 'cell_id' column.
    if "cell_id" not in table.obs.columns:
        raise ValueError("Tables['table'].obs must contain a 'cell_id' column.")
    # Ensure the index is unique.
    if table.obs.index.duplicated().any():
        raise ValueError("Tables['table'].obs index contains duplicates. They must be unique.")
    # Create a new layer 'raw' storing a copy of table.X (raw expression data).
    try:
        table.var["gene_id"] = table.var.index
        table.var.index = table.var["gene_names"]
    except:
        print("Warning: 'gene_names' column not found in Tables['table']. Using default index.")
    if table.X is None:
        raise ValueError("Tables['table'] must have an expression matrix in .X.")
    if not hasattr(table, "layers") or table.layers is None:
        table.layers = {}
    table.layers["raw"] = table.X.copy()

    # ----- 3. Move Images to Stainings -----
    try:
        morph_focus = sdata["morphology_focus"]
    except (KeyError, AttributeError):
        print("Warning: SpatialData does not have Images['morphology_focus'].")
        morph_focus = None

    if morph_focus is not None:
        # Assume morph_focus is a DataTree (or similar dict-like object) with keys for different scales.
        if "scale0" in morph_focus:
            default_image = morph_focus["scale0"]
        else:
            print("Warning: 'scale0' not found in Images['morphology_focus']. Using first available scale.")
            default_image = next(iter(morph_focus.values()))

        # Create or update the Stainings attribute.
        if not hasattr(sdata, "Stainings") or sdata.Stainings is None:
            sdata.Stainings = {}
        sdata.Stainings["default"] = default_image

    return sdata if copy else None


def process_image_data(sdata: sd.SpatialData) -> xr.DataArray:
    """
    Stitches image tiles from SpatialData object into a single global image.

    Args:
        sdata: The input SpatialData object containing image tiles.

    Returns
    -------
        An xarray DataArray representing the stitched global image.
    """
    tile_ids = list(sdata.images.data.keys())
    sample_tile = sdata[tile_ids[0]]
    tile_height = sample_tile.data.shape[1]
    tile_width = sample_tile.data.shape[2]

    x_offsets = {}
    y_offsets = {}
    for tid in tile_ids:
        transform_matrix = sdata[tid].transform["global"].matrix
        x_offsets[tid] = transform_matrix[0, 2]
        y_offsets[tid] = transform_matrix[1, 2]

    global_x_min = min(x_offsets.values())
    global_y_min = min(y_offsets.values())
    global_x_max = max(x_offsets[tid] + tile_width for tid in tile_ids)
    global_y_max = max(y_offsets[tid] + tile_height for tid in tile_ids)

    canvas_width = int(np.ceil(global_x_max - global_x_min))
    canvas_height = int(np.ceil(global_y_max - global_y_min))

    print(f"Global Image Canvas Dimensions: Width = {canvas_width}, Height = {canvas_height}")

    global_image_canvas = np.zeros((3, canvas_height, canvas_width), dtype=np.uint8)

    for tid in tile_ids:
        x_offset = int(np.round(x_offsets[tid] - global_x_min))
        y_offset = int(np.round(y_offsets[tid] - global_y_min))
        tile_image = sdata[tid].data
        global_image_canvas[:, y_offset : y_offset + tile_height, x_offset : x_offset + tile_width] = tile_image

    stitched_da = xr.DataArray(global_image_canvas, dims=("c", "y", "x"))
    print("Image stitching complete!")
    return stitched_da


def process_label_data(sdata: sd.SpatialData) -> xr.DataArray:
    """
    Stitches label tiles from SpatialData object into a single global label image.

    Args:
        sdata: The input SpatialData object containing label tiles.

    Returns
    -------
        An xarray DataArray representing the stitched global label image.
    """
    tile_ids = list(sdata.labels.data.keys())
    sample_label = sdata[tile_ids[0]]
    tile_height = sample_label.data.shape[0]
    tile_width = sample_label.data.shape[1]

    x_offsets = {}
    y_offsets = {}
    for tid in tile_ids:
        transform_matrix = sdata[f"{tid}"].transform["global"].matrix
        x_offsets[tid] = transform_matrix[0, 2]
        y_offsets[tid] = transform_matrix[1, 2]

    global_x_min = min(x_offsets.values())
    global_y_min = min(y_offsets.values())
    global_x_max = max(x_offsets[tid] + tile_width for tid in tile_ids)
    global_y_max = max(y_offsets[tid] + tile_height for tid in tile_ids)

    canvas_width = int(np.ceil(global_x_max - global_x_min))
    canvas_height = int(np.ceil(global_y_max - global_y_min))

    print(f"Global Label Canvas Dimensions: Width = {canvas_width}, Height = {canvas_height}")

    global_label_canvas = np.zeros((canvas_height, canvas_width), dtype=sample_label.data.dtype)

    for tid in tile_ids:
        x_offset = int(np.round(x_offsets[tid] - global_x_min))
        y_offset = int(np.round(y_offsets[tid] - global_y_min))
        tile_label = sdata[f"{tid}"].data
        global_label_canvas[y_offset : y_offset + tile_height, x_offset : x_offset + tile_width] = tile_label

    stitched_labels_da = xr.DataArray(global_label_canvas, dims=("y", "x"))
    print("Label stitching complete!")
    return stitched_labels_da


def process_transcript_data(sdata: sd.SpatialData) -> pd.DataFrame:
    """
    Processes transcript data from SpatialData object, concatenating tiles and
    performing necessary transformations.

    Args:
        sdata: The input SpatialData object containing transcript data.

    Returns
    -------
        A pandas DataFrame containing the processed transcript data.
    """
    tile_ids = list(sdata.points.data.keys())  # changed from sdata.labels to sdata.images
    alldf = []
    for id in tile_ids:
        transcripts = sdata[id].compute()
        alldf.append(transcripts)

    alltranscripts = pd.concat(alldf, ignore_index=True)

    alltranscripts["local_x"] = alltranscripts["x"]
    alltranscripts["local_y"] = alltranscripts["y"]

    del alltranscripts["y"]
    del alltranscripts["x"]

    alltranscripts.rename(columns={"x_global_px": "x", "y_global_px": "y", "target": "gene"}, inplace=True)

    alltranscripts["transcript_id"] = [i for i in range(len(alltranscripts))]
    alltranscripts["overlaps_cell"] = alltranscripts["cell_ID"] != 0
    alltranscripts["overlaps_nuclei"] = alltranscripts["CellComp"] == "Nuclear"
    alltranscripts["overlaps_nuclei"].fillna(False, inplace=True)
    alltranscripts["overlaps_nuclei"] = alltranscripts["overlaps_nuclei"].astype(bool)
    alltranscripts["control_probe"] = alltranscripts["gene"].str.startswith("NegPrb")

    sdata["table"].obsm["spatial"] = sdata["table"].obsm["global"]
    return alltranscripts


def cosmx_converter(sdata: sd.SpatialData, copy: bool = False) -> sd.SpatialData:
    """
    Converts CosMX data from a SpatialData object into a new SpatialData object
    with processed image, label, and transcript data.

    Args:
        sdata: The input SpatialData object containing the raw CosMX data.
        copy: If True, returns a copy of the SpatialData object with the
            converted data. If False, modifies the original SpatialData object
            in place and returns it.

    Returns
    -------
        A SpatialData object containing the converted data.  This is a *new*
        SpatialData object if copy is True, otherwise it's the *same* object
        as the input (modified in place).
    """
    sdata_out = sdata

    stitched_da = process_image_data(sdata_out)
    stitched_labels_da = process_label_data(sdata_out)
    parsed_transcripts = sd.models.PointsModel.parse(process_transcript_data(sdata_out))

    sdata_out.images = {
        "morphology_focus": Image2DModel.parse(data=stitched_da, scale_factors=(2, 2), c_coords=sdata_out[list(sdata_out.images.keys())[0]].c.data)
    }  # changed sdata to sdata_out
    sdata_out.labels = {"labels": Labels2DModel.parse(data=stitched_labels_da, scale_factors=(2, 2))}  # changed sdata to sdata_out
    sdata_out.points = {"transcripts": parsed_transcripts}

    return sdata if copy else None
