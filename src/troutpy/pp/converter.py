import copy

import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from spatialdata.models import Image2DModel, Labels2DModel, TableModel


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
    Stitches image tiles into a single global image.
    """
    fov_keys = list(sdata.images.data.keys())
    sample_raw = sdata.images.data[fov_keys[0]].data
    sample = sample_raw.compute() if hasattr(sample_raw, "compute") else sample_raw
    h, w = sample.shape[1:]
    x_offs, y_offs = {}, {}
    for key in fov_keys:
        M = sdata.images.data[key].transform["global"].matrix
        x_offs[key], y_offs[key] = M[0, 2], M[1, 2]
    x_min, y_min = min(x_offs.values()), min(y_offs.values())
    x_max = max(x_offs[key] + w for key in fov_keys)
    y_max = max(y_offs[key] + h for key in fov_keys)
    canvas_w = int(np.ceil(x_max - x_min))
    canvas_h = int(np.ceil(y_max - y_min))
    canvas = np.zeros((3, canvas_h, canvas_w), dtype=np.uint8)
    for key in fov_keys:
        tile_raw = sdata.images.data[key].data
        tile = tile_raw.compute() if hasattr(tile_raw, "compute") else tile_raw
        x0 = int(round(x_offs[key] - x_min))
        y0 = int(round(y_offs[key] - y_min))
        canvas[:, y0 : y0 + h, x0 : x0 + w] = tile
    return xr.DataArray(canvas, dims=("c", "y", "x"))


def process_label_data(sdata: sd.SpatialData) -> (xr.DataArray, dict):
    """
    Stitches label tiles, assigns unique IDs, and returns mapping from (fov, original_cell_ID) to new unique ID.
    """
    fov_keys = list(sdata.labels.data.keys())
    sample_raw = sdata.labels.data[fov_keys[0]].data
    sample = sample_raw.compute() if hasattr(sample_raw, "compute") else sample_raw
    h, w = sample.shape[:2]
    x_offs, y_offs = {}, {}
    for key in fov_keys:
        M = sdata.labels.data[key].transform["global"].matrix
        x_offs[key], y_offs[key] = M[0, 2], M[1, 2]
    x_min, y_min = min(x_offs.values()), min(y_offs.values())
    x_max = max(x_offs[key] + w for key in fov_keys)
    y_max = max(y_offs[key] + h for key in fov_keys)
    canvas_w = int(np.ceil(x_max - x_min))
    canvas_h = int(np.ceil(y_max - y_min))
    global_lbl = np.zeros((canvas_h, canvas_w), dtype=sample.dtype)

    next_id = 1
    id_map = {}
    for key in fov_keys:
        # derive numeric fov from key, e.g. '1_points' -> 1
        fov = int(str(key).split("_", 1)[0])
        tile_raw = sdata.labels.data[key].data
        tile = tile_raw.compute() if hasattr(tile_raw, "compute") else tile_raw
        unique_ids = np.unique(tile[tile > 0])
        # map using numeric fov
        tile_map = {(fov, orig): next_id + i for i, orig in enumerate(unique_ids)}
        id_map.update(tile_map)
        next_id += len(unique_ids)
        rel = np.zeros_like(tile)
        for (fv, orig), new in tile_map.items():
            rel[tile == orig] = new
        x0 = int(round(x_offs[key] - x_min))
        y0 = int(round(y_offs[key] - y_min))
        sub = global_lbl[y0 : y0 + h, x0 : x0 + w]
        mask = (sub == 0) & (rel > 0)
        sub[mask] = rel[mask]
        global_lbl[y0 : y0 + h, x0 : x0 + w] = sub
    return xr.DataArray(global_lbl, dims=("y", "x")), id_map


def process_transcript_data(sdata: sd.SpatialData) -> pd.DataFrame:
    """
    Concatenates transcript tiles, keeps fov (numeric) for remapping.
    """
    dfs = []
    for key in sdata.points.data.keys():
        fov = int(str(key).split("_", 1)[0])
        elem = sdata.points.data[key]
        raw = elem.data if hasattr(elem, "data") else elem
        df = raw.compute() if hasattr(raw, "compute") else raw
        df["fov"] = fov
        dfs.append(df)
    tx = pd.concat(dfs, ignore_index=True)
    tx["local_x"] = tx["x"]
    tx["local_y"] = tx["y"]
    tx.drop(columns=["x", "y"], inplace=True)
    tx.rename(columns={"x_global_px": "x", "y_global_px": "y", "target": "gene"}, inplace=True)
    tx["transcript_id"] = np.arange(len(tx))
    tx["overlaps_cell"] = tx["cell_ID"] != 0
    tx["overlaps_nuclei"] = tx["CellComp"] == "Nuclear"
    tx["overlaps_nuclei"].fillna(False, inplace=True)
    tx["control_probe"] = tx["gene"].str.startswith("NegPrb")
    if "global" in sdata["table"].obsm:
        sdata["table"].obsm["spatial"] = sdata["table"].obsm["global"].copy()
    return tx


def cosmx_converter(sdata: sd.SpatialData, copy_data: bool = False) -> sd.SpatialData:
    """Main converter: stitches, remaps cell_IDs by (fov,orig)->global."""
    sdata_out = copy.deepcopy(sdata) if copy_data else sdata
    # 1) stitch
    img_da = process_image_data(sdata_out)
    lbl_da, id_map = process_label_data(sdata_out)
    # 2) transcripts
    tx_df = process_transcript_data(sdata_out)
    # 3) remap transcripts
    tx_df["cell_ID"] = tx_df.apply(lambda r: id_map.get((r["fov"], r["cell_ID"]), 0), axis=1).astype(int)
    # 4) remap table
    if "cell_ID" in sdata_out.table.obs:
        if "fov" not in sdata_out.table.obs.columns:
            raise ValueError("table.obs must contain numeric 'fov' column to remap cell_IDs.")
        sdata_out.table.obs["cell_ID"] = sdata_out.table.obs.apply(lambda r: id_map.get((int(r["fov"]), r["cell_ID"]), 0), axis=1).astype(int)
    # 5) shift coords
    min_x, min_y = tx_df["x"].min(), tx_df["y"].min()
    tx_df["x"] -= min_x
    tx_df["y"] -= min_y
    if "spatial" in sdata_out.table.obsm:
        sdata_out.table.obsm["spatial"][:, 0] -= min_x
        sdata_out.table.obsm["spatial"][:, 1] -= min_y
    # 6) pack
    sdata_out.images = {
        "morphology_focus": Image2DModel.parse(data=img_da, scale_factors=(2, 2), c_coords=sdata_out[list(sdata_out.images.keys())[0]].c.data)
    }
    sdata_out.labels = {"labels": Labels2DModel.parse(data=lbl_da, scale_factors=(2, 2))}
    sdata_out.points = {"transcripts": sd.models.PointsModel.parse(tx_df)}
    # 7) shapes + metadata
    sdata_out["shapes"] = sd.to_polygons(sdata_out.labels["labels"])
    sdata_out["shapes"].index = sdata_out["shapes"].index.astype(int)

    # format data

    sdata_out["table"].obs["region"] = "shapes"
    original_table = sdata_out["table"].copy()
    try:
        del original_table.uns["spatialdata_attrs"]
    except:
        pass
    # re-parse that AnnData into a SpatialData table layer
    sdata_out["table"] = TableModel.parse(
        original_table,
        region="shapes",  # or whatever region_key you want
        region_key="region",
        instance_key="cell_ID",
    )
    return sdata_out if copy_data else None
