import geopandas as gpd
import numpy as np
from shapely import linearrings, polygons
from spatialdata import SpatialData
from spatialdata.models import ShapesModel


def _make_squares(centroid_coordinates: np.ndarray, half_widths: list[float]) -> ShapesModel:
    """Create square polygons based on centroid coordinates and half-widths."""
    linear_rings = []
    for centroid, half_width in zip(centroid_coordinates, half_widths, strict=False):
        min_coords = centroid - half_width
        max_coords = centroid + half_width

        linear_rings.append(
            linearrings(
                [
                    [min_coords[0], min_coords[1]],
                    [min_coords[0], max_coords[1]],
                    [max_coords[0], max_coords[1]],
                    [max_coords[0], min_coords[1]],
                ]
            )
        )
    s = polygons(linear_rings)
    polygon_series = gpd.GeoSeries(s)
    cell_polygon_table = gpd.GeoDataFrame(geometry=polygon_series)
    return ShapesModel.parse(cell_polygon_table)


def create_grid_squares(sdata: SpatialData, layer: str = "transcripts", square_size: float = 50) -> tuple[ShapesModel, np.ndarray]:
    """
    Generate a grid of square polygons covering the transcript space.

    Parameters
    ----------
    sdata:
        The spatial data object containing transcript coordinates.
    layer:
        The key to access transcript coordinates in sdata.
    square_size:
        The size of each square grid cell.

    Returns
    -------
    tuple: A ShapesModel containing the grid squares and an array of centroid coordinates.
    """
    transcripts = sdata.points[layer]
    x_min, y_min = transcripts[["x", "y"]].compute().min().values
    x_max, y_max = transcripts[["x", "y"]].compute().max().values

    x_coords = np.arange(x_min + square_size / 2, x_max, square_size)
    y_coords = np.arange(y_min + square_size / 2, y_max, square_size)

    centroid_coordinates = np.array([[x, y] for x in x_coords for y in y_coords])
    half_widths = [square_size / 2] * len(centroid_coordinates)

    return _make_squares(centroid_coordinates, half_widths), centroid_coordinates


def aggregate_extracellular_transcripts(
    sdata: SpatialData,
    layer: str = "transcripts",
    gene_key: str = "feature_name",
    method: str = "bin",
    square_size: float = 50,
    copy: bool = False,
    key_added: str | None = None,
):
    """
    Aggregate extracellular transcript counts into a grid of squares.

    Parameters
    ----------
    sdata
        The spatial data object.
    gene_key
        Column name where the gene assigned to each transcript is stored
    layer
        The key to access transcript coordinates in sdata.
    square_size
        The size of each square grid bin.
    method
        Strategy employed to aggregate extracellular transcripts
    copy
        Wether to return the sdata as a new object
    key_added
        Name of the table where to store the grouped extracellular transcripts .Default is 'segmentation_free_table'
    """
    if method == "bin":
        # Generate grid squares
        grid_squares, centroid_coordinates = create_grid_squares(sdata, layer, square_size)

        # Store generated squares in SpatialData
        sdata.shapes["grid_squares"] = grid_squares

        # Filter for non-extracellular transcripts
        sdata["extracellular_transcripts"] = sdata[layer][sdata[layer]["extracellular"].compute()]  # type: ignore ## modify this. Currently is badly implemented

        # Aggregate transcript counts by grid squares
        sdata_shapes = sdata.aggregate(values="extracellular_transcripts", by="grid_squares", value_key=gene_key, agg_func="count")

        # Store aggregated table in sdata.table['extracellular_table']
        if not key_added:
            key_added = "segmentation_free_table"
        sdata[key_added] = sdata_shapes["table"]
        sdata[key_added].obsm["spatial"] = centroid_coordinates  # type: ignore

    return sdata.copy() if copy else None
