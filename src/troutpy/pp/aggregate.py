import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc  # Assumed available based on usage in original code
from scipy.ndimage import maximum_filter
from scipy.spatial import KDTree
from shapely import linearrings, polygons
from shapely.geometry import box
from spatialdata import SpatialData
from spatialdata.models import ShapesModel
from spatialdata.models.models import ShapesModel  # For converting GeoDataFrame
from tqdm import tqdm


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
    sdata,
    layer: str = "transcripts",
    gene_key: str = "feature_name",
    method: str = "bin",
    square_size: float = 50,
    radius: float = 50,
    knn_k: int = 5,
    overlap_bin: bool = False,
    local_maxima_threshold: float = 0,
    extracellular_only: bool = True,
    copy: bool = False,
    key_added: str | None = None,
):
    """
    Aggregate transcript counts based on one of several methods.

    By default, only extracellular transcripts are used. Use the parameter
    `extracellular_only` to switch between processing only extracellular transcripts or all.

    Methods
    -------
      - "bin": Aggregates transcripts by grid squares (the original approach).
      - "radius": For each transcript, computes an expression signature based on transcripts within a given radius.
      - "knn": For each transcript, computes its expression signature based on its k-nearest neighbors.
      - "local_maxima": Computes a 2D histogram over the spatial extent (using extracellular counts only by default)
                        and selects grid cells that are local peaks and above a threshold as centers,
                        then aggregates transcripts within a given radius from the center.

    Parameters
    ----------
    sdata
        The spatial data object.
    layer
        The key to access transcript coordinates in sdata.
    gene_key
        Column name where the gene assigned to each transcript is stored.
    method
        Aggregation strategy. Options: "bin", "radius", "knn", "local_maxima"
    square_size
        Size of each square grid bin (used for "bin" and "local_maxima" methods).
    radius
        Radius for aggregating neighboring transcripts (used for "radius" and "local_maxima").
    knn_k
        Number of nearest neighbors (used for the "knn" method).
    overlap_bin
        (Not used in the "bin" method, but available for future updates.)
    local_maxima_threshold
        Minimum count threshold for a grid cell to be accepted as a local maximum.
    extracellular_only
        If True (default), only uses extracellular transcripts.
    copy
        If True, returns a new SpatialData object; otherwise, sdata is modified in place.
    key_added
        Key under which aggregated results are stored. Defaults to 'segmentation_free_table' if not provided.

    Returns
    -------
    sdata or None
        If copy is True, returns a new SpatialData object; otherwise, returns None.
    """
    # Select transcripts based on the extracellular_only flag.
    if extracellular_only:
        extracell = sdata[layer][sdata[layer]["extracellular"].compute()]  # type: ignore
    else:
        extracell = sdata[layer].compute() if hasattr(sdata[layer], "compute") else sdata[layer]

    if method == "bin":
        # --- BIN METHOD (original version) ---
        # Generate grid squares from the full transcript coordinates.
        df = sdata[layer].compute()
        x_min, x_max = df["x"].min(), df["x"].max()
        y_min, y_max = df["y"].min(), df["y"].max()

        xs = np.arange(x_min, x_max + square_size, square_size)
        ys = np.arange(y_min, y_max + square_size, square_size)

        grid_squares = []
        centroids = []
        for x in xs[:-1]:
            for y in ys[:-1]:
                square = box(x, y, x + square_size, y + square_size)
                grid_squares.append(square)
                centroids.append((x + square_size / 2, y + square_size / 2))

        # Convert list of geometries to GeoDataFrame.
        grid_gdf = gpd.GeoDataFrame({"geometry": grid_squares})
        # Parse the GeoDataFrame into a valid ShapesModel.
        sdata.shapes["grid_squares"] = ShapesModel.parse(grid_gdf)

        # Store extracellular transcripts for aggregation.
        sdata["extracellular_transcripts"] = extracell

        # Aggregate transcript counts by grid squares using sdata's built-in method.
        sdata_shapes = sdata.aggregate(values="extracellular_transcripts", by="grid_squares", value_key=gene_key, agg_func="count")

        if not key_added:
            key_added = "segmentation_free_table"
        sdata[key_added] = sdata_shapes["table"]
        sdata[key_added].obsm["spatial"] = np.array(centroids)  # type: ignore

    elif method == "radius":
        # --- RADIUS METHOD ---
        df = extracell.compute() if hasattr(extracell, "compute") else extracell

        df[gene_key] = df[gene_key].astype(str)
        gene_cat = pd.Categorical(df[gene_key])
        codes = gene_cat.codes
        unique_genes = np.array(gene_cat.categories)
        n_genes = len(unique_genes)

        coords = df[["x", "y"]].to_numpy()
        tree = KDTree(coords)
        all_indices = tree.query_ball_point(coords, r=radius)

        signatures = []
        for indices in tqdm(all_indices, desc="Computing radius-based signatures"):
            count = np.bincount(codes[indices], minlength=n_genes)
            signatures.append(count)

        signature_table = pd.DataFrame(signatures, index=df.index, columns=unique_genes)
        adata_sig = sc.AnnData(signature_table)

        if not key_added:
            key_added = "segmentation_free_table"
        sdata[key_added] = adata_sig
        sdata[key_added].obsm["spatial"] = coords  # type: ignore

    elif method == "knn":
        # --- KNN METHOD ---
        df = extracell.compute() if hasattr(extracell, "compute") else extracell

        df[gene_key] = df[gene_key].astype(str)
        gene_cat = pd.Categorical(df[gene_key])
        codes = gene_cat.codes
        unique_genes = np.array(gene_cat.categories)
        n_genes = len(unique_genes)

        coords = df[["x", "y"]].to_numpy()
        tree = KDTree(coords)
        distances, indices = tree.query(coords, k=knn_k)

        signatures = []
        for neighbor_indices in tqdm(indices, desc="Computing KNN-based signatures"):
            count = np.bincount(codes[neighbor_indices], minlength=n_genes)
            signatures.append(count)

        signature_table = pd.DataFrame(signatures, index=df.index, columns=unique_genes)
        adata_sig = sc.AnnData(signature_table)

        if not key_added:
            key_added = "segmentation_free_table"
        sdata[key_added] = adata_sig
        sdata[key_added].obsm["spatial"] = coords  # type: ignore

    elif method == "local_maxima":
        # --- LOCAL MAXIMA METHOD ---
        df = extracell
        x_min = df["x"].min().compute() if hasattr(df["x"].min(), "compute") else df["x"].min()
        x_max = df["x"].max().compute() if hasattr(df["x"].max(), "compute") else df["x"].max()
        y_min = df["y"].min().compute() if hasattr(df["y"].min(), "compute") else df["y"].min()
        y_max = df["y"].max().compute() if hasattr(df["y"].max(), "compute") else df["y"].max()

        x_bins = np.arange(x_min, x_max + square_size, square_size)
        y_bins = np.arange(y_min, y_max + square_size, square_size)
        hist, xedges, yedges = np.histogram2d(
            df["x"].compute() if hasattr(df["x"], "compute") else df["x"],
            df["y"].compute() if hasattr(df["y"], "compute") else df["y"],
            bins=[x_bins, y_bins],
        )

        local_max = hist == maximum_filter(hist, size=3)
        local_max &= hist >= local_maxima_threshold
        peak_indices = np.argwhere(local_max)

        centroids = []
        for i, j in peak_indices:
            cx = (xedges[i] + xedges[i + 1]) / 2
            cy = (yedges[j] + yedges[j + 1]) / 2
            centroids.append((cx, cy))
        centroids = np.array(centroids)

        df = df.compute() if hasattr(df, "compute") else df
        df[gene_key] = df[gene_key].astype(str)
        gene_cat = pd.Categorical(df[gene_key])
        codes = gene_cat.codes
        unique_genes = np.array(gene_cat.categories)
        n_genes = len(unique_genes)
        coords = df[["x", "y"]].to_numpy()
        tree = KDTree(coords)

        signatures = []
        for center in tqdm(centroids, desc="Aggregating local maxima bins"):
            indices = tree.query_ball_point(center, r=radius)
            count = np.bincount(codes[indices], minlength=n_genes)
            signatures.append(count)

        signature_table = pd.DataFrame(signatures, columns=unique_genes, index=range(len(centroids)))
        adata_sig = sc.AnnData(signature_table)

        if not key_added:
            key_added = "segmentation_free_table"
        sdata[key_added] = adata_sig
        sdata[key_added].obsm["spatial"] = centroids  # type: ignore

    else:
        raise ValueError(f"Unsupported method: {method}")

    return sdata.copy() if copy else None
