from copy import deepcopy

import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from shapely import linearrings, polygons
from shapely.geometry import box
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from spatialdata import SpatialData
from spatialdata.models import ShapesModel
from tqdm import tqdm


def _make_squares(centroid_coordinates: np.ndarray, half_widths: list[float]) -> ShapesModel:
    """Create square polygons centered at the given coordinates with the specified half-widths.

    Parameters
    ----------
    centroid_coordinates : numpy.ndarray
        Array of shape ``(N, 2)`` with ``[x, y]`` centroid coordinates for each square.
    half_widths : list of float
        Half side-length for each square, parallel to ``centroid_coordinates``.

    Returns
    -------
    ShapesModel
        SpatialData ShapesModel containing the ``N`` square polygons.
    """
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
    sdata : SpatialData
        The spatial data object containing transcript coordinates.
    layer : str
        The key to access transcript coordinates in sdata.
    square_size : float
        The size of each square grid cell.

    Returns
    -------
    ShapesModel
        Grid of square polygons covering the transcript bounding box.
    numpy.ndarray
        Array of shape ``(N, 2)`` with the ``[x, y]`` centroid coordinates of each square.
    """
    transcripts = sdata.points[layer]
    x_min, y_min = transcripts[["x", "y"]].compute().min().values
    x_max, y_max = transcripts[["x", "y"]].compute().max().values

    x_coords = np.arange(x_min + square_size / 2, x_max, square_size)
    y_coords = np.arange(y_min + square_size / 2, y_max, square_size)

    centroid_coordinates = np.array([[x, y] for x in x_coords for y in y_coords])
    half_widths = [square_size / 2] * len(centroid_coordinates)

    return _make_squares(centroid_coordinates, half_widths), centroid_coordinates


def aggregate_urna(
    sdata: SpatialData,
    layer: str = "transcripts",
    gene_key: str = "gene",
    method: str = "bin",
    square_size: float = 50,
    radius: float = 50,
    knn_k: int = 5,
    extracellular_only: bool = True,
    copy: bool = False,
    key_added: str | None = None,
    cell_type_key: str = "leiden",
    embedding_enabled: bool = True,
    fragment_k: int = 5,
    fragment_dist_thresh: float = 7.0,
    fragment_sim_thresh: float = 0.5,
):
    """
    Aggregate urna transcript counts into a grid of squares.

    Parameters
    ----------
    sdata
        The spatial data object.
    layer
        The key to access transcript coordinates in sdata.
    gene_key
        Column name where the gene assigned to each transcript is stored.
    method
        Aggregation strategy. Options: "bin", "radius", "knn", "fragments".
    square_size
        Size of each square grid bin (used for the "bin" method).
    radius
        Radius for aggregating neighboring transcripts (used for the "radius" method).
    knn_k
        Number of nearest neighbors (used for the "knn" method).
    extracellular_only
        If True (default), only uses extracellular transcripts.
    copy
        If True, returns a new SpatialData object; otherwise, sdata is modified in place.
    key_added
        Key under which aggregated results are stored. Defaults to 'segmentation_free_table' if not provided.
    cell_type_key
        Cell type key in sdata['table'] to use for gene embeddings (used for the "fragments" method).
    embedding_enabled
        Whether to compute gene embeddings for the "fragments" method.
    fragment_k
        Number of spatial nearest neighbors considered when linking transcripts into fragments (used for the "fragments" method).
    fragment_dist_thresh
        Maximum distance between transcripts to be connected into the same fragment (used for the "fragments" method).
    fragment_sim_thresh
        Minimum gene-embedding cosine similarity for two transcripts to be connected into the same fragment (used for the "fragments" method).

    Returns
    -------
    sdata or None
        If copy is True, returns a new SpatialData object; otherwise, returns None.
    """
    # Select transcripts based on the extracellular_only flag.
    if extracellular_only:
        extracell = sdata[layer][sdata[layer]["extracellular"]]  # type: ignore
    else:
        extracell = sdata[layer].compute() if hasattr(sdata[layer], "compute") else sdata[layer]

    if method == "bin":
        # --- BIN METHOD ---
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

    elif method == "fragments":
        df = extracell.compute() if hasattr(extracell, "compute") else extracell
        df = df.loc[:, ["x", "y", gene_key]].rename(columns={gene_key: "gene"})

        # Optional: compute gene embeddings from cell type annotations
        if embedding_enabled:
            assert "table" in sdata, "To compute gene embeddings, sdata must contain a 'table' AnnData."
            adata_ref = sdata["table"]
            assert cell_type_key in adata_ref.obs, f"'{cell_type_key}' not found in adata.obs"
            gene_embeddings = compute_gene_embeddings_from_anndata(adata_ref, cell_type_col=cell_type_key)
        else:
            gene_embeddings = None

        labels = aggregate_fragments(
            df,
            gene_embeddings=gene_embeddings,
            k=fragment_k,
            dist_thresh=fragment_dist_thresh,
            sim_thresh=fragment_sim_thresh,
        )
        df["fragment_id"] = labels

        # Compute signature per fragment
        signature_table = pd.crosstab(df["fragment_id"], df["gene"])
        adata_frag = sc.AnnData(signature_table)

        # Compute centroids
        centroids = compute_fragment_centroids(df, fragment_col="fragment_id")
        centroids = centroids.set_index("fragment_id")
        centroids = centroids.loc[adata_frag.obs.index.astype(int)]  # align ordering
        adata_frag.obsm["spatial"] = centroids[["x_centroid", "y_centroid"]].to_numpy()

        if not key_added:
            key_added = "segmentation_free_table"
        sdata[key_added] = adata_frag

    else:
        raise ValueError(f"Unsupported method: {method}")

    return deepcopy(sdata) if copy else None


def aggregate_fragments(df, gene_embeddings=None, k=5, dist_thresh=10.0, sim_thresh=0.5):
    """
    Group extracellular (unassigned) transcripts into fragments based on spatial and molecular similarity.

    Parameters
    ----------
    df
        DataFrame with columns ['x', 'y', 'gene']
    gene_embeddings
        Optional dictionary mapping gene names to embedding vectors.If None, one-hot encoding is used.
    k
        Number of nearest neighbors for spatial graph.
    dist_thresh
        Max distance threshold for spatial neighbors.
    sim_thresh
        Min cosine similarity threshold to form edge.

    Returns
    -------
    fragment_ids
        Array of fragment labels for each transcript (-1 = unassigned)
    """
    coords = df[["x", "y"]].values

    # Gene embeddings: use provided or one-hot encode
    if gene_embeddings is not None:
        embeddings = np.vstack([gene_embeddings[g] for g in df["gene"]])
    else:
        encoder = OneHotEncoder(sparse_output=False)
        embeddings = encoder.fit_transform(df[["gene"]])

    # Build spatial neighbor graph
    nn = NearestNeighbors(n_neighbors=k, radius=dist_thresh)
    nn.fit(coords)
    dists, indices = nn.kneighbors(coords)

    # Build sparse graph of edges passing distance and embedding similarity threshold
    rows, cols, data = [], [], []
    for i in range(len(coords)):
        for j_idx, j in enumerate(indices[i]):
            if i == j:
                continue
            if dists[i][j_idx] > dist_thresh:
                continue
            sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8)
            if sim >= sim_thresh:
                rows.append(i)
                cols.append(j)
                data.append(1)

    graph = csr_matrix((data, (rows, cols)), shape=(len(df), len(df)))

    # Connected components = fragments
    n_components, labels = connected_components(csgraph=graph, directed=False)

    return labels


def compute_gene_embeddings_from_anndata(adata, cell_type_col="cell_type"):
    """
    Compute gene embeddings from AnnData by fraction of expressing cells per cell type.

    Parameters
    ----------
    adata: anndata.AnnData
       AnnData object
    cell_type_col: str
       Column in adata.obs with cell type annotations

    Returns
    -------
    dict
       Mapping from gene name to a 1D :class:`numpy.ndarray` embedding vector.
    """
    assert cell_type_col in adata.obs, f"{cell_type_col} not in adata.obs"

    cell_types = adata.obs[cell_type_col].unique()
    genes = adata.var_names
    embedding = {}

    for gene_idx, gene in enumerate(genes):
        vec = []
        gene_expr = adata.X[:, gene_idx]
        if hasattr(gene_expr, "toarray"):  # handle sparse matrix
            gene_expr = gene_expr.toarray().flatten()
        for ct in cell_types:
            ct_mask = adata.obs[cell_type_col] == ct
            ct_expr = gene_expr[ct_mask]
            fraction_expressing = np.mean(ct_expr > 0)
            vec.append(fraction_expressing)
        embedding[gene] = np.array(vec)

    return embedding


def compute_fragment_centroids(df, fragment_col="fragment_id"):
    """Compute the mean (x, y) centroid for each transcript fragment.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least ``"x"``, ``"y"``, and the fragment ID column.
    fragment_col : str, optional
        Column name holding fragment identifiers. Defaults to ``"fragment_id"``.

    Returns
    -------
    centroids : pandas.DataFrame
        DataFrame with one row per fragment and columns
        ``[fragment_col, "x_centroid", "y_centroid"]``.
    """
    return df.groupby(fragment_col)[["x", "y"]].mean().rename(columns={"x": "x_centroid", "y": "y_centroid"}).reset_index()
