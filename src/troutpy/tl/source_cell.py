import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from sklearn.neighbors import KDTree
import spatialdata as sd
from spatialdata import SpatialData

def create_xrna_metadata(
    sdata: SpatialData,
    points_layer: str = 'transcripts',
    gene_key: str = 'feature_name',
    copy: bool = False
) -> SpatialData | None:
    """
    Creates a new table within the SpatialData object that contains a 'gene' column 
    with the unique gene names extracted from the specified points layer.

    Parameters:
    ----------
    sdata : SpatialData
        The SpatialData object to modify.
    
    points_layer : str, optional
        The name of the layer in `sdata.points` from which to extract gene names.
        Default is 'transcripts'.
    
    gene_key : str, optional
        The key in the `points_layer` dataframe that contains the gene names.
        Default is 'feature_name'.
    
    copy : bool, optional
        If `True`, returns a copy of the `SpatialData` object with the new table added.
        If `False`, modifies the original `SpatialData` object in place. Default is `False`.

    Returns:
    -------
    SpatialData | None
        If `copy` is `True`, returns a copy of the modified `SpatialData` object.
        Otherwise, returns `None`.

    Raises:
    ------
    ValueError
        If the specified points layer does not exist in `sdata.points`.
        If the `gene_key` column is not present in the specified points layer.

    Examples:
    --------
    Add a metadata table for genes in the 'transcripts' layer:
    >>> create_xrna_metadata(sdata, points_layer='transcripts', gene_key='feature_name')

    Modify a custom SpatialData layer and return a copy:
    >>> updated_sdata = create_xrna_metadata(sdata, points_layer='custom_layer', gene_key='gene_id', copy=True)

    Notes:
    -----
    - The function uses `scanpy` to create an AnnData object and integrates it into the SpatialData table model.
    - The unique gene names are extracted from the specified points layer and stored in the `.var` of the AnnData object.
    """
    # Check if the specified points layer exists
    if points_layer not in sdata.points:
        raise ValueError(f"Points layer '{points_layer}' not found in sdata.points.")
    
    # Extract unique gene names from the specified points layer
    points_data = sdata.points[points_layer]
    if gene_key not in points_data.columns:
        raise ValueError(f"The specified points layer '{points_layer}' does not contain a '{gene_key}' column.")
    
    unique_genes = points_data[gene_key].compute().unique().astype(str)
    
    # Create a DataFrame for unique genes
    gene_metadata = pd.DataFrame(index=unique_genes)

    # Convert to AnnData and then to SpatialData table model
    exrna_adata = sc.AnnData(var=gene_metadata)
    metadata_table = sd.models.TableModel.parse(exrna_adata)
    
    # Add the new table to the SpatialData object
    sdata.tables['xrna_metadata'] = metadata_table

    print(f"Added 'xrna_metadata' table with {len(unique_genes)} unique genes to the SpatialData object.")
    
    # Return copy or modify in place
    return sdata if copy else None

def compute_source_cells(
    sdata,
    expression_threshold=1,
    gene_id_column='feature_name',
    layer='transcripts',
    copy=False
):
    """
    Compute the source of extracellular RNA by linking detected extracellular transcripts 
    to specific cell types in the spatial data.

    Parameters:
    ----------
    sdata : SpatialData object
        The input spatial data object containing spatial transcriptomics data.
    expression_threshold : float, optional, default=1
        Threshold for filtering transcripts based on expression levels.
    gene_id_column : str, optional, default='feature_name'
        Column name for gene identifiers in the transcripts data.
    layer : str, optional, default='transcripts'
        Layer in `sdata.points` containing the transcript information.
    copy : bool, optional, default=False
        If True, returns a modified copy of the spatial data object. Otherwise, modifies in place.

    Returns:
    -------
    sdata : SpatialData object or None
        The modified spatial data object with added `source` metadata if `copy=True`. 
        Otherwise, modifies the input object in place and returns None.
    """

    # Create a copy of the table containing spatial transcriptomics data
    adata = sdata['table'].copy()
    adata.X = adata.layers['raw']  # Use the 'raw' layer for calculations

    # Generate a binary matrix where values above the threshold are set to True
    adata_bin = adata.copy()
    adata_bin.X = adata_bin.X > expression_threshold

    # Compute the proportion of cells expressing each feature per cell type
    proportions = get_proportion_expressed_per_cell_type(
        adata_bin,
        cell_type_key='cell type'
    )

    # Ensure the necessary `xrna_metadata` is present in `sdata`
    if 'xrna_metadata' not in sdata:
        create_xrna_metadata(sdata, points_layer='transcripts')

    # Create an output DataFrame and store computed proportions
    outtable = pd.DataFrame(index=sdata['xrna_metadata'].var.index)
    sdata['xrna_metadata'].varm['source'] = outtable.join(proportions).to_numpy()

    # Return the modified SpatialData object or None based on the `copy` parameter
    return sdata.copy() if copy else None

def distance_to_source_cell(
    sdata, 
    layer='transcripts', 
    xcoord='x', 
    ycoord='y',
    xcellcoord='x_centroid', 
    ycellcoord='y_centroid',
    gene_id_column='feature_name',copy=False):
    
    # transcripts
    adata_bin = sdata['table'].copy()
    adata_bin.X = sdata['table'].layers['raw']
    adata_bin.obs['x_centroid'] = [sp[0] for sp in adata_bin.obsm['spatial']]
    adata_bin.obs['y_centroid'] = [sp[1] for sp in adata_bin.obsm['spatial']]
    transcripts = sdata.points[layer].compute()
    extracellular_transcripts = transcripts[transcripts['extracellular']]
    # Filter extracellular transcripts to those in adata_bin
    #extracellular_transcripts = extracellular_transcripts[extracellular_transcripts["feature_name"].isin(adata_bin.var_names)]
    tranid = []
    dist = []
    cellids=[]

    for gene_of_interest in tqdm(adata_bin.var_names):
        gene_idx = np.where(adata_bin.var_names == gene_of_interest)[0][0]
        adata_filtered = adata_bin[adata_bin.X[:, gene_idx] > 0]#.copy()
        extracellular_transcripts_filtered = extracellular_transcripts[extracellular_transcripts[gene_id_column] == gene_of_interest].copy()
        # Only proceed if there are positive cells for the gene of interests
        if (adata_filtered.n_obs > 0) & (extracellular_transcripts_filtered.shape[0]>0) :
            # Extract coordinates of cells and transcripts
            cell_coords = np.array([adata_filtered.obs[xcellcoord], adata_filtered.obs[ycellcoord]]).T
            transcript_coords = np.array([extracellular_transcripts_filtered[xcoord], extracellular_transcripts_filtered[ycoord]]).T

            # Compute KDTree for nearest cell
            tree = KDTree(cell_coords)
            distances, closest_cells_indices = tree.query(transcript_coords, k=1)
            # Append results to lists
            tranid.extend(extracellular_transcripts_filtered['transcript_id'])
            dist.extend([d[0]for d in distances])
            cell_ids = adata_filtered.obs['cell_id'].values[closest_cells_indices.flatten()]
            cellids.extend(c[0] for c in cell_ids.reshape(closest_cells_indices.shape))
    # Create a dictionary to map transcript IDs to distances
    id2dist = dict(zip(tranid, dist))
    id2closeid = dict(zip(tranid,cellids))
    # Store the results in the DataFrame
    transcripts['distance_to_source_cell'] = transcripts['transcript_id'].map(id2dist)
    transcripts['closest_source_cell'] = transcripts['transcript_id'].map(id2closeid)
    sdata.points[layer] = sd.models.PointsModel.parse(transcripts)

    # add median distance_to_source_cell
    dist_to_source=transcripts.loc[:,[gene_id_column,'distance_to_source_cell']].groupby(gene_id_column).median()
    dist_to_source.columns=['median_distance_to_source_cell']
    sdata['xrna_metadata'].var=sdata['xrna_metadata'].var.join(dist_to_source)

    return sdata.copy() if copy else None

def compute_distant_cells_prop(sdata, layer='transcripts', gene_id_column='feature_name', threshold=30,copy=False):
    """
    Compute the proportion of transcripts for each gene that are located beyond a specified distance 
    from their closest source cell, and add the result to the metadata of the SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        A SpatialData object containing the spatial omics data.
    layer : str, optional
        The layer in `sdata.points` that contains the transcript data. Default is 'transcripts'.
    gene_id_column : str, optional
        Column name in the transcript data representing gene identifiers. Default is 'feature_name'.
    threshold : float, optional
        The distance threshold (in micrometers) to calculate the proportion of transcripts farther away 
        from their closest source cell. Default is 30.

    Returns
    -------
    None
        The function modifies the `sdata` object in place, adding the computed proportions as a new 
        column in `sdata['xrna_metadata'].var`.

    Notes
    -----
    - This function assumes that `sdata.points[layer]` contains a column `distance_to_source_cell` 
      with distances between transcripts and their closest source cells.
    - The resulting column is named `frac_beyond_<threshold>_from_source`.

    Example
    -------
    ```
    compute_source_cells_beyond_distance(sdata, layer='transcripts', threshold=30)
    ```
    """

    # Extract transcript data
    data = sdata.points[layer].compute()
    
    # Calculate the proportions of distances above the threshold
    proportions_above_threshold = (data.groupby(gene_id_column)['distance_to_source_cell']
    .apply(lambda x: (x > threshold).mean()))
    
    # Create a DataFrame and rename the column
    proportions_above_threshold = pd.DataFrame(proportions_above_threshold)
    proportions_above_threshold.columns = [f'frac_beyond_{threshold}_from_source']
    
    # Join the computed proportions with the metadata
    for column in proportions_above_threshold.columns:
        if column in sdata['xrna_metadata'].var.columns:
             sdata['xrna_metadata'].var=sdata['xrna_metadata'].var.drop([column],axis=1)
    sdata['xrna_metadata'].var = sdata['xrna_metadata'].var.join(proportions_above_threshold)

    return sdata.copy() if copy else None

def get_proportion_expressed_per_cell_type(adata,cell_type_key='cell type'):
    cell_types = adata.obs[cell_type_key].unique().dropna()
    proportions = pd.DataFrame(index=adata.var_names, columns=cell_types)
    for cell_type in cell_types:
        proportions[cell_type] = adata[adata.obs[cell_type_key] == cell_type].X.mean(axis=0).T
    return proportions
