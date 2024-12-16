import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import spatialdata as sd
from typing import List, Union, Tuple

def compute_extracellular_counts(data_extracell): # would be good to change the name of this function
    """
    Compute observed, expected, and fold ratio for extracellular transcript counts.

    Parameters:
    data_extracell (pd.DataFrame): Data with extracellular transcripts.

    Returns:
    pd.DataFrame: Dataframe with observed, expected counts, fold ratios, and gene categories.
    """
    extracellular_counts = data_extracell.groupby('feature_name').count()
    extracellular_counts = pd.DataFrame({'observed': extracellular_counts.iloc[:, 0]})
    extracellular_counts['expected'] = int(extracellular_counts['observed'].sum() / extracellular_counts.shape[0])
    
    # Calculate fold ratios
    extracellular_counts['fold_ratio'] = np.log(extracellular_counts['observed'] / extracellular_counts['expected'])
    
    # Map gene categories
    gene2cat = dict(zip(data_extracell['feature_name'], data_extracell['codeword_category']))
    extracellular_counts['codeword_category'] = extracellular_counts.index.map(gene2cat)
    
    return extracellular_counts

def define_extracellular(
    sdata, 
    layer: str = 'transcripts', 
    method: str = 'segmentation_free',
    min_prop_of_extracellular: float = 0.8, 
    unassigned_to_cell_tag: str = 'UNASSIGNED', 
    copy: bool = False
):
    """
    Define extracellular transcripts in spatial omics data.

    This function identifies extracellular transcripts based on the specified method and updates the spatial data object accordingly.

    Parameters:
        sdata : SpatialData
            A spatial data object containing transcriptomic information.
        layer : str, optional (default: 'transcripts')
            The layer in `sdata.points` containing the transcript data to process.
        method : str, optional (default: 'segmentation_free')
            The method to define extracellular transcripts. Options:
            - 'segmentation_free': Uses segmentation-free clustering results.
            - 'nuclei': Uses overlap with nuclear annotations to classify extracellular transcripts.
            - 'cells': Classifies transcripts not assigned to a cell as extracellular.
        min_prop_of_extracellular : float, optional (default: 0.8)
            Minimum proportion of transcripts in a cluster required to be extracellular for 
            it to be classified as such (used only with 'segmentation_free' method).
        unassigned_to_cell_tag : str, optional (default: 'UNASSIGNED')
            Tag indicating transcripts not assigned to any cell.
        copy : bool, optional (default: False)
            If True, returns a copy of the updated spatial data. 
            If False, updates the `sdata` object in-place.

    Returns:
        Optional[SpatialData]:
            If `copy` is True, returns a copy of the updated `sdata` object.
            Otherwise, updates the `sdata` object in-place and returns None.

    Notes:
        - The 'segmentation_free' method uses clustering results to determine extracellular transcripts.
        - The 'nuclei' method assumes transcripts outside nuclei are extracellular.
        - The 'cells' method classifies transcripts unassigned to cells as extracellular.

    Example:
        ```python
        updated_sdata = define_extracellular(
            sdata, method='segmentation_free', min_prop_of_extracellular=0.9, copy=True
        )
        ```
    """
    # Compute the data layer
    data = sdata.points[layer].compute()

    # Method: Segmentation-free clustering
    if method == 'segmentation_free':
        data['overlaps_cell'] = (data['cell_id'] != unassigned_to_cell_tag).astype(int)
        overlapping_cell = pd.crosstab(data['segmentation_free_clusters'], data['overlaps_cell'])

        # Compute proportions and define extracellular clusters
        cluster_totals = overlapping_cell.sum(axis=1)
        cluster_proportions = overlapping_cell.div(cluster_totals, axis=0)
        extracellular_clusters = cluster_proportions[cluster_proportions.loc[:,0] >= min_prop_of_extracellular].index
        data['extracellular'] = ~data['segmentation_free_clusters'].isin(extracellular_clusters)

    # Method: Based on nuclei overlap
    elif method == 'nuclei':
        data['extracellular'] = data['overlaps_nucleus'] != 1

    # Method: Based on cell assignment
    elif method == 'cells':
        data['extracellular'] = data['cell_id'] == unassigned_to_cell_tag

    # Unsupported method
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'segmentation_free', 'nuclei', and 'cells'.")

    # Update the spatial data object
    sdata.points[layer] = sd.models.PointsModel.parse(data)

    return sdata if copy else None

#


def compute_crosstab(data,xvar:str='',yvar:str=''):
    """Compute crosstabs for each given variable"""
    crosstab_data= pd.crosstab(data[xvar], data[yvar])
    return crosstab_data