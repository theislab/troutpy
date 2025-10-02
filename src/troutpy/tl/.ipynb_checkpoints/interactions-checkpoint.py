import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spatialdata import SpatialData
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree


####### LIKELY DEPRECATED#########
def get_number_of_communication_genes(
    source_proportions: pd.DataFrame,  # gene by source cell type
    target_proportions: pd.DataFrame,  # gene by target cell type
    source_proportion_threshold: float = 0.2,
    target_proportion_threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Compute the number of exchanged genes between any two cell types

    Parameters
    ----------
    source_proportions
        A data frame (Gene name x Cell Type) with proportion of cells per cell type expressing corresponding gene
    target_proportions
        pandas.DataFrame with proportion of cells per cell type being the physically clostest cell to transcripts of corresponding gene. Defaults to 0.2.
    source_proportion_threshold
        The threshold to consider a cell type to be a significant source of a gene. Defaults to 0.2.
    target_proportion_threshold
        The threshold to consider a cell type to be a significant target of a gene. Defaults to 0.2.

    Returns
    -------
    number_interactions_df:
        Dataframe with number of interactions
    """
    # filter the source and target cell types by defining signficant proportions
    source_binary = source_proportions > source_proportion_threshold
    target_binary = target_proportions > target_proportion_threshold

    # prepare dataframe to store the number of exchanged genes
    number_interactions_df = pd.DataFrame(index=source_binary.columns, columns=target_binary.columns)

    # loop through the source and target cell types to compute the number of
    # exchanged genes
    for col in source_binary.columns:
        sig_gene_source = source_binary.index[source_binary[col]]
        for col2 in target_binary.columns:
            sig_gene_target = target_binary.index[target_binary[col2]]
            number_interactions_df.loc[col, col2] = len(set(sig_gene_source).intersection(sig_gene_target))

    number_interactions_df = number_interactions_df[number_interactions_df.index]
    number_interactions_df.columns.name = "Target cell type"
    number_interactions_df.index.name = "Source cell type"
    return number_interactions_df


####### LIKELY DEPRECATED#########
def get_gene_interaction_strength(
    source_proportions: pd.DataFrame,  # gene by source cell type
    target_proportions: pd.DataFrame,  # gene by target cell type
    gene_symbol: str = "",  # Gene of interest
    return_interactions: bool = False,  # Flag to return interaction matrix
    save: bool = False,  # Flag to save the plot
    output_path: str = "",  # Directory to save the plot
    format: str = "pdf",  # Format to save the plot (e.g., pdf, png)
) -> None:
    """
    Calculates the interaction strength between source and target cell types for a specified gene by multiplying the proportions of the gene in the source and target cell types. The interaction matrix can be visualized using a chord diagram, with the option to save the resulting plot.

    Parameters
    ----------
    source_proportions
        A DataFrame where rows represent genes and columns represent source cell types. Each value indicates the proportion of the gene in the respective source cell type.
    target_proportions
        A DataFrame where rows represent genes and columns represent target cell types. Each value indicates the proportion of the gene in the respective target cell type.
    gene_symbol
        The gene symbol for which the interaction strength is to be computed and visualized (default: '').
    return_interactions
        If True, returns the interaction matrix as a NumPy array (default: False).
    save
        If True, saves the chord diagram plot to the specified output path (default: False).
    output_path
        The directory path where the plot will be saved. If `save=True`, this path will be used to store the file (default: ''). A 'figures' subdirectory is created if it doesn't exist.
    format: str
        The file format for saving the plot (e.g., 'pdf', 'png'). This is used only if `save=True` (default: 'pdf').

    Returns
    -------
    - None or np.ndarray

    Notes
    -----
    - The function computes the interaction matrix by multiplying the proportions of the gene in the source and target cell types.
    - The chord diagram visualizes the interaction strength between the cell types.
    - If `save=True`, the plot is saved in the specified format and location.
    """
    # Ensure the target proportions have the same cell type columns as the source proportions
    target_proportions = target_proportions[source_proportions.columns]

    # Get source and target proportions for the specified gene
    source_proportions_vals = source_proportions.loc[gene_symbol].values[:, None]
    target_proportions_vals = target_proportions.loc[gene_symbol].values[None, :]

    # Compute the interaction matrix (source proportions * target proportions)
    interactions = source_proportions_vals @ target_proportions_vals

    # Define the colormap and create color mappings for each cell type

    # Plot the interaction strength using a chord diagram
    #### work on this function ######
    # cmap = plt.get_cmap("tab20")
    # colors = [cmap(i) for i in range(interactions.shape[0])]
    # chord_diagram(interactions, source_proportions.columns.tolist(), directed=True, fontsize=8, colors=colors)
    plt.title(f"exotranscriptomic {gene_symbol} exchange", fontweight="bold")

    # Save the plot if the 'save' option is enabled
    if save:
        figures_path = os.path.join(output_path, "figures")
        os.makedirs(figures_path, exist_ok=True)  # Create 'figures' directory if it doesn't exist
        plt.savefig(os.path.join(figures_path, f"communication_profile_{gene_symbol}.{format}"))  # Save the figure

    # Show the plot
    plt.show()
    return pd.DataFrame(interactions, index=source_proportions.columns, columns=target_proportions.columns)


def communication_strength(sdata: SpatialData, source_layer: str = "source_score", target_layer: str = "target_score", copy: bool = False):
    """
    Compute a 3D interaction strength matrix from the source table in SpatialData.

    Parameters
    ----------
        sdata
            SpatialData object with a 'tables' attribute.
        source_layer
            Key to access the source table within sdata.tables.

    Returns
    -------
        sdata
            SpatialData object with computed interactions
    """
    source_table = sdata.tables[source_layer]
    target_table = sdata.tables[target_layer]
    interaction_strength = np.empty((source_table.shape[0], source_table.shape[1], target_table.shape[1]))

    for i, id in enumerate(tqdm(source_table.obs.index)):
        matmul_result = np.dot(np.array(source_table[id].X).T, np.array(target_table[id].X))
        interaction_strength[i, :, :] = matmul_result

    sdata["source_score"].uns["interaction_strength"] = interaction_strength

    return sdata.copy() if copy else None


def gene_specific_interactions(sdata, copy: bool = False, gene_key: str = "gene"):
    """
    Group the read-specific interaction scores into gene-specific scores

    Parameters
    ----------
        sdata: spatialdata.SpatialData
            A SpatialData object including precomputed communication strenghts for each exRNA
        copy: bool
            Wether to save the resulting sdata as a copy
        gene_key: str
            column in sdata['source_table'] containing gene assignment for each transcript


    Returns
    -------
        sdata
            A 3D array of shape (num_categories, H, W) containing mean values per category.
    """
    try:
        interaction_strength = sdata["source_score"].uns["interaction_strength"]
        source_table = sdata["source_score"]
    except:  # noqa: E722
        KeyError("Interaction streght is not computed. Please run troutpy.tl.compute_communication_strength first")

    categories = list(source_table.obs[gene_key])  # Extract categories
    unique_cats = np.unique(categories)  # Get unique categories

    # Initialize result 3D matrix (num_categories, H, W)
    result_matrix = np.zeros((len(unique_cats), interaction_strength.shape[1], interaction_strength.shape[2]))

    # Iterate over unique categories
    for i, cat in enumerate(tqdm(unique_cats, desc="Processing categories")):
        mask = np.array(categories) == cat  # Boolean mask
        subset = interaction_strength[mask, :, :]  # Extract relevant slices
        result_matrix[i] = np.mean(subset, axis=0)  # Compute mean over axis 0

    # save resulting matrix
    sdata["source_score"].uns["gene_interaction_strength"] = result_matrix
    sdata["source_score"].uns["gene_interaction_names"] = unique_cats

    return sdata.copy() if copy else None

def cell_contacts_with_urna_sources(
    sdata, 
    spatial_key: str = "spatial", 
    cell_type_key: str = "cell type", 
    distance: float = 50, 
    copy: bool = False,
    uns_prefix: str = "cell_contact"
):
    """
    Compute neighbor count matrices:
    1. cell-cell contact (spatial neighbors only)
    2. combined contact (spatial + uRNA neighbors)
    3. uRNA-specific contact (combined - cell-cell)

    Results are stored in sdata['table'].uns under separate keys: f"{uns_prefix}_cell_cell", f"{uns_prefix}_combined", f"{uns_prefix}_urna_specific".

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing:
        - sdata['table']: AnnData with cell_type annotations and spatial coordinates.
        - sdata['target_score']: AnnData with transcript-level target info
        - sdata['source_score']: AnnData with transcript-level source info
    spatial_key : str
        Key in sdata['table'].obsm containing spatial coordinates.
    cell_type_key : str
        Column in sdata['table'].obs with cell type labels.
    distance : float
        Radius to define spatial neighborhoods.
    copy : bool
        If True, return the matrices. If False, modify sdata in place and return None.
    uns_prefix : str
        Prefix for keys under which matrices will be saved in sdata['table'].uns.

    Returns
    -------
    dict of pd.DataFrame or None
        Dictionary with three DataFrames: {"cell_cell", "combined", "urna_specific"}
        if copy=True, otherwise None.
    """
    adata = sdata['table']
    cell_ids = adata.obs_names.values
    cell_types = adata.obs[cell_type_key].values
    unique_cell_types = np.unique(cell_types)
    
    # Step 1: Compute spatial neighbors
    coords = np.asarray(adata.obsm[spatial_key])
    tree = BallTree(coords)
    neighbors_idx = tree.query_radius(coords, r=distance)
    neighbors_dict = {cell_id: cell_ids[neighbors] for cell_id, neighbors in zip(cell_ids, neighbors_idx)}

    # Step 2: Merge source and target transcript info
    target_obs = sdata['target_score'].obs[['closest_cell', 'distance']].copy()
    target_obs.rename(columns={'closest_cell': 'target_cell'}, inplace=True)
    source_obs = sdata['source_score'].obs[['closest_cell']].copy()
    source_obs.rename(columns={'closest_cell': 'source_cell'}, inplace=True)

    trans_df = target_obs.join(source_obs, how='inner')
    trans_df = trans_df[trans_df['distance'] <= distance]

    # Step 3: Filter out transcripts where source already in neighborhood or == target
    def keep_transcript(row):
        target = row['target_cell']
        source = row['source_cell']
        return (source != target) and (source not in neighbors_dict[target])
    
    filtered_trans = trans_df[trans_df.apply(keep_transcript, axis=1)]

    # Step 4: Build extended neighborhoods (spatial + uRNA)
    extended_neighbors_dict = {cell_id: list(neighs) for cell_id, neighs in neighbors_dict.items()}
    for target_cell, group in filtered_trans.groupby('target_cell'):
        sources = group['source_cell'].values
        extended_neighbors_dict[target_cell] = np.unique(
            np.concatenate([extended_neighbors_dict[target_cell], sources])
        )

    # Step 5: Count matrices
    cell_type_map = dict(zip(cell_ids, cell_types))
    
    def count_matrix_from_neighbors(neigh_dict):
        mat = pd.DataFrame(0, index=unique_cell_types, columns=unique_cell_types, dtype=int)
        for source_cell, neighbors in neigh_dict.items():
            source_type = cell_type_map[source_cell]
            for target_cell in neighbors:
                try:
                    target_type = cell_type_map[target_cell]
                    mat.loc[source_type, target_type] += 1
                except:
                    pass
        return mat

    # cell-cell only
    cell_cell_mat = count_matrix_from_neighbors(neighbors_dict)
    # combined (spatial + uRNA)
    combined_mat = count_matrix_from_neighbors(extended_neighbors_dict)
    # uRNA-specific = combined - cell-cell
    urna_specific_mat = combined_mat - cell_cell_mat
    urna_specific_mat[urna_specific_mat < 0] = 0  # safety check

    # Save results under separate uns keys
    adata.uns[f"{uns_prefix}_cell_body"] = cell_cell_mat
    adata.uns[f"{uns_prefix}_combined"] = combined_mat
    adata.uns[f"{uns_prefix}_urna_specific"] = urna_specific_mat

    results = {
        "cell_body": cell_cell_mat,
        "combined": combined_mat,
        "urna_specific": urna_specific_mat
    }
    return results if copy else None


