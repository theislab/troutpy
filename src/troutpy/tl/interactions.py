import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spatialdata import SpatialData
from tqdm import tqdm


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
        pd.DataFrame with proportion of cells per cell type being the physically clostest cell to transcripts of corresponding gene. Defaults to 0.2.
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
    format (str, optional)
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


def compute_communication_strength(sdata: SpatialData, source_layer: str = "source_score", target_layer: str = "target_score", copy: bool = False):
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


def gene_specific_interactions(sdata, copy: bool = False):
    """
    Group the read-specific interaction scores into gene-specific scores

    Parameters
    ----------
        sdata
            A SpatialData object including precomputed communication strenghts for each exRNA
        copy
            Wether to save the resulting sdata as a copy

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

    categories = list(source_table.obs["feature_name"])  # Extract categories
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
