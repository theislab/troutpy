import os

import matplotlib.pyplot as plt
import pandas as pd

# function to compute the number of exchanged genes between any two cell types


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
    - source_proportions (pd.DataFrame): A data frame (Gene name x Cell Type) with proportion of cells per cell type expressing corresponding gene
    - target_proportions : A data frame
    - (Gene name x Cell Type) with proportion of cells per cell type being the physically clostest cell to transcripts of corresponding gene. Defaults to 0.2.
    - source_proportion_threshold (float, optional): The threshold to consider a cell type to be a significant source of a gene. Defaults to 0.2.
    - target_proportion_threshold (float, optional): The threshold to consider a cell type to be a significant target of a gene. Defaults to 0.2.

    Returns
    -------
    - pd.DataFrame: _description_
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
    - source_proportions (pd.DataFrame): A DataFrame where rows represent genes and columns represent source cell types. Each value indicates the proportion of the gene in the respective source cell type.
    - target_proportions (pd.DataFrame): A DataFrame where rows represent genes and columns represent target cell types. Each value indicates the proportion of the gene in the respective target cell type.
    - gene_symbol (str, optional): The gene symbol for which the interaction strength is to be computed and visualized (default: '').
    - return_interactions (bool, optional): If True, returns the interaction matrix as a NumPy array (default: False).
    - save (bool, optional): If True, saves the chord diagram plot to the specified output path (default: False).
    - output_path (str, optional): The directory path where the plot will be saved. If `save=True`, this path will be used to store the file (default: ''). A 'figures' subdirectory is created if it doesn't exist.
    - format (str, optional): The file format for saving the plot (e.g., 'pdf', 'png'). This is used only if `save=True` (default: 'pdf').

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
