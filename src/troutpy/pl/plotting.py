import numpy as np
import pandas as pd
import os
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from exrna.pp.compute import compute_crosstab
from typing import Optional, Union, Sequence, Tuple
from anndata import AnnData
from matplotlib.colors import Colormap, Normalize
from pathlib import Path




def sorted_heatmap(celltype_by_feature, output_path:str='',filename:str="Heatmap_target_cells_by_gene",format='pdf',cmap='viridis',vmax=None,save=False,figsize=(10, 10)):
    """
    Plots the heatmap of target cells by gene.

    Parameters:
    celltype_by_feature (pd.DataFrame): DataFrame showing the fraction of each feature by cell type.
    outpath_dummy (str): Path to save the output plots.
    """
    figures_path = os.path.join(output_path, 'figures')
    os.makedirs(figures_path, exist_ok=True)

    # Sort by maximum feature in cell types
    max_indices = np.argmax(celltype_by_feature.values, axis=1)
    celltype_by_feature = celltype_by_feature.iloc[np.argsort(max_indices)]
    celltype_by_feature.index = celltype_by_feature.index[np.argsort(max_indices)]

    # Heatmap plot
    plt.figure(figsize=figsize)
    sns.heatmap(celltype_by_feature, cmap=cmap, vmax=vmax)
    plt.ylabel(f'{celltype_by_feature.index.name}')
    plt.xlabel(f'{celltype_by_feature.columns.name}')
    plt.title(filename)
    if save==True:
        plt.savefig(os.path.join(figures_path, f'{filename}.{format}'))

def coupled_scatter(sdata, layer='extracellular_transcripts', output_path:str='', transcript_group='distance_to_source_cell', 
    save=True, format='pdf', xcoord='x', ycoord='y', xcellcoord='x_centroid', ycellcoord='y_centroid', 
    colormap='Blues', size=2, color_cells='red', figsize=(10, 7), vmax=None):
    """
    Plots a scatter plot of transcript locations and cell centroids, coloring the transcripts by a specific feature 
    (e.g., distance to the closest cell) and optionally saving the plot to a file.

    This function creates a scatter plot where transcripts are plotted according to their spatial coordinates (x, y), 
    and their color represents a feature, such as the distance to the nearest cell. Cell centroids are overlaid on the 
    plot with a specified color. The plot can be saved to a specified file path.

    Parameters:
    ----------
    sdata : dict-like spatial data object
        A spatial data object that contains transcript and cell information. The relevant data is accessed from:
        - `sdata['table']`: The cell information stored as an AnnData object.
        - `sdata['points'][layer]`: The transcript data layer.

    layer : str, optional
        The key for the layer in `sdata.points` that contains transcript data (default: 'extracellular_transcripts').

    output_path : str, optional
        The directory path where the plot will be saved. If not provided, the plot will not be saved (default: '').

    transcript_group : str, optional
        The key in the transcript data (e.g., distance to the source cell) to be used for coloring the scatter plot 
        (default: 'distance_to_source_cell').

    save : bool, optional
        Whether to save the plot to a file. If `True`, the plot is saved to `output_path` (default: True).

    format : str, optional
        The format for saving the plot (e.g., 'pdf', 'png'). This is only used if `save=True` (default: 'pdf').

    xcoord : str, optional
        The column name in the transcript data representing the x-coordinate (default: 'x').

    ycoord : str, optional
        The column name in the transcript data representing the y-coordinate (default: 'y').

    xcellcoord : str, optional
        The column name in the cell data representing the x-coordinate of cell centroids (default: 'x_centroid').

    ycellcoord : str, optional
        The column name in the cell data representing the y-coordinate of cell centroids (default: 'y_centroid').

    colormap : str, optional
        The colormap to use for coloring the transcripts based on the `transcript_group` values (default: 'Blues').

    size : float, optional
        The size of the scatter points for cells and transcripts. Transcripts are scaled down by 0.1 (default: 2).

    color_cells : str, optional
        The color to use for the cell centroids (default: 'red').

    figsize : tuple, optional
        The size of the figure in inches (width, height). This controls the dimensions of the plot (default: (10, 7)).

    vmax : float, optional
        The upper limit for the colormap. If provided, this limits the color scale to values below `vmax` (default: None).

    Returns:
    -------
    None
        The function generates a scatter plot and optionally saves it to the specified output path.

    Notes:
    -----
    - The transcript data and cell centroid data are extracted from `sdata`. 
    - The `vmax` parameter allows control over the maximum value of the color scale for better visualization control.
    - The plot is saved in the specified format and at the specified output path if `save=True`.
    """

    # Copy the AnnData object for cell data
    adata = sdata['table'].copy()

    # Use raw layer for transcript data
    adata.X = sdata['table'].layers['raw']

    # Extract x, y centroid coordinates from the cell data
    adata.obs['x_centroid'] = [sp[0] for sp in adata.obsm['spatial']]
    adata.obs['y_centroid'] = [sp[1] for sp in adata.obsm['spatial']]

    # Extract transcript data from the specified layer
    transcripts = sdata.points[layer].compute()

    # Create output directory if it doesn't exist
    figures_path = os.path.join(output_path, 'figures')
    os.makedirs(figures_path, exist_ok=True)

    # Create the scatter plot
    plt.figure(figsize=figsize)

    # Plot transcript locations, colored by the selected feature (transcript_group)
    plt.scatter(transcripts[xcoord], transcripts[ycoord], c=transcripts[transcript_group], s=size*0.1, cmap=colormap, vmax=vmax)

    # Plot cell centroids
    plt.scatter(adata.obs[xcellcoord], adata.obs[ycellcoord], s=size, color=color_cells)

    # Set plot title
    plt.title(f'{transcript_group}')

    # Save the plot if specified
    if save:
        plt.savefig(os.path.join(figures_path, f"Scatter_{transcript_group}_{colormap}.{format}"))
  
def heatmap(data,output_path:str='',save=False,figsize=None,tag='',title=None, cmap="RdBu_r", annot=False, cbar=True,vmax=None,vmin=0,
                 row_cluster=True,col_cluster=True):
    if figsize==None:
        figsize=(data.shape[1]/3,(data.shape[0]/7)+2)
    g=sns.clustermap(data, cmap=cmap, annot=annot, figsize=figsize,vmax=vmax,vmin=vmin,col_cluster=col_cluster,row_cluster=row_cluster)
    #plt.tight_layout()
    g.fig.suptitle(title) 
    if save==True:
        figures_path = os.path.join(output_path, 'figures')
        os.makedirs(figures_path, exist_ok=True)
        plt.savefig(os.path.join(figures_path, "heatmap_"+tag+".pdf"))
    plt.show()

def plot_crosstab(data, xvar: str = '', yvar: str = '', normalize=True, axis=1, kind='barh', 
                  save=True, figures_path: str = '', stacked=True, figsize=(6, 10), 
                  cmap='viridis', saving_format='pdf', sortby=None):
    """
    Plot a cross-tabulation between two variables in a dataset and visualize it as either a bar plot, horizontal bar plot, or heatmap.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset containing the variables for the cross-tabulation.
        
    xvar : str, optional (default: '')
        The variable to use on the x-axis for the cross-tabulation.
        
    yvar : str, optional (default: '')
        The variable to use on the y-axis for the cross-tabulation.
        
    normalize : bool, optional (default: True)
        Whether to normalize the cross-tabulated data (percentages). If True, the data will be normalized.
        
    axis : int, optional (default: 1)
        The axis to normalize across. Use `1` for row normalization and `0` for column normalization.
        
    kind : str, optional (default: 'barh')
        The kind of plot to generate. Options include:
        - 'barh': Horizontal bar plot
        - 'bar': Vertical bar plot
        - 'heatmap': Heatmap visualization
        - 'clustermap': Clustermap visualization
        
    save : bool, optional (default: True)
        If True, the plot will be saved to a file.
        
    figures_path : str, optional (default: '')
        The directory path where the figure should be saved. If not specified, the plot will be saved in the current directory.
        
    stacked : bool, optional (default: True)
        If True, the bar plots will be stacked. Only applicable for 'barh' and 'bar' plot kinds.
        
    figsize : tuple, optional (default: (6, 10))
        The size of the figure for the plot (width, height).
        
    cmap : str, optional (default: 'viridis')
        The colormap to use for the plot, especially for heatmap and clustermap visualizations.
        
    saving_format : str, optional (default: 'pdf')
        The format to save the plot in. Options include 'png', 'pdf', etc.
        
    sortby : str, optional (default: None)
        The column or row to sort the cross-tabulated data by before plotting.

    Returns:
    --------
    None
        This function generates a plot and optionally saves it to a file.
    """
    
    # Compute the crosstab data
    crosstab_data = compute_crosstab(data, xvar=xvar, yvar=yvar)
    
    # Normalize the data if required
    if normalize:
        crosstab_data = crosstab_data.div(crosstab_data.sum(axis=axis), axis=0)
        normtag = 'normalize'
    else:
        normtag = 'raw'
    
    # Sort the data if needed
    if sortby is not None:
        crosstab_data = crosstab_data.sort_values(by=sortby)
    
    # Generate the plot filename
    plot_filename = f"{kind}_{xvar}_{yvar}_{normtag}_{cmap}.{saving_format}"
    
    # Plot based on the selected kind
    if kind == 'barh':
        plt.figure()
        crosstab_data.plot(kind='barh', stacked=stacked, figsize=figsize, width=0.99, colormap=cmap)
        plt.title(f'{xvar}_vs_{yvar}')
        if save:
            plt.savefig(os.path.join(figures_path, plot_filename))
        plt.show()

    elif kind == 'bar':
        plt.figure()
        crosstab_data.plot(kind='bar', stacked=stacked, figsize=figsize, width=0.99, colormap=cmap)
        plt.title(f'{xvar}_vs_{yvar}')
        if save:
            plt.savefig(os.path.join(figures_path, plot_filename))
        plt.show()

    elif kind == 'heatmap':
        plt.figure()
        sns.heatmap(crosstab_data, figsize=figsize, cmap=cmap)
        plt.title(f'{xvar}_vs_{yvar}')
        if save:
            plt.savefig(os.path.join(figures_path, plot_filename))
        plt.show()

    elif kind == 'clustermap':
        plt.figure()
        sns.clustermap(crosstab_data, figsize=figsize, cmap=cmap)
        plt.title(f'{xvar}_vs_{yvar}')
        if save:
            plt.savefig(os.path.join(figures_path, plot_filename))
        plt.show()

def pie_of_positive(data, groupby: str = '', figures_path: str = '', save: bool = True):
    """
    Generates a pie chart showing the proportion of positive and negative values 
    for a specified categorical variable in the data.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data containing the categorical variable to group by.
    groupby : str, optional
        The column name in the data to group by (default is an empty string).
    figures_path : str, optional
        The path where the pie chart will be saved if `save` is True (default is an empty string).
    save : bool, optional
        Whether to save the figure as a PDF (default is True). If False, the chart is displayed without saving.

    Returns
    -------
    None
        The function generates and either saves or displays a pie chart, 
        depending on the value of the `save` parameter.
    """
    
    plt.figure()
    y = np.array([np.sum(data[groupby] == False), np.sum(data[groupby] == True)])
    mylabels = [f"{groupby}=False", f"{groupby}=True"]
    
    plt.pie(y, labels=mylabels, colors=['#a0b7e0', '#c5e493'])
    plt.title(f'Proportion of {groupby}')
    
    if save:
        plot_filename = f"pie_positivity_{groupby}_.pdf"
        plt.savefig(os.path.join(figures_path, plot_filename))

def genes_over_noise(sdata, scores_by_genes,layer='extracellular_transcripts', output_path:str='',save=True,format:str='pdf'):
    """
    This function plots log fold change per gene over noise using a boxplot.
    
    Parameters:
    - data_quantified: DataFrame containing the extracellular transcript data, including feature names and codeword categories.
    - scores_by_genes: DataFrame containing gene scores with feature names and log fold ratios.
    - output_path: Path to save the figure.
    """
    data_quantified=sdata.points[layer].compute()
    # Create the output directory for figures if it doesn't exist
    PATH_FIGURES = os.path.join(output_path, "figures")
    os.makedirs(PATH_FIGURES, exist_ok=True)

    # Map feature names to codeword categories
    feature2codeword = dict(zip(data_quantified['feature_name'], data_quantified['codeword_category']))
    scores_by_genes['codeword_category'] = scores_by_genes['feature_name'].map(feature2codeword)

    # Plot the boxplot
    sns.boxplot(
        data=scores_by_genes,
        y="codeword_category",
        x="log_fold_ratio",
        hue="codeword_category",
    )
    # Plot the reference line at x = 0
    plt.plot([0, 0], [*plt.gca().get_ylim()], "r--")
    if save==True:
    # Save the figure
        plt.savefig(os.path.join(PATH_FIGURES, f"boxplot_log_fold_change_per_gene{format}"), bbox_inches="tight", pad_inches=0)
    # Show the plot
    plt.show()

def moranI_histogram(svg_df, save=True, figures_path: str = '', bins: int = 200, format: str = 'pdf'):
    """
    Plots the distribution of Moran's I scores from a DataFrame.

    Parameters:
    -----------
    svg_df : pandas.DataFrame
        DataFrame containing a column 'I' with Moran's I scores.
    save : bool, optional, default=True
        Whether to save the plot as a file.
    figures_path : str, optional
        Path to save the figure. Only used if `save=True`.
    bins : int, optional, default=200
        Number of bins to use in the histogram.
    format : str, optional, default='pdf'
        Format in which to save the figure (e.g., 'pdf', 'png').

    Returns:
    --------
    None
    """
    # Check if figures_path exists if saving the figure
    if save and figures_path:
        if not os.path.exists(figures_path):
            raise ValueError(f"The provided path '{figures_path}' does not exist.")
    
    # Plot the distribution
    plt.figure(figsize=(8, 6))
    plt.hist(svg_df.sort_values(by='I', ascending=False)['I'], bins=bins)
    plt.xlabel("Moran's I")
    plt.ylabel("Frequency")
    plt.title("Distribution of Moran's I Scores")
    
    # Save the plot if requested
    if save:
        file_name = os.path.join(figures_path, f'barplot_moranI_by_gene.{format}')
        plt.savefig(file_name, format=format)
        print(f"Plot saved to: {file_name}")
    
    plt.show()

def proportion_above_threshold(
    df, 
    threshold_col='proportion_above_threshold', 
    feature_col='feature_name', 
    top_percentile=0.05, 
    bottom_percentile=0.05, 
    specific_transcripts=None, 
    figsize=(4, 10), 
    orientation='h', 
    bar_color="black", 
    title='Proportion of distant exRNa (>30um) from source', 
    xlabel='Proportion above threshold', 
    ylabel='Feature',
    save=False,
    output_path:str='',format='pdf'
):
    """
    Plots the top and bottom percentiles of features with the highest and lowest proportions above a threshold,
    or visualizes a specific list of transcripts.

    Parameters:
    - df: DataFrame containing feature proportions.
    - threshold_col: Column name for proportions above the threshold (default: 'proportion_above_threshold').
    - feature_col: Column name for feature names (default: 'feature_name').
    - top_percentile: Proportion (0-1) of features with the highest proportions to display (default: 0.05 for top 5%).
    - bottom_percentile: Proportion (0-1) of features with the lowest proportions to display (default: 0.05 for bottom 5%).
    - specific_transcripts: List of specific transcript names to plot (optional).
    - figsize: Tuple specifying the size of the plot (default: (4, 10)).
    - orientation: Orientation of the bars ('h' for horizontal, 'v' for vertical, default: 'h').
    - bar_color: Color of the bars (default: 'black').
    - title: Title of the plot (default: 'Proportion of distant exRNa (>30um) from source').
    - xlabel: Label for the x-axis (default: 'Proportion above threshold').
    - ylabel: Label for the y-axis (default: 'Feature').
    """
    df=df[~df[threshold_col].isna()]
    print(df.shape)
    # Filter for top and bottom percentiles if no specific transcripts are provided
    if specific_transcripts is None:
        top_cutoff = df[threshold_col].quantile(1 - top_percentile)
        bottom_cutoff = df[threshold_col].quantile(bottom_percentile)
        plot_data = pd.concat([
            df[df[threshold_col] >= top_cutoff],  # Top percentile
            df[df[threshold_col] <= bottom_cutoff]  # Bottom percentile
        ])
    else:
        plot_data = df[df[feature_col].isin(specific_transcripts)]

    # Plot
    plt.figure(figsize=figsize)
    if orientation=='h':
        plt.barh(plot_data['feature_name'],plot_data[threshold_col],color=bar_color)
    if orientation=='v':
        plt.bar(plot_data['feature_name'],plot_data[threshold_col],color=bar_color)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    figures_path = os.path.join(output_path, 'figures')
    os.makedirs(figures_path, exist_ok=True)
    filename=f'barplot_distant_from_source_min{bottom_percentile}_max{top_percentile}_{bar_color}'
    if save==True:
        plt.savefig(os.path.join(figures_path, f'{filename}.{format}'))
    plt.show()

def nmf_factors_exrna_cells_W(sdata,nmf_adata_key='nmf_data', save=True,saving_path='',spot_size:int=30,cmap='viridis'):
    # Plot the factors for each cell in a spatial plot
    adata=sdata[nmf_adata_key]
    W = adata.obsm['W_nmf']
    for factor in range(20):
        # Add the factor values to adata.obs for plotting
        adata.obs[f'NMF_factor_{factor + 1}'] = W[:, factor]
        # Plot spatial map of cells colored by this factor
        if save:
            sc.pl.spatial(adata, color=f'NMF_factor_{factor + 1}', cmap=cmap, title=f'NMF Factor {factor + 1}', spot_size=30,show=False)
            plt.savefig(saving_path+'/figures/'+ f'spatialnmf{factor}.png')
            plt.show()
        else:
            sc.pl.spatial(adata, color=f'NMF_factor_{factor + 1}', cmap=cmap, title=f'NMF Factor {factor + 1}', spot_size=spot_size)

def nmf_gene_contributions(sdata,nmf_adata_key='nmf_data', save=True, vmin=0.0, vmax=0.02,saving_path='',cmap='viridis',figsize=(5,5)):
    adata=sdata[nmf_adata_key]
    loadings=pd.DataFrame(adata.uns['H_nmf'],columns=adata.var.index)
    loadings_filtered=loadings.loc[:,np.max(loadings,axis=0)>0.05].transpose()
    figures_path = os.path.join(saving_path, 'figures')
    os.makedirs(figures_path, exist_ok=True)

    # Sort by maximum feature in cell types
    max_indices = np.argmax(loadings_filtered.values, axis=1)
    loadings_filtered = loadings_filtered.iloc[np.argsort(max_indices)]
    loadings_filtered.index = loadings_filtered.index[np.argsort(max_indices)]

    # Heatmap plot
    plt.figure(figsize=figsize)
    sns.heatmap(loadings_filtered, cmap=cmap, vmax=1)
    if save==True:
        plt.savefig(os.path.join(figures_path, "loadings_NMF.pdf"))
    plt.show()
    plt.close()  # Close the figure to avoid memory issues
    
def apply_exrnaH_to_cellular_to_create_cellularW(adata_extracellular_with_nmf, adata_annotated_cellular):
    H = adata_extracellular_with_nmf.uns['H_nmf']
    
    # Check the number of genes in adata_annotated and spots2region_output to match gene loadings (H)
    genes_spots2region = adata_extracellular_with_nmf.var_names
    genes_annotated = adata_annotated_cellular.var_names

    # Get intersection of genes between the two datasets
    common_genes = genes_annotated.intersection(genes_spots2region)

    # Filter both datasets to keep only common genes
    adata_annotated_cellular = adata_annotated_cellular[:, common_genes]
    H_filtered = H[:, np.isin(genes_spots2region, common_genes)]  # Filtered NMF gene loadings for common genes

    # Apply the NMF factors to the annotated dataset
    # Calculate the new W matrix by multiplying the annotated data with the filtered H
    W_annotated = adata_annotated_cellular.X @ H_filtered.T

    adata_annotated_cellular.obsm['factors']=pd.DataFrame(W_annotated,index=adata_annotated_cellular.obs.index)
    #print(W_annotated[:, 0].shape)
    # Add the factors as new columns in adata_annotated.obs
    for factor in range(W_annotated.shape[1]):
        adata_annotated_cellular.obs[f'NMF_factor_{factor + 1}'] = W_annotated[:, factor]

    return adata_annotated_cellular

def paired_nmf_factors(
    sdata, 
    layer='nmf_data', 
    n_factors=5,  # Number of NMF factors to plot
    figsize=(12, 6),  # Size of the figure
    spot_size_exrna=5,  # Spot size for extracellular transcripts
    spot_size_cells=10,  # Spot size for cell map
    cmap_exrna='YlGnBu',  # Colormap for extracellular transcripts
    cmap_cells='Reds',  # Colormap for cells
    vmax_exrna='p99',  # Maximum value for color scale (extracellular)
    vmax_cells=None,  # Maximum value for color scale (cells)
    save=False,
    output_path:str='',
    format='pdf'
):
    """
    Plots the spatial distribution of NMF factors for extracellular transcripts and cells.
    
    Parameters:
    ----------
    sdata : spatial data object
        The spatial data object containing both extracellular and cell data.
    
    layer : str, optional
        Layer in sdata to extract the NMF data from (default: 'nmf_data').
    
    n_factors : int, optional
        Number of NMF factors to plot (default: 5).
    
    figsize : tuple, optional
        Size of the figure for each subplot (default: (12, 6)).
    
    spot_size_exrna : float, optional
        Size of the spots for extracellular transcript scatter plot (default: 5).
    
    spot_size_cells : float, optional
        Size of the spots for cell scatter plot (default: 10).
    
    cmap_exrna : str, optional
        Colormap for the extracellular transcript NMF factors (default: 'YlGnBu').
    
    cmap_cells : str, optional
        Colormap for the cell NMF factors (default: 'Reds').
    
    vmax_exrna : str or float, optional
        Maximum value for extracellular transcript color scale (default: 'p99').
    
    vmax_cells : str or float, optional
        Maximum value for cell color scale (default: None).
    """

    # Extract NMF data from sdata
    adata = sdata[layer]
    adata_annotated = sdata['table']
    
    # Get the factors from the obsm attribute (NMF results)
    factors = pd.DataFrame(adata.obsm['W_nmf'], index=adata.obs.index)
    factors.columns = [f'NMF_factor_{fact+1}' for fact in factors.columns]
    
    # Add each NMF factor to adata.obs
    for f in factors.columns:
        adata.obs[f] = factors[f]
    
    # Loop over the specified number of NMF factors and plot
    for factor in range(n_factors):
        factor_name = f'NMF_factor_{factor + 1}'
        
        # Create a figure with a single subplot for each factor
        fig, axs = plt.subplots(1, 1, figsize=figsize)
        
        # Plot the spatial distribution for extracellular transcripts
        sc.pl.spatial(
            adata, color=factor_name, cmap=cmap_exrna, 
            title=f'NMF Factor {factor + 1} (Extracellular)', 
            ax=axs, show=False, spot_size=spot_size_exrna, vmax=vmax_exrna
        )
        
        # Overlay the cell spatial distribution
        sc.pl.spatial(
            adata_annotated, color=factor_name, cmap=cmap_cells, 
            title=f'NMF Factor cell-red/exRNa-blue {factor + 1}', 
            ax=axs, show=False, spot_size=spot_size_cells, vmax=vmax_cells
        )
        if save==True:
         if save:
            figures_path = os.path.join(output_path, 'figures')
            os.makedirs(figures_path, exist_ok=True)
            file_name = os.path.join(figures_path, f'Spatial_NMF Factor {factor + 1}.{format}')
            plt.savefig(file_name)

        # Adjust layout and show the combined plot
        plt.tight_layout()
        plt.show()

def W(adata, n_factors, save=True): # not very intuitive
    # Plot the spatial map of cells colored by each factor
    for factor in range(n_factors):
        sc.pl.spatial(adata, color=f'NMF_factor_{factor + 1}', cmap='plasma', title=f'NMF Factor {factor + 1}', spot_size=15, save=f'exo_to_cell_spatial_{factor}.png')

def spatial_interactions(
    sdata: AnnData,
    layer: str = 'extracellular_transcripts_enriched',
    gene: str = 'Arc',
    gene_key: str = 'feature_name',
    cell_id_key: str = 'cell_id',
    color_target:str='blue',
    color_source:str='red',
    color_transcript:str='green',
    spatial_key: str = 'spatial',
    img: Optional[Union[bool, Sequence]] = None,
    img_alpha: Optional[float] = None,
    image_cmap: Optional[Colormap] = None,
    size: Optional[Union[float, Sequence[float]]] = 8,
    alpha: float = 0.6,
    title: Optional[Union[str, Sequence[str]]] = None,
    legend_loc: Optional[str] = 'best',
    figsize: Tuple[float, float] = (10, 10),
    dpi: Optional[int] = 100,
    save: Optional[Union[str, Path]] = None,
    **kwargs
):
    # Extract relevant data
    transcripts = sdata.points[layer]
    trans_filt = transcripts[transcripts[gene_key] == gene]
    target_cells = trans_filt['closest_target_cell'].compute()
    source_cells = trans_filt['closest_source_cell'].compute()
    cell_positions = pd.DataFrame(sdata['table'].obsm[spatial_key], index=sdata.table.obs[cell_id_key], columns=['x', 'y'])

    # Plotting
    plt.figure(figsize=figsize, dpi=dpi)
    if img is not None:
        plt.imshow(img, alpha=img_alpha, cmap=image_cmap, **kwargs)
    plt.scatter(cell_positions['x'], cell_positions['y'], c='grey', s=0.6, alpha=alpha, **kwargs)
    plt.scatter(cell_positions.loc[target_cells, 'x'], cell_positions.loc[target_cells, 'y'], c=color_target, s=size, label='Target Cells', **kwargs)
    plt.scatter(cell_positions.loc[source_cells, 'x'], cell_positions.loc[source_cells, 'y'], c=color_source, s=size, label='Source Cells', **kwargs)
    plt.scatter(trans_filt['x'], trans_filt['y'], c=color_transcript, s=size*0.4, label='Transcripts', **kwargs)
    


    # Titles and Legends
    plt.title(title or gene)
    plt.legend(loc=legend_loc)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    # Save the plot if path provided
    if save:
        plt.savefig(save)
    plt.show()

def interactions_with_arrows(
    sdata: AnnData,
    layer: str = 'extracellular_transcripts_enriched',
    gene: str = 'Arc',
    gene_key: str = 'feature_name',
    cell_id_key: str = 'cell_id',
    color_target: str = 'blue',
    color_source: str = 'red',
    color_transcript:str='green',
    spatial_key: str = 'spatial',
    img: Optional[Union[bool, Sequence]] = None,
    img_alpha: Optional[float] = None,
    image_cmap: Optional[Colormap] = None,
    size: Optional[Union[float, Sequence[float]]] = 8,
    alpha: float = 0.6,
    title: Optional[Union[str, Sequence[str]]] = None,
    legend_loc: Optional[str] = 'best',
    figsize: Tuple[float, float] = (10, 10),
    dpi: Optional[int] = 100,
    save: Optional[Union[str, Path]] = None,
    **kwargs
):
    # Extract relevant data
    transcripts = sdata.points[layer]
    trans_filt = transcripts[transcripts[gene_key] == gene]
    target_cells = trans_filt['closest_target_cell'].compute()
    source_cells = trans_filt['closest_source_cell'].compute()
    cell_positions = pd.DataFrame(sdata['table'].obsm[spatial_key], index=sdata.table.obs[cell_id_key], columns=['x', 'y'])

    # Plotting
    plt.figure(figsize=figsize, dpi=dpi)
    if img is not None:
        plt.imshow(img, alpha=img_alpha, cmap=image_cmap, **kwargs)

    # Plot arrows between each paired source and target cell
    for source, target in zip(source_cells, target_cells):
        if source in cell_positions.index and target in cell_positions.index:
            if source!=target:
                 x_start, y_start = cell_positions.loc[source, 'x'], cell_positions.loc[source, 'y']
                 x_end, y_end = cell_positions.loc[target, 'x'], cell_positions.loc[target, 'y']
                 plt.arrow(x_start, y_start, x_end - x_start, y_end - y_start, color='black', alpha=0.8, head_width=8, head_length=8)
    
    # Plot source and target cells
    plt.scatter(cell_positions['x'], cell_positions['y'], c='grey', s=0.6, alpha=alpha, **kwargs)
    plt.scatter(cell_positions.loc[target_cells, 'x'], cell_positions.loc[target_cells, 'y'], c=color_target, s=size, label='Target Cells', **kwargs)
    plt.scatter(cell_positions.loc[source_cells, 'x'], cell_positions.loc[source_cells, 'y'], c=color_source, s=size, label='Source Cells', **kwargs)
    plt.scatter(trans_filt['x'], trans_filt['y'], c=color_transcript, s=size*0.4, label='Transcripts', **kwargs)
    
    # Titles and Legends
    plt.title(title or gene)
    plt.legend(loc=legend_loc)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    # Save the plot if path provided
    if save:
        plt.savefig(save)
    plt.show()
