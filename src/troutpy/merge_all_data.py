import numpy as np
import pandas as pd

def merge_all_sources(
    path_overexpression_scores_per_gene: str,
    path_spatially_variable_score_moranI_per_gene: str,
    path_extracellular_transcripts_misseg_labels: str,
    path_proportion_of_transcripts_away_from_source: str,
    path_proportion_of_grouped_exRNA: str,
) -> pd.DataFrame:
    
    df2 = pd.read_parquet(path_overexpression_scores_per_gene)
    df3 = pd.read_parquet(path_spatially_variable_score_moranI_per_gene)
    df3 = df3.reset_index().rename({"index": "feature_name"}, axis=1)
    df4 = pd.read_parquet(path_extracellular_transcripts_misseg_labels)
    df6 = pd.read_parquet(path_proportion_of_transcripts_away_from_source)
    df7 = pd.read_parquet(path_proportion_of_grouped_exRNA)

    df23 = pd.merge(df2, df3, on="feature_name")
    assert np.all(
        df23.sort_values(by="feature_name").loc[:, ["count",	"log_fold_ratio"]].values == \
        df2.sort_values(by="feature_name").loc[:, ["count",	"log_fold_ratio"]].values
    )
    assert np.all(
        df23.sort_values(by="feature_name").loc[:, ["I", "pval_norm", "var_norm", "pval_norm_fdr_bh"]].values == \
        df3.sort_values(by="feature_name").loc[:, ["I", "pval_norm", "var_norm", "pval_norm_fdr_bh"]].values
    )
    
    df_extracellular = df4.loc[(df4.cell_id == "UNASSIGNED") & (~df4.missegmentation_associated)]
    df_intracellular = df4.loc[(df4.cell_id != "UNASSIGNED") | (df4.missegmentation_associated)]
    assert len(np.intersect1d(df_intracellular.index, df_extracellular.index)) == 0

    df_extracellular_counts = df_extracellular.groupby("feature_name").feature_name.count()\
        .to_frame()\
        .rename({"feature_name": "counts_extracellular"}, axis=1)\
        .reset_index()

    df_intracellular_counts = df_intracellular.groupby("feature_name").feature_name.count()\
        .to_frame()\
        .rename({"feature_name": "counts_intracellular"}, axis=1)\
        .reset_index()

    df_counts = pd.merge(
        df_extracellular_counts,
        df_intracellular_counts,
        on="feature_name",
        how="left",
    )

    assert len(df_counts.feature_name.unique()) == len(df_extracellular_counts.feature_name.unique())

    df_counts.loc[:, "ratio_extra_over_intra"] = df_counts.counts_extracellular / df_counts.counts_intracellular
    df_counts.loc[:, "fraction_extra_over_all"] = df_counts.counts_extracellular / (df_counts.counts_intracellular + df_counts.counts_extracellular)
    
    df67 = pd.merge(df6, df7, on="feature_name")

    assert np.all(
        df6.sort_values(by="feature_name").loc[:, ["feature_name", "proportion_above_threshold", "bin"]].values ==\
        df67.sort_values(by="feature_name").loc[:, ["feature_name", "proportion_above_threshold", "bin"]].values
    )

    assert np.all(
        df7.sort_values(by="feature_name").loc[:, ["proportion_of_colocalized"]].values ==\
        df67.sort_values(by="feature_name").loc[:, ["proportion_of_colocalized"]]
    )
    
    df_tmp = pd.merge(
        df23,
        df_counts,
        on="feature_name"
    )
    
    df_all = pd.merge(df_tmp, df67, on="feature_name")
    
    return df_all