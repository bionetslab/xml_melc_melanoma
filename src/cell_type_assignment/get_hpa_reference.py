import pandas as pd
from pathlib import Path
import os

def get_hpa_reference(tissue, hpa_data_path, max_normalize=False):
    expr, meta = _load_data(tissue, hpa_data_path)
    expr_per_cell_type = _aggregate_expression_per_cell_type(expr, meta)
    if max_normalize:
        expr_per_cell_type /= expr_per_cell_type.max()
    return expr_per_cell_type


def _load_data(tissue, hpa_data_path):
    expr_file_name = os.path.join(hpa_data_path, 'rna_single_cell_type_tissue.tsv.zip')
    meta_file_name = os.path.join(hpa_data_path, 'rna_single_cell_cluster_description.tsv')
    # ---
    expr = pd.read_csv(expr_file_name, sep='\t')
    meta = pd.read_csv(meta_file_name, sep='\t')
    expr = expr[expr.Tissue == tissue.lower()]
    meta = meta[meta.Tissue == tissue.capitalize()]
    expr.reset_index(inplace=True, drop=True)
    meta.reset_index(inplace=True, drop=True)
    meta.index = meta.Cluster
    return expr, meta


def _aggregate_expression_per_cell_type(expr, meta):
    cell_types = list(set(meta['Cell type']))
    genes = list(set(expr['Gene name']))
    expr_per_cell_type = pd.DataFrame(0.0, index=cell_types, columns=genes)
    count_per_cell_type = pd.DataFrame(0, index=cell_types, columns=genes)
    for i in range(expr.shape[0]):
        gene = expr.loc[i, 'Gene name']
        ntpm = expr.loc[i, 'nTPM']
        cluster = expr.loc[i, 'Cluster']
        cell_type = meta.loc[cluster, 'Cell type']
        cell_count = meta.loc[cluster, 'Cell count']
        expr_per_cell_type.loc[cell_type, gene] += ntpm * cell_count
        count_per_cell_type.loc[cell_type, gene] += cell_count
    # Cell type nTPMs
    expr_per_cell_type /= count_per_cell_type
    return expr_per_cell_type
