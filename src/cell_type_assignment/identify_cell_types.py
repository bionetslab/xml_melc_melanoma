import pandas as pd
from .split_tuple import SplitTuple
from .cell_type_tree import CellTypeTree


def identify_cell_types(adata, reference_expr_per_cell_type, min_fold_change=2, z_score_cutoff=1.96):
    """

    :param adata: AnnData object containing single-cell data. Each variable name var_name must either correspond to
    a column in the reference gene expression data or adata.uns[var_name] must contain a (possibly singleton) list of
    column names from the reference gene expression data to which the variable is mapped.
    :type adata: adata.AnnData
    :param reference_expr_per_cell_type:
    :type reference_expr_per_cell_type:
    :param min_fold_change:
    :type min_fold_change:
    :param z_score_cutoff:
    :type z_score_cutoff:
    :return:
    :rtype:
    """
    split_tuples = []
    ct = ['unknown' for _ in range(adata.n_obs)]
    adata.obs['cell_type'] = ct #pd.Categorical(ct)
    done = False
    while not done:
        done = True
        candidate_split_tuples = initialize_split_tuples(adata, reference_expr_per_cell_type, min_fold_change)
        for split_tuple in candidate_split_tuples:
            split_tuple.fit_gaussian_mixtures(adata)
            if split_tuple.is_bimodal(z_score_cutoff):
                
                if len(split_tuples) == 0 or not split_tuple.mapped_genes in [split_tuples[i].mapped_genes for i in range(len(split_tuples))]:
                    split_tuple.assign_cell_type(adata, z_score_cutoff)
                    reference_expr_per_cell_type.drop(split_tuple.cell_type, axis=0, inplace=True)
                    split_tuples.append(split_tuple)
                    done = False
                    break
        remaining_cell_types = list(reference_expr_per_cell_type.index)
        if len(remaining_cell_types) == 1:
            done = True
    remaining_cells = adata.obs[adata.obs['cell_type'] == 'unknown'].index
    for cell in remaining_cells:
        adata.obs.loc[cell, 'cell_type'] = '|'.join(remaining_cell_types)
    return CellTypeTree(split_tuples, remaining_cell_types)


def initialize_split_tuples(adata, reference_expr_per_cell_type, min_fold_change):
    split_tuples = []
    for var in adata.var_names:
        mapped_genes = None
        if var in adata.uns:
            mapped_genes = adata.uns[var]
        split_tuples += [SplitTuple(var, cell_type, mapped_genes) for cell_type in reference_expr_per_cell_type.index]
    for split_tuple in split_tuples:
        split_tuple.compute_sense_and_specificity(reference_expr_per_cell_type)
    split_tuples = [split_tuple for split_tuple in split_tuples if split_tuple.is_specific(min_fold_change)]
    return sorted(split_tuples, key=lambda x: x.fold_change, reverse=True)
