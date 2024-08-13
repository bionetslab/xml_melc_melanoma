from sklearn.mixture import GaussianMixture
import numpy as np


class SplitTuple:
    def __init__(self, split_var, cell_type, mapped_genes=None):
        self.split_var = split_var
        self.cell_type = cell_type
        self.mapped_genes = mapped_genes
        if self.mapped_genes is None:
            self.mapped_genes = [split_var]
        self.sense = None
        self.fold_change = None
        self.aic_unimodal = None
        self.aic_bimodal = None
        self.means_bimodal = None
        self.stds_bimodal = None
        self.cutoff = None

    def is_fitted(self):
        return self.means_bimodal is not None

    def is_initialized(self):
        return self.fold_change is not None

    def is_specific(self, min_fold_change):
        if not self.is_initialized():
            raise RuntimeError('Sense and fold change have not been computed.')
        return self.fold_change >= min_fold_change

    def is_bimodal(self, z_score_cutoff=1.96):
        if not self.is_fitted():
            raise RuntimeError('Gaussian mixtures have not been fitted.')
        if self.aic_unimodal < self.aic_bimodal:
            return False
        cutoff_lower = self.means_bimodal[0] + z_score_cutoff * self.stds_bimodal[0]
        cutoff_upper = self.means_bimodal[1] - z_score_cutoff * self.stds_bimodal[1]
        return cutoff_lower < cutoff_upper

    def compute_sense_and_specificity(self, reference_expr_per_cell_type):
        reference_cell_type = reference_expr_per_cell_type.loc[self.cell_type, self.mapped_genes]
        reference_other_cell_types = reference_expr_per_cell_type.loc[:, self.mapped_genes].drop(self.cell_type, axis=0)
        if (reference_cell_type > reference_other_cell_types.max()).all():
            self.sense = 'positive'
            self.fold_change = (reference_cell_type / (reference_other_cell_types.max() + 1e-10)).min()
        elif (reference_cell_type < reference_other_cell_types.min()).all():
            self.sense = 'negative'
            self.fold_change = (reference_other_cell_types.min() / (reference_cell_type + 1e-10)).min()
        else:
            self.sense = 'undefined'
            self.fold_change = 0.0

    def fit_gaussian_mixtures(self, adata):
        data = adata[adata.obs['cell_type'] == 'unknown', self.split_var].X.toarray()
        gm_unimodal = GaussianMixture(n_components=1).fit(data)
        gm_bimodal = GaussianMixture(n_components=2).fit(data)
        self.aic_unimodal = gm_unimodal.aic(data)
        self.aic_bimodal = gm_bimodal.aic(data)

        mean_1 = gm_bimodal.means_[0][0]
        mean_2 = gm_bimodal.means_[1][0]
        stds = np.sqrt(gm_bimodal.covariances_)
        if mean_1 < mean_2:
            self.means_bimodal = [mean_1, mean_2]
            self.stds_bimodal = [stds[0][0][0], stds[1][0][0]]
        else:
            self.means_bimodal = [mean_2, mean_1]
            self.stds_bimodal = [stds[1][0][0], stds[0][0][0]]

    def has_cell_type(self, expr):
        if self.sense == 'positive':
            return expr > self.cutoff
        elif self.sense == 'negative':
            return expr < self.cutoff
        else:
            raise RuntimeError('Split sense undefined.')

    def assign_cell_type(self, adata, z_score_cutoff=1.96):
        if not self.is_fitted():
            raise RuntimeError('Gaussian mixtures have not been fitted.')
        if not self.is_initialized():
            raise RuntimeError('Sense and fold change have not been computed.')
        if self.sense == 'positive':
            self.cutoff = self.means_bimodal[1] - z_score_cutoff * self.stds_bimodal[1]
        elif self.sense == 'negative':
            self.cutoff = self.means_bimodal[0] + z_score_cutoff * self.stds_bimodal[0]
        data = adata[adata.obs['cell_type'] == 'unknown', self.split_var].to_df()
        for cell in data.index:
            if self.has_cell_type(data.loc[cell, self.split_var]):
                adata.obs.loc[cell, 'cell_type'] = self.cell_type

