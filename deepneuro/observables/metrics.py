import numpy as np
from scipy import stats

def get_correlation(a, b):
    return np.corrcoef(a[np.triu_indices(a.shape[-1])], b[np.triu_indices(a.shape[-1])])[0,1]

def get_ks_distance(a, b):
    d, pvalue = stats.ks_2samp(a.flatten(), b.flatten())
    return d