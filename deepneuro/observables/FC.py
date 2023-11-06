from nilearn.connectome import ConnectivityMeasure

def FC(ts):
    return ConnectivityMeasure(kind='correlation', standardize='zscore_sample').fit_transform(ts.transpose(0,2,1))
