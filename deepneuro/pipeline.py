import numpy as np
from nilearn.connectome import ConnectivityMeasure
from integration.integrators import integrate_hopf_euler_maruyama
from observables.metrics import get_correlation, get_ks_distance

def simulate_and_observe(params, preds, omegas, SC_matrices, G, observable, empirical):
    np.random.seed(18)

    # Get shape
    n_samples, nodes = preds.shape

    # Integrate
    X = integrate_hopf_euler_maruyama(params, preds, omegas, SC_matrices, np.full(len(preds), G))
    # Subsample
    X = X[:,:,::int(params['TR']/params['dt'])]

    # Get connectivity
    if np.isnan(X).any():
        print(f'NaN found! i={i}, G={G}')

    if observable == 'FC':
        predicted = ConnectivityMeasure(kind='correlation').fit_transform(X.transpose(0,2,1))
        # Get correlation between predicted and empirical
        correlation = np.array([get_correlation(empirical[i], predicted[i]) for i in range(n_samples)])
        return correlation
    else:
        predicted = np.array([observable(ts) for ts in X])
        # Get Kolmogorov-Smirnov distance between predicted and empirical
        distance = np.array([get_ks_distance(empirical[i], predicted[i]) for i in range(n_samples)])
        return distance