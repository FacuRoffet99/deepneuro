import numpy as np
from numba import jit

@jit(nopython=True)
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix
    corr_mat = np.corrcoef(x.flatten(), y.flatten())
    # Return entry [0,1]
    return corr_mat[0,1]

@jit(nopython=True)
def calc_length(start, end, step):
    # This fails for a negative step e.g., range(10, 0, -1).
    # From https://stackoverflow.com/questions/31839032/python-how-to-calculate-the-length-of-a-range-without-creating-the-range
    return (end - start - 1) // step + 1

@jit(nopython=True)
def swFCD(signal, windowSize=60, windowStep=20):  # Compute the FCD of an input BOLD signal
    (N, Tmax) = signal.shape
    lastWindow = Tmax - windowSize  # 190 = 220 - 30
    N_windows = calc_length(0, lastWindow, windowStep)  # N_windows = len(np.arange(0, lastWindow, windowStep))

    if not np.isnan(signal).any():  # No problems, go ahead!!!
        Isubdiag = np.tril_indices(N, k=-1)  # Indices of triangular lower part of matrix

        # For each pair of sliding windows calculate the FC at t and t2 and
        # compute the correlation between the two.
        cotsampling = np.zeros((int(N_windows*(N_windows-1)/2)))
        kk = 0
        ii2 = 0
        for t in range(0, lastWindow, windowStep):
            jj2 = 0
            sfilt = (signal[:, t:t+windowSize+1]).T  # Extracts a (sliding) window between t and t+windowSize (included)
            cc = np.corrcoef(sfilt, rowvar=False)  # Pearson correlation coefficients
            for t2 in range(0, lastWindow, windowStep):
                sfilt2 = (signal[:, t2:t2+windowSize+1]).T  # Extracts a (sliding) window between t2 and t2+windowSize (included)
                cc2 = np.corrcoef(sfilt2, rowvar=False)  # Pearson correlation coefficients

                # Numba doesn't like using tuples as indexes, this is a fix for that
                cc_diag = np.zeros(len(Isubdiag[0]))
                cc2_diag = np.zeros(len(Isubdiag[0]))
                for idx, (x, y) in enumerate(zip(Isubdiag[0], Isubdiag[1])):
                    cc_diag[idx] = cc[x, y]
                    cc2_diag[idx] = cc2[x,y]
                ca = pearson_r(cc_diag, cc2_diag)
                # ca = pearson_r(cc[Isubdiag],cc2[Isubdiag])  # Correlation between both FC

                if jj2 > ii2:  # Only keep the upper triangular part
                    cotsampling[kk] = ca
                    kk = kk+1
                jj2 = jj2+1
            ii2 = ii2+1

        return cotsampling
    else:
        return np.zeros((int(N_windows*(N_windows-1)/2)))