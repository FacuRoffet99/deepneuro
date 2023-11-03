import numpy as np
from numba import jit
from skimage.transform import resize
from .PIM import PhaseInteractionMatrix

def tril_indices_column(N, k=0):
    row_i, col_i = np.nonzero(
        np.tril(np.ones(N), k=k).T)  # Matlab works in column-major order, while Numpy works in row-major.
    Isubdiag = (col_i,
                row_i)  # Thus, I have to do this little trick: Transpose, generate the indices, and then "transpose" again...
    return Isubdiag

@jit(nopython=True)
def mean(x, axis=None):
    if axis == None:
        return np.sum(x, axis) / np.prod(x.shape)
    else:
        return np.sum(x, axis) / x.shape[axis]

@jit(nopython=True)
def numba_phFCD(phIntMatr_upTri, size_kk3):
    npattmax = phIntMatr_upTri.shape[0]
    phfcd = np.zeros((size_kk3))
    kk3 = 0

    for t in range(npattmax - 2):
        p1_sum = np.sum(phIntMatr_upTri[t:t + 3, :], axis=0)
        p1_norm = np.linalg.norm(p1_sum)
        for t2 in range(t + 1, npattmax - 2):
            p2_sum = np.sum(phIntMatr_upTri[t2:t2 + 3, :], axis=0)
            p2_norm = np.linalg.norm(p2_sum)

            dot_product = np.dot(p1_sum, p2_sum)
            phfcd[kk3] = dot_product / (p1_norm * p2_norm)
            kk3 += 1
    return phfcd

# From [Deco2019]: Comparing empirical and simulated FCD.
# For a single subject session where M time points were collected, the corresponding phase-coherence based
# FCD matrix is defined as a MxM symmetric matrix whose (t1, t2) entry is defined by the cosine similarity
# between the upper triangular parts of the 2 matrices dFC(t1) and dFC(t2) (previously defined; see above).
# For 2 vectors p1 and p2, the cosine similarity is given by (p1.p2)/(||p1||||p2||).
# Epochs of stable FC(t) configurations are reflected around the FCD diagonal in blocks of elevated
# inter-FC(t) correlations.

def phFCD(ts, discardOffset=10):  # Compute the FCD of an input BOLD signal
    phIntMatr = PhaseInteractionMatrix(ts)  # Compute the Phase-Interaction Matrix
    if not np.isnan(phIntMatr).any():  # No problems, go ahead!!!
        (N, Tmax) = ts.shape
        npattmax = Tmax - (2 * discardOffset - 1)  # calculates the size of phfcd vector
        size_kk3 = int((npattmax - 3) * (npattmax - 2) / 2)  # The int() is not needed because N*(N-1) is always even, but "it will produce an error in the future"...
        Isubdiag = tril_indices_column(N, k=-1)  # Indices of triangular lower part of matrix
        phIntMatr_upTri = np.zeros((npattmax, int(N * (N - 1) / 2)))  # The int() is not needed, but... (see above)
        for t in range(npattmax):
            phIntMatr_upTri[t,:] = phIntMatr[t][Isubdiag]
        phfcd = numba_phFCD(phIntMatr_upTri, size_kk3,)

    else:
        print('############ Warning!!! phFCD.from_fMRI: NAN found ############')
        phfcd = np.array([np.nan])
    # if saveMatrix:
    #     buildMatrixToSave(phfcd, npattmax - 2)
    return phfcd

def buildFullMatrix(FCD_data):
    LL = FCD_data.shape[0]
    # T is size of the matrix given the length of the lower/upper triangular part (displaced by 1)
    T = int((1. + np.sqrt(1. + 8. * LL)) / 2.)
    fcd_mat = np.zeros((T, T))
    fcd_mat[np.triu_indices(T, k=1)] = FCD_data
    fcd_mat += fcd_mat.T
    return fcd_mat

def reduced_phFCD(ts, size, order=3):
    observable = phFCD(ts)
    mat = buildFullMatrix(observable)
    resized = resize(mat, (size, size), order=order, anti_aliasing=True)
    return resized    