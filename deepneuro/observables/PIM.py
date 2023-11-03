import numpy as np
from numba import jit
from scipy import signal
from ..preprocessing.filters import demean

@jit(nopython=True)
def adif(a, b):
    if np.abs(a - b) > np.pi:
        c = 2 * np.pi - np.abs(a - b)
    else:
        c = np.abs(a - b)
    return c

@jit(nopython=True)
def numba_PIM(phases, N, Tmax, dFC, PhIntMatr, discardOffset=10):
  T = np.arange(discardOffset, Tmax - discardOffset + 1)
  for t in T:
    for i in range(N):
      for j in range(i+1):
        dFC[i, j] = np.cos(adif(phases[i, t - 1], phases[j, t - 1]))
        dFC[j, i] = dFC[i, j]
    PhIntMatr[t - discardOffset] = dFC
  return PhIntMatr

def PhaseInteractionMatrix(ts, discardOffset=10):  # Compute the Phase-Interaction Matrix of an input BOLD signal
    if not np.isnan(ts).any():  # No problems, go ahead!!!
        (N, Tmax) = ts.shape
        npattmax = Tmax - (2 * discardOffset - 1)  # calculates the size of phfcd matrix
        # Data structures we are going to need...
        phases = np.empty((N, Tmax))
        dFC = np.empty((N, N))
        PhIntMatr = np.empty((npattmax, N, N))

        for n in range(N):
            Xanalytic = signal.hilbert(demean(ts[n, :]))
            phases[n, :] = np.angle(Xanalytic)

        PhIntMatr = numba_PIM(phases, N, Tmax, dFC, PhIntMatr)

    else:
        print('############ Warning!!! PhaseInteractionMatrix.from_fMRI: NAN found ############')
        PhIntMatr = np.array([np.nan])
    # ======== sometimes we need to plot the matrix. To simplify the code, we save it here if needed...
    # if saveMatrix:
    #     import scipy.io as sio
    #     sio.savemat(save_file + '.mat', {name: PhIntMatr})
    return PhIntMatr