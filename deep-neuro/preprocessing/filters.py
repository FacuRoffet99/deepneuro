from scipy.signal import butter, detrend, filtfilt
import numpy.matlib as mtlib
import numpy as np

def demean(x,dim=0):
    dims = x.size
    return x - mtlib.tile(np.mean(x,dim), dims)  # repmat(np.mean(x,dim),dimrep)

def BandPassFilter(boldSignal, f_low, f_high, TR, k, removeStrongArtefacts=True):
    # Convenience method to apply a filter (always the same one) to all areas in a BOLD signal. For a single,
    # isolated area evaluation, better use the method below.
    (N, Tmax) = boldSignal.shape
    fnq = 1./(2.*TR)              # Nyquist frequency
    Wn = [f_low/fnq, f_high/fnq]                                   # butterworth bandpass non-dimensional frequency
    bfilt, afilt = butter(k,Wn, btype='band', analog=False)   # construct the filter
    # bfilt = bfilt_afilt[0]; afilt = bfilt_afilt[1]  # numba doesn't like unpacking...
    signal_filt = np.zeros(boldSignal.shape)
    for seed in range(N):
        if not np.isnan(boldSignal[seed, :]).any():  # No problems, go ahead!!!
            ts = demean(detrend(boldSignal[seed, :]))  # Probably, we do not need to demean here, detrend already does the job...

            if removeStrongArtefacts:
                ts[ts>3.*np.std(ts)] = 3.*np.std(ts)    # Remove strong artefacts
                ts[ts<-3.*np.std(ts)] = -3.*np.std(ts)  # Remove strong artefacts

            signal_filt[seed,:] = filtfilt(bfilt, afilt, ts, padlen=3*(max(len(bfilt),len(afilt))-1))  # Band pass filter. padlen modified to get the same result as in Matlab
        else:  # We've found problems, mark this region as "problematic", to say the least...
            print(f'############ Warning!!! BandPassFilter: NAN found at region {seed} ############')
            signal_filt[seed,0] = np.nan
    return signal_filt

def STDFilter(time_series):
    return np.array([ts/ts.std() for ts in time_series])

def AmplitudeFilter(time_series):
    return time_series/np.abs(time_series).max()