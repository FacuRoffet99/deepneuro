import matplotlib.pyplot as plt
import numpy as np

def plot_ts(ts, min=None, max=None):
    t = np.arange(0, ts.shape[1], 1)
    plt.plot(t[min:max], ts.T[min:max])
    plt.xlabel('Time')

def get_and_plot_psd(ts, TR=2, plot_w_until=0.5):
    # Sampling frequency (Hz)
    sampling_frequency = 1/TR  # Example sampling frequency
    max_freq_ids = np.zeros(len(ts), dtype=int)
    for i, t in enumerate(ts):
        # Compute the FFT
        fft_result = np.fft.fft(t)

        # Calculate the frequencies corresponding to the FFT result
        frequencies = np.fft.fftfreq(len(fft_result), 1.0 / sampling_frequency)

        # Calculate the PSD
        psd = np.abs(fft_result) ** 2
        # Plot only positive frequencies
        positive_frequencies = 2*np.pi* frequencies[:len(frequencies) // 2]
        positive_psd = psd[:len(psd) // 2]

        max_freq_ids[i] = positive_psd.argmax()

        # Create the plot
        # plt.figure(figsize=(8, 6))
        max_id = len(positive_frequencies[positive_frequencies<plot_w_until])
        plt.plot(positive_frequencies[:max_id], positive_psd[:max_id])

    plt.xlabel('w (rad/s)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title('Power Spectral Density (PSD) - Positive Frequencies')
    # Display the plot
    plt.show()

    return max_freq_ids, positive_frequencies