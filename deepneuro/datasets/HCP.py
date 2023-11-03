import tarfile
import os
import numpy as np

# Unzip original files
def unzip_files(ts_path, sc_path, save_path):
    tar = tarfile.open(ts_path)
    tar.extractall(save_path)
    tar.close()

    tar = tarfile.open(sc_path)
    tar.extractall(save_path)
    tar.close()

# Read ts files
def read_time_series(data_folder, nodes=25, length=4800, remove_first=4, max_files=None):
    # Get list of files
    data_files = sorted(os.listdir(data_folder))
    # Filter
    if max_files is None:
        n_files = len(data_files)
    else:
        n_files = max_files
    # Get shape of ts's
    lenght, nodes = np.loadtxt(data_folder+data_files[0]).shape
    # Create array of propper size
    time_series = np.ones((n_files, nodes, length))     # i x n x t
    # Read
    for i, f in enumerate(data_files[:n_files]):
        time_series[i,:,:] = np.loadtxt(data_folder+f).T
    # Slicing
    time_series = time_series[:,:,remove_first:]
    # Return ts's
    return time_series

# Read structural connectivity matrices
def read_sc_matrices(data_file, max_files=None):
    # Read data
    data = np.loadtxt(data_file)
    n_files, rows = data.shape
    nodes = int(np.sqrt(rows))
    # Filter
    if max_files is not None:
        n_files = max_files
        data = data[:n_files]
    # Create array of propper size
    SC_matrices = np.ones((n_files, nodes, nodes))    # i x n x n
    # Generate
    for n, d in enumerate(data):
        sc = np.abs(d.reshape(nodes, nodes))
        np.fill_diagonal(sc, 0)
        SC_matrices[n,:,:] = 0.2 * sc/sc.max()
    # Return SC matrices
    return SC_matrices

# Get HCP data
def readHCP(ts_path, sc_path, save_path, max_files=None):
    unzip_files(ts_path, sc_path, save_path)
    # Read ts's
    time_series = read_time_series(save_path+'/'+ts_path.split('/')[-1], max_files=None)
    # Obtain SC matrices
    SC_matrices = read_sc_matrices(save_path+'/'+sc_path.split('/')[-1], max_files=None)
    # Get mean SC matrix
    mean_SC = SC_matrices.mean(axis=0)
    return time_series, SC_matrices, mean_SC

