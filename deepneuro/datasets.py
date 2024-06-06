import mat73
import scipy.io

# ---------------------- HCP task ----------------------
def get_ts_hcp_task(data_root):
    '''
    Reads the HCP task dataset.

    Args:
        data_root (str): Folder containing the data files (.mat).

    Returns:
        time_series (dict): Dictionary where the keys are the tasks and the values are the arrays.
    '''
    time_series = {}
    time_series['memory'] = mat73.loadmat(data_root+'hcp1003_WM_LR_dbs80.mat')
    time_series['gambling'] = mat73.loadmat(data_root+'hcp1003_GAMBLING_LR_dbs80.mat')
    time_series['motor'] = mat73.loadmat(data_root+'hcp1003_MOTOR_LR_dbs80.mat')
    time_series['language'] = mat73.loadmat(data_root+'hcp1003_LANGUAGE_LR_dbs80.mat')
    time_series['social'] = mat73.loadmat(data_root+'hcp1003_SOCIAL_LR_dbs80.mat')
    time_series['relational'] = mat73.loadmat(data_root+'hcp1003_RELATIONAL_LR_dbs80.mat')
    time_series['emotion'] = mat73.loadmat(data_root+'hcp1003_EMOTION_LR_dbs80.mat')
    time_series['rest'] = mat73.loadmat(data_root+'hcp1003_REST1_LR_dbs80.mat')
    return time_series

def get_sc_hcp_task(matrix_file):
    SC = scipy.io.loadmat(matrix_file)['SC_dbs80FULL']   
    return SC 


# ---------------------- Stroke ----------------------
def get_ts_stroke(data_file):
    '''
    Reads the stroke dataset.

    Args:
        data_root (str): Data file (.mat).

    Returns:
        time_series (dict): Dictionary where the keys are the cohorts and the values are the arrays.
    '''
    data = mat73.loadmat(data_file)
    time_series = {}
    time_series['control'] = data['TS_control']
    time_series['accute'] = [i[0].T for i in data['Stroke_subjects_with_3_timepoints']]
    time_series['intermediate'] = [i[1].T for i in data['Stroke_subjects_with_3_timepoints']]
    time_series['chronic'] = [i[2].T for i in data['Stroke_subjects_with_3_timepoints']]
    return time_series

def get_sc_stroke(data_file):
    SC = mat73.loadmat(data_file)['SC_controls'] 
    return SC 



# ---------------------- HCP (THIS CODE IS OLD, MUST BE CHECKED BEFORE USED) ----------------------

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
def readHCP(ts_path, sc_path, save_path, max_files=None, nodes=25, length=4800):
    unzip_files(ts_path, sc_path, save_path)
    # Read ts's
    time_series = read_time_series(save_path+'/node_timeseries/3T_HCP1200_MSMAll_d'+str(nodes)+'_ts2/', max_files=None, nodes=nodes, length=length)
    # Obtain SC matrices
    SC_matrices = read_sc_matrices(save_path+'/netmats/3T_HCP1200_MSMAll_d'+str(nodes)+'_ts2/netmats1.txt', max_files=None)
    # Get mean SC matrix
    mean_SC = SC_matrices.mean(axis=0)
    return time_series, SC_matrices, mean_SC