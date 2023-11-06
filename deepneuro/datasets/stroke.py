import numpy as np
from .dataloader_stroke import *

def readStroke(data_file):
    dataSetLabels = ['HC', 'Acute', 'Intermediate', 'Chronic']

    Masks, SC_hc, TSs, TsControl, TsStroke, IDx = read_matlab_h5py(data_file)

    # Mix useful data
    SubjectData = sortInfo(Masks, TSs, IDx)

    # Convert data to numpy format and delete empty timesteps for stroke data
    subject_ids = np.array([k.split('_')[-1] for k in SubjectData.keys()])
    data_hc = np.array([ts for ts in TsControl]) # not used
    masks = np.array([SubjectData[id]['Mask'] for id in SubjectData])
    data_acute = np.array([SubjectData[id]['Ts'+str(0)] for id in SubjectData])[:,:,:896]
    data_inter = np.array([SubjectData[id]['Ts'+str(1)] for id in SubjectData])[:,:,:896]
    data_chronic = np.array([SubjectData[id]['Ts'+str(2)] for id in SubjectData])[:,:,:896]

    # Delete last node
    data_hc = data_hc[:,:-1,:]
    data_acute = data_acute[:,:-1,:]
    data_inter = data_inter[:,:-1,:]
    data_chronic = data_chronic[:,:-1,:]
    masks = masks[:,:-1,:-1]
    SC_hc = SC_hc[:-1,:-1]

    # Normalize SC
    norm_SC_hc = correctSC(SC_hc)

    # Estimate SC for stroke
    norm_SC_stroke = masks/100 * norm_SC_hc
    analyzeMatrix('SC', norm_SC_stroke)

    # Expand SC of HC to have one per subject
    norm_SC_hc = norm_SC_hc[np.newaxis].repeat(len(data_hc), axis=0)
    analyzeMatrix('SC', norm_SC_hc)

    return (data_hc, data_acute, data_inter, data_chronic), (norm_SC_hc, norm_SC_stroke)
