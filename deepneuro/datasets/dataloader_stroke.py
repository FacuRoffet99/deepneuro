# =====================================================================================
# Methods to input Stroke data
# =====================================================================================
import numpy as np
import scipy.io as sio
import os, csv
import numpy.random as rnd


base_folder = "/content/drive/MyDrive/data/Stroke/"
saveSufix = 'subj_'

delete235 = False


def characterizeConnectivityMatrix(C):
    return np.max(C), np.min(C), np.average(C), np.std(C), np.max(np.sum(C, axis=0)), np.average(np.sum(C, axis=0))


# def checkClassifications(subjects):
#     # ============================================================================
#     # This code is to check whether we have the information of the type of subject
#     # They can be one of: ???
#     # ============================================================================
#     input_classification = csv.reader(open(base_folder+"/subjects.csv", 'r'))
#     classification = dict((rows[0],rows[1]) for rows in input_classification)
#     mistery = []
#     for pos, subject in enumerate(subjects):
#         if subject in classification:
#             print('{}: Subject {} classified as {}'.format(pos, subject, classification[subject]))
#         else:
#             print('{}: Subject {} NOT classified'.format(pos, subject))
#             mistery.append(subject)
#     print("Misisng {} subjects:".format(len(mistery)), mistery)
#     print()
#     return classification


def getClassifications(Subjects):
    # ============================================================================
    # This code is to check whether we have the information of the type of subject
    # ============================================================================
    all_clasific = {'HC_' + str(pos): 'HC' for pos, ts in enumerate(TsControl)} | \
                   {id: 'Acute' for id in SubjectData}
    return all_clasific


# =====================================================================================
# Methods to input AD data
# =====================================================================================

# ===================== compute the Avg SC matrix over the HC sbjects
def computeAvgSC_HC_Matrix(classification, baseFolder):
    HC = [subject for subject in classification.keys() if classification[subject] == 'HC']
    print("SC + HC: {} (0)".format(HC[0]))
    sc_folder = baseFolder+'/'+HC[0]+"/DWI_processing"
    SC = np.loadtxt(sc_folder+"/connectome_weights.csv")

    sumMatrix = SC
    for subject in HC[1:]:
        print("SC + HC: {}".format(subject))
        sc_folder = baseFolder+'/'+subject+"/DWI_processing"
        SC = np.loadtxt(sc_folder+"/connectome_weights.csv")
        sumMatrix += SC
    return sumMatrix / len(HC)  # but we normalize it afterwards, so we probably do not need this...


# ===================== Load one specific subject data
# def loadSubjectData(subject, correcSCMatrix=True):
#
#     return SCnorm, abeta_burden, tau_burden, fullSeries


# ===================== Load all fMRI data
def load_fullCohort_fMRI(classification, baseFolder, cohort='HC'):
    if cohort == 'HC':
        all_fMRI = {'HC_'+str(pos): ts for pos, ts in enumerate(TsControl)}
    else:
        idx = dataSetLabels.index(cohort)
        all_fMRI = {id: SubjectData[id]['Ts'+str(idx-1)] for id in SubjectData}
    return all_fMRI


# ===================== Load all fMRI data
def load_all_HC_fMRI(classification, baseFolder):
    load_fullCohort_fMRI(classification, baseFolder, cohort='HC')


# ===================== Normalize a SC matrix
normalizationFactor = 0.2
avgHuman66 = 0.0035127188987848714
areasHuman66 = 66  # yeah, a bit redundant... ;-)
maxNodeInput66 = 0.7275543904602363
def correctSC(SC):
    N = SC.shape[0]
    logMatrix = np.log(SC+1)
    # areasSC = logMatrix.shape[0]
    # avgSC = np.average(logMatrix)
    # === Normalization ===
    # finalMatrix = normalizationFactor * logMatrix / logMatrix.max()  # normalize to the maximum, as in Gus' codes
    # finalMatrix = logMatrix * avgHuman66/avgSC * (areasHuman66*areasHuman66)/(areasSC * areasSC)  # normalize to the avg AND the number of connections...
    maxNodeInput = np.max(np.sum(logMatrix, axis=0))  # This is the same as np.max(logMatrix @ np.ones(N))
    finalMatrix = logMatrix * maxNodeInput66 / maxNodeInput
    return finalMatrix


def analyzeMatrix(name, C):
    max, min, avg, std, maxNodeInput, avgNodeInput = characterizeConnectivityMatrix(C)
    print(name + " => Shape:{}, Max:{}, Min:{}, Avg:{}, Std:{}".format(C.shape, max, min, avg, std), end='')
    print("  => impact=Avg*#:{}".format(avg*C.shape[0]), end='')
    print("  => maxNodeInputs:{}".format(maxNodeInput), end='')
    print("  => avgNodeInputs:{}".format(avgNodeInput))


# =====================================================================================
# cutTimeSeriesIfNeeded:
# This is used to avoid "infinite" computations for some cases (i.e., subjects) that have fMRI
# data that is way longer than any other subject, causing almost impossible computations to perform,
# because they last several weeks (~4 to 6), which seems impossible to complete with modern Windows SO,
# which restarts the computer whenever it want to perform supposedly "urgent" updates...
# =====================================================================================
force_Tmax = True
# This method is to perform the timeSeries cutting when excessively long...
def cutTimeSeriesIfNeeded(timeseries, limit_forcedTmax=200):
    if force_Tmax and timeseries.shape[1] > limit_forcedTmax:
        print(f"cutting lengthy timeseries: {timeseries.shape[1]} to {limit_forcedTmax}")
        timeseries = timeseries[:,0:limit_forcedTmax]
    return timeseries


# =====================================================================================
# Load HDF5 matlab file
# =====================================================================================
def checkNaNs(TsControl, SubjectData, TsStroke):
    for pos, ts in enumerate(TsControl):
        if np.isnan(ts).any():
            print(f'Control subject {pos} has NaN!')

    for pos, subj in enumerate(SubjectData):
        if np.isnan(SubjectData[subj]['Ts0']).any():
            print(f'Subject {subj} has NaN in Acute tiemseries!')
        if np.isnan(SubjectData[subj]['Ts1']).any():
            print(f'Subject {subj} has NaN in Intermediate timeseries!')
        if np.isnan(SubjectData[subj]['Ts2']).any():
            print(f'Subject {subj} has NaN in Chronic timeseries!')

    # We do not use TsStroke, but let's check it anyway...
    for pos, ts in enumerate(TsStroke):
        if np.isnan(ts).any():
            print(f'TsStroke subject {pos} has NaN!')


def read_matlab_h5py(filename):
    def fixNaN(ts):
        N = ts.shape[1]
        rows = np.where(np.isnan(ts))[0]
        idx = np.unique(rows)
        for id in idx:
            ts[id,:] = np.std(np.nanmean(ts[200:N, :], axis=0)) * rnd.randn(1, N)
        return ts

    def adjustLength(all_ts):
        N = all_ts[list(all_ts.keys())[0]][0].shape[0]
        longest = np.max([[all_ts[id][0].shape[1], all_ts[id][1].shape[1], all_ts[id][2].shape[1]] for id in all_ts])
        print(f'N = {N}, longest = {longest}')
        res = {}
        for id in all_ts:
            res[id] = [np.zeros((N,longest)), np.zeros((N,longest)), np.zeros((N,longest))]
            # effectiveLen = all_ts[id][0].shape[1] if
            res[id][0][:, :all_ts[id][0].shape[1]] = all_ts[id][0]
            res[id][1][:, :all_ts[id][1].shape[1]] = all_ts[id][1]
            res[id][2][:, :all_ts[id][2].shape[1]] = all_ts[id][2]
        print('done')
        return res

    import h5py
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        # print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        # a_group_key = list(f.keys())[0]
        # get the object type for a_group_key: usually group or dataset
        # print(type(f['subjects_idxs']))
        # If a_group_key is a dataset name,
        # this gets the dataset values and returns as a list
        # data = list(f[a_group_key])
        # preferred methods to get dataset values:
        # ds_obj = f[a_group_key]  # returns as a h5py dataset object
        # ds_arr = f[a_group_key][()]  # returns as a numpy array

        # ['Masks_stroke', 'SC_controls', 'Stroke_subjects_with_3_timepoints', 'TS_control', 'TS_stroke', 'subjects_idxs']

        _MasksList = list(f['Masks_stroke'])
        Masks = [np.array(f[_MasksList[i][0]]) for i in range(len(_MasksList))]

        SC = f['SC_controls'][()]

        _TsList = f['Stroke_subjects_with_3_timepoints'][()]
        TSs = {f[_TsList[3,i]][()][0,0]:
                   [fixNaN(np.array(f[_TsList[0,i]][()])),
                    fixNaN(np.array(f[_TsList[1,i]][()])),
                    fixNaN(np.array(f[_TsList[2,i]][()]))]
               for i in range(_TsList.shape[1])}
        TSs = adjustLength(TSs)

        _TsControl = list(f['TS_control'])
        TsControl = [fixNaN(np.array(f[_TsControl[i][0]]).T) for i in range(len(_TsControl))]

        _TsStroke = list(f['TS_stroke'])
        TsStroke = [fixNaN(np.array(f[_TsStroke[i][0]]).T) for i in range(len(_TsStroke))]

        _IDx = list(f['subjects_idxs'])
        IDx = [f[_IDx[i][0]][0,0] for i in range(len(_IDx))]

        if delete235:
            print('Region 235 should be deleted...')

    return Masks, SC, TSs, TsControl, TsStroke, IDx


def sortInfo(Masks, TSs, IDx):
    # We do not have all info for all subjects, so we need to sort it out...
    allData = {}
    for id in TSs:
        pos = IDx.index(id)
        allData[saveSufix + str(int(id))] = {
            'Mask': Masks[pos],
            'Ts0': TSs[id][0],
            'Ts1': TSs[id][1],
            'Ts2': TSs[id][2],
        }
    return allData


# _Masks, SC, _TSs, TsControl, _TsStroke, _IDx = read_matlab_h5py(base_folder+'for_gus_patow.mat')
# SubjectData = sortInfo(_Masks, _TSs, _IDx)  # SC and TsControl do not need sorting, and by now, _TsStroke is not used

dataSetLabels = ['HC', 'Acute', 'Intermediate', 'Chronic']

# =====================================================================================
# Some test/verification code
# =====================================================================================
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 15})

    analyzeMatrix('SC', SC)

    checkNaNs(TsControl, SubjectData)

    print('Done!')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
