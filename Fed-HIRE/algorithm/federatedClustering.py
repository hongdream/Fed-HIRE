import matplotlib
import os
os.environ["OMP_NUM_THREADS"] = '1'

from algorithm.experiment import *
from algorithm.MultiClustering import FCPL, fed_hire, MCPL
import numpy as np
import progressbar
matplotlib.use("Qt5Agg")


def fedClustering(L, rate, k_0=0.5, k_1=0.5):

    y_true = []
    for i, l in enumerate(L):
        # print(f"client{i + 1}: true k = {len(np.unique(l['label']))}")
        y_true.append(l["label"])
    y_true = list(itertools.chain.from_iterable(y_true))
    y_true = np.array(y_true)
    n_class = len(np.unique(y_true))

    Theta = []
    localK = []

    "FCPL on Client"
    for i, l in enumerate(progressbar.progressbar(L)):
        theta, label, k = FCPL(l["data"], rate, k_0)
        Theta.append(theta)
        l["localLabel"] = label
        localK.append(k)
    T = Theta
    T = np.concatenate(T)
    "MCPL on server"
    Label, K = MCPL(T, rate, k_1)

    "Fed-HIRE"
    labelData = np.stack(Label).T
    center_predict, w, C = fed_hire(labelData, n_class)

    "Local update"
    centroid = []
    for i in range(n_class):
        index = np.where(center_predict == i)
        centroid.append(np.mean(T[index[0], :], axis=0))
    index = []
    start = 0
    for k in localK:
        index.append([i for i in range(start, start + k)])
        start += k

    for i in range(len(L)):
        GranularLabel = np.zeros((L[i]['size'], len(Label)))
        predict = np.zeros((L[i]['size']))
        for j in range(len(Label)):
            granular_temp = Label[j][index[i]]
            predict_temp = center_predict[index[i]]
            for k in range(localK[i]):
                idc = np.where(L[i]["localLabel"] == k)
                GranularLabel[idc, j] = int(granular_temp[k])
                predict[idc] = predict_temp[k]
        L[i]["GranularLabel"] = GranularLabel.astype(int)
        L[i]["predict"] = predict.astype(int)

    return [L, centroid]




