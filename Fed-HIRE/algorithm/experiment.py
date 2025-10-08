import itertools
import os

from munkres import Munkres

os.environ["OMP_NUM_THREADS"] = '2'

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, fowlkes_mallows_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score

class GlobalConfig:
    cnt = -1

def labelMapping(y_true, y_pred):
    '''
    Remaps predicted cluster labels to best match ground-truth labels using the Hungarian algorithm.

    This is especially useful in unsupervised learning settings (e.g., clustering),
    where predicted labels may not have a one-to-one correspondence with true labels.


    :param y_true: 1D NumPy array of ground-truth labels
    :param y_pred: 1D NumPy array of predicted cluster labels
    :return: new_label: predicted labels remapped to best match ground-truth
    '''
    Label1 = np.unique(y_true)
    Label2 = np.unique(y_pred)
    G = np.zeros((len(Label1), len(Label2)))

    "Build the cost matrix G: entry (i, j) is the number of common samples between Label1[i] and Label2[j]"
    for i in range(len(Label1)):
        ind_cla1 = y_true == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(len(Label2)):
            ind_cla2 = y_pred == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)

    "Handle the case where number of clusters doesn't match number of true labels"
    if len(Label1) != len(Label2):
        max_size = max(len(Label1), len(Label2))
        padded_G = np.zeros((max_size, max_size))
        padded_G[:len(Label1), :len(Label2)] = G

        "Pad extra rows/columns with large cost to prevent matching to them unless necessary"
        padded_G[len(Label1):, :] = np.max(G) + 1
        padded_G[:, len(Label2):] = np.max(G) + 1
        G = padded_G

    "Perform optimal label assignment using Hungarian algorithm (Munkres implementation)"
    m = Munkres()
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    "Apply mapping to predicted labels"
    new_label = np.zeros(y_pred.shape)
    cnt = max(Label1)
    for i in range(len(Label2)):
        if c[i] < len(Label1):
            new_label[y_pred == Label2[i]] = Label1[c[i]]
        else:
            new_label[y_pred == Label2[i]] = GlobalConfig.cnt
            GlobalConfig.cnt -= 1

    return new_label.astype(int)


def purity(y_true, y_pred):
    clusters = np.unique(y_pred)
    y_true = np.reshape(y_true, (-1, 1))
    y_pred = np.reshape(y_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(y_pred == c)[0]
        y_temp = y_true[idx, :].reshape(-1)
        count.append(np.bincount(y_temp).max())
    return np.sum(count) / y_true.shape[0]

def ARI(y_true, y_pred, beta=1.):
    score = adjusted_rand_score(y_true, y_pred)
    return score

def NMI(y_true, y_pred):
    score = normalized_mutual_info_score(y_pred, y_true)
    return score

def AC(y_true, y_pred):
    score = accuracy_score(y_true, y_pred)
    return score

def validation(L):

    result = []
    y_true = []
    y_pred = []
    X = []
    for i, l in enumerate(L):
        y_true.append(l["label"])
        y_pred.append(l['predict'])
        X.append(l['data'])
    y_true = list(itertools.chain.from_iterable(y_true))
    y_pred = list(itertools.chain.from_iterable(y_pred))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    X = np.concatenate(X)


    y_pred = labelMapping(y_true, y_pred)

    result.append(purity(y_true, y_pred))
    result.append(ARI(y_true, y_pred))
    result.append(NMI(y_true, y_pred))
    result.append(AC(y_true, y_pred))

    return np.array(result), 'success'
