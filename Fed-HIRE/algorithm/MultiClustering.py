import itertools
import os
import statistics

os.environ["OMP_NUM_THREADS"] = '2'

import numpy as np
np.seterr(divide='ignore',invalid='ignore')
from sklearn.cluster import KMeans

def similarity(x, C, W):
    n, d = np.shape(C)
    s = np.zeros(n)
    for i in range(n):
        s[i] += np.sum(W[i, :] * ((x - C[i, :]) ** 2))
        s[i] *= -0.5
        s[i] = np.exp(s[i])
        if np.isnan(s[i]):
            s[i] = 0

    SUM = np.sum(s)
    s /= SUM
    return s


def FCPL(X, rate, k_0):
    n, d = np.shape(X)
    k = int(np.round((k_0 * n)))

    seed = np.random.choice(range(n), k, replace=False)

    totalData = X[seed, :]
    W = np.full((k, d), 1 / d).astype(float)
    cnt = np.full(k, 1).astype(int)
    belta = np.full(k, 1).astype(float)
    U = np.zeros((n, k)).astype(int)
    y = (np.ones(n) * -1).astype(int)
    for i in range(k):
        y[seed[i]] = i
        U[seed[i]][i] = 1
    theta = []
    granularity = []
    label = (np.ones((n)) * -1).astype(int)
    iter = 0
    while True:
        iter += 1
        noChange = True
        for i in range(n):
            C = totalData / np.repeat(np.expand_dims(np.sum(U, axis=0).T, axis=1), d, axis=1)
            s = similarity(X[i, :], C, W)
            g = 1 / (1 + np.exp(-10 * belta + 5))
            gamma = cnt / np.sum(cnt)
            s = (1 - gamma) * g * s
            sort_index = np.argsort(-s)
            v = sort_index[0]
            belta[v] += rate

            if len(sort_index) > 1:
                r = sort_index[1]
                belta[r] -= (rate * (s[r] / s[v]))

            cnt[v] += 1

            oldLabel = y[i]
            y[i] = v
            if oldLabel != v:
                totalData[v, :] += X[i, :]
                U[i][v] = 1
                if oldLabel != -1:
                    totalData[oldLabel, :] -= X[i, :]
                    U[i][oldLabel] = 0
                noChange = False
        if noChange == True:
            break
        else:
            M = np.zeros((k, d))
            F = np.zeros((k, d))
            H = np.zeros((k, d))
            C = totalData / np.repeat(np.expand_dims(np.sum(U, axis=0).T, axis=1), d, axis=1)
            for i in range(k):
                mu = []
                sigma = []
                index = np.where(y == i)[0]
                out_index = range(n)
                out_index = np.setdiff1d(out_index, index)
                count = len(index)
                M[i, :] = np.sum(np.exp(-0.5 * ((X[index, :] - C[i, :]) ** 2)), axis=0) / count
                mu.append(np.sum(X[index, :], axis=0) / count)
                mu.append(np.sum(X[out_index, :], axis=0) / (n - count))
                sigma.append(np.sum((X[index, :] - mu[0]) ** 2, axis=0) / (count - 1))
                sigma.append(np.sum((X[out_index, :] - mu[1]) ** 2, axis=0) / (n - count - 1))
                F[i, :] = np.sqrt(1 -
                                  np.sqrt((2 * np.sqrt(sigma[0]) * np.sqrt(sigma[1])) / (sigma[0] + sigma[1])) *
                                  np.exp(-(((mu[0] - mu[1]) ** 2) / (4 * (sigma[0] + sigma[1])))))

            H = M * F
            W = H / np.repeat(np.expand_dims(np.sum(H, axis=1), axis=1), d, axis=1)

    index = np.where(np.sum(U, axis=0) > 0)[0]
    length = len(index)
    for i in range(length):
        theta.append(totalData[index[i], :] / np.sum(U[:, index[i]]))
        label[np.where(U[:, index[i]] == 1)[0]] = i
    theta = np.stack(theta)
    k = length


    return theta, label, k

def calWeight(X, C, label):
    N, d = np.shape(X)

    alpha = np.zeros((len(C), d))
    beta = np.zeros((len(C), d))
    for k in range(len(C)):
        inside_index = np.where(label == k)
        outside_index = np.where(label != k)


        for r in range(d):
            if len(inside_index[0]) == 0 or len(outside_index[0]) == 0:
                alpha[k, r] = 0
                beta[k, r] = 0
                continue
            m = np.max(len(np.unique(X[:, r])))
            temp = 0
            for t in range(m):
                temp += (len(np.where(X[inside_index, r] == t)[1]) / len(inside_index[0]) -
                         len(np.where(X[outside_index, r] == t)[1]) / len(outside_index[0])) ** 2
            alpha[k, r] = np.sqrt(temp / 2)

            temp = 0
            for i in inside_index[0]:
                temp += len(np.where(X[inside_index, r] == X[i, r])[1]) / len(inside_index[0])
            beta[k, r] = temp / len(inside_index[0])

    H = alpha * beta

    w = H / np.tile(np.sum(H, axis=1)[:, np.newaxis], (1, d))

    return w

def labelassign(X, C, w):
    N, d = np.shape(X)
    k = np.size(C, axis=0)
    result = []
    for i in range(N):
        # 计算hamming distance
        distance = []
        for t in range(k):
            hamming_distance = (X[i, :] != C[t, :]).astype(float)
            hamming_distance *= w[t, :]
            distance.append(np.sum(hamming_distance))
        distance = np.array(distance)
        u = np.argmin(distance)
        result.append(u)
    return np.array(result)

def fed_hire(labelData, true_k):

    N, d = np.shape(labelData)

    unique_rows = np.unique(labelData, axis=0)
    C = unique_rows[np.random.choice(unique_rows.shape[0], true_k, replace=False)]

    convergence = False
    w = np.ones((true_k, d)) * (1 / d)
    result = labelassign(labelData, C, w)
    iter = 0
    while convergence == False:
        iter += 1
        old_result = result
        Centroid = []
        for t in range(true_k):
            index = np.where(result == t)
            if len(index[0]) == 0:
                Centroid.append(np.ones(d) * -1)
                continue
            modes = []
            for i in range(d):
                x = labelData[index, i].tolist()
                modes.append(statistics.mode(x[0]))
            Centroid.append(modes)
        Centroid = np.array(Centroid)
        w = calWeight(labelData, Centroid, old_result)

        result = labelassign(labelData, Centroid, w)
        if np.array_equal(result, old_result):
            convergence = True

    predict = result
    return predict, w, Centroid

def MCPL(X, rate, k_1):
    n, d = np.shape(X)
    k = int(np.round(k_1 * n))

    Label = []
    K = []
    epoch = 0
    while True:
        epoch += 1

        convergence = False
        seed = np.random.choice(range(n), k, replace=False)
        totalData = X[seed, :]
        W = np.full((k, d), 1 / d).astype(float)
        cnt = np.full(k, 1).astype(int)
        belta = np.full(k, 1).astype(float)
        U = np.zeros((n, k)).astype(int)
        y = (np.ones(n) * -1).astype(int)
        for i in range(k):
            y[seed[i]] = i
            U[seed[i]][i] = 1
        theta = []
        label = (np.ones((n)) * -1).astype(int)

        iter = 0
        while True:
            iter += 1
            noChange = True
            for i in range(n):
                C = totalData / np.repeat(np.expand_dims(np.sum(U, axis=0).T, axis=1), d, axis=1)
                s = similarity(X[i, :], C, W)
                g = 1 / (1 + np.exp(-10 * belta + 5))
                gamma = cnt / np.sum(cnt)
                s = (1 - gamma) * g * s
                sort_index = np.argsort(-s)
                v = sort_index[0]
                r = sort_index[1]
                belta[v] += rate
                belta[r] -= (rate * (s[r] / s[v]))
                cnt[v] += 1

                oldLabel = y[i]
                y[i] = v
                if oldLabel != v:
                    totalData[v, :] += X[i, :]
                    U[i][v] = 1
                    if oldLabel != -1:
                        totalData[oldLabel, :] -= X[i, :]
                        U[i][oldLabel] = 0
                    noChange = False
            if noChange == True:
                break
            else:
                M = np.zeros((k, d))
                F = np.zeros((k, d))
                H = np.zeros((k, d))
                C = totalData / np.repeat(np.expand_dims(np.sum(U, axis=0).T, axis=1), d, axis=1)
                for i in range(k):
                    mu = []
                    sigma = []
                    index = np.where(y == i)[0]
                    out_index = range(n)
                    out_index = np.setdiff1d(out_index, index)
                    count = len(index)
                    M[i, :] = np.sum(np.exp(-0.5 * ((X[index, :] - C[i, :]) ** 2)), axis=0) / count
                    mu.append(np.sum(X[index, :], axis=0) / count)
                    mu.append(np.sum(X[out_index, :], axis=0) / (n - count))
                    sigma.append(np.sum((X[index, :] - mu[0]) ** 2, axis=0) / (count - 1))
                    sigma.append(np.sum((X[out_index, :] - mu[1]) ** 2, axis=0) / (n - count - 1))
                    F[i, :] = np.sqrt(1 -
                                      np.sqrt((2 * np.sqrt(sigma[0]) * np.sqrt(sigma[1])) / (sigma[0] + sigma[1])) *
                                      np.exp(-(((mu[0] - mu[1]) ** 2) / (4 * (sigma[0] + sigma[1])))))

                H = M * F

                W = H / np.repeat(np.expand_dims(np.sum(H, axis=1), axis=1), d, axis=1)

        index = np.where(np.sum(U, axis=0) > 0)[0]
        length = len(index)
        for i in range(length):
            label[np.where(U[:, index[i]] == 1)[0]] = i

        if length == k:
            if epoch == 1:
                Label.append(label)
                K.append(length)
            convergence = True
        else:
            Label.append(label)
            K.append(length)
            k = length
        if length < 2:
            convergence = True
        if convergence == True:
            break

    return Label, K