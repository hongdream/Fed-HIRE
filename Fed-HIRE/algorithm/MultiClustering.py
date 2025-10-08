
import statistics
import time

import numpy as np
from numba import njit, prange
np.seterr(divide='ignore',invalid='ignore')
def similarity(x, C, W):
    '''
        Compute the similarity between a data sample `x` and a set of cluster centroids `C`

        :param x: A 1D NumPy array representing a single data sample
        :param C: A 2D NumPy array of cluster centroids
        :param W: A 2D NumPy array of feature-wise weights
        :return: A 1D NumPy array representing the normalized similarity of `x` to each centroid
    '''
    "Compute element-wise squared differences between x and each centroid in C"
    diff = (C - x) ** 2

    "Apply feature-wise weights to the squared differences"
    weighted_diff = W * diff

    "Sum weighted differences across features, apply Gaussian-like transformation"
    s = np.exp(-0.5 * np.sum(weighted_diff, axis=1))

    "Handle potential NaN values (e.g., from overflow or divide-by-zero)"
    s = np.nan_to_num(s)

    "Normalize similarities to make them sum to 1 (convert to probability distribution)"
    s /= np.sum(s)
    return s


def FCPL(X, rate, k_0, max_iter=30):
    '''
    FCPL (Fine-grained Competitive Penalized Learning) Clustering Algorithm.

    This function performs unsupervised clustering by dynamically adjusting cluster importance
    and feature weights, promoting fine-grained and discriminative partitioning of data.

    :param X: 2D NumPy array of input data
    :param rate: Learning rate for updating clusters importance
    :param k_0: Initial cluster rate (e.g., 0.1 means 10% of n_samples used to initialize clusters)
    :return:
        theta: Final cluster centroids
        label: Cluster assignment for each sample
        k: Final number of clusters
    '''

    n, d = np.shape(X)
    k = int(np.round((k_0 * n)))

    "Randomly select initial seeds (cluster representatives)"
    seed = np.random.choice(range(n), k, replace=False)

    "Initialize cluster statistics"
    totalData = X[seed, :]
    W = np.full((k, d), 1 / d).astype(float)
    cnt = np.full(k, 1).astype(int)
    belta = np.full(k, 1).astype(float)
    U = np.zeros((n, k)).astype(int)
    y = (np.ones(n) * -1).astype(int)
    y[seed] = np.arange(k)
    U[seed, np.arange(k)] = 1

    "Initialize outputs"
    theta = []
    granularity = []
    label = (np.ones((n)) * -1).astype(int)
    iter = 0

    "Main clustering loop"
    while True:
        iter += 1
        noChange = True
        for i in range(n):
            "Compute current centroids"
            C = totalData / np.repeat(np.expand_dims(np.sum(U, axis=0).T, axis=1), d, axis=1)

            "Compute similarity of sample x_i to each centroid"
            s = similarity(X[i, :], C, W)

            g = 1 / (1 + np.exp(-10 * belta + 5))
            gamma = cnt / np.sum(cnt)
            s = (1 - gamma) * g * s

            "Select the most similar cluster (winner) and the second-best (rival)"
            sort_index = np.argsort(-s)
            v = sort_index[0]
            belta[v] += rate

            "Penalize the nearest rival cluster"
            if len(sort_index) > 1:
                r = sort_index[1]
                belta[r] -= (rate * (s[r] / s[v]))

            cnt[v] += 1

            oldLabel = y[i]
            y[i] = v

            "Update statistics if assignment changed"
            if oldLabel != v:
                totalData[v, :] += X[i, :]
                U[i][v] = 1
                if oldLabel != -1:
                    totalData[oldLabel, :] -= X[i, :]
                    U[i][oldLabel] = 0
                noChange = False

        "Stop if no assignments changed"
        if noChange == True:
            break
        else:
            "Update feature-cluster weights"
            M = np.zeros((k, d))
            F = np.zeros((k, d))
            H = np.zeros((k, d))

            "Recompute centroids"
            C = totalData / np.repeat(np.expand_dims(np.sum(U, axis=0).T, axis=1), d, axis=1)
            for i in range(k):
                mu = []
                sigma = []
                index = np.where(y == i)[0]
                out_index = range(n)
                out_index = np.setdiff1d(out_index, index)
                count = len(index)

                "Compute similarity-based mean score for each feature"
                M[i, :] = np.sum(np.exp(-0.5 * ((X[index, :] - C[i, :]) ** 2)), axis=0) / count

                "Compute means and variances for intra- and inter-cluster comparison"
                mu.append(np.sum(X[index, :], axis=0) / count)
                mu.append(np.sum(X[out_index, :], axis=0) / (n - count))
                sigma.append(np.sum((X[index, :] - mu[0]) ** 2, axis=0) / (count - 1))
                sigma.append(np.sum((X[out_index, :] - mu[1]) ** 2, axis=0) / (n - count - 1))

                "Compute Hellinger distance-based discriminability for each feature"
                F[i, :] = np.sqrt(1 -
                                  np.sqrt((2 * np.sqrt(sigma[0]) * np.sqrt(sigma[1])) / (sigma[0] + sigma[1])) *
                                  np.exp(-(((mu[0] - mu[1]) ** 2) / (4 * (sigma[0] + sigma[1])))))

            H = M * F
            W = H / np.repeat(np.expand_dims(np.sum(H, axis=1), axis=1), d, axis=1)

        if iter > max_iter:
            break

    "Finalize output: extract centroids and labels for non-empty clusters"
    index = np.where(np.sum(U, axis=0) > 0)[0]
    length = len(index)
    for i in range(length):
        theta.append(totalData[index[i], :] / np.sum(U[:, index[i]]))
        label[np.where(U[:, index[i]] == 1)[0]] = i
    theta = np.stack(theta)
    k = length


    return theta, label, k

def calWeight(X, C, label):
    '''
    Computes feature weights for each cluster based on intra-cluster concentration and
    inter-cluster discrimination.

    The weights are calculated using two components:
    - Alpha: Measures the difference in feature distributions between inside and outside of the cluster.
    - Beta: Measures the concentration (stability) of feature values within the cluster.

    The final weight for each feature is the product of alpha and beta, normalized per cluster.

    :param X: 2D NumPy array, dataset with discrete feature values
    :param C: 2D NumPy array, cluster mode centroids
    :param label: 1D NumPy array, cluster assignment for each sample
    :return: 2D NumPy array, feature weights for each cluster
    '''
    N, d = X.shape
    K = len(C)

    alpha = np.zeros((K, d))
    beta = np.zeros((K, d))

    for k in range(K):
        idx_in = (label == k)
        idx_out = ~idx_in

        if np.sum(idx_in) == 0 or np.sum(idx_out) == 0:
            continue

        X_in = X[idx_in]
        X_out = X[idx_out]

        for r in range(d):
            vals = np.unique(X[:, r])
            max_val = int(np.max(vals)) + 1  # 保证 bincount 覆盖所有值

            "α: Distributional difference (Hellinger-like distance)"
            in_hist = np.bincount(X_in[:, r], minlength=max_val) / X_in.shape[0]
            out_hist = np.bincount(X_out[:, r], minlength=max_val) / X_out.shape[0]
            alpha[k, r] = np.sqrt(np.sum((in_hist - out_hist) ** 2) / 2)

            "β: Stability of the attribute values within the cluster (mean frequency of values)"
            x_r = X_in[:, r]
            counts = np.bincount(x_r, minlength=max_val)
            freqs = counts[x_r] / X_in.shape[0]
            beta[k, r] = np.mean(freqs)

    H = alpha * beta
    H_sum = np.sum(H, axis=1, keepdims=True) + 1e-10  # 防止除以 0
    w = H / H_sum

    return w

def labelassign(X, C, w):
    '''
        Assigns each sample in X to the most similar cluster in C based on a weighted similarity metric.


        :param X: 2D NumPy array of shape (N, d), where each row is a sample
        :param C: 2D NumPy array of shape (k, d), where each row is a cluster "centroid" (mode vector)
        :param w: 2D NumPy array of shape (k, d), feature weights for each cluster
        :return: 1D array of shape (N,), the assigned cluster index for each sample
    '''
    N, d = X.shape
    k = C.shape[0]

    "Broadcast comparison between samples and cluster centroids"
    same = (X[:, np.newaxis, :] == C[np.newaxis, :, :]).astype(float)

    "Multiply binary match matrix by feature weights to get weighted similarity"
    weighted_sim = same * w[np.newaxis, :, :]

    "Sum over features to get total similarity scores for each sample-cluster pair"
    similarity = np.sum(weighted_sim, axis=2)

    "Assign each sample to the cluster with the highest similarity score"
    result = np.argmax(similarity, axis=1)

    return result




def fed_hire(labelData, true_k):
    '''
    fed_hire (Federated Hierarchical Representative Ensemble) Clustering Algorithm.

    This function performs a clustering procedure over discrete label data ,
    using mode-based centroid updates and feature-cluster weighting. It is suitable for aggregating
    hierarchical cluster representations across distributed clients in a privacy-preserving setting.

    :param labelData: 2D NumPy array of shape (N, Δ)
    :param true_k: The desired number of clusters for aggregation
    :return:
        predict: Cluster assignment for each sample
        w: Learned feature weights per cluster
        Centroid: Mode-based centroids for each cluster
        status: 'success' if converged, 'error' if true_k > number of unique label vectors
    '''

    N, d = np.shape(labelData)

    "Identify all unique label vectors from the data"
    unique_rows = np.unique(labelData, axis=0)

    if true_k > unique_rows.shape[0]:
        return [], [], [], "error"

    "Randomly select initial centroids from the unique label vectors"
    C = unique_rows[np.random.choice(unique_rows.shape[0], true_k, replace=False)]

    "Initialize convergence flag, uniform feature-cluster weights, and initial label assignment"
    convergence = False
    w = np.ones((true_k, d)) * (1 / d)
    result = labelassign(labelData, C, w)
    iter = 0

    "Main clustering loop"
    while convergence == False:
        iter += 1
        old_result = result
        Centroid = []

        "For each cluster, compute the mode (most frequent label) per feature"
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

        "Recompute feature weights based on inter/intra-cluster statistics"
        w = calWeight(labelData, Centroid, old_result)

        "Reassign samples to clusters using updated centroids and weights"
        result = labelassign(labelData, Centroid, w)

        "Check for convergence (no label changes)"
        if np.array_equal(result, old_result):
            convergence = True

        if iter > 30:
            break

    predict = result
    return predict, w, Centroid, 'success'

def MCPL(X, rate, k_1):
    '''
    MCPL (Multi-granular Competitive Penalized Learning) Clustering Algorithm.

    This function performs a multi-stage clustering process that dynamically adjusts the number of clusters
    and refines feature importance through iterative competitive learning. It repeatedly reinitializes
    cluster centroids until the structure stabilizes, enabling fine-to-coarse exploration of data granularity.

    :param X: 2D NumPy array of input data
    :param rate: Learning rate for updating cluster importance
    :param k_1: Initial cluster ratio (e.g., 0.1 * n_samples)
    :return:
        Label: List of cluster assignments from each epoch
        K: List of effective cluster numbers after each epoch
    '''
    n, d = np.shape(X)
    k = int(np.round(k_1 * n))

    Label = []
    K = []
    epoch = 0

    while True:
        epoch += 1
        convergence = False

        "Step 1: Randomly initialize cluster seeds and stats"
        seed = np.random.choice(range(n), k, replace=False)
        totalData = X[seed, :]
        W = np.full((k, d), 1 / d).astype(float)
        cnt = np.full(k, 1).astype(int)
        belta = np.full(k, 1).astype(float)
        U = np.zeros((n, k)).astype(int)
        y = (np.ones(n) * -1).astype(int)

        y[seed] = np.arange(k)
        U[seed, np.arange(k)] = 1

        theta = []
        label = (np.ones((n)) * -1).astype(int)

        iter = 0
        while True:
            iter += 1
            noChange = True

            "step2: Assign samples to most competitive clusters"
            for i in range(n):
                C = totalData / np.repeat(np.expand_dims(np.sum(U, axis=0).T, axis=1), d, axis=1)
                s = similarity(X[i, :], C, W)

                "Compute similarity"

                g = 1 / (1 + np.exp(-10 * belta + 5))
                gamma = cnt / np.sum(cnt)
                s = (1 - gamma) * g * s

                "Choose the winning clusters and the nearest rival"
                sort_index = np.argsort(-s)
                v = sort_index[0]
                r = sort_index[1]

                "Update cluster importance"
                belta[v] += rate
                belta[r] -= (rate * (s[r] / s[v]))

                cnt[v] += 1

                oldLabel = y[i]
                y[i] = v

                "Update assignments and data sum if label changed"
                if oldLabel != v:
                    totalData[v, :] += X[i, :]
                    U[i][v] = 1
                    if oldLabel != -1:
                        totalData[oldLabel, :] -= X[i, :]
                        U[i][oldLabel] = 0
                    noChange = False

            "If all labels are stable, break inner loop"
            if noChange == True:
                break
            else:
                "step3: Update feature weights"
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

                    "Compute similarity-based feature mean"
                    M[i, :] = np.sum(np.exp(-0.5 * ((X[index, :] - C[i, :]) ** 2)), axis=0) / count

                    "Compute Hellinger-based feature discrimination"
                    mu.append(np.sum(X[index, :], axis=0) / count)
                    mu.append(np.sum(X[out_index, :], axis=0) / (n - count))
                    sigma.append(np.sum((X[index, :] - mu[0]) ** 2, axis=0) / (count - 1))
                    sigma.append(np.sum((X[out_index, :] - mu[1]) ** 2, axis=0) / (n - count - 1))
                    F[i, :] = np.sqrt(1 -
                                      np.sqrt((2 * np.sqrt(sigma[0]) * np.sqrt(sigma[1])) / (sigma[0] + sigma[1])) *
                                      np.exp(-(((mu[0] - mu[1]) ** 2) / (4 * (sigma[0] + sigma[1])))))

                H = M * F
                W = H / np.repeat(np.expand_dims(np.sum(H, axis=1), axis=1), d, axis=1)

            if iter > 30:
                break

        "Post-processing: Identify active clusters"
        index = np.where(np.sum(U, axis=0) > 0)[0]
        length = len(index)
        for i in range(length):
            label[np.where(U[:, index[i]] == 1)[0]] = i

        "Check convergence conditions"
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