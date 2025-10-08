


from algorithm.experiment import *
from algorithm.MultiClustering import FCPL, fed_hire, MCPL
import numpy as np
import progressbar


def fedClustering(L, rate, k_0=0.5, k_1=0.5):
    '''
    Federated Clustering Algorithm.

    Args:
        L (list):
            A list of dictionaries, where each dictionary represents a client.
            Each dictionary should at least contain the keys:
                - 'data': the local dataset (e.g., numpy array) for the client.
                - 'label': the ground-truth labels for the client's data.
                - 'size': the number of samples in the client's data.
        rate (float):
            A parameter for the clustering algorithm, representing the learning rate.
        k_0 (float, optional):
            Initial number of clusterlets for the local clustering step (client-side). Default is 0.5.
        k_1 (float, optional):
            Initial number of clusters for the global clustering step (server-side). Default is 0.5.

    Returns:
        L (list):
            The updated list of client dictionaries, each containing the results of the federated clustering,
            including predicted labels and granular labels.
        centroid (numpy.ndarray):
            The array of global cluster centroids computed after the federated clustering process.
        msg (str):
            Status message indicating success ('success') or error ('error').
    '''

    "1. Collect all true labels from every client for later evaluation"
    y_true = []
    for i, l in enumerate(L):
        y_true.append(l["label"])
    # Flatten the list
    y_true = list(itertools.chain.from_iterable(y_true))
    y_true = np.array(y_true)
    # Number of unique classes in the dataset
    n_class = len(np.unique(y_true))

    # To store parameters from each client's local clustering
    Theta = []
    # To store number of clusters found by each client
    localK = []

    "2. Run local clustering (FCPL) on each client"
    for i, l in enumerate(progressbar.progressbar(L)):
        # Run local clustering
        theta, label, k = FCPL(l["data"], rate, k_0)
        # Save local cluster parameters
        Theta.append(theta)
        l["localLabel"] = label
        # Save number of clusters detected locally
        localK.append(k)
    # T holds all local clustering results
    T = Theta

    "3. If no clustering results, return error"
    if len(T) == 0:
        return [], 'error'
    else:
        "4. Concatenate all local cluster centers into one array for global clustering"
        T = np.concatenate(T)

    "5. Run global clustering (MCPL) on server"
    Label, K = MCPL(T, rate, k_1)

    "6. Federated hierarchical assignment (Fed-HIRE) to refine global cluster assignments"
    labelData = np.stack(Label).T
    center_predict, w, C, msg = fed_hire(labelData, n_class)
    if msg == 'error':
        return [], 'error'

    "7. Update global cluster centers based on assignments"
    centroid = []
    for i in range(n_class):
        # Indices of points assigned to cluster i
        idx = np.where(center_predict == i)[0]
        # Compute mean as new centroid
        centroid.append(np.mean(T[idx], axis=0))
    centroid = np.array(centroid)

    "8. Build index slices: map each client's cluster indices in the concatenated array"
    index = []
    start = 0
    for k in localK:
        index.append(np.arange(start, start + k))
        start += k

    "9. For each client, assign granular and predicted labels based on global assignments"
    for i in range(len(L)):
        # Number of samples in this client
        li_size = L[i]['size']
        # Local cluster assignment for each sample in this client
        local_label = L[i]["localLabel"]
        # Total number of clusters
        num_labels = len(Label)
        # Store granular label matrix
        GranularLabel = np.zeros((li_size, num_labels), dtype=int)
        # Store predicted global cluster
        predict = np.zeros(li_size, dtype=int)

        # For each cluster, map global cluster assignment to local samples
        for j in range(num_labels):
            # Global cluster assignment for this client's clusters
            granular_temp = np.array(Label[j])[index[i]]
            # Predicted center assignment for this client's clusters
            predict_temp = center_predict[index[i]]

            # Map the global label to each local sample according to its local cluster assignment
            GranularLabel[:, j] = granular_temp[local_label]
            # Final predicted cluster for each sample
        predict[:] = predict_temp[local_label]

        # Save results back to client dictionary
        L[i]["GranularLabel"] = GranularLabel
        L[i]["predict"] = predict

    "10. Return updated client info, global centroids, and status"
    return L, "success"




