import time

import numpy as np

from algorithm.experiment import validation
from algorithm.federatedClustering import fedClustering

def main(dataset, path, T, c_num):
    '''
    Main function: Executes federated clustering multiple times and evaluates the performance
    using metrics such as Purity, ARI, NMI, and ACC.

    :param dataset: Name of the dataset
    :param path: Path to the local .npy file storing data for all clients
    :param T: Number of repetitions to run the clustering experiment
    :param c_num: Number of clients
    :return: None (results are printed to the console)
    '''

    Times = T       # Total number of experiments to run
    Result = []     # List to store results of each run
    t = 0           # Counter for completed runs

    while t < Times:
        L = []      # List to hold each client's data

        "Load data from .npy file (assumed to be a dictionary: key = client ID, value = data)"
        loadData = np.load(
            path,
            allow_pickle=True)
        for key in loadData:
            client = loadData[key].tolist()
            L.append(client)

        "Run federated clustering and record runtime"
        start_time = time.time()
        L, msg = fedClustering(L, 0.05, 0.5, 0.5)    # L = updated data, C = centroids, msg = status
        end_time = time.time()

        "Skip current run if clustering failed"
        if msg == 'error':
            continue

        print(f"Running time: {(end_time - start_time):.4f}")
        t += 1      # Increment count of successful runs

        "Evaluate clustering results using validation metrics"
        result, msg = validation(L)
        Result.append(result)

    "Aggregate results from all runs"
    Result = np.stack(Result)       # Convert list of results to NumPy array
    mean = np.mean(Result, axis=0)  # Compute mean for each metric
    std = np.std(Result, axis=0)    # Compute standard deviation for each metric

    "Print the final averaged results"
    print("==============================")
    print(f"Data: {dataset}, Client: {c_num}")
    print(f"Purity: {mean[0]:.3f}±{std[0]:.2f}")
    print(f"ARI: {mean[1]:.3f}±{std[1]:.2f}")
    print(f"NMI: {mean[2]:.3f}±{std[2]:.2f}")
    print(f"ACC: {mean[3]:.3f}±{std[3]:.2f}")
    print("==============================")


if __name__ == '__main__':
    # List of dataset names to be processed. H
    dataset = "EC"
    # Set the number of clients involved in the federated learning process.
    n_client = 8
    # Set the total number of clustering iterations to perform.
    t = 10
    # Construct the file path to the dataset based on its name.
    path = f"./dataset/{dataset}.npz"
    main(dataset, path, t, n_client)


