import numpy as np

from algorithm.experiment import validation
from algorithm.federatedClustering import fedClustering

if __name__ == '__main__':
    dataset = "US"

    L = []
    loadData = np.load(
        f'dataset/{dataset}.npz',
        allow_pickle=True)

    for key in loadData:
        client = loadData[key].tolist()
        L.append(client)

    Times = 5
    Result = []
    t = 0
    while t < Times:
        output = fedClustering(L, 0.05, 0.5, 0.5)

        L = output[0]
        C = output[1]

        result = validation(L, C, dataset)

        Result.append(result)
        t += 1
    Result = np.stack(Result)
    mean = np.mean(Result, axis=0)
    std = np.std(Result, axis=0)
    print(f"Purity: {mean[0]:.4f}")
    print(f"ARI: {mean[1]:.4f}")
    print(f"NMI: {mean[2]:.4f}")
    print(f"ACC: {mean[3]:.4f}")
