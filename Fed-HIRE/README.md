# One-shot Hierarchical Federated Clustering

## How to Run Fed-HIRE

Just run "main.py", then the experimental results will be displayed automatically. 



```
dataset = "xxx";
```

xxx: select the dataset you want to run

For detailed settings about the parameters, initialization, etc., please refer to the Proposed Method and the Appendix.

## File description

All the folders and files for implementing the proposed AFCL algorithm are introduced below:

- algorithm:  A folder containing the implementation functions and validation functions.
  - experiments.py: A validation function implements the performance indices evaluation.
  - federatedClustering.py: A function implements the Fed-HIRE framework.
  - MultiClustering.py: A function implements all the detailed algorithms.
- dataset: A folder contains non-IID datasets with incomplete clusters used in the corresponding paper.
- raw_dataset: A folder containing public/benchmark datasets used in the corresponding paper.
- main.py: A script to cluster different data sets in the Datasets folder using the proposed method.

## Experimental environment

- python 3.11
- numpy 1.26.4
- progressbar2 4.5.0
- munkres 1.1.4
- scikit-learn 1.5.2