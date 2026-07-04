# FGDF Experiment Reproduction Guide

## 1. Overview of the Experiment

The main workflow of the code is as follows:

1. Load the CSV dataset and split it into a training set and a test set at a `75% / 25%` ratio.
2. Further split the training set into a local training set and a validation set at an `80% / 20%` ratio.
3. Convert each sample into graph data:

   * Each sample forms one graph.
   * The number of nodes in the graph equals the number of features.
   * The input feature of each node is a copy of the original sample feature vector.
   * Sparse edges are constructed between nodes based on the distance between feature indices. By default, each node is connected to its nearest `k=3` nodes.

4. Distribute the training data to multiple clients in a class-balanced manner.
5. In each round of federated training:

   * The server distributes the global GAT model.
   * Clients perform local training.
   * Clients encrypt model weights using CKKS, while biases remain in plaintext.
   * The server aggregates weights in the ciphertext domain and aggregates biases in the plaintext domain.
   * Clients decrypt the aggregated global model.
   * The server updates the global model and evaluates it on the test set.

6. In each round, the following results are output and saved:

   * Test accuracy, recall, precision, F1 score, and loss.
   * Confusion matrix statistics: `TN / FP / FN / TP`.
   * Upload, download, and cumulative communication overhead.
   * Time consumption for local training, encryption, server aggregation, and decryption.
   * Privacy metrics such as `MI_mean`, `MI_max`, and `KL_sym`.

## 2. Environment Requirements

It is recommended to use an independent virtual environment. For the specific environment configuration, refer to the `FGDF.yaml` environment file.

## 3. Dataset Format

By default, the code reads a CSV file without a header. The dataset must also be preprocessed before running the experiment. Preprocessing includes normalization and setting the last column as the normal/attack label.

## 4. Key Hyperparameters

Main hyperparameters:

```python
batch_size = 32
learning_rate = 0.01
numEpoch = 30
local_epochs = 30
momentum = 0.9
number_of_samples = 100
number_of_clusters = 5
hidden_channels = 16
out_channels = num_of_classes
```

Differential privacy parameters:

```python
epsilon = 1
delta = 1e-5
sensitivity = 0.1
use_dp_noise = False
```

To enable differential privacy noise, change:

```python
use_dp_noise = False
```

to:

```python
use_dp_noise = True
```
