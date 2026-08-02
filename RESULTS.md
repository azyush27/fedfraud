# Results

## Centralized Baseline

- Precision: 0.3865
- Recall: 0.5811
- F1: 0.4642
- AUC: 0.6834

## Federated Learning (Milestone 3)

Simulated 4 banks via Flower's FedAvg strategy, 15 rounds, 2 local epochs/round.

| Round | AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| 1 | 0.5452 | 0.313 | 0.2534 | 0.5057 |
| 5 | 0.5528 | 0.309 | 0.2485 | 0.5287 |
| 15 | 0.5318 | 0.311 | 0.2497 | 0.5318 |

**Comparison to centralized baseline (Milestone 2):**

| Setting | AUC | F1 |
|---|---|---|
| Centralized (all data pooled) | 0.6834 | 0.4642 |
| Federated (4 non-IID banks, FedAvg) | 0.5318 | ~0.31 |

**Why the gap:** our 4 simulated banks have deliberately non-IID fraud rates
(8.7%–42.5%), reflecting realistic differences in customer risk profiles
across institutions. FedAvg averages model weights across clients trained
on very different local distributions, causing "client drift" -- a
well-documented limitation in federated learning literature. This gap is
expected, not a bug, and motivates future work with more advanced
strategies (FedProx, personalized federated learning) that explicitly
handle data heterogeneity.