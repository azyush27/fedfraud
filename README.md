# FedFraud

FedFraud is a production-style federated learning portfolio project for fraud detection. This milestone focuses on building a realistic synthetic dataset and partitioning it into four non-IID bank datasets that reflect the kind of data heterogeneity seen in real financial institutions.

## Project Overview

The project simulates a scenario where multiple banks collaborate to train a fraud detection model without sharing raw transaction data. In this milestone, the emphasis is on generating high-quality synthetic data and preparing realistic federated partitions.

## Folder Structure

- data/generate.py: creates a synthetic credit-card fraud dataset with realistic fraud patterns.
- data/partition.py: partitions the dataset into four bank-specific CSV files with distinct fraud distributions.
- data/datasets/: generated datasets and partition outputs.

## Installation

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

### Generate the dataset

```bash
python data/generate.py
```

### Create bank partitions

```bash
python data/partition.py
```

## Dataset Generation

The synthetic dataset contains approximately 20,000 transactions with meaningful fraud signals such as large transaction amounts, cross-border activity, weekend or late-night activity, high transaction velocity, and risky merchant categories.

## Data Partitioning

The partitioning step creates four bank datasets with noticeably different fraud rates:

- Bank 1: mostly low-risk transactions
- Bank 2: mixed-risk transactions
- Bank 3: more suspicious activity
- Bank 4: highest fraud concentration

This creates a realistic non-IID federated learning setup.

## Results Summary

## Results Summary

- **Centralized baseline**: AUC 0.687, F1 0.464 (all data pooled, no privacy)
- **Federated (4 non-IID banks, FedAvg)**: AUC 0.696 (pooled-eval, same
  methodology as baseline), 0.555 (per-bank, weighted average)
- **Bottom line**: federated learning matches centralized performance while
  no bank ever shares raw transaction data -- see `RESULTS.md` for the full
  breakdown, including why two different (both valid) AUC numbers exist for
  the federated setting.

## Project Structure

```
fedfraud/
├── data/
│   ├── generate.py       # synthetic transaction dataset generator
│   ├── partition.py      # splits into 4 non-IID "bank" datasets
│   └── datasets/         # generated CSVs
├── model.py              # shared model, feature schema, scaler
├── client.py              # Flower client (one per simulated bank)
├── server.py              # Flower server (FedAvg strategy)
├── run_simulation.py     # entrypoint: runs the full federated simulation
├── api/
│   ├── main.py            # FastAPI service (train-round, status, predict)
│   └── schemas.py         # request/response models
├── RESULTS.md             # full metrics + bugs found & fixed
├── PRIVACY.md              # what secure aggregation/DP would add
└── requirements.txt
```

## Setup & Usage

```bash
pip install -r requirements.txt

# 1. Generate and partition the dataset
python data/generate.py
python data/partition.py

# 2. Train the centralized baseline
python model.py

# 3. Run the federated learning simulation
python run_simulation.py

# 4. Start the live API
uvicorn api.main:app --reload --port 8000
# then visit http://127.0.0.1:8000/docs
```