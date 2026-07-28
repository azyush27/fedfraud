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
