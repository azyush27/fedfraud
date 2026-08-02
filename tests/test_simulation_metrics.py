from pathlib import Path

import numpy as np
import pandas as pd

from client import BankClient
from model import build_feature_matrix
from run_simulation import load_feature_schema


def test_feature_schema_matches_client_features(tmp_path: Path) -> None:
    data = pd.read_csv(Path("data/datasets/transactions.csv"))
    feature_columns = load_feature_schema()
    feature_matrix = build_feature_matrix(data, feature_columns=feature_columns)

    assert list(feature_matrix.columns) == feature_columns
    assert feature_matrix.shape[1] > 0


def test_client_fit_returns_metrics_and_parameters() -> None:
    feature_columns = load_feature_schema()
    client = BankClient("bank_1", Path("data/datasets/bank_1.csv"), feature_columns)
    params, num_examples, metrics = client.fit([], {})

    assert len(params) >= 1
    assert num_examples > 0
    assert metrics["bank"] == "bank_1"
    assert metrics["samples"] == num_examples
