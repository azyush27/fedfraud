from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from flwr.client import Client, NumPyClient
from flwr.common import Context

from model import build_feature_matrix


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "datasets"


class BankClient(NumPyClient):
    def __init__(self, bank_name: str, data_path: Path, feature_columns: list[str]) -> None:
        self.bank_name = bank_name
        self.data_path = data_path
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self.classifier = SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=5000,
            tol=1e-3,
            random_state=42,
        )
        self._load_local_data()

    def _load_local_data(self) -> None:
        data = pd.read_csv(self.data_path)
        self.X = build_feature_matrix(data, feature_columns=self.feature_columns)
        self.y = data["is_fraud"].astype(int).to_numpy()
        self.scaler.fit(self.X)
        self.X_scaled = self.scaler.transform(self.X)

    def _set_model_parameters(self, parameters: list[np.ndarray]) -> None:
        if len(parameters) == 0:
            self.classifier.coef_ = np.zeros((1, self.X_scaled.shape[1]), dtype=np.float64)
            self.classifier.intercept_ = np.zeros(1, dtype=np.float64)
            return

        coef = np.array(parameters[0], dtype=np.float64)
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)
        intercept = np.array(parameters[1], dtype=np.float64).reshape(1,)
        self.classifier.coef_ = coef
        self.classifier.intercept_ = intercept

    def get_parameters(self, config: dict[str, Any] | None = None) -> list[np.ndarray]:
        return [self.classifier.coef_.astype(np.float64), self.classifier.intercept_.astype(np.float64)]

    def fit(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        self._set_model_parameters(parameters)
        for _ in range(2):
            self.classifier.partial_fit(self.X_scaled, self.y, classes=np.array([0, 1]))
        local_count = len(self.X_scaled)
        return self.get_parameters(config), local_count, {"bank": self.bank_name, "samples": local_count}

    def evaluate(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[float, int, dict[str, Any]]:
        self._set_model_parameters(parameters)
        probabilities = self.classifier.predict_proba(self.X_scaled)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        f1 = f1_score(self.y, predictions, zero_division=0)
        auc = roc_auc_score(self.y, probabilities)
        return 0.0, len(self.X_scaled), {"f1": f1, "auc": auc, "bank": self.bank_name}


def client_fn(context: Context) -> Client:
    bank_name = f"bank_{context.node_id}"
    data_path = DATA_DIR / f"{bank_name}.csv"
    return BankClient(bank_name, data_path, []).to_client()
