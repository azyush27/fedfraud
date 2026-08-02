"""
Flower ClientApp -- one instance per simulated bank.

CONFIRMED working imports for flwr 1.32.x (check yours with `pip show flwr`):
    from flwr.client import NumPyClient, ClientApp
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
NOT flwr.clientapp / flwr.serverapp -- those modules don't exist and will
ImportError every time, which is what was happening before.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from model import build_model, get_params, set_params, local_train_step, evaluate, build_feature_matrix

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "datasets"
N_CLIENTS = 4


class BankClient(NumPyClient):
    def __init__(self, partition_id: int):
        df = pd.read_csv(DATA_DIR / f"bank_{partition_id + 1}.csv")
        X, y = build_feature_matrix(df)

        # IMPORTANT: bank CSVs are contiguous slices of a risk-score-sorted
        # dataset (see data/partition.py), so rows within each bank's file
        # are STILL ordered low-risk to high-risk. A naive X[:80%]/X[80%:]
        # split would train on that bank's lowest-risk rows and validate on
        # its highest-risk rows -- a distribution mismatch that tanks AUC
        # for reasons that have nothing to do with the model. Shuffle first.
        rng = np.random.default_rng(42 + partition_id)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]

        split = int(len(X) * 0.8)
        self.X_train, self.y_train = X[:split], y[:split]
        self.X_val, self.y_val = X[split:], y[split:]
        self.model = build_model()
        self.partition_id = partition_id

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        local_epochs = int(config.get("local_epochs", 1))
        local_train_step(self.model, self.X_train, self.y_train, local_epochs)
        return get_params(self.model), len(self.X_train), {
            "bank": f"bank_{self.partition_id + 1}",
            "local_fraud_rate": float(self.y_train.mean()),
        }

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        metrics = evaluate(self.model, self.X_val, self.y_val)
        loss = 1.0 - metrics["f1"]
        return loss, len(self.X_val), metrics


def client_fn(context: Context):
    partition_id = context.node_config.get("partition-id", 0) % N_CLIENTS
    return BankClient(partition_id).to_client()


app = ClientApp(client_fn=client_fn)


if __name__ == "__main__":
    c = BankClient(partition_id=0)
    params = get_params(build_model())
    new_params, n, metrics = c.fit(params, {"local_epochs": 2})
    print("fit() metrics:", metrics, "| n_examples:", n)
    _, _, eval_metrics = c.evaluate(new_params, {})
    print("evaluate() metrics:", eval_metrics)
