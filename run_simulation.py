from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation

from client import client_fn
from model import build_feature_matrix
from server import make_server_app


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "datasets"
RESULTS_PATH = ROOT / "RESULTS.md"


def load_feature_schema() -> list[str]:
    data = pd.read_csv(DATA_DIR / "transactions.csv")
    feature_matrix = build_feature_matrix(data)
    return feature_matrix.columns.tolist()


def load_baseline_metrics(path: Path) -> tuple[float, float]:
    results_text = path.read_text(encoding="utf-8")
    f1_match = re.search(r"- F1: ([0-9.]+)", results_text)
    auc_match = re.search(r"- AUC: ([0-9.]+)", results_text)
    if not f1_match or not auc_match:
        raise ValueError("Unable to parse baseline metrics from RESULTS.md")
    return float(f1_match.group(1)), float(auc_match.group(1))


def main() -> None:
    feature_columns = load_feature_schema()
    print("Starting federated simulation...")

    from client import BankClient

    def wrapped_client_fn(context):
        return BankClient(f"bank_{context.node_id}", DATA_DIR / f"bank_{context.node_id}.csv", feature_columns).to_client()

    client_app = ClientApp(client_fn=wrapped_client_fn)
    server_app = make_server_app()

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=4,
        backend_config={"num_cpus": 1},
    )

    strategy = server_app._server.strategy
    print("Federated learning completed.")
    for round_idx, metrics in sorted(strategy.round_metrics.items()):
        if "f1" in metrics and "auc" in metrics:
            print(f"Round {round_idx}: F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")

    baseline_f1, baseline_auc = load_baseline_metrics(RESULTS_PATH)
    last_metrics = strategy.round_metrics[max(strategy.round_metrics)]
    print(f"Final federated F1: {last_metrics['f1']:.4f} (baseline {baseline_f1:.4f})")
    print(f"Final federated AUC: {last_metrics['auc']:.4f} (baseline {baseline_auc:.4f})")


if __name__ == "__main__":
    main()
