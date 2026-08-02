"""
FastAPI layer over the federated fraud detection pipeline.

This implements the SAME FedAvg logic your Flower simulation uses
(model.py's build_model/local_train_step/get_params/set_params) directly,
rather than reaching into a running flwr.simulation instance -- that would
require Flower's simulation runtime (and Ray) to stay alive between HTTP
requests, which doesn't fit a normal request/response API. Your
run_simulation.py remains the actual Flower-based evidence for the
technical writeup; this endpoint demonstrates the identical algorithm live.

Run with:
    uvicorn api.main:app --reload --port 8000
Then open http://127.0.0.1:8000/docs for interactive testing.
"""
import copy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd

from model import (
    build_model, get_params, set_params, local_train_step, evaluate,
    build_feature_matrix, build_single_feature_vector,
)
from api.schemas import TrainRoundResponse, StatusResponse, PredictRequest, PredictResponse

app = FastAPI(title="Federated Fraud Detection API", version="1.0.0")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "datasets"
N_CLIENTS = 4

_BANKS = []
for i in range(1, N_CLIENTS + 1):
    df = pd.read_csv(DATA_DIR / f"bank_{i}.csv")
    X, y = build_feature_matrix(df)
    # Shuffle before splitting -- bank CSVs are risk-score sorted, so a
    # naive slice would train on low-risk rows and validate on high-risk
    # ones (the exact bug caught and fixed in Milestone 3's client.py).
    rng = np.random.default_rng(42 + i)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(len(X) * 0.8)
    _BANKS.append({
        "X_train": X[:split], "y_train": y[:split],
        "X_val": X[split:], "y_val": y[split:],
    })

_global_model = build_model()
_round = 0
_history: list[TrainRoundResponse] = []


def _weighted_average(param_list, weights):
    total = sum(weights)
    avg = []
    for i in range(len(param_list[0])):
        stacked = np.stack([p[i] * w for p, w in zip(param_list, weights)])
        avg.append(stacked.sum(axis=0) / total)
    return avg


@app.get("/")
def health():
    return {"status": "ok", "current_round": _round, "n_banks": N_CLIENTS}


@app.post("/train-round", response_model=TrainRoundResponse)
def train_round(local_epochs: int = 2):
    global _round, _global_model

    current_params = get_params(_global_model)
    client_updates, weights, per_bank_fraud_rate = [], [], {}

    for i, bank in enumerate(_BANKS):
        local_model = copy.deepcopy(_global_model)
        set_params(local_model, [p.copy() for p in current_params])
        local_train_step(local_model, bank["X_train"], bank["y_train"], local_epochs)
        client_updates.append(get_params(local_model))
        weights.append(len(bank["X_train"]))
        per_bank_fraud_rate[f"bank_{i + 1}"] = float(bank["y_train"].mean())

    new_params = _weighted_average(client_updates, weights)
    set_params(_global_model, new_params)
    _round += 1

    X_val_all = np.concatenate([b["X_val"] for b in _BANKS])
    y_val_all = np.concatenate([b["y_val"] for b in _BANKS])
    metrics = evaluate(_global_model, X_val_all, y_val_all)

    result = TrainRoundResponse(
        round=_round, global_metrics=metrics, per_bank_fraud_rate=per_bank_fraud_rate
    )
    _history.append(result)
    return result


@app.get("/status", response_model=StatusResponse)
def status():
    return StatusResponse(current_round=_round, history=_history)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _round == 0:
        raise HTTPException(400, "No training rounds have run yet -- call /train-round first")
    X = build_single_feature_vector(req.model_dump())
    prob = float(_global_model.predict_proba(X)[0, 1])
    return PredictResponse(fraud_probability=prob, is_fraud=prob >= 0.5, round_used=_round)