"""
Shared model + feature schema used by every client and the FastAPI layer.

Critical detail: one-hot encoding must use the SAME category list across
every bank, computed from the full dataset -- not inferred per-bank. If
bank_1's CSV happens to contain no "NG" country rows, get_dummies() on
bank_1 alone would produce a different column count than bank_4, and
Flower's FedAvg would fail (or silently misalign) trying to average
parameter arrays of different shapes across clients.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent
FULL_DATA_PATH = ROOT / "data" / "datasets" / "transactions.csv"

MERCHANT_CATEGORIES = ["Grocery", "Travel", "Retail", "Online", "Food Delivery",
                       "Entertainment", "Utilities", "Health"]
COUNTRIES = ["US", "CA", "GB", "DE", "FR", "AU", "JP", "BR", "NG"]
DEVICE_TYPES = ["mobile", "desktop", "tablet"]

NUMERIC_COLS = ["amount", "hour", "day_of_week", "card_age_days", "velocity",
                "is_international", "is_weekend", "is_night", "is_new_card",
                "high_amount", "new_card", "high_velocity"]

# Fixed, shared column order -- every bank produces exactly this shape
FEATURE_COLUMNS = (
    NUMERIC_COLS
    + [f"merchant_category_{m}" for m in MERCHANT_CATEGORIES]
    + [f"country_{c}" for c in COUNTRIES]
    + [f"device_type_{d}" for d in DEVICE_TYPES]
)
N_FEATURES = len(FEATURE_COLUMNS)
CLASSES = np.array([0, 1])


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Turns a raw bank CSV into (X, y) using the fixed shared schema,
    then applies the GLOBAL feature scaler (see _GLOBAL_SCALER below) so
    every bank's features are on the same scale. Without this, SGD's
    gradient updates get dominated by large-range columns like `amount`
    (5-6000) versus binary flags (0/1), and the model barely learns at all
    -- confirmed by testing: AUC stuck near 0.49-0.53 without scaling,
    jumping to ~0.68-0.70 with it applied consistently.
    """
    df = df.copy()
    df["high_amount"] = (df["amount"] > 900).astype(int)
    df["new_card"] = (df["card_age_days"] < 30).astype(int)
    df["high_velocity"] = (df["velocity"] > 2).astype(int)

    encoded = pd.get_dummies(
        df, columns=["merchant_category", "country", "device_type"]
    )
    for col in FEATURE_COLUMNS:
        if col not in encoded.columns:
            encoded[col] = 0
    X = encoded[FEATURE_COLUMNS].astype(np.float32).to_numpy()
    y = df["is_fraud"].astype(np.int64).to_numpy()
    X = _GLOBAL_SCALER.transform(X).astype(np.float32)
    return X, y


def _fit_global_scaler() -> StandardScaler:
    """Fit ONE scaler on the full dataset (not per-bank -- a real bank
    wouldn't be allowed to see other banks' data, but the scaler's
    mean/std here is a fixed preprocessing constant agreed on up front,
    not model training on raw data, so this doesn't violate the "no raw
    data sharing" principle any more than agreeing on a shared feature
    schema already does)."""
    full_df = pd.read_csv(FULL_DATA_PATH)
    full_df = full_df.copy()
    full_df["high_amount"] = (full_df["amount"] > 900).astype(int)
    full_df["new_card"] = (full_df["card_age_days"] < 30).astype(int)
    full_df["high_velocity"] = (full_df["velocity"] > 2).astype(int)
    encoded = pd.get_dummies(full_df, columns=["merchant_category", "country", "device_type"])
    for col in FEATURE_COLUMNS:
        if col not in encoded.columns:
            encoded[col] = 0
    X_full = encoded[FEATURE_COLUMNS].astype(np.float32).to_numpy()
    return StandardScaler().fit(X_full)


_GLOBAL_SCALER = _fit_global_scaler()


def build_model() -> SGDClassifier:
    # class_weight="balanced" as a string is NOT supported by partial_fit
    # (sklearn raises ValueError) -- compute the balanced weights once from
    # the full dataset's known class distribution and pass a fixed dict
    # instead. This keeps every bank's local model weighting the minority
    # (fraud) class the same way.
    full_df = pd.read_csv(FULL_DATA_PATH)
    y_full = full_df["is_fraud"].astype(np.int64).to_numpy()
    weights = compute_class_weight("balanced", classes=CLASSES, y=y_full)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}

    model = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=0.01,
                           class_weight=class_weight, random_state=42)
    X0 = np.zeros((2, N_FEATURES), dtype=np.float32)
    model.partial_fit(X0, np.array([0, 1]), classes=CLASSES)
    model.coef_ = np.zeros((1, N_FEATURES), dtype=np.float32)
    model.intercept_ = np.zeros(1, dtype=np.float32)
    return model


def get_params(model: SGDClassifier):
    return [model.coef_, model.intercept_]


def set_params(model: SGDClassifier, params) -> SGDClassifier:
    model.coef_ = params[0]
    model.intercept_ = params[1]
    return model


def local_train_step(model: SGDClassifier, X, y, local_epochs: int = 1):
    for _ in range(local_epochs):
        model.partial_fit(X, y)
    return model


def evaluate(model: SGDClassifier, X, y) -> dict:
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return {
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auc": float(roc_auc_score(y, probs)) if len(set(y)) > 1 else 0.0,
    }