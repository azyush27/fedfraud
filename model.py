from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "datasets" / "transactions.csv"
RESULTS_PATH = ROOT / "RESULTS.md"


def load_training_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the full synthetic dataset and prepare features and labels."""
    data = pd.read_csv(path)

    data = data.copy()
    data["high_amount"] = (data["amount"] > 900).astype(int)
    data["new_card"] = (data["card_age_days"] < 30).astype(int)
    data["high_velocity"] = (data["velocity"] > 2).astype(int)

    feature_frame = data.drop(columns=["transaction_id", "customer_id", "is_fraud"])
    categorical_columns = ["merchant_category", "country", "device_type"]
    encoded_features = pd.get_dummies(feature_frame, columns=categorical_columns, drop_first=False)
    encoded_features = encoded_features.astype(float)

    target = data["is_fraud"].astype(int)
    return encoded_features, target


def train_centralized_baseline(data_path: Path) -> dict[str, float]:
    """Train a centralized baseline SGD classifier and return evaluation metrics."""
    X, y = load_training_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=5000,
            tol=1e-3,
            random_state=42,
        ),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "auc": roc_auc_score(y_test, probabilities),
    }
    return metrics


def write_results(path: Path, metrics: dict[str, float]) -> None:
    """Persist the centralized baseline metrics to a markdown file."""
    content = "# Results\n\n"
    content += "## Centralized Baseline\n\n"
    content += f"- Precision: {metrics['precision']:.4f}\n"
    content += f"- Recall: {metrics['recall']:.4f}\n"
    content += f"- F1: {metrics['f1']:.4f}\n"
    content += f"- AUC: {metrics['auc']:.4f}\n"
    path.write_text(content, encoding="utf-8")


def main() -> None:
    """Train the centralized baseline and write evaluation metrics."""
    metrics = train_centralized_baseline(DATA_PATH)
    write_results(RESULTS_PATH, metrics)

    print("Centralized baseline metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
