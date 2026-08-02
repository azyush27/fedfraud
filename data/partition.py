from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "datasets"
INPUT_PATH = DATA_DIR / "transactions.csv"


def build_risk_score(df: pd.DataFrame) -> np.ndarray:
    risk_score = (
        (df["amount"] > 900).astype(int) * 0.22
        + df["is_international"].astype(int) * 0.18
        + df["is_night"].astype(int) * 0.12
        + (df["velocity"] > 2).astype(int) * 0.16
        + (df["country"] != "US").astype(int) * 0.10
        + df["merchant_category"].isin(["Travel", "Online", "Food Delivery", "Entertainment"]).astype(int) * 0.15
        + (df["card_age_days"] < 30).astype(int) * 0.08
        + (df["device_type"] != "desktop").astype(int) * 0.06
        + df["is_new_card"].astype(int) * 0.05
        + df["is_weekend"].astype(int) * 0.04
    )
    return np.clip(risk_score.to_numpy(), 0.0, 1.0)


def partition_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df_with_risk = df.copy()
    df_with_risk["risk_score"] = build_risk_score(df_with_risk)
    ordered_df = df_with_risk.sort_values("risk_score", ascending=True, kind="mergesort").reset_index(drop=True)
    bank_names = ["bank_1", "bank_2", "bank_3", "bank_4"]
    banks = {}
    target_size = len(ordered_df) // len(bank_names)
    for idx, bank_name in enumerate(bank_names):
        start, end = idx * target_size, (idx + 1) * target_size
        banks[bank_name] = ordered_df.iloc[start:end].drop(columns=["risk_score"]).copy()
    return banks


if __name__ == "__main__":
    dataset = pd.read_csv(INPUT_PATH)
    bank_datasets = partition_dataset(dataset)
    for bank_name, bank_df in bank_datasets.items():
        bank_df.to_csv(DATA_DIR / f"{bank_name}.csv", index=False)
        print(f"{bank_name}: {len(bank_df):,} txns, fraud rate = {bank_df['is_fraud'].mean()*100:.2f}%")