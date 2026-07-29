from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def build_synthetic_dataset(n_transactions: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic credit-card fraud dataset."""
    rng = np.random.default_rng(seed)

    merchants = ["Grocery", "Travel", "Retail", "Online", "Food Delivery", "Entertainment", "Utilities", "Health"]
    countries = ["US", "CA", "GB", "DE", "FR", "AU", "JP", "BR", "NG"]
    risky_merchants = ["Travel", "Online", "Food Delivery", "Entertainment"]

    customer_ids = rng.integers(1000, 5000, size=n_transactions)
    amount = np.clip(rng.lognormal(mean=0.6, sigma=0.85, size=n_transactions), 5, 6000)
    merchant_category = rng.choice(merchants, size=n_transactions, p=[0.25, 0.15, 0.20, 0.12, 0.10, 0.08, 0.05, 0.05])
    hour = rng.integers(0, 24, size=n_transactions)
    day_of_week = rng.integers(0, 7, size=n_transactions)
    country = rng.choice(countries, size=n_transactions, p=[0.55, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02])
    device_type = rng.choice(["mobile", "desktop", "tablet"], size=n_transactions, p=[0.45, 0.40, 0.15])
    card_age_days = np.clip(rng.exponential(scale=180.0, size=n_transactions), 1, 2000)
    velocity = rng.poisson(lam=1.3, size=n_transactions) + (rng.random(n_transactions) < 0.15).astype(int)
    is_international = (rng.random(n_transactions) < 0.14).astype(int)
    is_weekend = (day_of_week >= 5).astype(int)
    is_night = ((hour <= 5) | (hour >= 23)).astype(int)
    is_new_card = (rng.random(n_transactions) < 0.12).astype(int)

    risk_score = (
        (amount > 900).astype(int) * 0.22
        + is_international * 0.18
        + is_night * 0.12
        + (velocity > 2).astype(int) * 0.16
        + (country != "US").astype(int) * 0.10
        + np.isin(merchant_category, risky_merchants).astype(int) * 0.15
        + (card_age_days < 30).astype(int) * 0.08
        + (device_type != "desktop").astype(int) * 0.06
        + is_new_card * 0.05
        + is_weekend * 0.04
    )
    fraud_prob = np.clip(0.01 + risk_score * 0.9, 0.01, 0.95)
    is_fraud = (rng.random(n_transactions) < fraud_prob).astype(int)

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(1, n_transactions + 1),
            "customer_id": customer_ids,
            "amount": amount.round(2),
            "merchant_category": merchant_category,
            "hour": hour,
            "day_of_week": day_of_week,
            "country": country,
            "device_type": device_type,
            "card_age_days": card_age_days.round(1),
            "velocity": velocity,
            "is_international": is_international,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "is_new_card": is_new_card,
            "is_fraud": is_fraud,
        }
    )
    return df


def main() -> None:
    """Generate and save the synthetic dataset."""
    dataset = build_synthetic_dataset()
    output_path = DATA_DIR / "transactions.csv"
    dataset.to_csv(output_path, index=False)
    fraud_rate = dataset["is_fraud"].mean() * 100
    print(f"Generated {len(dataset):,} transactions at {output_path}")
    print(f"Fraud rate in the full dataset: {fraud_rate:.2f}%")


if __name__ == "__main__":
    main()
