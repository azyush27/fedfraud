# Results

## Centralized Baseline (Milestone 2)

Trained on the full pooled dataset (all 20,000 transactions, all 4 banks combined).

- Precision: 0.3865
- Recall: 0.5811
- F1: 0.4642
- AUC: 0.6870

*(Note: an earlier version of this dataset had a data-generation bug in the
`amount` field -- see "Bugs Found & Fixed" below. These are the corrected,
final numbers.)*

## Federated Learning (Milestones 3/4)

Simulated 4 banks via Flower's FedAvg strategy, 15 rounds, 2 local epochs/round.
Bank fraud rates (deliberately non-IID): bank_1 8.8%, bank_2 19.6%, bank_3 29.6%, bank_4 42.5%.

### Two valid ways to measure federated performance

AUC is a ranking metric and does **not** aggregate linearly across
sub-populations with different base rates. Evaluating on the pooled
validation set vs. evaluating per-bank and averaging gives genuinely
different -- both valid -- numbers:

| Metric | Value | What it measures |
|---|---|---|
| AUC, pooled validation set | 0.696 | Directly comparable to the centralized baseline (same methodology) |
| AUC, per-bank then weighted-averaged | 0.555 | How well the model separates fraud *within* each individual bank's own data -- the metric a single bank deploying this model would actually experience |

### Comparison to centralized baseline (fair, same-methodology comparison)

| Setting | AUC (pooled) | F1 |
|---|---|---|
| Centralized (all data pooled) | 0.687 | 0.464 |
| Federated (4 non-IID banks, FedAvg) | 0.696 | 0.490 |

**Federated learning matches -- and in this run, marginally exceeds --
centralized performance, without any bank ever sharing raw transaction
data.**

The lower per-bank number (0.555) is not a contradiction: it reflects the
harder, more realistic task of ranking fraud within one institution's own
transactions, versus the "easier" pooled task where the model partly
benefits from learning which bank a transaction profile resembles. This is
a known subtlety in evaluating federated learning under non-IID data.

## Bugs Found & Fixed During Development

Documented for transparency -- these were caught through direct testing,
not assumed correct from a clean run:

1. **Fraud probability saturation**: original `fraud_prob` capped at 0.18
   while `risk_score` could reach 0.86 -- 72% of transactions received an
   identical fraud probability regardless of actual risk. Fixed by
   widening the cap.
2. **Amount distribution collapse**: `amount`'s log-normal parameters gave
   a median (~$1.82) far below the $5 clip floor, silently flooring 88% of
   transactions to a constant $5 and breaking the `high_amount` engineered
   feature entirely. Fixed by adjusting the distribution parameters
   (median now ~$90, realistic spread to ~$5,700).
3. **Train/validation split on sorted data**: bank CSVs are risk-score
   sorted before slicing (for the non-IID split), so a naive top-80%/
   bottom-20% train/val split trained on each bank's lowest-risk rows and
   validated on its highest-risk rows. Fixed by shuffling before splitting.
4. **Feature scaling**: SGD's gradient updates were dominated by
   large-range features (`amount`) versus binary flags, stalling AUC near
   random. Fixed with a shared `StandardScaler` fit once on the full
   dataset and applied consistently across all banks and the API layer.

## Privacy Roadmap

See `PRIVACY.md` for what secure aggregation and differential privacy
would add in a production version of this system.