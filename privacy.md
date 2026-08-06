# Privacy Roadmap

This document explains what this project protects today, and what a
production deployment would add on top. It's written so the gap between
"hackathon demo" and "bank-grade system" is explicit, not glossed over.

## What this build already protects

**Raw transaction data never leaves each bank.** In the federated
simulation and the live API, every bank trains locally on its own CSV.
Only model weight updates (a small array of numbers) are sent to the
central aggregator and averaged into the global model. No transaction
row, customer ID, or amount is ever transmitted between banks or to the
server.

This is the core guarantee federated learning provides, and it's real --
not simulated. You could inspect network traffic between clients and
server in this system and never see a single transaction.

## What raw model weights can still leak, in theory

Weight updates are not raw data, but they're not perfectly private
either. Two categories of attack exist in the research literature that
this build does **not** defend against:

**Membership inference**: an attacker who can query the model repeatedly,
or who sees enough weight updates, can sometimes infer whether a
*specific* transaction was in a bank's training data, even without
seeing the transaction itself. This matters if, say, a competitor bank
could infer "Bank X saw a fraudulent transaction from this specific
customer" just from how the shared model's weights shifted.

**Gradient inversion**: in some settings (small batches, no
regularization, certain model architectures), it's possible to
partially reconstruct approximate input data from a single gradient
update. This is more of a concern for deep neural networks trained on
tiny batches than for the simple linear model used here, but it's a
known class of attack worth naming.

## What would close these gaps in production

**Secure aggregation**: a cryptographic protocol where the central server
receives only the *sum* of all clients' updates, and mathematically
cannot see any individual bank's update in isolation -- even the
aggregator is blind to who contributed what. Flower has experimental
support for this (`flwr.common.secagg`); it wasn't implemented here to
keep the build scoped to demonstrating the core federated architecture
within the hackathon timeline.

**Differential privacy (DP)**: adding carefully calibrated statistical
noise to each bank's local update before it's sent, so that no single
transaction's presence or absence can be confidently inferred from the
aggregated result -- even by an adversary with unlimited queries. The
standard approach for this is DP-SGD (e.g. via the `Opacus` library for
PyTorch), which bounds and clips per-example gradient contributions
before adding noise. This comes with a real accuracy tradeoff: more
noise means stronger privacy guarantees but a less accurate model, and
choosing that tradeoff (the "privacy budget," typically denoted ε) is
itself a deployment decision each bank would need to sign off on.

**Secure communication channel**: in production, weight updates would
need to travel over TLS between banks and the aggregator (this demo runs
entirely on localhost, so this wasn't a concern here, but it's a basic
requirement for any real deployment).

**Audit logging and access control**: production systems would need to
log every aggregation round, restrict who can query the model API, and
rate-limit prediction requests to reduce the surface area for the
membership-inference style attacks described above.

## Why this scoping is the right call for this build

Implementing secure aggregation and differential privacy correctly is a
substantial engineering effort on its own -- entire papers and library
releases are dedicated to getting the privacy/accuracy tradeoff right.
Attempting to bolt them on under hackathon time constraints risks either
not finishing, or finishing with a broken/misconfigured implementation
that gives a false sense of security, which is arguably worse than
clearly stating "not implemented yet."

This project's contribution is demonstrating that the federated
*architecture* itself -- no raw data sharing, model weights only,
FedAvg aggregation across genuinely non-IID bank data -- works and
achieves performance comparable to a centralized model (see
`RESULTS.md`). Secure aggregation and DP are the natural next layer on
top of an architecture that's already proven to work.