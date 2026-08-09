# 🛡️ FraudShield — Real-Time Fraud Detection System

> **A deep-learning fraud detection system: real-time transaction scoring plus offline graph-based fraud-ring detection.**
> FT-Transformer + Temporal GNN + Anomaly Head · FastAPI serving · MLflow tracking · Docker · CI/CD

[![CI/CD](https://github.com/achraf-gasmi/neural-fraud-detector/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/achraf-gasmi/neural-fraud-detector/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

Card-fraud detection is a harder ML problem than it looks, for reasons that break naive approaches:

- **Severe class imbalance.** Real-world fraud rates sit around 0.1–2% of transactions. A model optimized for accuracy can classify everything as legitimate and still be "99% accurate" while catching zero fraud — this is why AUPRC, not accuracy or even AUROC, has to be the metric that drives model selection here.
- **Every false positive has a cost.** Blocking a legitimate transaction means an angry customer, a support ticket, and sometimes a lost customer. A fraud model that isn't precise enough to run at a usably low false-positive rate doesn't get deployed, no matter how good its recall looks in isolation.
- **Fraud looks like a single bad transaction, but it's often a ring.** Card testing, account takeover, and merchant collusion frequently share infrastructure — the same stolen-card testing device, the same drop IP, the same colluding merchant — across many "independent-looking" transactions and accounts. A model that scores each transaction in isolation, with no visibility into who/what it's connected to, structurally cannot see this.
- **Fraud patterns evolve.** Static rules rot as fraud adapts to evade them. A model needs some signal beyond "does this match a known fraud label" — a sense of what *normal* looks like — to have a chance against patterns it wasn't explicitly trained on.
- **Real-time authorization has a latency budget.** A card-present or card-not-present authorization decision has to return in milliseconds, which rules out anything that requires expensive graph traversal or ensemble lookups at request time.
- **Disputes and compliance need a reason, not just a score.** When a transaction is declined or a chargeback is disputed, "the model said 0.91" isn't good enough — you need to know *why*.

FraudShield's architecture is a direct response to these constraints, not a generic classifier bolted onto a fraud dataset.

## How It Addresses Them

Two tiers, each doing the job it's actually good at:

**1. Real-time path — FT-Transformer, scores in isolation, low-latency.**
Every transaction is scored on its own tabular features (amount, velocity, geo, temporal, entity-level) by an FT-Transformer, trained with focal loss to handle the class imbalance without throwing away the minority signal via naive oversampling. Gradient-based feature attribution is returned with every score so a declined transaction comes with a reason, not just a number. This is the path `api/main.py` serves — it doesn't touch the graph, so it stays fast.

**2. Batch/offline path — the same FT-Transformer fused with a Temporal GNN over the transaction graph.**
Periodically (not per-request), `training/train_graph.py` trains and `scripts/score_with_graph.py` runs a hybrid model that also sees each transaction's neighborhood: the user, merchant, device, and IP it touched, and other transactions those entities were involved in, via heterogeneous graph attention (GAT). This is where shared-device/shared-IP/shared-merchant fraud rings actually become visible — signal the real-time, single-transaction path structurally cannot have. It costs more to run, so it runs offline/in batch rather than in the authorization path.

**Anomaly head, in both models.** Alongside the classification head, an auxiliary reconstruction head learns to reconstruct *normal* transactions. The reconstruction-error signal doesn't depend on having seen a labeled example of a given fraud pattern before, which is the model's main lever against fraud that doesn't match historical labels.

### What Makes This Different From a Notebook Classifier

| Typical tabular-fraud notebook | FraudShield |
|---|---|
| Download a Kaggle dataset | Generate synthetic data with 5 explicit fraud scenarios and realistic entity/velocity structure |
| Train one sklearn model | FT-Transformer (real-time) + FT-Transformer/GNN hybrid (batch, fraud-ring detection) |
| `model.pkl` | FastAPI service + Prometheus metrics + Docker + CI/CD |
| Optimize accuracy | Optimize AUPRC — the metric that actually matches the imbalance |
| Single-transaction view only | Explicit graph model for shared-device/IP/merchant fraud rings |

---

## System Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          FraudShield System                               │
├─────────────┬─────────────────────────────┬───────────────────────────────┤
│  Data Layer │        Model Layer           │        Serving Layer         │
│             │                               │                             │
│  Synthetic  │  Real-time (per transaction): │  FastAPI /score             │
│  Generator  │   FT-Transformer + Anomaly    │  + /score/batch             │
│  (5 fraud   │   Head → P(fraud), <15ms      │  + /metrics (Prometheus)    │
│  scenarios) │                               │                             │
│             │  Batch/offline (periodic):    │  Streamlit Dashboard        │
│  Feature    │   FT-Transformer + Temporal   │                             │
│  Pipeline   │   GNN + Anomaly Head over the │  MLflow Tracking            │
│  + Graph    │   transaction graph → ring    │                             │
│  Builder    │   detection                   │  Docker + GitHub Actions    │
└─────────────┴─────────────────────────────┴───────────────────────────────┘
```

### Model — FT-Transformer (+ optional Temporal GNN) + Anomaly Head

```
Raw Transaction Features (49-dim)
        │
        ├──► FT-Transformer (tabular encoder)          [real-time + batch]
        │     ├── Feature Tokenizer (per-feature learned embedding)
        │     ├── 3× Transformer blocks (Pre-Norm, Multi-Head Attention)
        │     └── CLS token → (B, d_token) representation
        │
        ├──► Temporal GNN (heterogeneous GAT)           [batch only]
        │     ├── user / merchant / device / ip entity nodes
        │     ├── transaction ↔ entity edges + temporal "precedes" edges
        │     └── sinusoidal time encoding per transaction node
        │
        └──► Gated Fusion Layer (real-time: transformer only; batch: both)
              ├── Classification Head → P(fraud)
              └── Anomaly Head → reconstruction loss
                    (learns "normal" — robust to novel fraud patterns)
```

**Why FT-Transformer over XGBoost/LightGBM?** GBMs treat features independently. The FT-Transformer tokenizes each feature into a learned embedding and applies attention across all features, capturing interaction patterns GBMs miss, and it composes naturally with the GNN fusion layer for the batch path.

**Why a GNN at all, if it can't run in the request path?** Because the fraud that matters most — coordinated rings, not one-off card testing — is defined by *shared infrastructure across transactions*, which a single-transaction model cannot represent no matter how good its features are. Running it offline is a real, common production pattern: score fast in the authorization path, then re-score/flag in batch for review, chargebacks, or delayed settlement.

**Why Focal Loss over weighted BCE?** Focal loss dynamically down-weights easy negatives `(1 - p_t)^γ`, focusing gradient updates on hard, ambiguous transactions — more adaptive than static class weighting, and it avoids the duplication/overfitting risk of naive oversampling.

**Why AUPRC as the primary metric?** AUROC is optimistic under class imbalance — a model that scores all negatives near 0 and positives near 1 can hit AUROC ≈ 0.99 even at a 1% fraud rate and still be useless at the precision/recall tradeoff a fraud team actually has to operate at. AUPRC reflects that tradeoff directly.

**Why synthetic data?** Real fraud datasets are heavily sanitized, legally restricted, and rarely include the entity-level (device/IP/merchant) graph structure needed to demonstrate ring detection at all. The synthetic generator explicitly encodes 5 real fraud mechanics (card-testing velocity, geo-velocity anomalies, bust-out spending curves, identity theft, merchant collusion) so the model — and its evaluation — has ground truth to check against.

---

## 📁 Project Structure

```
neural-fraud-detector/
├── data/
│   ├── generator/
│   │   └── synthesizer.py        # Synthetic data engine (5 fraud scenarios)
│   └── pipeline/
│       ├── features.py           # Feature engineering (rolling, geo, temporal) — 49 features
│       └── graph_builder.py      # HeteroData graph construction (PyG) for the batch model
│
├── models/
│   ├── transformer.py            # FT-Transformer from scratch
│   ├── gnn.py                    # Temporal heterogeneous GNN (GAT)
│   └── hybrid.py                 # Full hybrid model (FT-Transformer + GNN + anomaly head)
│
├── training/
│   ├── train.py                  # Real-time model training (FT-Transformer only, fast baseline)
│   ├── train_graph.py            # Batch/hybrid model training (FT-Transformer + GNN)
│   ├── losses.py                 # Focal loss + combined reconstruction loss
│   └── metrics.py                # AUPRC, F1 sweep, metric tracker
│
├── api/
│   └── main.py                   # FastAPI serving: /score, /score/batch, /metrics, /health
│
├── dashboard/
│   └── app.py                    # Streamlit monitoring dashboard
│
├── tests/
│   └── test_all.py               # Unit tests (data, features, both models, losses, metrics)
│
├── scripts/
│   ├── run_pipeline.py           # End-to-end: generate data → features → build graphs
│   ├── smoke_train.py            # CI smoke training test (real-time model)
│   ├── score_with_graph.py       # Offline batch scoring with the hybrid model
│   └── validate_data.py          # Data quality validation
│
├── docker/
│   ├── Dockerfile.train          # GPU training image (CUDA)
│   ├── Dockerfile.api            # CPU API serving image
│   ├── docker-compose.yml        # Full stack orchestration
│   └── prometheus.yml            # Scrape config for the API's /metrics endpoint
│
├── .github/workflows/
│   └── ci_cd.yml                 # CI (test + lint) → CD (build + deploy)
│
├── LICENSE
└── configs/
    └── config.yaml                # Hydra config (model, training, data)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (optional but recommended for training)
- Docker + Docker Compose

### 1. Install Dependencies

```bash
git clone https://github.com/achraf-gasmi/neural-fraud-detector.git
cd neural-fraud-detector

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121   # or the cpu index if you have no GPU
pip install -r requirements.txt

# Only needed for the hybrid/graph training path (training/train_graph.py) —
# pyg-lib backs PyG's heterogeneous NeighborLoader. Pick the wheel matching
# your torch build; see https://data.pyg.org/whl/ for other versions.
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### 2. Generate Data, Run the Feature Pipeline, and Build the Graph

```bash
python scripts/run_pipeline.py
# 1. Generates 500K synthetic transactions across 5 fraud scenarios
# 2. Runs feature engineering (rolling windows, geo-velocity, cyclic encoding) → 49 features
# 3. Builds the transaction graph (users/merchants/devices/IPs) for the hybrid model
# Saves train/val/test splits + graphs to data/processed/
```

### 3. Train

```bash
# Start MLflow tracking server (Windows: add --workers 1)
mlflow server --port 5000 --workers 1

# In a new terminal — fast baseline: FT-Transformer only
python -m training.train

# Full hybrid model: FT-Transformer + Temporal GNN (needs step 2's graphs + pyg-lib)
python -m training.train_graph

# View experiments at http://localhost:5000
```

### 4. Serve (Real-Time Path)

```bash
python -m api.main

curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "txn_id": "test_001",
    "timestamp": "2024-01-15T14:32:00",
    "user_id": "user_001",
    "merchant_id": "merch_001",
    "amount": 2500.00,
    "currency": "USD",
    "merchant_category": "cryptocurrency",
    "merchant_country": "RU",
    "merchant_lat": 55.75, "merchant_lon": 37.62,
    "user_lat": 40.71, "user_lon": -74.00,
    "device_id": "new_device_xyz",
    "ip_address": "185.220.101.1",
    "user_age": 42,
    "credit_limit": 5000.00
  }'
```

**Response:**
```json
{
  "txn_id": "test_001",
  "fraud_probability": 0.9134,
  "is_fraud": true,
  "risk_level": "CRITICAL",
  "threshold_used": 0.5,
  "latency_ms": 12.4,
  "top_risk_factors": [
    {"feature": "geo_velocity_kmh", "importance": 0.412, "value": 892.3},
    {"feature": "unique_devices_1h", "importance": 0.318, "value": 1.0},
    {"feature": "merchant_category", "importance": 0.201, "value": 10.0}
  ]
}
```

### 5. Score in Batch (Fraud-Ring / Hybrid Path)

```bash
python scripts/score_with_graph.py \
  --checkpoint checkpoints/best_hybrid_model.pt \
  --graph-dir data/processed/graphs \
  --split test \
  --output data/processed/graph_scores.csv
```

### 6. Dashboard

```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

### 7. Full Stack with Docker

```bash
cd docker
docker compose up api dashboard mlflow
# optional: metrics scraping
docker compose --profile monitoring up prometheus
```

---

## 🔬 Data Engineering

### Synthetic Data Generator

Five realistic fraud scenarios, each mimicking real-world patterns:

| Scenario | Description | Key Signals |
|---|---|---|
| **Card Testing** | Rapid micro-transactions to verify stolen card | High velocity, tiny amounts, multiple merchants |
| **Account Takeover** | Fraudster logs in from new device/location | Geo-velocity anomaly, unknown device, new IP |
| **Bust-Out** | Max out all credit rapidly then disappear | Spike in velocity, amounts near credit limit |
| **Identity Theft** | Large purchases on new accounts | New account + immediate high-value txns |
| **Merchant Collusion** | Fake transactions for cashback | Round amounts, same merchant, periodic timing |

### Feature Engineering

**49 features** across 6 categories:

- **Rolling windows** (1h, 6h, 24h, 168h): txn count, sum, max, std, unique merchants/devices/IPs
- **Geo-velocity**: km/h between consecutive transactions — catches impossible location jumps
- **Temporal**: cyclic sin/cos encoding of hour, day-of-week, month — no discontinuity at boundaries
- **Amount**: log transform, credit ratio, personal z-score, round-amount flag
- **Entity-level**: user age, credit limit
- **Categorical**: merchant category, merchant country

### Graph Construction (Batch Path)

`data/pipeline/graph_builder.py` builds a heterogeneous graph per split: `transaction` nodes (the 49 features above) connected to `user`, `merchant`, `device`, and `ip` entity nodes, plus `precedes` edges linking each user's transactions in temporal order. `training/train_graph.py` trains the hybrid model over this graph using mini-batch neighborhood sampling (PyG `NeighborLoader`), so training scales past what fits in memory as a single batch.

---

## 📊 Evaluation

Primary metric: **AUPRC** (Area Under Precision-Recall Curve) — see [The Problem](#the-problem) for why accuracy/AUROC alone are misleading here. `training/metrics.py` also reports AUROC, F1 at the optimal threshold, precision, recall, and false-positive rate, and `training/train.py` / `training/train_graph.py` log all of these to MLflow every epoch.

This repo doesn't ship pre-baked benchmark numbers — a number without the run that produced it isn't verifiable, and the honest state is: run it yourself and see what you get. To reproduce:

```bash
python scripts/run_pipeline.py        # generates data + features + graphs
python -m training.train              # real-time model — check MLflow at localhost:5000
python -m training.train_graph        # hybrid model — logged to a separate MLflow experiment
```

Both scripts save the best checkpoint (by validation AUPRC) to `checkpoints/`, plus a per-epoch metric history CSV, so a full run is fully auditable after the fact.

---

## 🔧 MLOps Stack

| Component | Tool | Purpose |
|---|---|---|
| Experiment tracking | MLflow | Log metrics, params, artifacts per run (separate experiments for the real-time and hybrid models) |
| Config management | Hydra | Reproducible, composable experiment configs |
| Serving | FastAPI + Uvicorn | REST API with single + batch scoring |
| Metrics | Prometheus | `/metrics` endpoint on the API (request counts, latency histogram, fraud-flagged counter) |
| Containerization | Docker | Reproducible environments |
| Orchestration | Docker Compose | Multi-service stack (API + Dashboard + MLflow + optional Prometheus) |
| CI/CD | GitHub Actions | Auto-test on PR, build + deploy on merge to main |
| Dashboard | Streamlit | Live monitoring, score distribution, fraud alerts |

### CI/CD Pipeline

```
PR opened
    │
    ├── Lint (ruff)
    ├── Unit tests (pytest) — covers both the real-time and hybrid model paths
    ├── Data pipeline validation
    └── Model smoke test (2 epochs, real-time model)
            │
    Merge to main
            │
    ├── Build API Docker image → ghcr.io
    ├── Build Trainer Docker image → ghcr.io
    └── Deploy via SSH → production server
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Smoke tests only (fast, CI-friendly)
pytest tests/ -v -k "smoke"

# With coverage report
pytest tests/ --cov=. --cov-report=html
```

Test coverage includes:
- ✅ Data generator (schema, fraud rate, scenario coverage, null checks)
- ✅ Feature pipeline (temporal splits, no leakage, feature columns, scaling)
- ✅ FT-Transformer (output shape, gradient flow, variable batch sizes)
- ✅ Real-time model — `TabularFraudDetector` (forward pass, probability range)
- ✅ Hybrid model — `FraudDetector` over a real graph (forward pass, probability range, entity-embedding indexing regression test)
- ✅ Loss functions (focal loss weighting behavior, combined loss structure)
- ✅ Metrics (perfect/random classifier baselines, early stopping logic)

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

## Contact

Achraf Gasmi — [achrafgasmi58@gmail.com](mailto:achrafgasmi58@gmail.com) · [linkedin.com/in/achraf-gasmi](https://linkedin.com/in/achraf-gasmi)
