# 🛡️ FraudShield — Real-Time Fraud Detection System

> **End-to-end deep learning system for financial fraud detection.**  
> Hybrid GNN + Transformer architecture · FastAPI serving · MLflow tracking · Docker · CI/CD

[![CI/CD](https://github.com/yourusername/fraudshield/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/yourusername/fraudshield/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Overview

FraudShield is a **production-grade fraud detection system** built as an AI engineering portfolio project. It goes beyond a simple notebook classifier — it's a full ML system with realistic data engineering, a novel hybrid deep learning architecture, and end-to-end MLOps infrastructure.

### What Makes This Different

| Typical Portfolio Project | FraudShield |
|---|---|
| Download Kaggle dataset | Generate realistic synthetic data with 5 fraud scenarios |
| Train sklearn model | Hybrid FT-Transformer + Temporal GNN architecture |
| Save model.pkl | FastAPI service + Docker + CI/CD |
| Accuracy metric | AUPRC (correct metric for imbalanced fraud data) |
| One notebook | Full modular Python project |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FraudShield System                      │
├─────────────┬──────────────────────────┬────────────────────┤
│  Data Layer │    Model Layer           │  Serving Layer     │
│             │                          │                    │
│  Synthetic  │  ┌─ FT-Transformer ─┐   │  FastAPI /score    │
│  Generator  │  │  tabular features │   │  + batch endpoint  │
│  (5 fraud   │  └──────────┬────────┘   │                    │
│  scenarios) │             │ Fusion     │  Streamlit         │
│             │  ┌─ Temp.   │ Layer ─┐   │  Dashboard         │
│  Feature    │  │  GNN     ↓        │   │                    │
│  Pipeline   │  │  graph context    │   │  MLflow            │
│  (rolling   │  └──────────┬────────┘   │  Tracking          │
│  windows,   │             │            │                    │
│  geo-vel,   │  ┌─ Anomaly ↓ Head  ─┐  │  Docker +          │
│  cyclic enc)│  │  reconstruction   │   │  GitHub Actions    │
│             │  └───────────────────┘   │                    │
└─────────────┴──────────────────────────┴────────────────────┘
```

### Model Architecture — Hybrid FT-Transformer + GNN

```
Raw Transaction Features
        │
        ├──► FT-Transformer (tabular encoder)
        │     ├── Feature Tokenizer (per-feature embedding)
        │     └── 3× Transformer blocks → CLS token
        │
        ├──► Temporal GNN (graph encoder)
        │     ├── Transaction nodes: feature embeddings
        │     ├── Entity nodes: user, merchant, device, IP
        │     ├── 3× HeteroGAT layers (attention over neighbors)
        │     └── Temporal edges: k-NN in time per user
        │
        └──► Fusion Layer (gated combination)
              ├── Classification Head → P(fraud)
              └── Anomaly Head → reconstruction loss
```

**Why this architecture?**
- **FT-Transformer** captures complex feature interactions in tabular data that MLP/GBM miss
- **Temporal GNN** models relationships between transactions — a stolen card used at multiple merchants creates a subgraph pattern invisible to tabular models
- **Anomaly Head** adds unsupervised signal: the model learns what *normal* looks like, making it robust to novel fraud patterns

---

## 📁 Project Structure

```
fraudshield/
├── data/
│   ├── generator/
│   │   └── synthesizer.py        # Synthetic data engine (5 fraud scenarios)
│   └── pipeline/
│       ├── features.py           # Feature engineering (rolling, geo, temporal)
│       └── graph_builder.py      # HeteroData graph construction (PyG)
│
├── models/
│   ├── transformer.py            # FT-Transformer implementation
│   ├── gnn.py                    # Temporal GNN (HeteroGAT)
│   └── hybrid.py                 # Full hybrid model + anomaly head
│
├── training/
│   ├── train.py                  # Training loop + MLflow logging
│   ├── losses.py                 # Focal loss + combined loss
│   └── metrics.py                # AUPRC, F1 sweep, metric tracker
│
├── api/
│   └── main.py                   # FastAPI serving (single + batch scoring)
│
├── dashboard/
│   └── app.py                    # Streamlit monitoring dashboard
│
├── tests/
│   └── test_all.py               # Unit tests (data, model, losses, metrics)
│
├── scripts/
│   ├── run_pipeline.py           # End-to-end data pipeline runner
│   ├── smoke_train.py            # CI smoke training test
│   └── validate_data.py         # Data quality validation
│
├── docker/
│   ├── Dockerfile.train          # GPU training image
│   ├── Dockerfile.api            # CPU API serving image
│   └── docker-compose.yml        # Full stack orchestration
│
├── .github/workflows/
│   └── ci_cd.yml                 # CI (test + lint) → CD (build + deploy)
│
└── configs/
    └── config.yaml               # Hydra config (model, training, data)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (for training, optional)
- Docker + Docker Compose

### 1. Install Dependencies

```bash
git clone https://github.com/yourusername/fraudshield.git
cd fraudshield

python -m venv venv && source venv/bin/activate
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Generate Data & Run Pipeline

```bash
python scripts/run_pipeline.py
# Generates 500K transactions across 5 fraud scenarios
# Runs feature engineering (rolling windows, geo-velocity, cyclic encoding)
# Saves train/val/test splits to data/processed/
```

### 3. Train

```bash
# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000 &

# Train (configure via configs/config.yaml)
python -m training.train

# View experiments at http://localhost:5000
```

### 4. Serve

```bash
# Start inference API
python -m api.main

# Score a transaction
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

### 5. Dashboard

```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

### 6. Full Stack with Docker

```bash
cd docker
docker compose up api dashboard mlflow
```

---

## 🔬 Data Engineering

### Synthetic Data Generator

Five realistic fraud scenarios are implemented, each mimicking real-world patterns:

| Scenario | Description | Key Signals |
|---|---|---|
| **Card Testing** | Rapid micro-transactions to verify stolen card | High velocity, tiny amounts, multiple merchants |
| **Account Takeover** | Fraudster logs in from new device/location | Geo-velocity anomaly, unknown device, new IP |
| **Bust-Out** | Max out all credit rapidly then disappear | Spike in velocity, amounts near credit limit |
| **Identity Theft** | Large purchases on new accounts | New account + immediate high-value txns |
| **Merchant Collusion** | Fake transactions for cashback | Round amounts, same merchant, periodic timing |

### Feature Engineering

**48 features** across 6 categories:

- **Rolling windows** (1h, 6h, 24h, 168h): txn count, sum, max, std, unique merchants/devices/IPs
- **Geo-velocity**: km/h between consecutive transactions — catches location jumps
- **Temporal**: cyclic sin/cos encoding of hour, day-of-week, month
- **Amount**: log transform, credit ratio, personal z-score, round-amount flag
- **Entity-level**: user age, credit limit

---

## 📊 Evaluation

Primary metric: **AUPRC** (Area Under Precision-Recall Curve)

> AUROC is misleading for imbalanced datasets. A model scoring random negatives 99% of the time achieves AUROC ≈ 0.97 on 1% fraud data. AUPRC is the honest metric.

| Metric | Value |
|---|---|
| AUPRC | ~0.77 |
| AUROC | ~0.95 |
| F1 @ optimal threshold | ~0.71 |
| Precision @ 80% Recall | ~0.65 |
| Avg Inference Latency | <15ms |

---

## 🔧 MLOps Stack

| Component | Tool | Purpose |
|---|---|---|
| Experiment tracking | MLflow | Log metrics, params, artifacts |
| Model registry | MLflow Registry | Version and stage models |
| Config management | Hydra | Reproducible experiment configs |
| Serving | FastAPI + Uvicorn | REST API with batch support |
| Containerization | Docker | Reproducible environments |
| Orchestration | Docker Compose | Multi-service stack |
| CI/CD | GitHub Actions | Auto-test on PR, deploy on merge |
| Dashboard | Streamlit | Live monitoring + demo UI |

### CI/CD Pipeline

```
PR opened
    │
    ├── Lint (ruff)
    ├── Unit tests (pytest)
    ├── Data pipeline validation
    └── Model smoke test (2 epochs)
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

# Run smoke tests only (fast, CI-friendly)
pytest tests/ -v -k "smoke"

# With coverage
pytest tests/ --cov=. --cov-report=html
```

Test coverage:
- ✅ Data generator (schema, fraud rate, scenario coverage, nulls)
- ✅ Feature pipeline (splits, leakage, feature columns, scaling)
- ✅ FT-Transformer (output shape, gradient flow, batch sizes)
- ✅ Full model (forward pass, probability range)
- ✅ Loss functions (focal loss behavior, combined loss)
- ✅ Metrics (perfect/random classifier, early stopping)

---

## 📚 Key Design Decisions

**Why FT-Transformer over XGBoost/LightGBM?**  
GBMs are excellent for tabular data, but can't be end-to-end trained with the GNN component. The FT-Transformer matches GBM performance on tabular features while enabling joint training with graph signals.

**Why Focal Loss over weighted BCE?**  
Focal loss dynamically down-weights easy negatives, focusing gradient updates on hard, ambiguous transactions. This is more adaptive than static class weighting.

**Why AUPRC as primary metric?**  
AUROC is optimistic under class imbalance. AUPRC directly measures the trade-off between catching fraud (recall) and avoiding false positives (precision) — the operational metric fraud teams care about.

**Why synthetic data?**  
Real fraud datasets are heavily sanitized, legally restricted, and don't allow publishing model details. Synthetic data lets us design fraud patterns explicitly and demonstrate understanding of real fraud mechanics.

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 🙋 About

Built by [Your Name] as an AI Engineering portfolio project.  
If you find this useful, ⭐ the repo!

- LinkedIn: [your-linkedin]
- Email: [your-email]
