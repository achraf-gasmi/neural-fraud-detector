# 🛡️ Neural Fraud Detector — Real-Time Fraud Detection System

> **End-to-end deep learning system for financial fraud detection.**  
> FT-Transformer + Anomaly Head · FastAPI serving · MLflow tracking · Docker · CI/CD

[![CI/CD](https://github.com/achraf-gasmi/neural-fraud-detector/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/achraf-gasmi/neural-fraud-detector/actions)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Overview

**Neural Fraud Detector** is a production-grade fraud detection system built as an AI Engineering portfolio project. It goes beyond a simple notebook classifier — it's a full ML system with realistic data engineering, a novel hybrid deep learning architecture, and end-to-end MLOps infrastructure.

### What Makes This Different

| Typical Portfolio Project | Neural Fraud Detector |
|---|---|
| Download Kaggle dataset | Generate realistic synthetic data with 5 fraud scenarios |
| Train sklearn model | FT-Transformer + Anomaly Head architecture |
| Save model.pkl | FastAPI service + Docker + CI/CD |
| Accuracy metric | AUPRC (correct metric for imbalanced fraud data) |
| One notebook | Full modular Python project |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Neural Fraud Detector System                │
├─────────────┬──────────────────────────┬────────────────────┤
│  Data Layer │    Model Layer           │  Serving Layer     │
│             │                          │                    │
│  Synthetic  │  ┌─ FT-Transformer ─┐   │  FastAPI /score    │
│  Generator  │  │  tabular features │   │  + batch endpoint  │
│  (5 fraud   │  └──────────┬────────┘   │                    │
│  scenarios) │             │ Fusion     │  Streamlit         │
│             │             │ Layer      │  Dashboard         │
│  Feature    │             ↓            │                    │
│  Pipeline   │  ┌─ Anomaly Head    ─┐  │  MLflow            │
│  (rolling   │  │  reconstruction   │  │  Tracking          │
│  windows,   │  └──────────┬────────┘  │                    │
│  geo-vel,   │             │            │  Docker +          │
│  cyclic enc)│             ↓            │  GitHub Actions    │
│             │    P(fraud) score         │                    │
└─────────────┴──────────────────────────┴────────────────────┘
```

### Model Architecture — FT-Transformer + Anomaly Head

```
Raw Transaction Features (48-dim)
        │
        ├──► FT-Transformer (tabular encoder)
        │     ├── Feature Tokenizer (per-feature learned embedding)
        │     ├── 3× Transformer blocks (Pre-Norm, Multi-Head Attention)
        │     └── CLS token → (B, d_token) representation
        │
        └──► Fusion Layer
              ├── Classification Head → P(fraud)
              └── Anomaly Head → reconstruction loss
                    (learns "normal" — robust to novel fraud patterns)
```

**Why this architecture?**
- **FT-Transformer** captures complex feature interactions in tabular data that MLP/GBM miss, by tokenizing each feature into a learned embedding before applying attention
- **Anomaly Head** adds unsupervised signal via reconstruction loss: the model jointly learns what *normal* looks like alongside supervised fraud classification — making it robust to novel, unseen fraud patterns
- **Focal Loss** dynamically down-weights easy negatives, focusing gradient updates on hard ambiguous transactions — more adaptive than static class weighting

---

## 📁 Project Structure

```
neural-fraud-detector/
├── data/
│   ├── generator/
│   │   └── synthesizer.py        # Synthetic data engine (5 fraud scenarios)
│   └── pipeline/
│       ├── features.py           # Feature engineering (rolling, geo, temporal)
│       └── graph_builder.py      # HeteroData graph construction (PyG)
│
├── models/
│   ├── transformer.py            # FT-Transformer from scratch
│   ├── gnn.py                    # Temporal GNN (HeteroGAT)
│   └── hybrid.py                 # Full hybrid model + anomaly head
│
├── training/
│   ├── train.py                  # Training loop + MLflow logging
│   ├── losses.py                 # Focal loss + combined reconstruction loss
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
│   └── validate_data.py          # Data quality validation
│
├── docker/
│   ├── Dockerfile.train          # GPU training image (CUDA)
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
- CUDA-capable GPU (optional but recommended)
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

# Install PyTorch with CUDA (RTX 30/40/50 series — adjust cu version as needed)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

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
# Start MLflow tracking server (Windows: add --workers 1)
mlflow server --port 5000 --workers 1

# In a new terminal — train (configure via configs/config.yaml)
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

Five realistic fraud scenarios, each mimicking real-world patterns:

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
- **Geo-velocity**: km/h between consecutive transactions — catches impossible location jumps
- **Temporal**: cyclic sin/cos encoding of hour, day-of-week, month — no discontinuity at boundaries
- **Amount**: log transform, credit ratio, personal z-score, round-amount flag
- **Entity-level**: user age, credit limit

---

## 📊 Results

Primary metric: **AUPRC** (Area Under Precision-Recall Curve)

> AUROC is misleading for imbalanced datasets. AUPRC directly measures the trade-off between catching fraud (recall) and avoiding false positives (precision) — the operational metric fraud teams care about.

### Test Set Performance (506K transactions, 2.77% fraud rate)

| Metric | Value |
|---|---|
| **AUPRC** | **0.9676** |
| **AUROC** | **0.9947** |
| **F1 @ optimal threshold** | **0.9439** |
| **Precision** | **98.25%** |
| **Recall** | **90.82%** |
| **False Positive Rate** | **0.05%** |
| True Positives | 2,018 |
| False Positives | 36 |
| Training time (RTX 5060, CPU-only epoch ~11min → GPU ~45s) | **~15 minutes** |
| Avg inference latency | **< 15ms** |

> **In production terms:** at 1M transactions/day, this model would flag only ~500 legitimate transactions as fraud while catching ~18,000 fraudulent ones.

### Training Curve

| Epoch | Val AUPRC | Val F1 |
|---|---|---|
| 1 | 0.9204 | 0.8997 |
| 5 | 0.9596 | 0.9306 |
| 8 | 0.9636 | 0.9352 |
| **12 (best)** | **0.9648** | **0.9381** |
| 22 (early stop) | 0.9579 | 0.9394 |

---

## 🔧 MLOps Stack

| Component | Tool | Purpose |
|---|---|---|
| Experiment tracking | MLflow | Log metrics, params, artifacts per run |
| Config management | Hydra | Reproducible, composable experiment configs |
| Serving | FastAPI + Uvicorn | REST API with single + batch scoring |
| Containerization | Docker | Reproducible environments |
| Orchestration | Docker Compose | Multi-service stack (API + Dashboard + MLflow) |
| CI/CD | GitHub Actions | Auto-test on PR, build + deploy on merge to main |
| Dashboard | Streamlit | Live monitoring, score distribution, fraud alerts |

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

# Smoke tests only (fast, CI-friendly)
pytest tests/ -v -k "smoke"

# With coverage report
pytest tests/ --cov=. --cov-report=html
```

Test coverage includes:
- ✅ Data generator (schema, fraud rate, scenario coverage, null checks)
- ✅ Feature pipeline (temporal splits, no leakage, feature columns, scaling)
- ✅ FT-Transformer (output shape, gradient flow, variable batch sizes)
- ✅ Full model (forward pass, probability range [0,1])
- ✅ Loss functions (focal loss weighting behavior, combined loss structure)
- ✅ Metrics (perfect/random classifier baselines, early stopping logic)

---

## 📚 Key Design Decisions

**Why FT-Transformer over XGBoost/LightGBM?**
GBMs are strong on tabular data but treat features independently. The FT-Transformer tokenizes each feature into a learned embedding and applies attention across all features — capturing interaction patterns GBMs miss. It also enables end-to-end training with auxiliary heads.

**Why Focal Loss over weighted BCE?**
Focal loss dynamically down-weights easy negatives `(1 - p_t)^γ`, focusing gradient updates on hard, ambiguous transactions. This is more adaptive than static class weighting and was shown to outperform it in class-imbalanced settings.

**Why AUPRC as primary metric?**
AUROC is optimistic under class imbalance — a model that scores all negatives near 0 and positives near 1 will achieve AUROC ≈ 0.99 even at 1% fraud rate. AUPRC reflects the precision-recall tradeoff that fraud operations teams actually optimize for.

**Why synthetic data?**
Real fraud datasets are heavily sanitized, legally restricted, and can't be published. Synthetic data lets us explicitly encode real fraud mechanics (card testing velocity, geo-velocity anomalies, bust-out spending curves) and validate the model against known patterns.

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 🙋 About

Built by **Achraf Gasmi** as an AI Engineering portfolio project.
If you find this useful, please ⭐ the repo!

- 🔗 LinkedIn: [linkedin.com/in/achraf-gasmi](https://linkedin.com/in/achraf-gasmi)
- 📧 Email: [achrafgasmi58@gmail.com]