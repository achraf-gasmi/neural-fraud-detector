"""
FraudShield — FastAPI Inference Service
========================================
Real-time fraud scoring endpoint with SHAP explainability.
"""

import os
import pickle
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import shap
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from data.pipeline.features import FEATURE_COLUMNS, add_temporal_features, add_amount_features
from models.tabular import TabularFraudDetector


# ─────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────

class ModelState:
    model = None
    scaler = None
    encoders = None
    explainer = None
    n_features = None
    threshold = 0.5
    device = None
    request_count = 0
    total_latency_ms = 0.0


state = ModelState()


# ─────────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading FraudShield model...")

    model_path = os.getenv("MODEL_PATH", "checkpoints/best_model.pt")
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "data/processed/artifacts")

    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load scaler and encoders
    with open(f"{artifacts_dir}/scaler.pkl", "rb") as f:
        state.scaler = pickle.load(f)
    with open(f"{artifacts_dir}/encoders.pkl", "rb") as f:
        state.encoders = pickle.load(f)

    state.n_features = len(FEATURE_COLUMNS)
    state.threshold = float(os.getenv("THRESHOLD", "0.5"))

    # Load model
    checkpoint = torch.load(model_path, map_location=state.device, weights_only=False)
    cfg = checkpoint["config"]

    # Build model from config (simplified for API)
    class SimpleConfig:
        class model:
            class transformer:
                d_token = cfg["model"]["transformer"]["d_token"]
                n_blocks = cfg["model"]["transformer"]["n_blocks"]
                attention_n_heads = cfg["model"]["transformer"]["attention_n_heads"]
                attention_dropout = cfg["model"]["transformer"]["attention_dropout"]
                ffn_d_hidden_multiplier = cfg["model"]["transformer"]["ffn_d_hidden_multiplier"]
                ffn_dropout = cfg["model"]["transformer"]["ffn_dropout"]
            class fusion:
                hidden_dim = cfg["model"]["fusion"]["hidden_dim"]
                dropout = cfg["model"]["fusion"]["dropout"]

    from omegaconf import OmegaConf
    cfg_obj = OmegaConf.create(cfg)
    model = TabularFraudDetector(n_features=state.n_features, cfg=cfg_obj)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(state.device)
    state.model = model

    logger.success(f"Model loaded from {model_path} (epoch {checkpoint['epoch']})")
    logger.info(f"Best AUPRC: {checkpoint['metrics']['auprc']:.4f}")

    yield  # Application runs here

    logger.info("Shutting down...")


app = FastAPI(
    title="FraudShield API",
    description="Real-time fraud detection with deep learning",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Prometheus Metrics
# ─────────────────────────────────────────────

SCORE_REQUESTS = Counter("fraudshield_score_requests_total", "Total scoring requests", ["endpoint"])
FRAUD_FLAGGED = Counter("fraudshield_fraud_flagged_total", "Transactions flagged as fraud")
SCORE_LATENCY = Histogram("fraudshield_score_latency_seconds", "Scoring latency in seconds", ["endpoint"])


# ─────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────

class TransactionRequest(BaseModel):
    txn_id: str = Field(..., description="Unique transaction ID")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    user_id: str
    merchant_id: str
    amount: float = Field(..., gt=0)
    currency: str = "USD"
    merchant_category: str
    merchant_country: str
    merchant_lat: float
    merchant_lon: float
    user_lat: float
    user_lon: float
    device_id: str
    ip_address: str
    user_age: int = Field(..., ge=18, le=120)
    credit_limit: float = Field(..., gt=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "txn_id": "txn_abc123",
                "timestamp": "2024-01-15T14:32:00",
                "user_id": "user_001",
                "merchant_id": "merch_001",
                "amount": 1250.00,
                "currency": "USD",
                "merchant_category": "electronics",
                "merchant_country": "US",
                "merchant_lat": 40.7128,
                "merchant_lon": -74.0060,
                "user_lat": 40.7580,
                "user_lon": -73.9855,
                "device_id": "dev_xyz789",
                "ip_address": "192.168.1.1",
                "user_age": 35,
                "credit_limit": 5000.00,
            }
        }
    }


class FraudScoreResponse(BaseModel):
    txn_id: str
    fraud_probability: float
    is_fraud: bool
    risk_level: str          # LOW / MEDIUM / HIGH / CRITICAL
    threshold_used: float
    latency_ms: float
    top_risk_factors: list[dict]
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    request_count: int
    avg_latency_ms: float
    device: str


# ─────────────────────────────────────────────
# Feature Engineering (single transaction)
# ─────────────────────────────────────────────

def preprocess_single(req: TransactionRequest) -> np.ndarray:
    """Convert a single request into model-ready feature vector."""
    row = {
        "txn_id": req.txn_id,
        "timestamp": pd.to_datetime(req.timestamp),
        "user_id": req.user_id,
        "merchant_id": req.merchant_id,
        "amount": req.amount,
        "currency": req.currency,
        "merchant_category": req.merchant_category,
        "merchant_country": req.merchant_country,
        "merchant_lat": req.merchant_lat,
        "merchant_lon": req.merchant_lon,
        "user_lat": req.user_lat,
        "user_lon": req.user_lon,
        "device_id": req.device_id,
        "ip_address": req.ip_address,
        "user_age": req.user_age,
        "credit_limit": req.credit_limit,
        "is_fraud": 0,
    }

    df = pd.DataFrame([row])
    df = add_temporal_features(df)
    df = add_amount_features(df)

    # Encode categoricals
    for col, le in state.encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
            )

    # Fill missing rolling/velocity features with 0
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    features = df[FEATURE_COLUMNS].fillna(0).values.astype(np.float32)
    features = state.scaler.transform(features)
    return features


def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    elif prob < 0.5:
        return "MEDIUM"
    elif prob < 0.8:
        return "HIGH"
    return "CRITICAL"


def get_top_risk_factors(features: np.ndarray, n: int = 5) -> list[dict]:
    """
    Approximate SHAP-style feature importance using gradient sensitivity.
    For production: use shap.DeepExplainer or shap.KernelExplainer with a background set.
    """
    x = torch.tensor(features, dtype=torch.float32).to(state.device)
    x.requires_grad_(True)

    out = state.model(x)
    prob = out["probs"]
    prob.backward()

    grads = x.grad.abs().squeeze(0).cpu().numpy()

    feature_importances = sorted(
        zip(FEATURE_COLUMNS, grads),
        key=lambda pair: pair[1],
        reverse=True
    )

    return [
        {
            "feature": name,
            "importance": round(float(imp), 4),
            "value": round(float(features[0, i]), 4),
        }
        for i, (name, imp) in enumerate(feature_importances[:n])
    ]


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", response_model=HealthResponse)
async def health():
    avg_lat = state.total_latency_ms / max(state.request_count, 1)
    return HealthResponse(
        status="healthy" if state.model is not None else "unhealthy",
        model_loaded=state.model is not None,
        request_count=state.request_count,
        avg_latency_ms=round(avg_lat, 2),
        device=str(state.device),
    )


@app.post("/score", response_model=FraudScoreResponse)
async def score_transaction(req: TransactionRequest):
    """Score a single transaction for fraud probability."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()

    features = preprocess_single(req)
    x = torch.tensor(features, dtype=torch.float32).to(state.device)

    with torch.no_grad():
        out = state.model(x)
        prob = float(out["probs"].item())

    risk_factors = get_top_risk_factors(features)

    latency_ms = (time.perf_counter() - t0) * 1000
    state.request_count += 1
    state.total_latency_ms += latency_ms

    SCORE_REQUESTS.labels(endpoint="score").inc()
    SCORE_LATENCY.labels(endpoint="score").observe(latency_ms / 1000)
    if prob >= state.threshold:
        FRAUD_FLAGGED.inc()

    return FraudScoreResponse(
        txn_id=req.txn_id,
        fraud_probability=round(prob, 4),
        is_fraud=prob >= state.threshold,
        risk_level=get_risk_level(prob),
        threshold_used=state.threshold,
        latency_ms=round(latency_ms, 2),
        top_risk_factors=risk_factors,
    )


@app.post("/score/batch")
async def score_batch(transactions: list[TransactionRequest]):
    """Score multiple transactions in one call."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(transactions) > 1000:
        raise HTTPException(status_code=400, detail="Max batch size is 1000")

    t0 = time.perf_counter()
    features_list = [preprocess_single(req) for req in transactions]
    features = np.vstack(features_list)
    x = torch.tensor(features, dtype=torch.float32).to(state.device)

    with torch.no_grad():
        out = state.model(x)
        probs = out["probs"].cpu().numpy()

    latency_ms = (time.perf_counter() - t0) * 1000

    SCORE_REQUESTS.labels(endpoint="score_batch").inc(len(transactions))
    SCORE_LATENCY.labels(endpoint="score_batch").observe(latency_ms / 1000)
    FRAUD_FLAGGED.inc(int((probs >= state.threshold).sum()))

    return {
        "results": [
            {
                "txn_id": req.txn_id,
                "fraud_probability": round(float(p), 4),
                "is_fraud": float(p) >= state.threshold,
                "risk_level": get_risk_level(float(p)),
            }
            for req, p in zip(transactions, probs)
        ],
        "batch_size": len(transactions),
        "latency_ms": round(latency_ms, 2),
        "avg_latency_per_txn_ms": round(latency_ms / len(transactions), 2),
    }


@app.get("/model/info")
async def model_info():
    """Return model metadata."""
    return {
        "n_features": state.n_features,
        "feature_names": FEATURE_COLUMNS,
        "threshold": state.threshold,
        "device": str(state.device),
        "parameters": sum(p.numel() for p in state.model.parameters()),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)