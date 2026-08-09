"""
FraudShield — Offline Batch Scoring with the Hybrid (GNN) Model
===================================================================
The real-time API (api/main.py) scores one transaction at a time and can't
cheaply pull in a live graph neighborhood, so it uses the fast tabular-only
model. This script runs the full hybrid FT-Transformer + GNN model (trained
via training/train_graph.py) over an entire precomputed graph at once —
e.g. as a nightly/hourly batch job — to catch fraud rings that share a
device, IP, or merchant, which the real-time path can't see.

Usage:
    python scripts/score_with_graph.py \
        --checkpoint checkpoints/best_hybrid_model.pt \
        --graph-dir data/processed/graphs \
        --split test \
        --output data/processed/graph_scores.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle

import pandas as pd
import torch
from loguru import logger

from data.pipeline.graph_builder import load_graph
from models.hybrid import FraudDetector
from training.metrics import compute_all_metrics, print_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_hybrid_model.pt")
    parser.add_argument("--graph-dir", default="data/processed/graphs")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = load_graph(f"{args.graph_dir}/{args.split}_graph.pt").to(device)
    with open(f"{args.graph_dir}/{args.split}_txn_ids.pkl", "rb") as f:
        txn_ids = pickle.load(f)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    entity_counts = checkpoint["entity_counts"]
    cfg = checkpoint["config"]
    tc, gc, fc, ac = (cfg["model"]["transformer"], cfg["model"]["gnn"],
                       cfg["model"]["fusion"], cfg["model"]["anomaly_head"])

    n_features = graph["transaction"].x.shape[1]
    model = FraudDetector(
        n_features=n_features,
        transformer_d_token=tc["d_token"], transformer_n_blocks=tc["n_blocks"],
        transformer_n_heads=tc["attention_n_heads"], transformer_attn_dropout=tc["attention_dropout"],
        transformer_ffn_d_hidden_mult=tc["ffn_d_hidden_multiplier"], transformer_ffn_dropout=tc["ffn_dropout"],
        gnn_hidden_channels=gc["hidden_channels"], gnn_num_layers=gc["num_layers"],
        gnn_heads=gc["heads"], gnn_dropout=gc["dropout"], gnn_time_encoding_dim=gc["time_encoding_dim"],
        n_users=entity_counts["user"], n_merchants=entity_counts["merchant"],
        n_devices=entity_counts["device"], n_ips=entity_counts["ip"],
        fusion_hidden_dim=fc["hidden_dim"], fusion_dropout=fc["dropout"],
        anomaly_hidden_dim=ac["reconstruction_dim"], use_anomaly_head=ac["enabled"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    logger.success(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
                    f"(val AUPRC {checkpoint['metrics']['auprc']:.4f})")

    logger.info(f"Scoring {graph['transaction'].x.shape[0]:,} transactions "
                f"from the {args.split} split (full graph context)...")
    with torch.no_grad():
        out = model(graph)
        probs = out["probs"].cpu().numpy()

    y_true = graph["transaction"].y.cpu().numpy()
    results = pd.DataFrame({
        "txn_id": txn_ids,
        "is_fraud_true": y_true,
        "fraud_probability": probs,
        "is_fraud_pred": (probs >= args.threshold).astype(int),
    }).sort_values("fraud_probability", ascending=False)

    if (y_true >= 0).all() and y_true.max() > 0:
        metrics = compute_all_metrics(y_true, probs, threshold=args.threshold)
        print_metrics(metrics, prefix=f"Hybrid model — {args.split}")

    logger.info(f"Top {args.top_n} highest-risk transactions:")
    print(results.head(args.top_n).to_string(index=False))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        results.to_csv(args.output, index=False)
        logger.success(f"Full scored output written to {args.output}")


if __name__ == "__main__":
    main()
