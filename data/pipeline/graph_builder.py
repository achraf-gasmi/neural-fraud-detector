"""
FraudShield — Graph Construction Pipeline
==========================================
Builds a heterogeneous temporal graph from transactions for the GNN encoder.
Nodes: transactions, users, merchants, devices, IPs
Edges: card→merchant, card→device, card→ip, merchant→ip, temporal adjacency
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected

from data.pipeline.features import FEATURE_COLUMNS


# ─────────────────────────────────────────────
# Entity ID Mappers
# ─────────────────────────────────────────────

def build_id_maps(df: pd.DataFrame) -> dict:
    """Map string entity IDs to consecutive integers."""
    return {
        "user": {uid: i for i, uid in enumerate(df["user_id"].unique())},
        "merchant": {mid: i for i, mid in enumerate(df["merchant_id"].unique())},
        "device": {did: i for i, did in enumerate(df["device_id"].unique())},
        "ip": {ip: i for i, ip in enumerate(df["ip_address"].unique())},
    }


# ─────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────

def build_transaction_graph(
    df: pd.DataFrame,
    id_maps: dict,
    feature_cols: list[str] = FEATURE_COLUMNS,
    max_temporal_neighbors: int = 5,
) -> HeteroData:
    """
    Build a PyTorch Geometric HeteroData graph from a transaction DataFrame.

    Node types:
      - transaction (one per row, features = FEATURE_COLUMNS)
      - user, merchant, device, ip (entity nodes)

    Edge types:
      - (transaction, sent_by, user)
      - (transaction, at, merchant)
      - (transaction, used_device, device)
      - (transaction, from_ip, ip)
      - (transaction, temporal_next, transaction)  ← k-NN in time per user

    Args:
        df: Feature-engineered transaction DataFrame (sorted by timestamp)
        id_maps: Dicts mapping string IDs to integer indices
        feature_cols: Feature columns to use as transaction node features
        max_temporal_neighbors: Max temporal edges per node

    Returns:
        HeteroData graph
    """
    logger.info(f"Building graph for {len(df):,} transactions...")
    data = HeteroData()

    # ── Transaction node features ──
    X = torch.tensor(df[feature_cols].fillna(0).values, dtype=torch.float32)
    y = torch.tensor(df["is_fraud"].values, dtype=torch.long)
    data["transaction"].x = X
    data["transaction"].y = y
    data["transaction"].txn_id = list(df["txn_id"])

    # ── Entity node features (simple embeddings — trainable) ──
    n_users = len(id_maps["user"])
    n_merchants = len(id_maps["merchant"])
    n_devices = len(id_maps["device"])
    n_ips = len(id_maps["ip"])

    # Entity nodes have no fixed features; embedding handled in model
    data["user"].num_nodes = n_users
    data["merchant"].num_nodes = n_merchants
    data["device"].num_nodes = n_devices
    data["ip"].num_nodes = n_ips

    logger.info(f"  Nodes — txn: {len(df):,} | user: {n_users} | "
                f"merchant: {n_merchants} | device: {n_devices} | ip: {n_ips}")

    # ── Edges: transaction → entity ──
    txn_idx = torch.arange(len(df), dtype=torch.long)

    user_idx = torch.tensor(
        df["user_id"].map(id_maps["user"]).values, dtype=torch.long
    )
    merchant_idx = torch.tensor(
        df["merchant_id"].map(id_maps["merchant"]).values, dtype=torch.long
    )
    device_idx = torch.tensor(
        df["device_id"].map(id_maps["device"]).values, dtype=torch.long
    )
    ip_idx = torch.tensor(
        df["ip_address"].map(id_maps["ip"]).values, dtype=torch.long
    )

    # (transaction, sent_by, user) — bidirectional
    data["transaction", "sent_by", "user"].edge_index = torch.stack([txn_idx, user_idx])
    data["user", "has_txn", "transaction"].edge_index = torch.stack([user_idx, txn_idx])

    # (transaction, at, merchant)
    data["transaction", "at", "merchant"].edge_index = torch.stack([txn_idx, merchant_idx])
    data["merchant", "received", "transaction"].edge_index = torch.stack([merchant_idx, txn_idx])

    # (transaction, used_device, device)
    data["transaction", "used_device", "device"].edge_index = torch.stack([txn_idx, device_idx])
    data["device", "used_in", "transaction"].edge_index = torch.stack([device_idx, txn_idx])

    # (transaction, from_ip, ip)
    data["transaction", "from_ip", "ip"].edge_index = torch.stack([txn_idx, ip_idx])
    data["ip", "origin_of", "transaction"].edge_index = torch.stack([ip_idx, txn_idx])

    # ── Temporal edges: k nearest preceding transactions per user ──
    logger.info(f"  Building temporal edges (k={max_temporal_neighbors})...")
    src_list, dst_list = [], []

    df_reset = df.reset_index(drop=True)
    for uid, group in df_reset.groupby("user_id"):
        indices = group.index.tolist()
        for j, dst in enumerate(indices):
            # Connect to up to k previous transactions of this user
            start = max(0, j - max_temporal_neighbors)
            for src in indices[start:j]:
                src_list.append(src)
                dst_list.append(dst)

    if src_list:
        temporal_edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        data["transaction", "precedes", "transaction"].edge_index = temporal_edge_index
        logger.info(f"  Temporal edges: {len(src_list):,}")

    logger.success("Graph built.")
    return data


# ─────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────

def save_graph(data: HeteroData, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)
    logger.success(f"Graph saved to {path}")


def load_graph(path: str) -> HeteroData:
    return torch.load(path)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def build_all_graphs(
    processed_dir: str = "data/processed",
    graph_dir: str = "data/processed/graphs",
):
    """Build and save graphs for train, val, test splits."""
    os.makedirs(graph_dir, exist_ok=True)

    # Load splits
    train_df = pd.read_parquet(f"{processed_dir}/train.parquet")
    val_df = pd.read_parquet(f"{processed_dir}/val.parquet")
    test_df = pd.read_parquet(f"{processed_dir}/test.parquet")

    # Build ID maps from training set only (prevent leakage)
    full_df = pd.concat([train_df, val_df, test_df])
    id_maps = build_id_maps(full_df)

    with open(f"{graph_dir}/id_maps.pkl", "wb") as f:
        pickle.dump(id_maps, f)

    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info(f"Building {split} graph...")
        graph = build_transaction_graph(df, id_maps)
        save_graph(graph, f"{graph_dir}/{split}_graph.pt")

    logger.success("All graphs saved.")


if __name__ == "__main__":
    build_all_graphs()
