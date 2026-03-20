"""
FraudShield — Test Suite
=========================
Tests for: data generator, feature pipeline, model architecture, API endpoints.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_raw_df():
    """Generate a tiny synthetic dataset for testing."""
    from data.generator.synthesizer import generate_dataset
    df = generate_dataset(
        n_users=50,
        n_merchants=20,
        n_transactions=1000,
        fraud_rate=0.02,
        output_path="/tmp/test_transactions.parquet",
        seed=0,
    )
    return df


@pytest.fixture(scope="module")
def processed_splits(small_raw_df):
    """Run the feature pipeline on the small dataset."""
    from data.pipeline.features import run_pipeline
    train, val, test = run_pipeline(
        raw_path="/tmp/test_transactions.parquet",
        output_dir="/tmp/fraudshield_test",
        artifacts_dir="/tmp/fraudshield_test/artifacts",
    )
    return train, val, test


@pytest.fixture
def sample_features(processed_splits):
    from data.pipeline.features import FEATURE_COLUMNS
    train, _, _ = processed_splits
    return torch.tensor(train[FEATURE_COLUMNS].values[:16], dtype=torch.float32)


# ─────────────────────────────────────────────
# Data Generator Tests
# ─────────────────────────────────────────────

class TestDataGenerator:

    def test_generates_correct_shape(self, small_raw_df):
        assert len(small_raw_df) >= 900, "Should generate close to 1000 transactions"
        assert "is_fraud" in small_raw_df.columns
        assert "txn_id" in small_raw_df.columns

    def test_fraud_rate_approximately_correct(self, small_raw_df):
        actual_rate = small_raw_df["is_fraud"].mean()
        # Allow ±2x variance due to small dataset
        assert 0.005 <= actual_rate <= 0.15, f"Fraud rate {actual_rate:.3%} out of range"

    def test_all_scenarios_present(self, small_raw_df):
        fraud_df = small_raw_df[small_raw_df["is_fraud"] == 1]
        scenarios = fraud_df["fraud_scenario"].unique()
        assert len(scenarios) >= 3, f"Expected ≥3 fraud scenarios, got: {scenarios}"

    def test_no_null_critical_columns(self, small_raw_df):
        critical = ["txn_id", "user_id", "merchant_id", "amount", "timestamp"]
        for col in critical:
            assert small_raw_df[col].isna().sum() == 0, f"Nulls in {col}"

    def test_amounts_positive(self, small_raw_df):
        assert (small_raw_df["amount"] > 0).all(), "All amounts should be positive"

    def test_timestamps_monotonic_after_sort(self, small_raw_df):
        sorted_df = small_raw_df.sort_values("timestamp")
        assert (sorted_df["timestamp"].diff().iloc[1:] >= pd.Timedelta(0)).all()


# ─────────────────────────────────────────────
# Feature Pipeline Tests
# ─────────────────────────────────────────────

class TestFeaturePipeline:

    def test_splits_non_empty(self, processed_splits):
        train, val, test = processed_splits
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_no_data_leakage(self, processed_splits):
        """Train must end before val, val before test (temporal split)."""
        train, val, test = processed_splits
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()

    def test_feature_columns_present(self, processed_splits):
        from data.pipeline.features import FEATURE_COLUMNS
        train, _, _ = processed_splits
        for col in FEATURE_COLUMNS:
            assert col in train.columns, f"Missing feature: {col}"

    def test_no_inf_values(self, processed_splits):
        from data.pipeline.features import FEATURE_COLUMNS
        train, val, test = processed_splits
        for split_name, df in [("train", train), ("val", val), ("test", test)]:
            inf_count = np.isinf(df[FEATURE_COLUMNS].values).sum()
            assert inf_count == 0, f"Inf values in {split_name} features"

    def test_features_scaled(self, processed_splits):
        """After StandardScaler, features should have ~0 mean."""
        from data.pipeline.features import FEATURE_COLUMNS
        train, _, _ = processed_splits
        means = train[FEATURE_COLUMNS].mean()
        # Allow some slack — not all features will be perfectly zero-mean
        assert (means.abs() < 2).all(), "Features appear unscaled"


# ─────────────────────────────────────────────
# Model Architecture Tests
# ─────────────────────────────────────────────

class TestFTTransformer:

    def test_output_shape(self, sample_features):
        from models.transformer import FTTransformer
        n_features = sample_features.shape[1]
        model = FTTransformer(n_features=n_features, d_token=64, n_blocks=2, attention_n_heads=4)
        out = model(sample_features)
        assert out.shape == (16, 64), f"Expected (16, 64), got {out.shape}"

    def test_gradients_flow(self, sample_features):
        from models.transformer import FTTransformer
        n_features = sample_features.shape[1]
        model = FTTransformer(n_features=n_features, d_token=64, n_blocks=2, attention_n_heads=4)
        out = model(sample_features)
        loss = out.mean()
        loss.backward()
        # Check at least one parameter has a gradient
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients flowing through FTTransformer"

    def test_different_batch_sizes(self, sample_features):
        from models.transformer import FTTransformer
        n_features = sample_features.shape[1]
        model = FTTransformer(n_features=n_features, d_token=64, n_blocks=1, attention_n_heads=4)
        for bs in [1, 4, 32]:
            x = torch.randn(bs, n_features)
            out = model(x)
            assert out.shape == (bs, 64)


class TestTabularFraudDetector:

    @pytest.fixture
    def model_and_features(self, sample_features):
        from omegaconf import OmegaConf
        from training.train import TabularFraudDetector

        cfg = OmegaConf.create({
            "model": {
                "transformer": {
                    "d_token": 64,
                    "n_blocks": 2,
                    "attention_n_heads": 4,
                    "attention_dropout": 0.1,
                    "ffn_d_hidden_multiplier": 1.333,
                    "ffn_dropout": 0.1,
                },
                "fusion": {"hidden_dim": 128, "dropout": 0.2},
            }
        })
        n_features = sample_features.shape[1]
        model = TabularFraudDetector(n_features=n_features, cfg=cfg)
        return model, sample_features

    def test_forward_shape(self, model_and_features):
        model, X = model_and_features
        out = model(X)
        assert "logits" in out
        assert "probs" in out
        assert "reconstruction" in out
        assert out["logits"].shape == (16, 1)
        assert out["probs"].shape == (16,)

    def test_probs_in_range(self, model_and_features):
        model, X = model_and_features
        out = model(X)
        probs = out["probs"].detach()
        assert (probs >= 0).all() and (probs <= 1).all()

    @pytest.mark.smoke
    def test_smoke_forward_pass(self, model_and_features):
        """Quick smoke test — just verify it doesn't crash."""
        model, X = model_and_features
        out = model(X)
        assert out["probs"].shape[0] == X.shape[0]


# ─────────────────────────────────────────────
# Loss Function Tests
# ─────────────────────────────────────────────

class TestLossFunctions:

    def test_focal_loss_higher_for_hard_examples(self):
        from training.losses import FocalLoss
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

        # Easy correct prediction (high prob, label=1)
        easy_logit = torch.tensor([[5.0]])
        easy_label = torch.tensor([1])
        easy_loss = loss_fn(easy_logit, easy_label)

        # Hard wrong prediction (low prob, label=1)
        hard_logit = torch.tensor([[-1.0]])
        hard_label = torch.tensor([1])
        hard_loss = loss_fn(hard_logit, hard_label)

        assert hard_loss > easy_loss, "Focal loss should penalize hard examples more"

    def test_combined_loss_returns_dict(self):
        from training.losses import CombinedFraudLoss
        loss_fn = CombinedFraudLoss()
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8,))
        recon = torch.randn(8, 32)
        original = torch.randn(8, 32)

        losses = loss_fn(logits, targets, recon, original)
        assert "total_loss" in losses
        assert "focal_loss" in losses
        assert "anomaly_loss" in losses

    def test_anomaly_loss_only_on_normal(self):
        """Anomaly head should primarily reconstruct legitimate transactions."""
        from training.losses import CombinedFraudLoss
        loss_fn = CombinedFraudLoss(anomaly_weight=1.0)
        logits = torch.randn(8, 1)
        targets = torch.zeros(8, dtype=torch.long)  # all normal
        recon = torch.zeros(8, 16)
        original = torch.zeros(8, 16)

        losses = loss_fn(logits, targets, recon, original)
        assert losses["anomaly_loss"].item() == pytest.approx(0.0, abs=1e-5)


# ─────────────────────────────────────────────
# Metrics Tests
# ─────────────────────────────────────────────

class TestMetrics:

    def test_perfect_classifier(self):
        from training.metrics import compute_all_metrics
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])
        metrics = compute_all_metrics(y_true, y_scores)
        assert metrics["auprc"] > 0.99
        assert metrics["auroc"] > 0.99

    def test_random_classifier(self):
        from training.metrics import compute_all_metrics
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_scores = np.random.uniform(0, 1, 1000)
        metrics = compute_all_metrics(y_true, y_scores)
        # Random classifier AUPRC ≈ fraud rate
        assert 0.3 < metrics["auprc"] < 0.7

    def test_metric_tracker_early_stopping(self):
        from training.metrics import MetricTracker
        tracker = MetricTracker(patience=3)

        # Improving
        assert tracker.update({"auprc": 0.5}, 1) is True
        assert tracker.update({"auprc": 0.6}, 2) is True

        # Stagnating — should trigger early stop after patience
        tracker.update({"auprc": 0.59}, 3)
        tracker.update({"auprc": 0.58}, 4)
        tracker.update({"auprc": 0.57}, 5)
        assert tracker.should_stop() is True
