"""
FraudShield — Monitoring Dashboard
====================================
Real-time monitoring of fraud detection model:
- Live transaction scoring demo
- Model performance metrics
- Score distribution & drift detection
- Fraud scenario breakdown
"""

import time
import random
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta

# ─── Page Config ───
st.set_page_config(
    page_title="FraudShield Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# ─── Custom CSS ───
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .fraud-alert {
        background: linear-gradient(135deg, #3d1515, #5c1f1f);
        border: 1px solid #ff4444;
        border-radius: 8px;
        padding: 12px;
    }
    .safe-alert {
        background: linear-gradient(135deg, #0d3d1e, #1a5c30);
        border: 1px solid #00cc66;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1e2130/6c7ae0?text=FraudShield", width=200)
    st.markdown("---")

    st.subheader("⚙️ Settings")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
    n_demo_txns = st.slider("Demo batch size", 10, 200, 50)

    st.markdown("---")
    st.subheader("📊 API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success("🟢 API Online")
        st.metric("Requests served", health.get("request_count", 0))
        st.metric("Avg latency", f"{health.get('avg_latency_ms', 0):.1f} ms")
    except Exception:
        st.error("🔴 API Offline")
        st.info("Start the API with:\n```\npython -m api.main\n```")


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.title("🛡️ FraudShield — Real-Time Monitoring")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live Scoring", "📈 Model Performance", "📊 Analytics", "⚠️ Alerts"
])


# ─────────────────────────────────────────────
# Tab 1: Live Scoring
# ─────────────────────────────────────────────

with tab1:
    st.subheader("Score a Transaction")

    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Amount ($)", min_value=0.01, value=1250.0)
        merchant_category = st.selectbox("Merchant Category", [
            "grocery", "gas_station", "restaurant", "online_retail",
            "electronics", "travel", "pharmacy", "entertainment",
            "clothing", "luxury_goods", "cryptocurrency", "money_transfer"
        ])
        user_age = st.number_input("User Age", min_value=18, max_value=100, value=35)

    with col2:
        credit_limit = st.number_input("Credit Limit ($)", min_value=100.0, value=5000.0)
        merchant_country = st.selectbox("Merchant Country", ["US", "GB", "FR", "CN", "RU", "NG", "BR"])
        device_id = st.text_input("Device ID", value="dev_abc123")

    with col3:
        ip_address = st.text_input("IP Address", value="192.168.1.1")
        merchant_lat = st.number_input("Merchant Lat", value=40.71)
        merchant_lon = st.number_input("Merchant Lon", value=-74.00)

    if st.button("🔍 Score Transaction", use_container_width=True, type="primary"):
        payload = {
            "txn_id": f"txn_{random.randint(10000, 99999)}",
            "timestamp": datetime.now().isoformat(),
            "user_id": "user_demo_001",
            "merchant_id": "merch_demo_001",
            "amount": amount,
            "currency": "USD",
            "merchant_category": merchant_category,
            "merchant_country": merchant_country,
            "merchant_lat": merchant_lat,
            "merchant_lon": merchant_lon,
            "user_lat": 40.75,
            "user_lon": -73.98,
            "device_id": device_id,
            "ip_address": ip_address,
            "user_age": user_age,
            "credit_limit": credit_limit,
        }

        try:
            with st.spinner("Scoring..."):
                resp = requests.post(f"{API_URL}/score", json=payload, timeout=10)
                result = resp.json()

            prob = result["fraud_probability"]
            risk = result["risk_level"]

            # Result display
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("Fraud Probability", f"{prob:.1%}")
            col_r2.metric("Risk Level", risk)
            col_r3.metric("Decision", "🚨 FRAUD" if result["is_fraud"] else "✅ LEGIT")
            col_r4.metric("Latency", f"{result['latency_ms']:.1f} ms")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Fraud Score (%)"},
                delta={"reference": threshold * 100},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ff4444" if prob >= threshold else "#00cc66"},
                    "steps": [
                        {"range": [0, 30], "color": "#0d3d1e"},
                        {"range": [30, 50], "color": "#3d3d0d"},
                        {"range": [50, 80], "color": "#3d1515"},
                        {"range": [80, 100], "color": "#5c0000"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": threshold * 100,
                    },
                },
            ))
            fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

            # Top risk factors
            st.subheader("🔍 Top Risk Factors")
            factors_df = pd.DataFrame(result["top_risk_factors"])
            fig2 = px.bar(
                factors_df, x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale="Reds",
                title="Feature Importance (Gradient Sensitivity)"
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=300)
            st.plotly_chart(fig2, use_container_width=True)

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API. Make sure it's running on port 8000.")
        except Exception as e:
            st.error(f"Error: {e}")


# ─────────────────────────────────────────────
# Tab 2: Model Performance
# ─────────────────────────────────────────────

with tab2:
    st.subheader("Model Performance Metrics")

    # Simulated metrics (replace with real MLflow data in production)
    @st.cache_data(ttl=60)
    def get_simulated_history():
        epochs = list(range(1, 31))
        return pd.DataFrame({
            "epoch": epochs,
            "train_auprc": [0.4 + 0.45 * (1 - np.exp(-e/8)) + np.random.normal(0, 0.01) for e in epochs],
            "val_auprc": [0.35 + 0.42 * (1 - np.exp(-e/10)) + np.random.normal(0, 0.015) for e in epochs],
            "train_loss": [0.8 * np.exp(-e/12) + 0.05 + np.random.normal(0, 0.005) for e in epochs],
            "val_loss": [0.85 * np.exp(-e/14) + 0.06 + np.random.normal(0, 0.008) for e in epochs],
            "val_f1": [0.3 + 0.5 * (1 - np.exp(-e/9)) + np.random.normal(0, 0.02) for e in epochs],
        })

    history = get_simulated_history()

    fig = make_subplots(rows=1, cols=2, subplot_titles=["AUPRC (Primary Metric)", "Loss"])
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["train_auprc"], name="Train AUPRC",
                             line=dict(color="#6c7ae0")), row=1, col=1)
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["val_auprc"], name="Val AUPRC",
                             line=dict(color="#00cc66")), row=1, col=1)
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["train_loss"], name="Train Loss",
                             line=dict(color="#ff8c00")), row=1, col=2)
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["val_loss"], name="Val Loss",
                             line=dict(color="#ff4444")), row=1, col=2)

    fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # Simulated PR curve
    st.subheader("Precision-Recall Curve")
    recalls = np.linspace(0, 1, 100)
    precisions = 0.9 * np.exp(-2 * recalls) + 0.1
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recalls, y=precisions, fill="tozeroy",
                                line=dict(color="#6c7ae0"), name=f"AUPRC ≈ 0.77"))
    fig_pr.add_vline(x=0.8, line_dash="dash", line_color="white", annotation_text="80% Recall")
    fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=350,
                         paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                         plot_bgcolor="rgba(15,17,23,1)")
    st.plotly_chart(fig_pr, use_container_width=True)


# ─────────────────────────────────────────────
# Tab 3: Analytics
# ─────────────────────────────────────────────

with tab3:
    st.subheader("Transaction Analytics")

    # Simulated score distribution
    np.random.seed(42)
    legit_scores = np.random.beta(1.5, 8, 10000)
    fraud_scores = np.random.beta(5, 2, 150)
    all_scores = np.concatenate([legit_scores, fraud_scores])
    labels = ["Legitimate"] * 10000 + ["Fraud"] * 150

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=legit_scores, name="Legitimate",
                                    marker_color="#00cc66", opacity=0.7, nbinsx=50))
    fig_dist.add_trace(go.Histogram(x=fraud_scores, name="Fraud",
                                    marker_color="#ff4444", opacity=0.7, nbinsx=50))
    fig_dist.add_vline(x=threshold, line_dash="dash", line_color="white",
                       annotation_text=f"Threshold: {threshold:.2f}")
    fig_dist.update_layout(barmode="overlay", title="Fraud Score Distribution",
                           height=350, paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                           plot_bgcolor="rgba(15,17,23,1)")
    st.plotly_chart(fig_dist, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Fraud by scenario
        scenarios = ["Card Testing", "Account Takeover", "Bust-Out", "Identity Theft", "Merchant Collusion"]
        counts = [30, 25, 20, 15, 10]
        fig_pie = px.pie(values=counts, names=scenarios, title="Fraud by Scenario",
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Fraud over time
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        daily_fraud = np.random.poisson(5, 30)
        fig_time = px.line(x=dates, y=daily_fraud, title="Daily Fraud Events",
                           labels={"x": "Date", "y": "Count"})
        fig_time.update_traces(line_color="#ff4444")
        fig_time.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                               plot_bgcolor="rgba(15,17,23,1)", height=350)
        st.plotly_chart(fig_time, use_container_width=True)


# ─────────────────────────────────────────────
# Tab 4: Alerts
# ─────────────────────────────────────────────

with tab4:
    st.subheader("⚠️ Recent High-Risk Transactions")

    # Simulated alerts
    alerts = pd.DataFrame({
        "Time": [datetime.now() - timedelta(minutes=i*7) for i in range(10)],
        "TXN ID": [f"txn_{random.randint(100000, 999999)}" for _ in range(10)],
        "Amount ($)": [round(random.uniform(500, 5000), 2) for _ in range(10)],
        "Risk Score": [round(random.uniform(0.7, 0.99), 3) for _ in range(10)],
        "Scenario": random.choices(
            ["Card Testing", "Account Takeover", "Bust-Out", "Identity Theft"], k=10
        ),
        "Status": random.choices(["🚨 Blocked", "⚠️ Review", "🚨 Blocked"], k=10),
    })
    alerts = alerts.sort_values("Risk Score", ascending=False)
    st.dataframe(alerts, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.info("🔧 In production: alerts are fed from Kafka consumer → PostgreSQL → this dashboard via WebSocket.")


# ─── Auto-refresh ───
if auto_refresh:
    time.sleep(5)
    st.rerun()
