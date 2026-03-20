"""
FraudShield — Synthetic Data Generator
=======================================
Generates realistic transaction data with multiple fraud scenarios.
Each scenario mimics real-world fraud patterns observed in production systems.
"""

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger
from tqdm import tqdm

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)


# ─────────────────────────────────────────────
# Entity Profiles
# ─────────────────────────────────────────────

@dataclass
class UserProfile:
    user_id: str
    name: str
    age: int
    country: str
    city: str
    lat: float
    lon: float
    credit_limit: float
    avg_txn_amount: float
    std_txn_amount: float
    preferred_categories: list
    typical_hours: list        # hours of day they usually transact
    devices: list
    ips: list
    is_compromised: bool = False  # flag for account takeover scenario


@dataclass
class MerchantProfile:
    merchant_id: str
    name: str
    category: str
    country: str
    lat: float
    lon: float
    avg_ticket: float
    is_colluding: bool = False   # flag for merchant collusion scenario


MERCHANT_CATEGORIES = [
    "grocery", "gas_station", "restaurant", "online_retail",
    "electronics", "travel", "pharmacy", "entertainment",
    "clothing", "luxury_goods", "cryptocurrency", "money_transfer"
]

HIGH_RISK_CATEGORIES = {"cryptocurrency", "money_transfer", "luxury_goods"}

CATEGORY_AVG_AMOUNTS = {
    "grocery": (45, 30),
    "gas_station": (55, 20),
    "restaurant": (35, 25),
    "online_retail": (80, 60),
    "electronics": (250, 200),
    "travel": (400, 350),
    "pharmacy": (30, 20),
    "entertainment": (25, 15),
    "clothing": (75, 50),
    "luxury_goods": (800, 600),
    "cryptocurrency": (500, 400),
    "money_transfer": (300, 250),
}


# ─────────────────────────────────────────────
# Profile Generators
# ─────────────────────────────────────────────

def generate_users(n: int) -> list[UserProfile]:
    logger.info(f"Generating {n} user profiles...")
    users = []
    for _ in range(n):
        lat = round(random.uniform(-60, 70), 4)
        lon = round(random.uniform(-180, 180), 4)
        # Spending power distribution (log-normal)
        credit_limit = round(np.random.lognormal(8.5, 0.8), 2)
        avg_amt = round(credit_limit * random.uniform(0.005, 0.03), 2)

        # Each user has 1-3 devices and 1-4 IP addresses
        n_devices = random.randint(1, 3)
        n_ips = random.randint(1, 4)

        users.append(UserProfile(
            user_id=str(uuid.uuid4())[:12],
            name=fake.name(),
            age=random.randint(18, 80),
            country=fake.country_code(),
            city=fake.city(),
            lat=lat,
            lon=lon,
            credit_limit=credit_limit,
            avg_txn_amount=avg_amt,
            std_txn_amount=avg_amt * 0.5,
            preferred_categories=random.sample(MERCHANT_CATEGORIES, k=random.randint(3, 7)),
            typical_hours=random.sample(range(6, 23), k=random.randint(4, 8)),
            devices=[f"dev_{uuid.uuid4().hex[:8]}" for _ in range(n_devices)],
            ips=[f"{random.randint(1,254)}.{random.randint(0,254)}.{random.randint(0,254)}.{random.randint(1,254)}"
                 for _ in range(n_ips)],
        ))
    return users


def generate_merchants(n: int) -> list[MerchantProfile]:
    logger.info(f"Generating {n} merchant profiles...")
    merchants = []
    for _ in range(n):
        category = random.choice(MERCHANT_CATEGORIES)
        avg_mu, avg_sigma = CATEGORY_AVG_AMOUNTS[category]
        lat = round(random.uniform(-60, 70), 4)
        lon = round(random.uniform(-180, 180), 4)
        merchants.append(MerchantProfile(
            merchant_id=str(uuid.uuid4())[:10],
            name=fake.company(),
            category=category,
            country=fake.country_code(),
            lat=lat,
            lon=lon,
            avg_ticket=round(abs(np.random.normal(avg_mu, avg_sigma)), 2),
        ))
    return merchants


# ─────────────────────────────────────────────
# Normal Transaction Generator
# ─────────────────────────────────────────────

def generate_normal_transaction(
    user: UserProfile,
    merchant: MerchantProfile,
    timestamp: datetime,
) -> dict:
    """Generate a single realistic normal transaction."""
    amount = max(0.5, np.random.normal(user.avg_txn_amount, user.std_txn_amount))
    amount = min(amount, user.credit_limit * 0.3)  # cap at 30% of credit limit

    device = random.choice(user.devices)
    ip = random.choice(user.ips)

    return {
        "txn_id": str(uuid.uuid4())[:16],
        "timestamp": timestamp.isoformat(),
        "user_id": user.user_id,
        "merchant_id": merchant.merchant_id,
        "amount": round(amount, 2),
        "currency": "USD",
        "merchant_category": merchant.category,
        "merchant_country": merchant.country,
        "merchant_lat": merchant.lat,
        "merchant_lon": merchant.lon,
        "user_lat": user.lat,
        "user_lon": user.lon,
        "device_id": device,
        "ip_address": ip,
        "user_age": user.age,
        "credit_limit": user.credit_limit,
        "is_fraud": 0,
        "fraud_scenario": "none",
    }


# ─────────────────────────────────────────────
# Fraud Scenario Generators
# ─────────────────────────────────────────────

def fraud_card_testing(
    user: UserProfile,
    merchants: list[MerchantProfile],
    start_time: datetime,
) -> list[dict]:
    """
    Card testing: fraudster verifies stolen card with many small rapid transactions
    before making large purchases. Signature: high velocity, small amounts, multiple merchants.
    """
    txns = []
    n_test = random.randint(5, 20)
    foreign_ip = f"185.{random.randint(1,254)}.{random.randint(0,254)}.{random.randint(1,254)}"
    foreign_device = f"dev_{uuid.uuid4().hex[:8]}"

    for i in range(n_test):
        merchant = random.choice(merchants)
        timestamp = start_time + timedelta(seconds=random.randint(10, 120) * i)
        amount = round(random.uniform(0.5, 5.0), 2)  # micro-transactions

        txn = generate_normal_transaction(user, merchant, timestamp)
        txn.update({
            "txn_id": str(uuid.uuid4())[:16],
            "amount": amount,
            "ip_address": foreign_ip,
            "device_id": foreign_device,
            "is_fraud": 1,
            "fraud_scenario": "card_testing",
        })
        txns.append(txn)

    # Then a large transaction after testing
    big_merchant = random.choice([m for m in merchants if m.category in HIGH_RISK_CATEGORIES])
    big_txn = generate_normal_transaction(user, big_merchant, timestamp + timedelta(minutes=5))
    big_txn.update({
        "amount": round(random.uniform(500, user.credit_limit * 0.8), 2),
        "ip_address": foreign_ip,
        "device_id": foreign_device,
        "is_fraud": 1,
        "fraud_scenario": "card_testing",
    })
    txns.append(big_txn)
    return txns


def fraud_account_takeover(
    user: UserProfile,
    merchants: list[MerchantProfile],
    start_time: datetime,
) -> list[dict]:
    """
    Account takeover: attacker logs in from new location/device, makes large purchases.
    Signature: geo velocity anomaly, new device, change in spending pattern.
    """
    txns = []
    # Completely different location
    fraud_lat = round(random.uniform(-60, 70), 4)
    fraud_lon = round(random.uniform(-180, 180), 4)
    fraud_device = f"dev_{uuid.uuid4().hex[:8]}"
    fraud_ip = f"77.{random.randint(1,254)}.{random.randint(0,254)}.{random.randint(1,254)}"

    n_txns = random.randint(3, 8)
    for i in range(n_txns):
        merchant = random.choice(merchants)
        timestamp = start_time + timedelta(minutes=random.randint(2, 15) * i)
        amount = round(random.uniform(user.avg_txn_amount * 3, user.credit_limit * 0.5), 2)

        txn = generate_normal_transaction(user, merchant, timestamp)
        txn.update({
            "txn_id": str(uuid.uuid4())[:16],
            "amount": amount,
            "user_lat": fraud_lat,
            "user_lon": fraud_lon,
            "device_id": fraud_device,
            "ip_address": fraud_ip,
            "merchant_category": random.choice(list(HIGH_RISK_CATEGORIES)),
            "is_fraud": 1,
            "fraud_scenario": "account_takeover",
        })
        txns.append(txn)
    return txns


def fraud_bust_out(
    user: UserProfile,
    merchants: list[MerchantProfile],
    start_time: datetime,
) -> list[dict]:
    """
    Bust-out fraud: builds credit history, then maxes out all credit rapidly and disappears.
    Signature: sudden spike in spending velocity and amounts, near credit limit.
    """
    txns = []
    n_txns = random.randint(10, 25)
    cumulative = 0

    for i in range(n_txns):
        if cumulative >= user.credit_limit * 0.95:
            break
        merchant = random.choice(merchants)
        timestamp = start_time + timedelta(hours=random.uniform(0.5, 3) * i)
        remaining = user.credit_limit * 0.95 - cumulative
        amount = round(min(random.uniform(100, 800), remaining), 2)
        cumulative += amount

        txn = generate_normal_transaction(user, merchant, timestamp)
        txn.update({
            "txn_id": str(uuid.uuid4())[:16],
            "amount": amount,
            "is_fraud": 1,
            "fraud_scenario": "bust_out",
        })
        txns.append(txn)
    return txns


def fraud_identity_theft(
    user: UserProfile,
    merchants: list[MerchantProfile],
    start_time: datetime,
) -> list[dict]:
    """
    Identity theft on new accounts: large purchases immediately after account creation.
    Signature: new account + immediate high-value purchases in luxury/crypto.
    """
    txns = []
    fraud_device = f"dev_{uuid.uuid4().hex[:8]}"
    fraud_ip = f"91.{random.randint(1,254)}.{random.randint(0,254)}.{random.randint(1,254)}"

    high_risk_merchants = [m for m in merchants if m.category in HIGH_RISK_CATEGORIES]
    if not high_risk_merchants:
        high_risk_merchants = merchants

    n_txns = random.randint(2, 5)
    for i in range(n_txns):
        merchant = random.choice(high_risk_merchants)
        timestamp = start_time + timedelta(minutes=random.randint(1, 30) * i)
        amount = round(random.uniform(300, user.credit_limit * 0.6), 2)

        txn = generate_normal_transaction(user, merchant, timestamp)
        txn.update({
            "txn_id": str(uuid.uuid4())[:16],
            "amount": amount,
            "device_id": fraud_device,
            "ip_address": fraud_ip,
            "merchant_category": merchant.category,
            "is_fraud": 1,
            "fraud_scenario": "identity_theft",
        })
        txns.append(txn)
    return txns


def fraud_merchant_collusion(
    user: UserProfile,
    merchant: MerchantProfile,
    start_time: datetime,
) -> list[dict]:
    """
    Merchant collusion: merchant and cardholder collude to generate fake transactions
    for cashback or chargebacks. Signature: round amounts, same merchant, periodic timing.
    """
    txns = []
    n_txns = random.randint(3, 10)
    # Suspiciously round amounts
    round_amounts = [round(random.choice([50, 100, 200, 250, 500]) * random.uniform(0.9, 1.1), 2)
                     for _ in range(n_txns)]

    for i, amount in enumerate(round_amounts):
        timestamp = start_time + timedelta(days=i * random.randint(1, 3))
        txn = generate_normal_transaction(user, merchant, timestamp)
        txn.update({
            "txn_id": str(uuid.uuid4())[:16],
            "amount": amount,
            "merchant_id": merchant.merchant_id,
            "is_fraud": 1,
            "fraud_scenario": "merchant_collusion",
        })
        txns.append(txn)
    return txns


# ─────────────────────────────────────────────
# Main Generator
# ─────────────────────────────────────────────

SCENARIO_GENERATORS = {
    "card_testing": fraud_card_testing,
    "account_takeover": fraud_account_takeover,
    "bust_out": fraud_bust_out,
    "identity_theft": fraud_identity_theft,
    "merchant_collusion": fraud_merchant_collusion,
}


def generate_dataset(
    n_users: int = 5000,
    n_merchants: int = 500,
    n_transactions: int = 500_000,
    fraud_rate: float = 0.015,
    fraud_scenario_weights: Optional[dict] = None,
    output_path: str = "data/raw/transactions.parquet",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a full synthetic fraud dataset.

    Args:
        n_users: Number of unique user profiles
        n_merchants: Number of unique merchant profiles
        n_transactions: Approximate total transactions (including fraud bursts)
        fraud_rate: Target fraction of fraudulent transactions
        fraud_scenario_weights: Dict mapping scenario name to weight
        output_path: Where to save the parquet file
        seed: Random seed for reproducibility

    Returns:
        DataFrame with all transactions
    """
    np.random.seed(seed)
    random.seed(seed)

    if fraud_scenario_weights is None:
        fraud_scenario_weights = {
            "card_testing": 0.30,
            "account_takeover": 0.25,
            "bust_out": 0.20,
            "identity_theft": 0.15,
            "merchant_collusion": 0.10,
        }

    users = generate_users(n_users)
    merchants = generate_merchants(n_merchants)

    # Date range: 1 year of data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    total_seconds = int((end_date - start_date).total_seconds())

    all_transactions = []

    # ── Normal transactions ──
    n_normal = int(n_transactions * (1 - fraud_rate))
    logger.info(f"Generating {n_normal:,} normal transactions...")

    for _ in tqdm(range(n_normal), desc="Normal txns"):
        user = random.choice(users)
        # Prefer merchants in their preferred categories
        preferred = [m for m in merchants if m.category in user.preferred_categories]
        merchant = random.choice(preferred if preferred else merchants)

        # Realistic timing: prefer user's typical hours
        rand_hour = random.choice(user.typical_hours) if random.random() < 0.8 else random.randint(0, 23)
        rand_day_offset = random.randint(0, 364)
        timestamp = start_date + timedelta(days=rand_day_offset, hours=rand_hour,
                                           minutes=random.randint(0, 59))

        txn = generate_normal_transaction(user, merchant, timestamp)
        all_transactions.append(txn)

    # ── Fraud transactions ──
    n_fraud_events = int(n_transactions * fraud_rate / 5)  # each event generates ~5 txns
    scenario_names = list(fraud_scenario_weights.keys())
    scenario_probs = list(fraud_scenario_weights.values())

    logger.info(f"Generating ~{n_fraud_events * 5:,} fraud transactions across {n_fraud_events} events...")

    for _ in tqdm(range(n_fraud_events), desc="Fraud events"):
        user = random.choice(users)
        merchant = random.choice(merchants)
        scenario = random.choices(scenario_names, weights=scenario_probs, k=1)[0]
        fraud_time = start_date + timedelta(seconds=random.randint(0, total_seconds))

        generator_fn = SCENARIO_GENERATORS[scenario]
        if scenario == "merchant_collusion":
            fraud_txns = generator_fn(user, merchant, fraud_time)
        else:
            fraud_txns = generator_fn(user, merchants, fraud_time)

        all_transactions.extend(fraud_txns)

    # ── Assemble DataFrame ──
    logger.info("Assembling DataFrame...")
    df = pd.DataFrame(all_transactions)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values("timestamp").reset_index(drop=True)

    fraud_count = df["is_fraud"].sum()
    actual_rate = fraud_count / len(df)
    logger.info(f"Dataset: {len(df):,} transactions | {fraud_count:,} fraud ({actual_rate:.2%})")
    logger.info(f"Scenario breakdown:\n{df[df.is_fraud==1]['fraud_scenario'].value_counts()}")

    # Save
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.success(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    df = generate_dataset(
        n_users=5000,
        n_merchants=500,
        n_transactions=500_000,
        fraud_rate=0.015,
        output_path="data/raw/transactions.parquet",
    )
    print(df.head())
    print(df.dtypes)