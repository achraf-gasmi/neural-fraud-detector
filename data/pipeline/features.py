"""
FraudShield — Feature Engineering Pipeline
===========================================
Transforms raw transactions into rich feature vectors for the model.
Includes rolling aggregations, geo-velocity, and entity-level statistics.
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os


def haversine_distance(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_rolling_features(df: pd.DataFrame, windows_hours: list) -> pd.DataFrame:
    """
    Rolling window features per user.

    Root cause of previous errors:
    - Rolling .apply() on string columns → TypeError (needs numeric)
    - pd.concat / DataFrame(dict, index=df.index) with duplicate timestamps → ValueError

    Fix: encode strings to int codes, use a separate DatetimeIndex df for rolling,
    extract .values (numpy) from every result, assign back positionally to the
    RangeIndex main df. No index alignment ever happens.
    """
    logger.info(f"Computing rolling features for windows: {windows_hours}h ...")

    # Clean RangeIndex — guarantees no duplicate index problems downstream
    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    # Integer-encode entity string IDs (rolling requires numeric dtype)
    df["_merchant_int"] = df["merchant_id"].astype("category").cat.codes.astype(float)
    df["_device_int"]   = df["device_id"].astype("category").cat.codes.astype(float)
    df["_ip_int"]       = df["ip_address"].astype("category").cat.codes.astype(float)

    # Separate df with DatetimeIndex used only for rolling computation
    df_ts = df[["user_id", "amount", "_merchant_int", "_device_int", "_ip_int"]].copy()
    df_ts.index = pd.DatetimeIndex(df["timestamp"])
    df_ts.index.name = None

    for w in windows_hours:
        window = f"{w}h"
        logger.info(f"  Rolling {window} ...")
        grp = df_ts.groupby("user_id")

        def extract(col, func=None, apply_fn=None):
            r = grp[col].rolling(window, closed="left")
            r = getattr(r, func)() if func else r.apply(apply_fn, raw=True)
            return r.reset_index(level=0, drop=True).sort_index().values

        df[f"txn_count_{w}h"]          = extract("amount", func="count")
        df[f"txn_sum_{w}h"]            = extract("amount", func="sum")
        df[f"txn_max_{w}h"]            = extract("amount", func="max")
        df[f"txn_std_{w}h"]            = extract("amount", func="std")
        df[f"unique_merchants_{w}h"]   = extract("_merchant_int", apply_fn=lambda x: float(len(set(x))))
        df[f"unique_devices_{w}h"]     = extract("_device_int",   apply_fn=lambda x: float(len(set(x))))
        df[f"unique_ips_{w}h"]         = extract("_ip_int",       apply_fn=lambda x: float(len(set(x))))

    df = df.drop(columns=["_merchant_int", "_device_int", "_ip_int"], errors="ignore")
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing geo-velocity features...")
    df = df.sort_values(["user_id", "timestamp"]).copy()

    df["prev_merchant_lat"] = df.groupby("user_id")["merchant_lat"].shift(1)
    df["prev_merchant_lon"] = df.groupby("user_id")["merchant_lon"].shift(1)
    df["prev_timestamp"]    = df.groupby("user_id")["timestamp"].shift(1)

    time_diff_hours = (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds() / 3600
    time_diff_hours = time_diff_hours.replace(0, 1e-6)

    geo_dist = haversine_distance(
        df["merchant_lat"].values, df["merchant_lon"].values,
        df["prev_merchant_lat"].fillna(df["merchant_lat"]).values,
        df["prev_merchant_lon"].fillna(df["merchant_lon"]).values,
    )

    df["geo_velocity_kmh"]          = geo_dist / time_diff_hours
    df["geo_distance_km"]           = geo_dist
    df["time_since_last_txn_hours"] = time_diff_hours
    df["user_merchant_distance_km"] = haversine_distance(
        df["user_lat"].values, df["user_lon"].values,
        df["merchant_lat"].values, df["merchant_lon"].values,
    )
    df = df.drop(columns=["prev_merchant_lat", "prev_merchant_lon", "prev_timestamp"])
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding temporal features...")
    df = df.copy()
    ts = df["timestamp"]
    df["hour"]        = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_month"]= ts.dt.day
    df["month"]       = ts.dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]    = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding amount features...")
    df = df.copy()
    df["amount_to_credit_ratio"] = df["amount"] / df["credit_limit"].clip(lower=1)
    df["log_amount"]             = np.log1p(df["amount"])
    user_stats = df.groupby("user_id")["amount"].agg(["mean", "std"]).rename(
        columns={"mean": "user_mean_amount", "std": "user_std_amount"}
    )
    df = df.join(user_stats, on="user_id")
    df["amount_zscore"] = (df["amount"] - df["user_mean_amount"]) / df["user_std_amount"].clip(lower=1e-6)
    df = df.drop(columns=["user_mean_amount", "user_std_amount"])
    df["is_round_amount"] = ((df["amount"] % 50 == 0) | (df["amount"] % 100 == 0)).astype(int)
    return df


def encode_categoricals(df, fit=True, encoders=None):
    logger.info("Encoding categoricals...")
    cat_cols = ["merchant_category", "merchant_country", "currency"]
    df = df.copy()
    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in cat_cols:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
            )
    return df, encoders


FEATURE_COLUMNS = [
    "amount", "log_amount", "amount_to_credit_ratio", "amount_zscore", "is_round_amount",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_night",
    "geo_velocity_kmh", "geo_distance_km", "time_since_last_txn_hours", "user_merchant_distance_km",
    "user_age", "credit_limit",
    "merchant_category", "merchant_country",
    "txn_count_1h", "txn_sum_1h", "txn_max_1h", "txn_std_1h",
    "unique_merchants_1h", "unique_devices_1h", "unique_ips_1h",
    "txn_count_6h", "txn_sum_6h", "txn_max_6h", "txn_std_6h",
    "unique_merchants_6h", "unique_devices_6h", "unique_ips_6h",
    "txn_count_24h", "txn_sum_24h", "txn_max_24h", "txn_std_24h",
    "unique_merchants_24h", "unique_devices_24h", "unique_ips_24h",
    "txn_count_168h", "txn_sum_168h", "txn_max_168h", "txn_std_168h",
    "unique_merchants_168h", "unique_devices_168h", "unique_ips_168h",
]


def run_pipeline(
    raw_path="data/raw/transactions.parquet",
    output_dir="data/processed",
    artifacts_dir="data/processed/artifacts",
    windows_hours=None,
    fit=True,
):
    if windows_hours is None:
        windows_hours = [1, 6, 24, 168]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    logger.info(f"Loading raw data from {raw_path}...")
    df = pd.read_parquet(raw_path)
    logger.info(f"Loaded {len(df):,} rows")

    df = add_temporal_features(df)
    df = add_amount_features(df)
    df = add_rolling_features(df, windows_hours)
    df = add_velocity_features(df)
    df, encoders = encode_categoricals(df, fit=fit)

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)

    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    scaler = StandardScaler()
    if fit:
        train_df[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    val_df[FEATURE_COLUMNS]  = scaler.transform(val_df[FEATURE_COLUMNS])
    test_df[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])

    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    logger.info(f"Train fraud rate: {train_df['is_fraud'].mean():.3%}")

    train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
    val_df.to_parquet(f"{output_dir}/val.parquet", index=False)
    test_df.to_parquet(f"{output_dir}/test.parquet", index=False)

    with open(f"{artifacts_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{artifacts_dir}/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    logger.success("Pipeline complete. Artifacts saved.")
    return train_df, val_df, test_df


if __name__ == "__main__":
    train, val, test = run_pipeline()
    print(f"n_features: {len(FEATURE_COLUMNS)}")