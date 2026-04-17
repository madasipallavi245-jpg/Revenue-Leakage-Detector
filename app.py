"""
================================================================
  Revenue Leakage Detection — Streamlit App
  Model: Fixed VAE (leakage_vae_fixed.keras)
================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import tensorflow as tf
import keras
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Revenue Leakage Detector",
    page_icon="🔍",
    layout="wide",
)

# ── 1. CUSTOM CLASS DEFINITION (MUST BE TOP LEVEL) ───────────
# This ensures Keras can "see" the VAE class when loading the model
@keras.saving.register_keras_serializable()
class VAE(keras.Model): 
    def __init__(self, encoder, decoder, kl_weight=0.001, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(self.kl_weight * kl_loss)
        return reconstruction

# ── 2. DATA LOADING FUNCTIONS ────────────────────────────────
@st.cache_resource
def load_artefacts():
    """
    Load model, scaler, and config once and cache them.
    """
    BASE = os.path.dirname(__file__)

    # Load config
    with open(os.path.join(BASE, "leakage_vae_config.json"), "r") as f:
        config = json.load(f)

    # Load scaler
    scaler = joblib.load(os.path.join(BASE, "leakage_robust_scaler.pkl"))

    # Load VAE model
    vae = keras.models.load_model(
        os.path.join(BASE, "leakage_vae_fixed.keras"),
        compile=False
    )

    return vae, scaler, config

# ── 3. FEATURE ENGINEERING ───────────────────────────────────
def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Target derivation
    if "expected_total" in df.columns and "payment_value" in df.columns:
        df["leakage_amount"] = df["payment_value"] - df["expected_total"]
        df["has_leakage"]    = (df["leakage_amount"].abs() > 1.0).astype(int)

    # Payment features
    if "payment_value" in df.columns and "price" in df.columns:
        df["payment_residual"] = (
            df["payment_value"] - df["price"] - df["freight_value"]
        ).abs()

    if "payment_installments" in df.columns:
        safe_inst = df["payment_installments"].replace(0, 1)
        df["installment_unit_price"]    = df["payment_value"] / safe_inst
        df["price_per_installment_gap"] = (
            (df["price"] / safe_inst) - df["installment_unit_price"]
        ).abs()

    # Date features
    DATE_FMT  = "%d-%m-%Y %H.%M"
    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format=DATE_FMT, errors="coerce")

    if "order_approved_at" in df.columns and "order_purchase_timestamp" in df.columns:
        df["approval_delay_mins"] = (
            (df["order_approved_at"] - df["order_purchase_timestamp"])
            .dt.total_seconds() / 60
        )
        df["approval_delay_mins"] = df["approval_delay_mins"].fillna(df["approval_delay_mins"].median())

    if "order_delivered_customer_date" in df.columns:
        df["delivery_gap_days"] = (
            (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"])
            .dt.total_seconds() / 86400
        ).fillna(0)

    if "order_purchase_timestamp" in df.columns:
        df["purchase_hour"]    = df["order_purchase_timestamp"].dt.hour
        df["purchase_weekday"] = df["order_purchase_timestamp"].dt.dayofweek
        df["purchase_month"]   = df["order_purchase_timestamp"].dt.month

    df.drop(columns=[c for c in date_cols if c in df.columns], inplace=True)

    # Numeric features
    if "price" in df.columns and "total_items_count" in df.columns:
        df["price_per_item"]      = df["price"] / df["total_items_count"]
        df["expected_item_total"] = df["price"] * df["total_items_count"]

    if "freight_value" in df.columns and "price" in df.columns:
        df["freight_ratio"] = df["freight_value"] / (df["price"] + 0.01)

    for dim in ["product_length_cm", "product_height_cm", "product_width_cm", "product_weight_g"]:
        if dim in df.columns:
            df[dim] = df[dim].fillna(df[dim].median())

    if all(c in df.columns for c in ["product_length_cm", "product_height_cm", "product_width_cm"]):
        df["volume_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
        df.drop(columns=["product_length_cm", "product_height_cm", "product_width_cm"], inplace=True)

    # Drop ID / leaking columns
    drop_cols = ["order_id", "customer_id", "product_id", "customer_zip_code_prefix", "leakage_amount", "expected_total"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # OHE
    cat_cols = [c for c in ["order_status", "payment_type", "customer_state", "product_category_name_english"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

    return df

# ── 4. PREDICTION LOGIC ──────────────────────────────────────
def predict_leakage(df_features: pd.DataFrame, vae, scaler, config) -> pd.DataFrame:
    CLIP_VAL     = config["clip_val"]
    threshold    = config["vae_threshold"]
    use_focused  = config["use_focused_error"]
    pay_idx      = config["payment_col_indices"]

    if "has_leakage" in df_features.columns:
        df_features = df_features.drop(columns=["has_leakage"])

    X_scaled = scaler.transform(df_features)
    X_scaled = np.clip(X_scaled, -CLIP_VAL, CLIP_VAL)
    X_recon  = vae.predict(X_scaled, verbose=0)

    if use_focused and pay_idx:
        scores = np.mean(np.square(X_scaled[:, pay_idx] - X_recon[:, pay_idx]), axis=1)
    else:
        scores = np.mean(np.square(X_scaled - X_recon), axis=1)

    flags = (scores > threshold).astype(int)

    return pd.DataFrame({
        "risk_score":    np.round(scores, 6),
        "leakage_flag":  flags,
        "risk_level":    pd.cut(scores, bins=[-np.inf, threshold * 0.5, threshold, threshold * 2, np.inf],
                                labels=["Low", "Medium", "High", "Critical"])
    })

# ════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ════════════════════════════════════════════════════════════

st.title("🔍 Revenue Leakage Detection")

# Load artefacts
try:
    vae, scaler, config = load_artefacts()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# File upload
uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    st.write(f"Loaded {len(df_raw)} rows.")

    if st.button("Detect Leakage", type="primary"):
        df_features = engineer_features(df_raw)
        results     = predict_leakage(df_features, vae, scaler, config)
        
        df_out = pd.concat([df_raw.reset_index(drop=True), results], axis=1)
        st.subheader("Results")
        st.dataframe(df_out[df_out["leakage_flag"] == 1])

        # Download button
        csv_out = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv_out, "results.csv", "text/csv")