import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import tensorflow as tf
import keras
import warnings

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Revenue Leakage Detector",
    page_icon="🔍",
    layout="wide",
)

# ── 1. VAE CLASS DEFINITION (Must be Top Level) ──────────────
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

# ── 2. ROBUST DATA LOADING ──────────────────────────────────
@st.cache_resource
def load_artefacts():
    # Use absolute path to the directory containing app.py
    BASE = os.path.dirname(os.path.abspath(__file__))
    
    files = {
        "config": os.path.join(BASE, "leakage_vae_config.json"),
        "scaler": os.path.join(BASE, "leakage_robust_scaler.pkl"),
        "model":  os.path.join(BASE, "leakage_vae_fixed.keras")
    }

    # DEBUG: Check if files actually exist
    missing_files = [name for name, path in files.items() if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(f"Missing files in GitHub: {', '.join(missing_files)}")

    # Load config
    with open(files["config"], "r") as f:
        config = json.load(f)

    # Load scaler
    scaler = joblib.load(files["scaler"])

    # Load VAE model
    vae = keras.models.load_model(files["model"], compile=False)

    return vae, scaler, config

# ── 3. FEATURE ENGINEERING ───────────────────────────────────
def engineer_features(df_raw: pd.DataFrame):
    df = df_raw.copy()
    
    # Target derivation (for internal metrics)
    if "expected_total" in df.columns and "payment_value" in df.columns:
        df["has_leakage"] = (abs(df["payment_value"] - df["expected_total"]) > 1.0).astype(int)

    # Drop non-numeric and identifying columns
    drop_cols = ["order_id", "customer_id", "product_id", "order_status", 
                 "payment_type", "customer_state", "product_category_name_english"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    # Remove any remaining text/object columns
    df = df.select_dtypes(include=[np.number])
    
    df = df.fillna(0)
    return df

# ── 4. PREDICTION LOGIC ──────────────────────────────────────
def predict_leakage(df_features, vae, scaler, config):
    X_scaled = scaler.transform(df_features)
    X_scaled = np.clip(X_scaled, -config["clip_val"], config["clip_val"])
    
    X_recon = vae.predict(X_scaled, verbose=0)
    scores = np.mean(np.square(X_scaled - X_recon), axis=1)
    flags = (scores > config["vae_threshold"]).astype(int)

    return pd.DataFrame({
        "risk_score": np.round(scores, 6),
        "leakage_flag": flags,
        "risk_level": pd.cut(scores, bins=[-np.inf, config["vae_threshold"]*0.5, 
                                           config["vae_threshold"], config["vae_threshold"]*2, np.inf],
                             labels=["Low", "Medium", "High", "Critical"])
    })

# ════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ════════════════════════════════════════════════════════════

st.title("🔍 Revenue Leakage Detection")

try:
    vae, scaler, config = load_artefacts()
    st.success("Model and Configuration loaded successfully!")
except Exception as e:
    st.error(f"Setup Error: {e}")
    st.info("Ensure the .keras, .pkl, and .json files are uploaded to your GitHub repository.")
    st.stop()

uploaded = st.file_uploader("Upload your transactions CSV", type=["csv"])

if uploaded:
    df_raw = pd.read_csv(uploaded)
    if st.button("Start Detection"):
        with st.spinner("Analyzing..."):
            try:
                df_feats = engineer_features(df_raw)
                results = predict_leakage(df_feats, vae, scaler, config)
                df_final = pd.concat([df_raw, results], axis=1)
                
                st.subheader("Anomalous Transactions Detected")
                st.dataframe(df_final[df_final["leakage_flag"] == 1])
            except Exception as e:
                st.error(f"Processing Error: {e}")