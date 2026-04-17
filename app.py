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

# ── 1. VAE CLASS DEFINITION ──────────────────────────────────
# This must be at the top level for Keras to load the model
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
    # Use absolute paths to find files in the same directory as app.py
    BASE = os.path.dirname(os.path.abspath(__file__))
    
    config_path = os.path.join(BASE, "leakage_vae_config.json")
    scaler_path = os.path.join(BASE, "leakage_robust_scaler.pkl")
    model_path  = os.path.join(BASE, "leakage_vae_fixed.keras")

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load VAE model
    vae = keras.models.load_model(model_path, compile=False)

    return vae, scaler, config

# ── 3. FEATURE ENGINEERING ───────────────────────────────────
def engineer_features(df_raw: pd.DataFrame):
    df = df_raw.copy()
    
    # Feature engineering logic from your notebook
    if "expected_total" in df.columns and "payment_value" in df.columns:
        df["has_leakage"] = (abs(df["payment_value"] - df["expected_total"]) > 1.0).astype(int)

    # Drop non-numeric/ID columns for prediction
    drop_cols = ["order_id", "customer_id", "product_id", "order_status", 
                 "payment_type", "customer_state", "product_category_name_english"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    # Convert date strings to numbers or drop them
    for col in df.select_dtypes(include=['object']).columns:
        df.drop(columns=[col], inplace=True)

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
    st.success("All systems go! Model and artefacts loaded.")
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

uploaded = st.file_uploader("Upload your transaction CSV", type=["csv"])

if uploaded:
    df_raw = pd.read_csv(uploaded)
    if st.button("Analyze Data"):
        with st.spinner("Analyzing..."):
            df_feats = engineer_features(df_raw)
            results = predict_leakage(df_feats, vae, scaler, config)
            df_final = pd.concat([df_raw, results], axis=1)
            
            st.subheader("Results Overview")
            st.write(f"Flagged {results['leakage_flag'].sum()} suspicious transactions.")
            st.dataframe(df_final[df_final["leakage_flag"] == 1])