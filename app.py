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

# ── 1. VAE CLASS (Must be at the top level) ──────────────────
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

# ── 2. LOADING LOGIC ────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    # This looks exactly where app.py is sitting
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    config_path = os.path.join(base_path, "leakage_vae_config.json")
    scaler_path = os.path.join(base_path, "leakage_robust_scaler.pkl")
    model_path  = os.path.join(base_path, "leakage_vae_fixed.keras")

    # Load artifacts
    with open(config_path, "r") as f:
        config = json.load(f)
    scaler = joblib.load(scaler_path)
    vae = keras.models.load_model(model_path, compile=False)

    return vae, scaler, config

# ── 3. UI ───────────────────────────────────────────────────
st.title("🔍 Revenue Leakage Detector")

# DEBUGGER: If this fails, this list will tell us why
with st.expander("System File Check"):
    current_files = os.listdir(os.path.dirname(os.path.abspath(__file__)))
    st.write("Files Streamlit sees in GitHub:", current_files)

try:
    vae, scaler, config = load_artefacts()
    st.success("All models loaded!")
except Exception as e:
    st.error(f"File Error: {e}")
    st.stop()

# ... (Add your upload and prediction code here)