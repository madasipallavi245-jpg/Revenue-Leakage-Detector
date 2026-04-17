import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import tensorflow as tf
import keras

# ── 1. VAE CLASS ─────────────────────────────────────────────
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

# ── 2. LOADING WITH DEBUGGING ────────────────────────────────
@st.cache_resource
def load_artefacts():
    # Force the path to the current working directory
    base_path = os.getcwd()
    
    config_fn = "leakage_vae_config.json"
    scaler_fn = "leakage_robust_scaler.pkl"
    model_fn  = "leakage_vae_fixed.keras"

    # Load artifacts
    with open(os.path.join(base_path, config_fn), "r") as f:
        config = json.load(f)
    scaler = joblib.load(os.path.join(base_path, scaler_fn))
    vae = keras.models.load_model(os.path.join(base_path, model_fn), compile=False)

    return vae, scaler, config

# ── 3. UI ────────────────────────────────────────────────────
st.title("🔍 Revenue Leakage Detector")

# DEBUGGER: Show what files Streamlit actually sees
with st.expander("Folder Debugger (Check if files exist)"):
    st.write("Current Directory:", os.getcwd())
    st.write("Files found in folder:", os.listdir(os.getcwd()))

try:
    vae, scaler, config = load_artefacts()
    st.success("Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# ... (Add your file_uploader and predict code here) ...