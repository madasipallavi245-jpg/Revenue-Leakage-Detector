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
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Revenue Leakage Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Load model + artefacts ───────────────────────────────────
@st.cache_resource
def load_model():
    """
    Load once and cache.
    @st.cache_resource means this runs only on first app load,
    not on every user interaction — keeps the app fast.
    """
    import tensorflow as tf
    from tensorflow import keras

    # ── Define VAE class ──────────────────────────────────────
    @keras.saving.register_keras_serializable(package="", name="VAE")
    class VAE(keras.Model):
        def __init__(self, input_dim=129, latent_dim=16, **kwargs):
            super().__init__(**kwargs)
            self.input_dim  = input_dim
            self.latent_dim = latent_dim

            # Encoder
            self.enc1        = keras.layers.Dense(64, activation="relu")
            self.enc2        = keras.layers.Dense(32, activation="relu")
            self.z_mean_l    = keras.layers.Dense(latent_dim, name="z_mean")
            self.z_log_var_l = keras.layers.Dense(latent_dim, name="z_log_var")

            # Decoder
            self.dec1    = keras.layers.Dense(32, activation="relu")
            self.dec2    = keras.layers.Dense(64, activation="relu")
            self.dec_out = keras.layers.Dense(input_dim, activation="linear")

        def encode(self, x):
            h = self.enc1(x)
            h = self.enc2(h)
            return self.z_mean_l(h), self.z_log_var_l(h)

        def reparameterize(self, z_mean, z_log_var):
            eps = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * eps

        def decode(self, z):
            h = self.dec1(z)
            h = self.dec2(h)
            return self.dec_out(h)

        def call(self, inputs):
            z_mean, z_log_var = self.encode(inputs)
            z = self.reparameterize(z_mean, z_log_var)
            return self.decode(z)

        def get_config(self):
            config = super().get_config()
            config.update({
                "input_dim":  self.input_dim,
                "latent_dim": self.latent_dim,
            })
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    BASE = os.path.dirname(__file__)

    # Load config (thresholds, column indices, etc.)
    with open(os.path.join(BASE, "leakage_vae_config.json"), "r") as f:
        config = json.load(f)

    # Load scaler
    scaler = joblib.load(os.path.join(BASE, "leakage_robust_scaler.pkl"))

    # Load VAE — decorator above registers the class with registered_name="VAE"
    # which exactly matches what is stored in leakage_vae_fixed.keras
    vae = keras.models.load_model(
        os.path.join(BASE, "leakage_vae_fixed.keras"),
        compile=False,
        safe_mode=False,
    )

    return vae, scaler, config


# ── Feature engineering — mirrors your notebook Cell 2 exactly ──
def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduces the exact same feature engineering steps
    from your Kaggle notebook Cell 2 (the working VAE pipeline).
    """
    df = df_raw.copy()

    # ── Target (we derive it but don't use as input) ──────────
    if "expected_total" in df.columns and "payment_value" in df.columns:
        df["leakage_amount"] = df["payment_value"] - df["expected_total"]
        df["has_leakage"]    = (df["leakage_amount"].abs() > 1.0).astype(int)

    # ── KEY FIX 2 features ────────────────────────────────────
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

    # ── Date features ─────────────────────────────────────────
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
        df["approval_delay_mins"] = df["approval_delay_mins"].fillna(
            df["approval_delay_mins"].median()
        )

    if "order_delivered_customer_date" in df.columns:
        df["delivery_gap_days"] = (
            (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"])
            .dt.total_seconds() / 86400
        ).fillna(0)

    if "order_delivered_carrier_date" in df.columns:
        df["carrier_pickup_days"] = (
            (df["order_delivered_carrier_date"] - df["order_purchase_timestamp"])
            .dt.total_seconds() / 86400
        ).fillna(0)

    if "order_purchase_timestamp" in df.columns:
        df["purchase_hour"]    = df["order_purchase_timestamp"].dt.hour
        df["purchase_weekday"] = df["order_purchase_timestamp"].dt.dayofweek
        df["purchase_month"]   = df["order_purchase_timestamp"].dt.month

    df.drop(columns=[c for c in date_cols if c in df.columns], inplace=True)

    # ── Numeric features ──────────────────────────────────────
    if "price" in df.columns and "total_items_count" in df.columns:
        df["price_per_item"]      = df["price"] / df["total_items_count"]
        df["expected_item_total"] = df["price"] * df["total_items_count"]

    if "freight_value" in df.columns and "price" in df.columns:
        df["freight_ratio"] = df["freight_value"] / (df["price"] + 0.01)

    for dim in ["product_length_cm", "product_height_cm",
                "product_width_cm", "product_weight_g"]:
        if dim in df.columns:
            df[dim] = df[dim].fillna(df[dim].median())

    if all(c in df.columns for c in
           ["product_length_cm", "product_height_cm", "product_width_cm"]):
        df["volume_cm3"] = (
            df["product_length_cm"] *
            df["product_height_cm"] *
            df["product_width_cm"]
        )
        df.drop(columns=["product_length_cm",
                         "product_height_cm",
                         "product_width_cm"], inplace=True)

    if "product_category_name_english" in df.columns:
        df["product_category_name_english"] = \
            df["product_category_name_english"].fillna("unknown")

    # ── Drop ID / leaking columns ─────────────────────────────
    drop_cols = [
        "order_id", "customer_id", "customer_unique_id", "product_id",
        "customer_city", "product_category_name",
        "leakage_amount", "expected_total", "customer_zip_code_prefix",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ── OHE ───────────────────────────────────────────────────
    cat_cols = [c for c in ["order_status", "payment_type",
                             "customer_state",
                             "product_category_name_english"]
                if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

    return df


def predict_leakage(df_features: pd.DataFrame,
                    vae, scaler, config) -> pd.DataFrame:
    """
    Run the exact same inference pipeline as your notebook Cell 7.
    Returns original df with risk_score and leakage_flag columns added.
    """
    CLIP_VAL    = config["clip_val"]
    threshold   = config["vae_threshold"]
    use_focused = config["use_focused_error"]
    pay_idx     = config["payment_col_indices"]

    # Separate label if present
    labels = None
    if "has_leakage" in df_features.columns:
        labels = df_features["has_leakage"].values
        df_features = df_features.drop(columns=["has_leakage"])

    # Scale
    X_scaled = scaler.transform(df_features)
    X_scaled = np.clip(X_scaled, -CLIP_VAL, CLIP_VAL)

    # Reconstruct
    X_recon = vae.predict(X_scaled, verbose=0)

    if use_focused and pay_idx:
        # Payment-focused reconstruction error (better F1)
        scores = np.mean(
            np.square(X_scaled[:, pay_idx] - X_recon[:, pay_idx]),
            axis=1
        )
    else:
        # Full reconstruction error
        scores = np.mean(np.square(X_scaled - X_recon), axis=1)

    flags = (scores > threshold).astype(int)

    result = pd.DataFrame({
        "risk_score":   np.round(scores, 6),
        "leakage_flag": flags,
        "risk_level":   pd.cut(
            scores,
            bins=[-np.inf, threshold * 0.5,
                  threshold, threshold * 2, np.inf],
            labels=["Low", "Medium", "High", "Critical"]
        ),
    })

    if labels is not None:
        result["actual_label"] = labels

    return result


# ════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ════════════════════════════════════════════════════════════

st.title("🔍 Revenue Leakage Detection")
st.markdown(
    "Upload your orders CSV → the Fixed VAE model scans every row "
    "and flags transactions where **payment_value ≠ expected_total**."
)

# ── Sidebar — model info ─────────────────────────────────────
with st.sidebar:
    st.header("Model Info")
    st.markdown("""
    **Model:** Fixed VAE  
    **Architecture:** 3-layer encoder/decoder  
    **Latent dim:** 16  
    **Scaler:** RobustScaler + clip ±15  
    **Key feature:** payment_residual  
    **Strategy:** Payment-focused reconstruction error  
    """)
    st.divider()
    st.markdown("**What the risk levels mean:**")
    st.markdown("""
    - 🟢 **Low** — Normal transaction  
    - 🟡 **Medium** — Monitor  
    - 🔴 **High** — Review manually  
    - 🚨 **Critical** — Block / refund  
    """)
    st.divider()
    st.markdown("**EDA Findings:**")
    st.markdown("""
    - Installments are #1 cause  
    - Boleto = zero leakage  
    - 93% are overcharges  
    - Jan/Feb highest risk  
    """)

# ── Load model ───────────────────────────────────────────────
try:
    with st.spinner("Loading VAE model..."):
        vae, scaler, config = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.info(
        "Make sure **leakage_vae_fixed.keras**, "
        "**leakage_robust_scaler.pkl**, and "
        "**leakage_vae_config.json** are in the same folder as app.py"
    )
    st.stop()

# ── File upload ──────────────────────────────────────────────
st.subheader("Step 1 — Upload your orders CSV")
uploaded = st.file_uploader(
    "Upload CSV file (same format as Revenue_Leakage_Detection_Dataset.csv)",
    type=["csv"]
)

if uploaded is not None:
    # ── Load raw data ─────────────────────────────────────────
    df_raw = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df_raw):,} rows × {df_raw.shape[1]} columns")

    # Keep original columns for display
    display_cols = []
    for c in ["order_id", "payment_value", "expected_total",
              "payment_type", "payment_installments",
              "price", "freight_value", "order_status"]:
        if c in df_raw.columns:
            display_cols.append(c)

    df_display = df_raw[display_cols].copy() if display_cols else df_raw.copy()

    with st.expander("Preview uploaded data (first 5 rows)"):
        st.dataframe(df_raw.head())

    # ── Run prediction ────────────────────────────────────────
    st.subheader("Step 2 — Run detection")

    if st.button("Detect Leakage", type="primary"):
        with st.spinner("Engineering features and running VAE..."):
            try:
                df_features = engineer_features(df_raw.copy())
                results     = predict_leakage(df_features, vae, scaler, config)
                df_out      = pd.concat([df_display.reset_index(drop=True),
                                         results.reset_index(drop=True)],
                                        axis=1)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        # ── Summary metrics ───────────────────────────────────
        st.subheader("Step 3 — Results")

        n_flagged  = results["leakage_flag"].sum()
        n_total    = len(results)
        rate       = n_flagged / n_total * 100
        n_critical = (results["risk_level"] == "Critical").sum()
        n_high     = (results["risk_level"] == "High").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total orders", f"{n_total:,}")
        c2.metric("Flagged as leakage", f"{n_flagged}",
                  delta=f"{rate:.2f}% of orders",
                  delta_color="inverse")
        c3.metric("Critical risk", f"{n_critical}",
                  delta="Block immediately", delta_color="inverse")
        c4.metric("High risk", f"{n_high}",
                  delta="Review manually", delta_color="inverse")

        # ── Risk breakdown chart ──────────────────────────────
        st.subheader("Risk level breakdown")
        risk_counts = results["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        order = ["Critical", "High", "Medium", "Low"]
        risk_counts["Risk Level"] = pd.Categorical(
            risk_counts["Risk Level"], categories=order, ordered=True
        )
        risk_counts = risk_counts.sort_values("Risk Level")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3))
        colors_map = {
            "Critical": "#E24B4A",
            "High":     "#EF9F27",
            "Medium":   "#378ADD",
            "Low":      "#1D9E75",
        }
        clrs = [colors_map.get(r, "#888780")
                for r in risk_counts["Risk Level"]]
        ax.barh(risk_counts["Risk Level"], risk_counts["Count"],
                color=clrs, edgecolor="none")
        ax.set_xlabel("Number of orders")
        ax.set_title("Orders by risk level", fontweight="bold")
        for i, (_, row) in enumerate(risk_counts.iterrows()):
            ax.text(row["Count"] + 0.5, i,
                    str(row["Count"]), va="center", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()

        # ── Reconstruction error distribution ─────────────────
        st.subheader("Reconstruction error distribution")
        fig2, ax2 = plt.subplots(figsize=(9, 3))
        normal_scores  = results[results["leakage_flag"] == 0]["risk_score"]
        flagged_scores = results[results["leakage_flag"] == 1]["risk_score"]
        ax2.hist(normal_scores,  bins=60, color="#378ADD",
                 alpha=0.6, label="Normal", density=True)
        if len(flagged_scores) > 0:
            ax2.hist(flagged_scores, bins=20, color="#E24B4A",
                     alpha=0.8, label="Flagged", density=True)
        ax2.axvline(config["vae_threshold"], color="#EF9F27",
                    linewidth=2, linestyle="--",
                    label=f"Threshold ({config['threshold_pct']}th pct)")
        ax2.set_xlabel("Reconstruction error (risk score)")
        ax2.set_ylabel("Density")
        ax2.legend()
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        st.pyplot(fig2)
        plt.close()

        # ── Flagged orders table ──────────────────────────────
        st.subheader("Flagged orders")
        flagged_df = df_out[df_out["leakage_flag"] == 1].sort_values(
            "risk_score", ascending=False
        )

        if len(flagged_df) == 0:
            st.success("No leakage detected in this batch.")
        else:
            def colour_risk(val):
                colours = {
                    "Critical": "background-color:#FCEBEB;color:#791F1F",
                    "High":     "background-color:#FAEEDA;color:#633806",
                    "Medium":   "background-color:#E6F1FB;color:#0C447C",
                    "Low":      "background-color:#EAF3DE;color:#27500A",
                }
                return colours.get(val, "")

            st.dataframe(
                flagged_df.style.applymap(
                    colour_risk, subset=["risk_level"]
                ),
                use_container_width=True,
            )

        # ── Actual vs predicted (if labels available) ─────────
        if "actual_label" in results.columns:
            from sklearn.metrics import classification_report, confusion_matrix
            import seaborn as sns

            st.subheader("Model evaluation (labels detected in CSV)")
            report = classification_report(
                results["actual_label"],
                results["leakage_flag"],
                target_names=["Normal", "Leakage"],
                output_dict=True
            )
            st.dataframe(pd.DataFrame(report).T.round(3))

            fig3, ax3 = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(results["actual_label"],
                                  results["leakage_flag"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                        xticklabels=["Normal", "Leakage"],
                        yticklabels=["Normal", "Leakage"],
                        ax=ax3, cbar=False)
            ax3.set_title("Confusion Matrix", fontweight="bold")
            ax3.set_xlabel("Predicted")
            ax3.set_ylabel("Actual")
            st.pyplot(fig3)
            plt.close()

        # ── Download results ──────────────────────────────────
        st.subheader("Download results")
        csv_out = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download full results CSV",
            data=csv_out,
            file_name="leakage_detection_results.csv",
            mime="text/csv",
        )

else:
    st.info(
        "No file uploaded yet. "
        "Upload your orders CSV above to get started."
    )
    st.markdown("""
    **Expected columns in your CSV:**
    `order_id`, `payment_value`, `expected_total`, `price`, 
    `freight_value`, `payment_installments`, `payment_type`, 
    `order_status`, `total_items_count`, `customer_state`, 
    `product_category_name_english` and the other columns 
    from the original dataset.
    """)