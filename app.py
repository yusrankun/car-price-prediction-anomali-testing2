import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load trained pipeline model ---
try:
    model = joblib.load("best_model_LightGBM.pkl")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- Pastikan model berupa pipeline dengan preprocessing ---
if not hasattr(model, "named_steps") or "preprocessor" not in model.named_steps:
    st.error("Model tidak mengandung preprocessing pipeline. Harap ekspor ulang model dengan preprocessing.")
    st.stop()

# --- Ambil info dari preprocessing ---
preprocessor = model.named_steps["preprocessor"]
expected_cols = preprocessor.feature_names_in_

# Ambil fitur numerik dan kategorikal dari pipeline
num_cols = preprocessor.transformers_[0][2]
cat_cols = preprocessor.transformers_[1][2]

# Tentukan kolom biner (Yes/No) jika ada
binary_cols = [col for col in cat_cols if col in ["Leather interior", "is_premium", "Right_hand_drive"]]

# --- Streamlit UI ---
st.title("🚗 Car Price Prediction App (LightGBM)")

st.markdown("Masukkan informasi mobil di bawah ini untuk memprediksi harga:")

user_input = {}

# --- Dynamic input generation ---
for col in expected_cols:
    if col in num_cols:
        if col in ["Engine volume"]:  # float kolom
            user_input[col] = st.number_input(col, step=0.1, format="%.2f")
        else:
            user_input[col] = st.number_input(col, step=1)
    elif col in binary_cols:
        user_input[col] = st.radio(col, ["Yes", "No"])
    elif col in cat_cols:
        user_input[col] = st.text_input(col)
    else:
        user_input[col] = st.text_input(col)

# --- Convert to DataFrame ---
input_df = pd.DataFrame([user_input])

# --- Preprocessing: Convert binary cols to int ---
binary_map = {"Yes": 1, "No": 0}
for col in binary_cols:
    if col in input_df.columns:
        input_df[col] = input_df[col].map(binary_map)

# --- Reorder columns to match training ---
input_df = input_df.reindex(columns=expected_cols)

# --- Handle missing values and type ---
for col in num_cols:
    if col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

for col in cat_cols:
    if col in input_df.columns:
        input_df[col] = input_df[col].fillna("Unknown")

# --- Optional: convert integer cols ---
int_cols = [col for col in num_cols if col != "Engine volume"]
float_cols = ["Engine volume"]

for col in int_cols:
    if col in input_df.columns:
        input_df[col] = input_df[col].astype(int)

for col in float_cols:
    if col in input_df.columns:
        input_df[col] = input_df[col].astype(float)

# --- Predict button ---
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Prediksi Harga Mobil: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
