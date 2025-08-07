import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load model ---
try:
    model = joblib.load("best_model_LightGBM.pkl")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- Cek pipeline ---
if not hasattr(model, "named_steps") or "preprocessor" not in model.named_steps:
    st.error("Model tidak memiliki preprocessing pipeline.")
    st.stop()

# --- Ambil struktur dari preprocessor ---
preprocessor = model.named_steps["preprocessor"]
expected_cols = preprocessor.feature_names_in_
num_cols = preprocessor.transformers_[0][2]
cat_cols = preprocessor.transformers_[1][2]

# --- Definisi nilai-nilai kategori ---
manufacturer_options = ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ',
                        'OPEL', 'Rare', 'BMW', 'AUDI', 'NISSAN', 'SUBARU', 'KIA', 'MITSUBISHI', 'SSANGYONG', 'VOLKSWAGEN']

model_options = ['Rare', 'FIT', 'Santa FE', 'Prius', 'Sonata', 'Camry', 'E 350', 'Elantra',
                 'Highlander', 'X5', 'H1', 'Aqua', 'Civic', 'Tucson', 'Cruze', 'Fusion', 'REXTON', 'Actyon', 'Optima']

category_options = ['Jeep', 'Hatchback', 'Sedan', 'Rare', 'Universal', 'Coupe', 'Minivan']

drive_options = ['4x4', 'Front', 'Rear']

fuel_gear_options = ['Hybrid_Automatic', 'Petrol_Tiptronic', 'Petrol_Variator', 'Petrol_Automatic',
                     'Diesel_Automatic', 'CNG_Manual', 'Rare', 'CNG_Automatic', 'Hybrid_Tiptronic',
                     'Hybrid_Variator', 'Petrol_Manual', 'LPG_Automatic', 'Diesel_Manual']

doors_category_options = ['4-5', '2-3']

# Kolom biner
binary_cols = ['Leather interior', 'is_premium', 'Right_hand_drive']
binary_map = {"Yes": 1, "No": 0}

# Kolom float
float_cols = ['Engine volume']
# Kolom integer (selain float)
int_cols = [col for col in num_cols if col not in float_cols]

# --- UI ---
st.title("🚗 Car Price Prediction (LightGBM)")

user_input = {}

# --- Input dinamis berdasarkan jenis kolom ---
for col in expected_cols:
    if col in float_cols:
        user_input[col] = st.number_input(col, step=0.1, format="%.2f")
    elif col in int_cols:
        user_input[col] = st.number_input(col, step=1)
    elif col in binary_cols:
        user_input[col] = st.radio(col, ["Yes", "No"])
    elif col == "Manufacturer":
        user_input[col] = st.selectbox(col, manufacturer_options)
    elif col == "Model":
        user_input[col] = st.selectbox(col, model_options)
    elif col == "Category":
        user_input[col] = st.selectbox(col, category_options)
    elif col == "Drive wheels":
        user_input[col] = st.selectbox(col, drive_options)
    elif col == "fuel_gear":
        user_input[col] = st.selectbox(col, fuel_gear_options)
    elif col == "Doors_category":
        user_input[col] = st.selectbox(col, doors_category_options)
    else:
        user_input[col] = st.text_input(col)

# --- Convert ke DataFrame ---
input_df = pd.DataFrame([user_input])

# --- Convert kolom biner ke angka ---
for col in binary_cols:
    if col in input_df:
        input_df[col] = input_df[col].map(binary_map)

# --- Reorder dan konversi tipe data ---
input_df = input_df.reindex(columns=expected_cols)

for col in int_cols:
    if col in input_df:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0).astype(int)

for col in float_cols:
    if col in input_df:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0.0).astype(float)

for col in cat_cols:
    if col in input_df:
        input_df[col] = input_df[col].fillna("Unknown")

# --- Prediksi ---
if st.button("Predict Price"):
    try:
        pred = model.predict(input_df)[0]
        st.success(f"💰 Perkiraan Harga Mobil: **${pred:,.2f}**")
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}")
