import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Car Price Prediction", layout="centered")
st.title("🚗 Car Price Prediction App (LightGBM)")
st.markdown("Masukkan detail mobil di bawah ini untuk memprediksi harga jual.")

# --- Load Model Pipeline ---
model_pipeline = joblib.load("best_model_LightGBM.pkl")

if not hasattr(model_pipeline, "predict"):
    st.error("Model pipeline tidak valid.")
    st.stop()

# Ambil preprocessor dan kolom
preprocessor = model_pipeline.named_steps["preprocessor"]
num_cols = preprocessor.transformers_[0][2]
cat_cols = preprocessor.transformers_[1][2]

# --- Static category options ---
category_options = {
    "Leather interior": ["Yes", "No"],
    "Right_hand_drive": ["Yes", "No"],
    "is_premium": ["Yes", "No"],
    "Manufacturer": ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA',
                     'MERCEDES-BENZ', 'OPEL', 'Rare', 'BMW', 'AUDI', 'NISSAN',
                     'SUBARU', 'KIA', 'MITSUBISHI', 'SSANGYONG', 'VOLKSWAGEN'],
    "Model": ['Rare', 'FIT', 'Santa FE', 'Prius', 'Sonata', 'Camry', 'E 350', 'Elantra',
              'Highlander', 'X5', 'H1', 'Aqua', 'Civic', 'Tucson', 'Cruze', 'Fusion',
              'REXTON', 'Actyon', 'Optima'],
    "Category": ['Jeep', 'Hatchback', 'Sedan', 'Rare', 'Universal', 'Coupe', 'Minivan'],
    "Drive wheels": ['4x4', 'Front', 'Rear'],
    "fuel_gear": ['Hybrid_Automatic', 'Petrol_Tiptronic', 'Petrol_Variator', 'Petrol_Automatic',
                  'Diesel_Automatic', 'CNG_Manual', 'Rare', 'CNG_Automatic', 'Hybrid_Tiptronic',
                  'Hybrid_Variator', 'Petrol_Manual', 'LPG_Automatic', 'Diesel_Manual'],
    "Doors_category": ['4-5', '2-3']
}

# --- Input Form ---
user_input = {}

for col in num_cols:
    if col == "volume_per_cylinder":
        user_input[col] = st.number_input(col, value=1.0, min_value=0.0)
    else:
        user_input[col] = st.number_input(col, value=0, step=1)

for col in cat_cols:
    if col in category_options:
        user_input[col] = st.selectbox(col, category_options[col])
    else:
        user_input[col] = st.text_input(col, "")

# --- Preprocess binary yes/no ---
binary_map = {"Yes": 1, "No": 0}
for binary_col in ["Leather interior", "Right_hand_drive", "is_premium"]:
    if binary_col in user_input:
        user_input[binary_col] = binary_map.get(user_input[binary_col], 0)

# --- Create DataFrame ---
input_df = pd.DataFrame([user_input])

# --- Predict ---
if st.button("Predict Price"):
    try:
        log_prediction = model_pipeline.predict(input_df)[0]
        prediction = np.expm1(log_prediction)
        st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
