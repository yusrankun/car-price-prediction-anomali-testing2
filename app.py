import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load trained model pipeline ===
model = joblib.load("best_model_LightGBM.pkl")

# === Check if model is pipeline and extract info ===
if not hasattr(model, 'named_steps'):
    st.error("Model does not contain preprocessing pipeline. Please re-export with preprocessing included.")
    st.stop()

preprocessor = model.named_steps['preprocessor']
expected_cols = preprocessor.feature_names_in_
num_cols = preprocessor.transformers_[0][2]
cat_cols = preprocessor.transformers_[1][2]

# === Manual categorical options (based on training set) ===
cat_options = {
    "Manufacturer": ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA',
                     'MERCEDES-BENZ', 'OPEL', 'Rare', 'BMW', 'AUDI', 'NISSAN',
                     'SUBARU', 'KIA', 'MITSUBISHI', 'SSANGYONG', 'VOLKSWAGEN'],
    "Model": ['Rare', 'FIT', 'Santa FE', 'Prius', 'Sonata', 'Camry', 'E 350',
              'Elantra', 'Highlander', 'X5', 'H1', 'Aqua', 'Civic', 'Tucson',
              'Cruze', 'Fusion', 'REXTON', 'Actyon', 'Optima'],
    "Category": ['Jeep', 'Hatchback', 'Sedan', 'Rare', 'Universal', 'Coupe', 'Minivan'],
    "Drive wheels": ['4x4', 'Front', 'Rear'],
    "fuel_gear": ['Hybrid_Automatic', 'Petrol_Tiptronic', 'Petrol_Variator',
                  'Petrol_Automatic', 'Diesel_Automatic', 'CNG_Manual', 'Rare',
                  'CNG_Automatic', 'Hybrid_Tiptronic', 'Hybrid_Variator',
                  'Petrol_Manual', 'LPG_Automatic', 'Diesel_Manual'],
    "Doors_category": ['4-5', '2-3'],
    "Leather interior": ['Yes', 'No'],
    "Right_hand_drive": ['Yes', 'No'],
    "is_premium": ['Yes', 'No']
}

# === Streamlit UI ===
st.title("🚗 Car Price Prediction App (LightGBM)")
st.markdown("Masukkan detail mobil di bawah ini untuk memprediksi harga jual.")

user_input = {}

# === Dynamic input based on expected columns ===
for col in expected_cols:
    if col in cat_options:
        user_input[col] = st.selectbox(f"{col}", options=cat_options[col])
    elif col in num_cols:
        user_input[col] = st.number_input(f"{col}", value=0)
    else:
        user_input[col] = st.text_input(f"{col}", value="")

# === Convert input to DataFrame ===
input_df = pd.DataFrame([user_input])

# === Map 'Yes'/'No' to binary ===
binary_cols = ['Leather interior', 'Right_hand_drive', 'is_premium']
for col in binary_cols:
    if col in input_df.columns:
        input_df[col] = input_df[col].map({'Yes': 1, 'No': 0}).astype(int)

# === Reorder columns and handle types ===
input_df = input_df.reindex(columns=expected_cols)

# Coerce numerics
for col in num_cols:
    input_df[col] = pd.to_numeric(input_df[col], errors='coerce').astype(float)

# Fill categorical NAs
for col in cat_cols:
    input_df[col] = input_df[col].fillna("Unknown")

# === Predict ===
if st.button("🔮 Predict Price"):
    try:
        log_prediction = model.predict(input_df)[0]
        prediction = np.expm1(log_prediction)  # reverse log1p
        st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
