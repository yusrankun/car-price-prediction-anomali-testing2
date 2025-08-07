import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Car Price Prediction", layout="centered")

# --- Load trained model ---
model = joblib.load("best_model_LightGBM.pkl")

# --- Check model is a pipeline ---
if not hasattr(model, "named_steps"):
    st.error("Model does not contain preprocessing pipeline.")
    st.stop()

# --- Extract expected input columns ---
preprocessor = model.named_steps["preprocessor"]
expected_cols = list(preprocessor.feature_names_in_)
num_cols = preprocessor.transformers_[0][2]
cat_cols = preprocessor.transformers_[1][2]

# --- Define binary and dropdown columns ---
binary_cols = ['Leather interior', 'is_premium', 'Right_hand_drive']
binary_map = {'Yes': 1, 'No': 0}

dropdown_options = {
    "Manufacturer": ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ', 'OPEL',
                     'Rare', 'BMW', 'AUDI', 'NISSAN', 'SUBARU', 'KIA', 'MITSUBISHI', 'SSANGYONG', 'VOLKSWAGEN'],
    "Model": ['Rare', 'FIT', 'Santa FE', 'Prius', 'Sonata', 'Camry', 'E 350', 'Elantra', 'Highlander', 'X5',
              'H1', 'Aqua', 'Civic', 'Tucson', 'Cruze', 'Fusion', 'REXTON', 'Actyon', 'Optima'],
    "Category": ['Jeep', 'Hatchback', 'Sedan', 'Rare', 'Universal', 'Coupe', 'Minivan'],
    "Drive wheels": ['4x4', 'Front', 'Rear'],
    "fuel_gear": ['Hybrid_Automatic', 'Petrol_Tiptronic', 'Petrol_Variator', 'Petrol_Automatic',
                  'Diesel_Automatic', 'CNG_Manual', 'Rare', 'CNG_Automatic', 'Hybrid_Tiptronic',
                  'Hybrid_Variator', 'Petrol_Manual', 'LPG_Automatic', 'Diesel_Manual'],
    "Doors_category": ['4-5', '2-3']
}

# --- Identify float/int columns ---
float_cols = ['engine volume']  # <- tambahkan di sini kolom float lainnya jika ada
int_cols = [col for col in num_cols if col not in float_cols and col not in binary_cols]

# --- Streamlit UI ---
st.title("🚗 Car Price Prediction - LightGBM")

st.markdown("Masukkan detail mobil di bawah ini:")

user_input = {}

# --- Input fields dynamically generated ---
for col in expected_cols:
    if col in float_cols:
        user_input[col] = st.number_input(f"{col}", step=0.1, format="%.2f")
    elif col in int_cols:
        user_input[col] = st.number_input(f"{col}", step=1)
    elif col in binary_cols:
        user_input[col] = st.radio(f"{col}", ['Yes', 'No'])
    elif col in dropdown_options:
        user_input[col] = st.selectbox(f"{col}", dropdown_options[col])
    elif col in cat_cols:
        user_input[col] = st.text_input(f"{col}")
    else:
        user_input[col] = st.text_input(f"{col}")

# --- Convert input to DataFrame ---
input_df = pd.DataFrame([user_input])

# --- Convert binary columns Yes/No to 1/0 ---
for col in binary_cols:
    if col in input_df:
        input_df[col] = input_df[col].map(binary_map)

# --- Type conversions ---
for col in float_cols:
    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

for col in int_cols:
    input_df[col] = pd.to_numeric(input_df[col], errors='coerce', downcast='integer')

# --- Ensure correct column order ---
input_df = input_df.reindex(columns=expected_cols)

# --- Prediction ---
if st.button("🔍 Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
