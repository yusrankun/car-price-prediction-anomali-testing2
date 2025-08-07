import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load trained model pipeline ---
model = joblib.load('best_model_LightGBM.pkl')

# --- Check if model is a pipeline ---
if not hasattr(model, 'named_steps'):
    st.error("Model does not contain preprocessing pipeline. Please re-export with preprocessing included.")
    st.stop()

# --- Extract preprocessing steps ---
preprocessor = model.named_steps['preprocessor']
expected_cols = list(preprocessor.feature_names_in_)
num_cols = preprocessor.transformers_[0][2]
cat_cols = preprocessor.transformers_[1][2]

# --- Define categorical choices ---
cat_options = {
    'Manufacturer': ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ', 'OPEL',
                     'Rare', 'BMW', 'AUDI', 'NISSAN', 'SUBARU', 'KIA', 'MITSUBISHI', 'SSANGYONG', 'VOLKSWAGEN'],
    'Model': ['Rare', 'FIT', 'Santa FE', 'Prius', 'Sonata', 'Camry', 'E 350', 'Elantra', 'Highlander', 'X5',
              'H1', 'Aqua', 'Civic', 'Tucson', 'Cruze', 'Fusion', 'REXTON', 'Actyon', 'Optima'],
    'Category': ['Jeep', 'Hatchback', 'Sedan', 'Rare', 'Universal', 'Coupe', 'Minivan'],
    'Drive wheels': ['4x4', 'Front', 'Rear'],
    'fuel_gear': ['Hybrid_Automatic', 'Petrol_Tiptronic', 'Petrol_Variator', 'Petrol_Automatic', 'Diesel_Automatic',
                  'CNG_Manual', 'Rare', 'CNG_Automatic', 'Hybrid_Tiptronic', 'Hybrid_Variator', 'Petrol_Manual',
                  'LPG_Automatic', 'Diesel_Manual'],
    'Doors_category': ['4-5', '2-3']
}

# --- Define binary columns ---
binary_cols = ['Leather interior', 'Right_hand_drive', 'is_premium']
binary_map = {'Yes': 1, 'No': 0}

# --- Define float & int columns ---
float_cols = ['engine volume'] if 'engine volume' in expected_cols else []
int_cols = [col for col in num_cols if col not in float_cols and col not in binary_cols]

# --- Streamlit UI ---
st.title("🚗 Car Price Prediction with LightGBM")

user_input = {}

# --- Dynamic form based on expected columns ---
for col in expected_cols:
    if col in binary_cols:
        user_input[col] = st.selectbox(f"{col}", ['Yes', 'No'])
    elif col in cat_options:
        user_input[col] = st.selectbox(f"{col}", cat_options[col])
    elif col in int_cols:
        user_input[col] = st.number_input(f"{col}", value=0, step=1)
    elif col in float_cols:
        user_input[col] = st.number_input(f"{col}", value=0.0, step=0.1)
    else:
        # Default to text input for any unknowns
        user_input[col] = st.text_input(f"{col}", value="")

# --- Convert to DataFrame ---
input_df = pd.DataFrame([user_input])
input_df = input_df.reindex(columns=expected_cols)

# --- Convert binary values ---
for col in binary_cols:
    if col in input_df.columns:
        input_df[col] = input_df[col].map(binary_map)

# --- Convert data types ---
for col in float_cols:
    if col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

for col in int_cols:
    if col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce', downcast='integer')

# --- Fill missing categorical with "Unknown" ---
input_df[cat_cols] = input_df[cat_cols].fillna("Unknown")

# --- Predict ---
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
