import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load trained pipeline model ---
model = joblib.load("best_model_LightGBM.pkl")

# --- Check if model is pipeline ---
if hasattr(model, 'named_steps'):
    preprocessor = model.named_steps['preprocessor']
    expected_cols = preprocessor.feature_names_in_
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]
else:
    st.error("Model does not contain preprocessing pipeline. Please re-export with preprocessing included.")
    st.stop()

# --- Title ---
st.title("🚗 Car Price Prediction")

# --- Fixed categories for categorical selectboxes ---
category_options = {
    "Manufacturer": ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ', 'OPEL', 'Rare', 'BMW', 'AUDI', 'NISSAN', 'SUBARU', 'KIA', 'MITSUBISHI', 'SSANGYONG', 'VOLKSWAGEN'],
    "Model": ['Rare', 'FIT', 'Santa FE', 'Prius', 'Sonata', 'Camry', 'E 350', 'Elantra', 'Highlander', 'X5', 'H1', 'Aqua', 'Civic', 'Tucson', 'Cruze', 'Fusion', 'REXTON', 'Actyon', 'Optima'],
    "Category": ['Jeep', 'Hatchback', 'Sedan', 'Rare', 'Universal', 'Coupe', 'Minivan'],
    "Drive wheels": ['4x4', 'Front', 'Rear'],
    "fuel_gear": ['Hybrid_Automatic', 'Petrol_Tiptronic', 'Petrol_Variator', 'Petrol_Automatic', 'Diesel_Automatic', 'CNG_Manual', 'Rare', 'CNG_Automatic', 'Hybrid_Tiptronic', 'Hybrid_Variator', 'Petrol_Manual', 'LPG_Automatic', 'Diesel_Manual'],
    "Doors_category": ['4-5', '2-3']
}

# --- UI input ---
user_input = {}
for col in expected_cols:
    if col in num_cols:
        user_input[col] = st.number_input(f"{col}", value=0.0)
    elif col in cat_cols:
        # Handle specific binary columns
        if col in ['Leather interior', 'is_premium', 'Right_hand_drive']:
            user_input[col] = st.selectbox(f"{col}", ['Yes', 'No'])
            user_input[col] = 1 if user_input[col] == 'Yes' else 0
        elif col in category_options:
            user_input[col] = st.selectbox(f"{col}", category_options[col])
        else:
            user_input[col] = st.text_input(f"{col}", value="Unknown")  # fallback

# --- Convert input to DataFrame ---
input_df = pd.DataFrame([user_input])

# --- Ensure order of columns matches training ---
input_df = input_df.reindex(columns=expected_cols)

# --- Convert numeric columns to float ---
input_df[num_cols] = input_df[num_cols].apply(pd.to_numeric, errors='coerce')

# --- Fill missing categorical with placeholder ---
input_df[cat_cols] = input_df[cat_cols].fillna("Unknown")

# --- Predict ---
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
