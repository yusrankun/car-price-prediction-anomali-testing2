import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load trained pipeline model ---
model = joblib.load("best_model_LightGBM.pkl")

# --- Get preprocessing info ---
if hasattr(model, 'named_steps'):
    preprocessor = model.named_steps['preprocessor']
    expected_cols = preprocessor.feature_names_in_
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]
else:
    st.error("Model does not contain preprocessing pipeline.")
    st.stop()

# --- Kolom numerik yang tetap float ---
float_cols = ['Engine volume']
int_cols = [col for col in num_cols if col not in float_cols]

# --- Dropdown options untuk categorical (isi sesuai kebutuhan) ---
dropdown_options = {
    'Manufacturer': ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ', 'OPEL',
                     'Rare', 'BMW', 'AUDI', 'NISSAN', 'SUBARU', 'KIA', 'MITSUBISHI', 'SSANGYONG', 'VOLKSWAGEN'],
    'Model': ['Rare', 'FIT', 'Santa FE', 'Prius', 'Sonata', 'Camry', 'E 350', 'Elantra', 'Highlander', 'X5', 'H1',
              'Aqua', 'Civic', 'Tucson', 'Cruze', 'Fusion', 'REXTON', 'Actyon', 'Optima'],
    'Category': ['Jeep', 'Hatchback', 'Sedan', 'Rare', 'Universal', 'Coupe', 'Minivan'],
    'Drive wheels': ['4x4', 'Front', 'Rear'],
    'fuel_gear': ['Hybrid_Automatic', 'Petrol_Tiptronic', 'Petrol_Variator', 'Petrol_Automatic',
                  'Diesel_Automatic', 'CNG_Manual', 'Rare', 'CNG_Automatic', 'Hybrid_Tiptronic',
                  'Hybrid_Variator', 'Petrol_Manual', 'LPG_Automatic', 'Diesel_Manual'],
    'Doors_category': ['4-5', '2-3']
}

# --- Streamlit UI ---
st.title("🚗 Car Price Prediction (LightGBM)")

user_input = {}

for col in expected_cols:
    # Numeric input
    if col in int_cols:
        user_input[col] = st.number_input(f"{col}", value=0, step=1, format="%d")
    elif col in float_cols:
        user_input[col] = st.number_input(f"{col}", value=0.0)

    # Binary inputs
    elif col in ['Leather interior', 'is_premium']:
        user_input[col] = 1 if st.selectbox(f"{col}", ['No', 'Yes']) == 'Yes' else 0

    # Dropdown (if predefined options available)
    elif col in dropdown_options:
        user_input[col] = st.selectbox(f"{col}", dropdown_options[col])

    # Default text input
    else:
        user_input[col] = st.text_input(f"{col}", "")

# --- Convert to DataFrame ---
input_df = pd.DataFrame([user_input])
input_df = input_df.reindex(columns=expected_cols)

# --- Handle types explicitly ---
input_df[int_cols] = input_df[int_cols].astype(int)
input_df[float_cols] = input_df[float_cols].astype(float)
input_df[cat_cols] = input_df[cat_cols].fillna("Unknown")

# --- Predict ---
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Car Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
