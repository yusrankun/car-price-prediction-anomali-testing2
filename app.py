import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load trained model ---
model = joblib.load('best_model_LightGBM.pkl')
# --- Check if model is pipeline ---
if hasattr(model, 'named_steps'):
    preprocessor = model.named_steps['preprocessor']
    expected_cols = preprocessor.feature_names_in_
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]
else:
    st.error("Model does not contain preprocessing pipeline. Please re-export with preprocessing included.")
    st.stop()
# --- Get expected columns from the pipeline ---
preprocessor = model.named_steps['preprocessor']
expected_cols = preprocessor.feature_names_in_
num_cols = preprocessor.transformers_[0][2]
cat_cols = preprocessor.transformers_[1][2]

# --- Streamlit UI ---
st.title("Car Price Prediction - LightGBM Model")

# --- Dynamic form input based on expected columns ---
user_input = {}
for col in expected_cols:
    if col in num_cols:
        user_input[col] = st.number_input(f"{col}", value=0.0)
    elif col in cat_cols:
        user_input[col] = st.text_input(f"{col}", value="")

# --- Convert input to DataFrame ---
input_df = pd.DataFrame([user_input])

# --- Reorder columns to match training ---
input_df = input_df.reindex(columns=expected_cols)

# --- Handle missing / unknown values ---
input_df[num_cols] = input_df[num_cols].apply(pd.to_numeric, errors='coerce')
input_df[cat_cols] = input_df[cat_cols].fillna("Unknown")

# --- Predict ---
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Car Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
