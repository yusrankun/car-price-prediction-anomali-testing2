import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan expected columns
model, expected_cols = joblib.load('best_model_LightGBM.pkl')

st.title("Car Price Prediction 🚗")

# Form input user
st.header("Input Car Features")

# Contoh input field sesuai fitur
Levy = st.number_input("Levy", min_value=0.0, value=0.0)
Leather_interior = st.selectbox("Leather Interior", ["Yes", "No"])
Mileage = st.number_input("Mileage (km)", min_value=0)
Doors = st.selectbox("Doors", [2, 3, 4, 5])
Airbags = st.number_input("Number of Airbags", min_value=0)
Right_hand_drive = st.selectbox("Right Hand Drive", ["Yes", "No"])
volume_per_cylinder = st.number_input("Volume per Cylinder", min_value=0.0)
car_age = st.number_input("Car Age", min_value=0)
is_premium = st.selectbox("Is Premium", ["Yes", "No"])
Model_encoded = st.number_input("Model Encoded", min_value=0)
Manufacturer = st.selectbox("Manufacturer", ['Toyota', 'BMW', 'Mercedes', 'Other'])  # ganti sesuai datamu
Model = st.text_input("Model")
Category = st.selectbox("Category", ['Sedan', 'SUV', 'Hatchback', 'Other'])
Drive_wheels = st.selectbox("Drive Wheels", ['Front', 'Rear', '4WD'])
fuel_gear = st.text_input("Fuel + Gear Type")  # kolom gabungan jika kamu buat seperti itu
Doors_category = st.selectbox("Doors Category", ['Few', 'Normal', 'Many'])  # ganti sesuai datamu

# Buat DataFrame dari input
input_dict = {
    'Levy': Levy,
    'Leather interior': Leather_interior,
    'Mileage': Mileage,
    'Doors': Doors,
    'Airbags': Airbags,
    'Right_hand_drive': Right_hand_drive,
    'volume_per_cylinder': volume_per_cylinder,
    'car_age': car_age,
    'is_premium': is_premium,
    'Model_encoded': Model_encoded,
    'Manufacturer': Manufacturer,
    'Model': Model,
    'Category': Category,
    'Drive wheels': Drive_wheels,
    'fuel_gear': fuel_gear,
    'Doors_category': Doors_category
}

input_df = pd.DataFrame([input_dict])

# Reindex untuk cocokkan kolom ke pipeline
input_df = input_df.reindex(columns=expected_cols)

# Prediksi
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
