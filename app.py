import streamlit as st
import pandas as pd
import joblib

# Load model pipeline
model_pipeline = joblib.load("best_model_LightGBM.pkl")

st.title("🚗 Car Price Prediction App")

st.markdown("Masukkan detail mobil di bawah ini untuk memprediksi harganya.")

# Input fields

Levy = st.number_input("Levy", min_value=0.0, step=10.0)
Leather_interior = st.selectbox("Leather interior", ["Yes", "No"])
Mileage = st.number_input("Mileage (km)", min_value=0.0, step=1000.0)
Doors = st.number_input("Doors", min_value=2, max_value=5)
Airbags = st.number_input("Airbags", min_value=0, step=1)
Right_hand_drive = st.selectbox("Right hand drive", ["Yes", "No"])
volume_per_cylinder = st.number_input("Volume per cylinder", min_value=0.0, step=0.1)
car_age = st.number_input("Car Age", min_value=0, max_value=50)
is_premium = st.selectbox("Is Premium?", ["Yes", "No"])
Model_encoded = st.number_input("Model encoded", min_value=0)

# --- Categorical columns from your list
Manufacturer = st.selectbox("Manufacturer", [
    'LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ',
    'OPEL', 'Rare', 'BMW', 'AUDI', 'NISSAN', 'SUBARU', 'KIA', 'MITSUBISHI',
    'SSANGYONG', 'VOLKSWAGEN'
])

Model = st.selectbox("Model", [
    'Rare', 'FIT', 'Santa FE', 'Prius', 'Sonata', 'Camry', 'E 350', 'Elantra',
    'Highlander', 'X5', 'H1', 'Aqua', 'Civic', 'Tucson', 'Cruze', 'Fusion',
    'REXTON', 'Actyon', 'Optima'
])

Category = st.selectbox("Category", [
    'Jeep', 'Hatchback', 'Sedan', 'Rare', 'Universal', 'Coupe', 'Minivan'
])

Drive_wheels = st.selectbox("Drive wheels", ['4x4', 'Front', 'Rear'])

fuel_gear = st.selectbox("Fuel / Gear", [
    'Hybrid_Automatic', 'Petrol_Tiptronic', 'Petrol_Variator', 'Petrol_Automatic',
    'Diesel_Automatic', 'CNG_Manual', 'Rare', 'CNG_Automatic', 'Hybrid_Tiptronic',
    'Hybrid_Variator', 'Petrol_Manual', 'LPG_Automatic', 'Diesel_Manual'
])

Doors_category = st.selectbox("Doors category", ['4-5', '2-3'])

# Predict button
if st.button("Predict"):
    try:
        input_data = pd.DataFrame([{
            "Levy": Levy,
            "Leather interior": Leather_interior,
            "Mileage": Mileage,
            "Doors": Doors,
            "Airbags": Airbags,
            "Right_hand_drive": Right_hand_drive,
            "volume_per_cylinder": volume_per_cylinder,
            "car_age": car_age,
            "is_premium": is_premium,
            "Model_encoded": Model_encoded,
            "Manufacturer": Manufacturer,
            "Model": Model,
            "Category": Category,
            "Drive wheels": Drive_wheels,
            "fuel_gear": fuel_gear,
            "Doors_category": Doors_category
        }])

        prediction = model_pipeline.predict(input_data)[0]
        st.success(f"💰 Predicted Price: {prediction:,.2f} GEL")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
