import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline model
model_pipeline = joblib.load("best_model_LightGBM.pkl")

# Judul Aplikasi
st.title("Prediksi Harga Mobil")

# Input dari user
st.header("Masukkan Detail Mobil:")

levy = st.number_input("Levy", value=0.0)
leather = st.selectbox("Leather interior", ["Yes", "No"])
mileage = st.number_input("Mileage", value=0.0)
doors = st.number_input("Doors", value=4)
airbags = st.number_input("Airbags", value=2)
right_hand_drive = st.selectbox("Right hand drive", ["Yes", "No"])
volume_per_cylinder = st.number_input("Volume per Cylinder", value=0.0)
car_age = st.number_input("Car Age", value=5)
is_premium = st.selectbox("Is Premium", ["Yes", "No"])
model_encoded = st.number_input("Model Encoded", value=0)

manufacturer = st.text_input("Manufacturer", "Toyota")
model = st.text_input("Model", "Corolla")
category = st.text_input("Category", "Sedan")
drive_wheels = st.text_input("Drive wheels", "Front")
fuel_gear = st.text_input("Fuel/Gear", "Petrol/Auto")
doors_category = st.text_input("Doors Category", "4/5")

# Mapping Yes/No to 1/0
leather = 1 if leather == "Yes" else 0
right_hand_drive = 1 if right_hand_drive == "Yes" else 0
is_premium = 1 if is_premium == "Yes" else 0

# Buat DataFrame dari input user
input_data = pd.DataFrame([{
    "Levy": levy,
    "Leather interior": leather,
    "Mileage": mileage,
    "Doors": doors,
    "Airbags": airbags,
    "Right_hand_drive": right_hand_drive,
    "volume_per_cylinder": volume_per_cylinder,
    "car_age": car_age,
    "is_premium": is_premium,
    "Model_encoded": model_encoded,
    "Manufacturer": manufacturer,
    "Model": model,
    "Category": category,
    "Drive wheels": drive_wheels,
    "fuel_gear": fuel_gear,
    "Doors_category": doors_category
}])

# Tampilkan input
st.subheader("Data yang Dimasukkan:")
st.write(input_data)

# Prediksi saat tombol ditekan
if st.button("Prediksi Harga"):
    try:
        prediction = model_pipeline.predict(input_data)[0]
        st.success(f"Prediksi Harga Mobil: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
