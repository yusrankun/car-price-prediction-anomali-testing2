import streamlit as st
import pandas as pd
import joblib

# Load model LightGBM
model = joblib.load('lightgbm_best_model.pkl')

st.title("Prediksi Harga Mobil - LightGBM")

# Sidebar: pilih input mode
input_mode = st.sidebar.radio("Pilih input data:", ['Manual Input', 'Upload File CSV'])

if input_mode == 'Manual Input':
    st.header("Input Data Mobil")

    # Contoh input form
    mileage = st.number_input("Mileage", min_value=0)
    levy = st.number_input("Levy", min_value=0)
    model_encoded = st.number_input("Model (Encoded)", min_value=0)
    airbags = st.number_input("Jumlah Airbags", min_value=0)
    leather = st.selectbox("Leather Interior", ['Yes', 'No'])
    car_age = st.slider("Usia Mobil", 0, 30, 5)
    is_premium = st.selectbox("Premium Brand", ['Yes', 'No'])

    # Simpan dalam dataframe
    input_df = pd.DataFrame([{
        'num__Mileage': mileage,
        'num__Levy': levy,
        'num__Model_encoded': model_encoded,
        'num__Airbags': airbags,
        'num__Leather interior': 1 if leather == 'Yes' else 0,
        'num__car_age': car_age,
        'num__is_premium': 1 if is_premium == 'Yes' else 0
        # Tambahkan fitur lain sesuai fitur yang digunakan model
    }])

    if st.button("Prediksi Harga"):
        prediction = model.predict(input_df)[0]
        st.success(f"Estimasi harga mobil: **{prediction:,.0f}**")

elif input_mode == 'Upload File CSV':
    st.header("Upload File CSV")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file:
        df_new = pd.read_csv(uploaded_file)
        predictions = model.predict(df_new)
        df_new['Predicted_Price'] = predictions

        st.subheader("Hasil Prediksi")
        st.dataframe(df_new)

        csv = df_new.to_csv(index=False).encode('utf-8')
        st.download_button("Download hasil", csv, "hasil_prediksi.csv", "text/csv")

