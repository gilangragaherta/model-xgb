import streamlit as st
import pickle
import pandas as pd

# Perbarui path ke lokasi yang benar
MODEL_PATH = "/Users/macbook/Documents/Purwadhika/Final Project/customer_churn_model_xgb.sav"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as model_file:
        loaded_model = pickle.load(model_file)
    return loaded_model

best_model = load_model()

# UI dengan Streamlit
st.title("\U0001F4CA Buxton Store Customer Churn Prediction")
st.write("Masukkan data pelanggan untuk memprediksi kemungkinan churn.")

st.sidebar.header("\U0001F522 Masukkan Data Pelanggan")

# Input user dengan batasan yang disesuaikan
tenure = st.sidebar.number_input("Tenure (bulan)", min_value=0, max_value=61, value=12)
preferred_login_device = st.sidebar.selectbox("Preferred Login Device", ["Mobile Phone", "Computer"])
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
warehouse_to_home = st.sidebar.number_input("Jarak Warehouse ke Rumah", min_value=5, max_value=127, value=10)
preferred_payment_mode = st.sidebar.selectbox("Preferred Payment Mode", ["Debit Card", "CC", "UPI", "E Wallet", "COD"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
hour_spend_on_app = st.sidebar.number_input("Jam Dihabiskan di Aplikasi", min_value=0, max_value=5, value=2)
number_of_device_registered = st.sidebar.number_input("Jumlah Device Terdaftar", min_value=1, max_value=6, value=2)
preferred_order_cat = st.sidebar.selectbox("Preferred Order Category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery","Others"])
satisfaction_score = st.sidebar.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married"])
number_of_address = st.sidebar.number_input("Jumlah Alamat Tersimpan", min_value=1, max_value=22, value=3)
complain = st.sidebar.selectbox("Pernah Komplain?", [0, 1])
order_amount_hike = st.sidebar.number_input("Kenaikan Order dari Tahun Lalu (%)", min_value=11, max_value=26, value=15)
coupon_used = st.sidebar.number_input("Kupon Digunakan", min_value=0, max_value=16, value=5)
order_count = st.sidebar.number_input("Jumlah Order", min_value=1, max_value=16, value=10)
day_since_last_order = st.sidebar.number_input("Hari Sejak Order Terakhir", min_value=0, max_value=46, value=7)
cashback_amount = st.sidebar.number_input("Cashback Amount", min_value=0.0, max_value=324.99, value=100.0)

# Buat DataFrame dari input user
df_input = pd.DataFrame({
    'Tenure': [tenure],
    'PreferredLoginDevice': [preferred_login_device],
    'CityTier': [city_tier],
    'WarehouseToHome': [warehouse_to_home],
    'PreferredPaymentMode': [preferred_payment_mode],
    'Gender': [gender],
    'HourSpendOnApp': [hour_spend_on_app],
    'NumberOfDeviceRegistered': [number_of_device_registered],
    'PreferedOrderCat': [preferred_order_cat],
    'SatisfactionScore': [satisfaction_score],
    'MaritalStatus': [marital_status],
    'NumberOfAddress': [number_of_address],
    'Complain': [complain],
    'OrderAmountHikeFromlastYear': [order_amount_hike],
    'CouponUsed': [coupon_used],
    'OrderCount': [order_count],
    'DaySinceLastOrder': [day_since_last_order],
    'CashbackAmount': [cashback_amount]
})

st.subheader("\U0001F4C4 Data yang Dimasukkan")
st.write(df_input)

# Prediksi Churn
if st.button("\U0001F52E Prediksi Churn"):
    try:
        prediction_proba = best_model.predict_proba(df_input)[0][1]  # Probabilitas churn
        prediction_class = best_model.predict(df_input)[0]  # Hasil klasifikasi

        st.subheader("\U0001F4CA Hasil Prediksi")
        
        if prediction_class == 1:
            st.error(f"⚠️ Pelanggan Berisiko Churn ({prediction_proba:.2%})")
        else:
            st.success(f"✅ Pelanggan Tidak Churn ({prediction_proba:.2%})")

        # Visualisasi
        st.progress(int(prediction_proba * 100))

    except Exception as e:
        st.error(f"Terjadi error dalam prediksi: {e}")
