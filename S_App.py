import streamlit as st
import joblib
import pandas as pd
import os

# ⛑ Tambahan penting agar joblib.load() tidak gagal:
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Debug (boleh dihapus nanti)
# st.write("Current directory:", os.getcwd())
# st.write("Files in directory:", os.listdir("."))

# Tampilkan versi lib (debug)
import sklearn
import imblearn
# st.write("scikit-learn version:", sklearn.__version__)
# st.write("imbalanced-learn version:", imblearn.__version__)

# Path ke model
MODEL_PATH = os.path.join("models", "customerchurn_model_xgb.joblib")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file tidak ditemukan di {MODEL_PATH}")
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Load model
best_model = load_model()
if best_model is None:
    st.stop()

# Streamlit App UI
st.title("📊 Buxton Store - Prediksi Customer Churn")
st.write("Masukkan data pelanggan untuk memprediksi kemungkinan *churn*.")

st.sidebar.header("🧾 Input Data Pelanggan")

# Input sidebar
tenure = st.sidebar.number_input("Tenure (bulan)", 0, 61, 12)
preferred_login_device = st.sidebar.selectbox("Preferred Login Device", ["Mobile Phone", "Computer"])
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
warehouse_to_home = st.sidebar.number_input("Jarak Warehouse ke Rumah", 5, 127, 10)
preferred_payment_mode = st.sidebar.selectbox("Preferred Payment Mode", ["Debit Card", "CC", "UPI", "E Wallet", "COD"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
hour_spend_on_app = st.sidebar.number_input("Jam di Aplikasi", 0, 5, 2)
number_of_device_registered = st.sidebar.number_input("Jumlah Device Terdaftar", 1, 6, 2)
preferred_order_cat = st.sidebar.selectbox("Preferred Order Category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"])
satisfaction_score = st.sidebar.slider("Satisfaction Score", 1, 5, 3)
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married"])
number_of_address = st.sidebar.number_input("Jumlah Alamat", 1, 22, 3)
complain = st.sidebar.selectbox("Pernah Komplain?", [0, 1])
order_amount_hike = st.sidebar.number_input("Kenaikan Order Tahun Lalu (%)", 11, 26, 15)
coupon_used = st.sidebar.number_input("Kupon Digunakan", 0, 16, 5)
order_count = st.sidebar.number_input("Jumlah Order", 1, 16, 10)
day_since_last_order = st.sidebar.number_input("Hari Sejak Order Terakhir", 0, 46, 7)
cashback_amount = st.sidebar.number_input("Cashback", 0.0, 324.99, 100.0)

# Dataframe dari input
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

st.subheader("📋 Data yang Dimasukkan")
st.write(df_input)

# Prediksi
if st.button("🔮 Prediksi Churn"):
    try:
        proba = best_model.predict_proba(df_input)[0][1]
        pred = best_model.predict(df_input)[0]

        st.subheader("📈 Hasil Prediksi")
        if pred == 1:
            st.error(f"⚠️ Pelanggan berisiko *churn* ({proba:.2%})")
        else:
            st.success(f"✅ Pelanggan tidak *churn* ({proba:.2%})")

        st.progress(int(proba * 100))

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
