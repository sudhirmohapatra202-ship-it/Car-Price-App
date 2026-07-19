# app.py
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# ---- Load model, scaler, encoders, feature order ----
@st.cache_resource
def load_artifacts():
    model = joblib.load("car_price_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("label_encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, encoders, feature_columns

model, scaler, encoders, feature_columns = load_artifacts()

st.title("🚗 Car Price Predictor")
st.write("Enter the car details below to get an estimated selling price.")

st.divider()

# ---- Inputs ----
car_name = st.selectbox("Car Brand", list(encoders["Car_Name"].classes_))
year = st.number_input("Manufacture Year", min_value=2000, max_value=2026, value=2018)
km_driven = st.number_input("KM Driven", min_value=0, max_value=500000, value=50000, step=1000)
fuel = st.selectbox("Fuel Type", list(encoders["fuel"].classes_))
seller_type = st.selectbox("Seller Type", list(encoders["seller_type"].classes_))
transmission = st.selectbox("Transmission", list(encoders["transmission"].classes_))
owner = st.selectbox("Owner", list(encoders["owner"].classes_))
seats = st.number_input("Seats", min_value=2, max_value=10, value=5)
engine_cc = st.number_input("Engine (CC)", min_value=600, max_value=5000, value=1200)

st.divider()

if st.button("Predict Price", type="primary"):
    try:
        # Build a single-row DataFrame with raw values, in training column order
        raw_input = {
            "Car_Name": car_name,
            "year": year,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "seats": seats,
            "Engine (CC)": engine_cc,
        }
        input_df = pd.DataFrame([raw_input])[feature_columns]

        # Apply the SAME label encoders used in training
        for col, le in encoders.items():
            input_df[col] = le.transform(input_df[col])

        # Apply the SAME scaler used in training
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_columns)

        # Predict
        prediction = model.predict(input_scaled)[0]

        st.success(f"### Estimated Selling Price: ₹{prediction:,.0f}")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.divider()
st.caption("Model: Decision Tree Regressor | Built with scikit-learn + Streamlit")
