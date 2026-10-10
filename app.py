from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CarDekho Price Predictor", page_icon="🚗")
st.title("CarDekho Used Car Price Predictor")
st.write("Enter the car details to get an estimated selling price.")

bundle_path = Path(__file__).parent / "car_price_bundle.joblib"
bundle = joblib.load(bundle_path)

with st.form("car_details"):
    year = st.number_input(
        "Year",
        min_value=bundle["year_min"],
        max_value=bundle["year_max"],
        value=bundle["year_max"],
        step=1,
    )
    km_driven = st.number_input(
        "Kilometers driven",
        min_value=bundle["km_min"],
        max_value=bundle["km_max"],
        value=min(max(60000, bundle["km_min"]), bundle["km_max"]),
        step=1000,
    )
    fuel = st.selectbox("Fuel", bundle["options"]["fuel"])
    seller_type = st.selectbox("Seller type", bundle["options"]["seller_type"])
    transmission = st.selectbox("Transmission", bundle["options"]["transmission"])
    owner = st.selectbox("Owner history", bundle["options"]["owner"])
    brand = st.selectbox("Brand", bundle["options"]["brand"])

    submitted = st.form_submit_button("Estimate price")

if submitted:
    car = pd.DataFrame([{
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "brand": brand,
    }])

    estimated_price = bundle["model"].predict(car)[0]
    st.metric("Estimated selling price", f"₹{estimated_price:,.0f}")
    st.caption("This is an estimate based on the dataset and may differ from an actual sale price.")
