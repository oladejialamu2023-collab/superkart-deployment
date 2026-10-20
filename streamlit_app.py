import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart Sales Forecast")

api_root = st.sidebar.text_input("Backend URL", "http://backend:7860")

with st.st.form("forecast_form") if hasattr(st, "form") else st.form("forecast_form"):
    payload = {
        "Product_Weight": st.number_input("Product Weight", value=12.66),
        "Product_Sugar_Content": st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"]),
        "Product_Allocated_Area": st.number_input("Allocated Area", value=0.027),
        "Product_MRP": st.number_input("Product MRP", value=117.08),
        "Store_Size": st.selectbox("Store Size", ["Small", "Medium", "High"]),
        "Store_Location_City_Type": st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"]),
        "Store_Type": st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"]),
        "Product_Id_char": st.selectbox("Product Prefix", ["FD", "DR", "NC"]),
        "Store_Age_Years": st.number_input("Store Age", value=16),
        "Product_Type_Category": st.selectbox("Product Category", ["Perishables", "Non Perishables"])
    }
    submitted = st.form_submit_button("Predict Sales")

if submitted:
    try:
        response = requests.post(f"{api_root}/v1/rental", json=payload, timeout=30)
        response.raise_for_status()
        st.metric("Predicted Product-Store Sales", f"{response.json()['predicted_sales']:,.2f}")
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")

st.subheader("Batch Prediction")
upload = st.file_uploader("Upload a CSV with the required model features", type="csv")

if upload is not None and st.button("Run Batch Prediction"):
    response = requests.post(f"{api_root}/v1/rentalbatch", files={"file": upload.getvalue()}, timeout=60)
    if response.ok:
        st.dataframe(pd.DataFrame.from_dict(response.json(), orient="index", columns=["Predicted_Sales"]))
    else:
        st.error(response.text)
