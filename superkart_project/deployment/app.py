import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="vijayendras/superkart-sales-model", filename="SuperKart-Sales-Forecast-model-v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("SuperKart Sales Forecast App")
st.write("""
This application predicts the likelihood of a machine failing based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
Product_Weight = st.number_input("Weight of the Product", min_value=0.1, value=0.1)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "No Sugar", "Regular"])
Product_Allocated_Area = st.number_input("Product Allocated Area ( Ratio of the area in store allocated to the product)", min_value=0.01, value=0.01)
Product_Type = st.selectbox("Product Type (kind of product)", ["Fruits and Vegetables", "Snack Foods", "Dairy", "Frozen Foods", "Baking Goods", "Canned", "Meat", "Soft Drinks", "Breads", "Hard Drinks", "Starchy Foods", "Breakfast", "Seafood"])
Product_MRP = st.number_input("MRP of the Product", min_value=0.1, value=0.1)
Store_Id = st.selectbox("Store Id", ["OUT001", "OUT002", "OUT003", "OUT004"])
Store_Size = st.selectbox("Store Size", ["High", "Medium", "Low"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"])

# Assemble input into DataFrame

input_data = pd.DataFrame([{
    'Product_Weight': Product_Weight,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_Type': Product_Type,
    'Product_MRP': Product_MRP,
    'Store_Id': Store_Id,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type
}])

if st.button("Predict superkart Package Taken"):
    prediction = model.predict(input_data)[0]
    result = target
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
