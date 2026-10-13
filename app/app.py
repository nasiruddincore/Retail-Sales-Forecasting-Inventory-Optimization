import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inventory_logic import calculate_inventory_policy
from src.train_model import make_features

st.set_page_config(page_title="Retail Inventory Optimizer", layout="wide")
st.title("📦 Retail Sales Forecasting & Inventory Optimization")

MODEL_PATH = "models/retail_forecast_model.pkl"
DATA_PATH = "data/retail_timeseries.csv"

if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Please run `src/simulate_data.py` and `src/train_model.py` first.")
    st.stop()

artifact = joblib.load(MODEL_PATH)
model, feat_cols, resid_std = artifact["model"], artifact["features"], artifact["resid_std"]
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

st.sidebar.header("⚙️ Supply Chain Controls")
store_sel = st.sidebar.selectbox("Select Store", df["store_id"].unique())
sku_sel = st.sidebar.selectbox("Select SKU", df["item_id"].unique())
on_hand = st.sidebar.number_input("Current On-Hand Stock", min_value=0, value=50)
lead_time = st.sidebar.slider("Supplier Lead Time (Days)", 1, 14, 3)
service_level = st.sidebar.selectbox("Target Service Level", [0.90, 0.95, 0.99], index=1)

sku_df = df[(df["store_id"] == store_sel) & (df["item_id"] == sku_sel)].copy()

if st.button("Generate Forecast & Restock Plan"):
    # Create features for inference
    feat_df = make_features(sku_df).dropna()
    recent_data = feat_df.tail(14) 
    
    # ML Prediction (Forecasting next 14 days baseline)
    predictions = np.maximum(0, model.predict(recent_data[feat_cols]))
    
    annual_demand = sku_df["qty_sold"].sum() * (365 / len(sku_df))
    policy = calculate_inventory_policy(
        predictions, resid_std, on_hand, lead_time, service_level, annual_demand
    )
    
    st.subheader(f"📊 Inventory Replenishment Plan: {store_sel} | {sku_sel}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recommended Order Qty", f"{policy['Recommended Order Qty']} units")
    c2.metric("Reorder Point (ROP)", f"{policy['Reorder Point (ROP)']} units")
    c3.metric("Safety Stock", f"{policy['Safety Stock']} units")
    c4.metric("EOQ", f"{policy['EOQ']} units")
    
    st.divider()
    st.subheader("📈 14-Day Demand Forecast Preview")
    chart_data = pd.DataFrame({"Day": [f"Day +{i+1}" for i in range(14)], "Forecasted Sales": predictions})
    st.bar_chart(chart_data.set_index("Day"))