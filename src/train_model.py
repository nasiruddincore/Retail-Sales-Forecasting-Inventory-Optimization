import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

def make_features(g):
    g = g.sort_values("date")
    for L in (1, 7, 14): 
        g[f"lag_{L}"] = g["qty_sold"].shift(L)
    for W in (7, 14, 28):
        g[f"rollmean_{W}"] = g["qty_sold"].shift(1).rolling(W).mean()
        g[f"rollstd_{W}"]  = g["qty_sold"].shift(1).rolling(W).std()
    g["dow"] = g["date"].dt.dayofweek
    return g

def train_forecaster():
    print("🧠 Loading data and building features...")
    df = pd.read_csv("data/retail_timeseries.csv", parse_dates=["date"])
    
    # Remove stockout-censored rows (clean demand)
    df = df[df["stockout_flag"] == 0].copy()
    
    # Apply feature engineering per SKU-store
    df = df.groupby(["store_id", "item_id"], group_keys=False).apply(make_features).dropna()
    
    feat_cols = [c for c in df.columns if c not in ["qty_sold", "date", "store_id", "item_id"]]
    X, y = df[feat_cols], df["qty_sold"]
    groups = df["store_id"].astype(str) + "_" + df["item_id"].astype(str)
    
    gss = GroupShuffleSplit(test_size=0.2, random_state=13)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    Xtr, Xte, ytr, yte = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]
    
    print("🏋️ Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=13)
    rf.fit(Xtr, ytr)
    
    pred = rf.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    resid_std = float(np.std(yte - pred)) # Used for safety stock uncertainty
    
    print(f"📊 Model MAE: {mae:.2f} units")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": rf, "features": feat_cols, "resid_std": resid_std}, "models/retail_forecast_model.pkl")
    print("✅ Model saved to models/retail_forecast_model.pkl")

if __name__ == "__main__":
    train_forecaster()