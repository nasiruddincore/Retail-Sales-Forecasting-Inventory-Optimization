import pandas as pd
import numpy as np
import os

def generate_retail_data(days=365):
    print("🛒 Generating synthetic retail sales dataset...")
    np.random.seed(42)
    
    dates = pd.date_range(start="2024-01-01", periods=days, freq="D")
    stores = ['S001', 'S002']
    items = ['SKU_101', 'SKU_102', 'SKU_103'] 
    
    data = []
    for store in stores:
        for item in items:
            base_sales = 50 if item == 'SKU_101' else (20 if item == 'SKU_102' else 5)
            
            for d in dates:
                dow = d.dayofweek
                weekend_boost = 1.5 if dow >= 5 else 1.0 # Higher sales on weekends
                on_promo = 1 if np.random.rand() > 0.85 else 0
                promo_lift = 2.0 if on_promo else 1.0
                
                # Simulate stockout (item unavailable)
                stockout_flag = 1 if np.random.rand() > 0.95 else 0
                
                if stockout_flag:
                    qty = 0
                else:
                    qty = int(np.random.poisson(lam=base_sales) * weekend_boost * promo_lift)
                
                price = 19.99 if item == 'SKU_101' else (49.99 if item == 'SKU_102' else 9.99)
                if on_promo: price *= 0.8
                
                data.append({
                    'store_id': store, 'item_id': item, 'date': d, 
                    'qty_sold': qty, 'on_promo': on_promo, 
                    'price': round(price, 2), 'stockout_flag': stockout_flag
                })
                
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/retail_timeseries.csv', index=False)
    print(f"✅ Dataset created: data/retail_timeseries.csv ({len(df)} records)")

if __name__ == "__main__":
    generate_retail_data()