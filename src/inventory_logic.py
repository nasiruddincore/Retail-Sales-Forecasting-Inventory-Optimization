import numpy as np
from scipy.stats import norm

def calculate_inventory_policy(forecast_horizon, resid_std, on_hand, lead_time, 
                               service_level=0.95, annual_demand=10000, 
                               ordering_cost=50, unit_cost=20, holding_rate=0.2):
    """ Converts demand forecast into inventory restock metrics. """
    z = norm.ppf(service_level)
    
    # Demand during lead time
    mu_L = np.sum(forecast_horizon[:lead_time])
    # Standard deviation during lead time
    sigma_L = resid_std * (lead_time ** 0.5)
    
    SS = z * sigma_L                # Safety Stock
    ROP = mu_L + SS                 # Reorder Point
    H = unit_cost * holding_rate    # Holding Cost
    
    # Economic Order Quantity
    EOQ = np.sqrt((2 * annual_demand * ordering_cost) / H) if H > 0 else mu_L
    
    # How much to actually order today
    order_qty = max(0, max(EOQ, ROP - on_hand))
    
    return {
        "Expected Lead Time Demand": round(mu_L, 2),
        "Safety Stock": round(SS, 2),
        "Reorder Point (ROP)": round(ROP, 2),
        "EOQ": round(EOQ, 2),
        "Recommended Order Qty": int(order_qty)
    }