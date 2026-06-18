# Retail Demand Forecasting API
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Retail Demand Forecasting API",
              description="Predict product demand and get inventory recommendations",
              version="1.0")

# Load model
with open("../models/advanced_lgbm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load inventory data
inventory_df = pd.read_csv("../data/inventory_optimization.csv")


class DemandInput(BaseModel):
    month: int
    year: int
    wday: int
    day: int
    week: int
    quarter: int
    is_weekend: int
    is_month_start: int
    is_month_end: int
    has_event: int
    is_snap: int
    avg_price: float
    sales_lag_1: float
    sales_lag_7: float
    sales_lag_28: float
    sales_rolling_7: float
    sales_rolling_28: float
    cat_encoded: int
    store_encoded: int
    state_encoded: int
    dept_encoded: int


@app.get("/")
def home():
    return {
        "message": "Retail Demand Forecasting API",
        "endpoints": ["/predict", "/inventory/{item_id}", "/health"]
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": "Advanced LightGBM"}


@app.post("/predict")
def predict_demand(data: DemandInput):
    input_df = pd.DataFrame([data.dict()])
    
    prediction = model.predict(input_df)[0]
    prediction = max(0, prediction)
    
    estimated_revenue = prediction * data.avg_price
    monthly_order = prediction * 30
    
    return {
        "predicted_demand": round(float(prediction), 2),
        "estimated_revenue": round(float(estimated_revenue), 2),
        "recommended_monthly_order": round(float(monthly_order), 2)
    }


@app.get("/inventory/{item_id}")
def get_inventory(item_id: str):
    item_data = inventory_df[inventory_df['item_id'] == item_id]
    
    if item_data.empty:
        return {"error": "Item not found"}
    
    result = item_data.to_dict(orient='records')
    return {"item_id": item_id, "data": result}


@app.get("/inventory/category/{cat_id}")
def get_category_inventory(cat_id: str):
    cat_data = inventory_df[inventory_df['cat_id'] == cat_id]
    
    if cat_data.empty:
        return {"error": "Category not found"}
    
    summary = {
        "category": cat_id,
        "total_items": len(cat_data),
        "avg_daily_demand": round(cat_data['avg_daily_demand'].mean(), 2),
        "avg_safety_stock": round(cat_data['safety_stock'].mean(), 2),
        "avg_order_quantity": round(cat_data['order_quantity'].mean(), 2),
        "avg_monthly_revenue": round(cat_data['monthly_revenue'].mean(), 2)
    }
    return summary