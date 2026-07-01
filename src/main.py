# Retail Demand Forecasting API
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="Retail Demand Forecasting API",
    description="Predict product demand, get inventory recommendations and reorder alerts",
    version="2.0"
)

# Load model
with open("../models/advanced_lgbm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load data
inventory_df = pd.read_csv("../data/inventory_optimization.csv")
reorder_df = pd.read_csv("../data/reorder_schedule.csv")
revenue_df = pd.read_csv("../data/revenue_forecast_30days.csv")
abc_df = pd.read_csv("../data/abc_inventory_analysis.csv")


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
        "message": "Retail Demand Forecasting API v2.0",
        "endpoints": [
            "/predict",
            "/inventory/{item_id}",
            "/inventory/category/{cat_id}",
            "/reorder/urgent",
            "/reorder/summary",
            "/revenue/forecast",
            "/abc/{item_id}",
            "/health"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Advanced LightGBM",
        "version": "2.0"
    }


@app.post("/predict")
def predict_demand(data: DemandInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    prediction = max(0, prediction)
    return {
        "predicted_demand": round(float(prediction), 2),
        "estimated_revenue": round(float(prediction * data.avg_price), 2),
        "recommended_monthly_order": round(float(prediction * 30), 2)
    }


@app.get("/inventory/{item_id}")
def get_inventory(item_id: str):
    item_data = inventory_df[inventory_df['item_id'] == item_id]
    if item_data.empty:
        return {"error": "Item not found"}
    return {"item_id": item_id, "data": item_data.to_dict(orient='records')}


@app.get("/inventory/category/{cat_id}")
def get_category_inventory(cat_id: str):
    cat_data = inventory_df[inventory_df['cat_id'] == cat_id]
    if cat_data.empty:
        return {"error": "Category not found"}
    return {
        "category": cat_id,
        "total_items": len(cat_data),
        "avg_daily_demand": round(cat_data['avg_daily_demand'].mean(), 2),
        "avg_safety_stock": round(cat_data['safety_stock'].mean(), 2),
        "avg_order_quantity": round(cat_data['order_quantity'].mean(), 2),
        "avg_monthly_revenue": round(cat_data['monthly_revenue'].mean(), 2)
    }


@app.get("/reorder/urgent")
def get_urgent_reorders():
    urgent = reorder_df[reorder_df['priority'] == 'Urgent'].nsmallest(20, 'days_until_reorder')
    return {
        "total_urgent_items": len(reorder_df[reorder_df['priority'] == 'Urgent']),
        "top_20_urgent": urgent[['item_id', 'store_id', 'cat_id', 'order_quantity', 'days_until_reorder']].to_dict(orient='records')
    }


@app.get("/reorder/summary")
def get_reorder_summary():
    return {
        "urgent": int((reorder_df['priority'] == 'Urgent').sum()),
        "soon": int((reorder_df['priority'] == 'Soon').sum()),
        "normal": int((reorder_df['priority'] == 'Normal').sum()),
        "total_items": len(reorder_df)
    }


@app.get("/revenue/forecast")
def get_revenue_forecast():
    return {
        "total_30day_revenue": round(float(revenue_df['total_monthly_revenue'].sum()), 2),
        "by_category": revenue_df[['cat_id', 'total_monthly_revenue', 'revenue_pct']].to_dict(orient='records')
    }


@app.get("/abc/{item_id}")
def get_abc_classification(item_id: str):
    item_data = abc_df[abc_df['item_id'] == item_id]
    if item_data.empty:
        return {"error": "Item not found"}
    return {
        "item_id": item_id,
        "abc_class": item_data['abc_class'].values[0],
        "projected_annual_value": round(float(item_data['projected_annual_value'].mean()), 2),
        "stockout_risk_score": round(float(item_data['stockout_risk_score'].mean()), 2)
    }