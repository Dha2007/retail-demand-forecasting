# Retail Demand Forecasting Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Retail Demand Forecasting Dashboard",
    page_icon="🛒",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("../data/featured_sample.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['LTV'] = df['sales'] * df['avg_price']
    return df

@st.cache_data
def load_inventory():
    return pd.read_csv("../data/inventory_optimization.csv")

@st.cache_data
def load_store_results():
    return pd.read_csv("../data/store_forecast_results.csv")

@st.cache_resource
def load_model():
    with open("../models/advanced_lgbm_model.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
inventory = load_inventory()
store_results = load_store_results()
model = load_model()

# Title
st.title("🛒 Retail Demand Forecasting & Inventory Optimization")
st.markdown("---")

# KPI Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Items", "30,490")
with col2:
    st.metric("Total Stores", "10")
with col3:
    st.metric("Avg Daily Sales", "34,341")
with col4:
    st.metric("Best Model MAE", "1.18")

st.markdown("---")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", [
    "Overview",
    "Sales Analysis",
    "Inventory Optimization",
    "Demand Forecast",
    "Predict Demand"
])

if page == "Overview":
    st.header("Overview")

    col1, col2 = st.columns(2)

    with col1:
        # Sales by category
        cat_sales = df.groupby('cat_id')['sales'].sum().reset_index()
        fig = px.pie(cat_sales, values='sales', names='cat_id',
                    title="Sales by Category",
                    color_discrete_sequence=['#0891B2', '#7C3AED', '#059669'])
        st.plotly_chart(fig)

    with col2:
        # Sales by state
        state_sales = df.groupby('state_id')['sales'].sum().reset_index()
        fig = px.bar(state_sales, x='state_id', y='sales',
                    title="Sales by State",
                    color='state_id',
                    color_discrete_sequence=['#0891B2', '#7C3AED', '#059669'])
        st.plotly_chart(fig)

elif page == "Sales Analysis":
    st.header("Sales Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Monthly sales trend
        monthly = df.groupby(['year', 'month'])['sales'].sum().reset_index()
        fig = px.line(monthly, x='month', y='sales', color='year',
                     title="Monthly Sales by Year")
        st.plotly_chart(fig)

    with col2:
        # Sales by weekday
        weekday_sales = df.groupby('weekday')['sales'].mean().reset_index()
        fig = px.bar(weekday_sales, x='weekday', y='sales',
                    title="Average Sales by Weekday",
                    color_discrete_sequence=['#0891B2'])
        st.plotly_chart(fig)

    # Store sales
    store_sales = df.groupby('store_id')['sales'].sum().reset_index()
    fig = px.bar(store_sales, x='store_id', y='sales',
                title="Total Sales by Store",
                color='store_id')
    st.plotly_chart(fig)

elif page == "Inventory Optimization":
    st.header("Inventory Optimization")

    col1, col2 = st.columns(2)

    with col1:
        # Average order quantity by category
        cat_inv = inventory.groupby('cat_id')['order_quantity'].mean().reset_index()
        fig = px.bar(cat_inv, x='cat_id', y='order_quantity',
                    title="Average Order Quantity by Category",
                    color_discrete_sequence=['#0891B2'])
        st.plotly_chart(fig)

    with col2:
        # Safety stock by category
        cat_safety = inventory.groupby('cat_id')['safety_stock'].mean().reset_index()
        fig = px.bar(cat_safety, x='cat_id', y='safety_stock',
                    title="Average Safety Stock by Category",
                    color_discrete_sequence=['#7C3AED'])
        st.plotly_chart(fig)

    # Top 10 items needing restocking
    st.subheader("Top 10 Items Needing Restocking")
    top_items = inventory.nlargest(10, 'reorder_point')[
        ['item_id', 'store_id', 'avg_daily_demand',
         'safety_stock', 'reorder_point', 'order_quantity']]
    st.dataframe(top_items)

elif page == "Demand Forecast":
    st.header("Demand Forecast Results")

    col1, col2 = st.columns(2)

    with col1:
        # Store MAE results
        fig = px.bar(store_results, x='store_id', y='MAE',
                    title="Forecast MAE by Store",
                    color='MAE',
                    color_continuous_scale='viridis')
        st.plotly_chart(fig)

    with col2:
        # Store RMSE results
        fig = px.bar(store_results, x='store_id', y='RMSE',
                    title="Forecast RMSE by Store",
                    color='RMSE',
                    color_continuous_scale='plasma')
        st.plotly_chart(fig)

    # Model comparison
    st.subheader("Model Comparison")
    model_comp = pd.DataFrame({
        'Model': ['Prophet', 'Basic LightGBM', 'Advanced LightGBM'],
        'MAE': [12.0, 1.2, 1.18],
        'RMSE': [17.04, 2.49, 2.46]
    })
    st.dataframe(model_comp)

    fig = px.bar(model_comp, x='Model', y='MAE',
                title="Model Comparison - MAE",
                color_discrete_sequence=['#0891B2'])
    st.plotly_chart(fig)

elif page == "Predict Demand":
    st.header("Predict Product Demand")

    col1, col2 = st.columns(2)

    with col1:
        month = st.slider("Month", 1, 12, 6)
        year = st.slider("Year", 2011, 2016, 2015)
        day = st.slider("Day", 1, 31, 15)
        week = st.slider("Week", 1, 52, 25)
        wday = st.slider("Day of Week", 1, 7, 3)
        quarter = st.slider("Quarter", 1, 4, 2)

    with col2:
        is_weekend = st.selectbox("Is Weekend?", [0, 1])
        is_month_start = st.selectbox("Is Month Start?", [0, 1])
        is_month_end = st.selectbox("Is Month End?", [0, 1])
        has_event = st.selectbox("Has Event?", [0, 1])
        is_snap = st.selectbox("Is SNAP Day?", [0, 1])
        avg_price = st.number_input("Average Price ($)", value=3.5)

    col3, col4 = st.columns(2)
    with col3:
        sales_lag_1 = st.number_input("Sales Yesterday", value=5)
        sales_lag_7 = st.number_input("Sales 7 Days Ago", value=5)
        sales_lag_28 = st.number_input("Sales 28 Days Ago", value=5)

    with col4:
        sales_rolling_7 = st.number_input("7 Day Rolling Avg", value=5.0)
        sales_rolling_28 = st.number_input("28 Day Rolling Avg", value=5.0)
        cat_encoded = st.selectbox("Category", [0, 1, 2],
                                   format_func=lambda x: ['FOODS', 'HOBBIES', 'HOUSEHOLD'][x])
        store_encoded = st.selectbox("Store", list(range(10)))
        state_encoded = st.selectbox("State", [0, 1, 2],
                                     format_func=lambda x: ['CA', 'TX', 'WI'][x])
        dept_encoded = st.selectbox("Department", list(range(7)))

    if st.button("Predict Demand"):
        input_data = pd.DataFrame([{
            'month': month, 'year': year, 'wday': wday,
            'day': day, 'week': week, 'quarter': quarter,
            'is_weekend': is_weekend, 'is_month_start': is_month_start,
            'is_month_end': is_month_end, 'has_event': has_event,
            'is_snap': is_snap, 'avg_price': avg_price,
            'sales_lag_1': sales_lag_1, 'sales_lag_7': sales_lag_7,
            'sales_lag_28': sales_lag_28, 'sales_rolling_7': sales_rolling_7,
            'sales_rolling_28': sales_rolling_28, 'cat_encoded': cat_encoded,
            'store_encoded': store_encoded, 'state_encoded': state_encoded,
            'dept_encoded': dept_encoded
        }])

        prediction = model.predict(input_data)[0]
        prediction = max(0, prediction)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Demand", f"{round(prediction, 0)} units")
        with col2:
            st.metric("Estimated Revenue", f"${round(prediction * avg_price, 2)}")
        with col3:
            st.metric("Recommended Order", f"{round(prediction * 30, 0)} units/month")