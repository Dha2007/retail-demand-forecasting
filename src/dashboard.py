import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Retail Demand Forecasting & Inventory Optimization", layout="wide")

# Resolve data directory relative to this file's repo root (src/.. -> repo root -> data)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


@st.cache_data
def load_csv(filename):
    return pd.read_csv(os.path.join(DATA_DIR, filename))


st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    [
        "Overview",
        "Sales Trends",
        "Demand Forecasting",
        "Price & Promotions",
        "Inventory Optimization",
        "Reorder Alerts",
    ],
)

st.title("Retail Demand Forecasting & Inventory Optimization")

# ---------------------------------------------------------------- OVERVIEW --
if page == "Overview":
    st.header("Overview")

    dept_sales = load_csv("department_sales.csv")
    state_cat_sales = load_csv("state_category_sales.csv")
    monthly_sales = load_csv("monthly_sales.csv")

    total_sales = monthly_sales["monthly_sales"].sum()
    avg_monthly = monthly_sales.groupby(["year", "month"])["monthly_sales"].sum().mean()
    top_dept = dept_sales.sort_values("total_sales", ascending=False).iloc[0]
    top_state = state_cat_sales.groupby("state_id")["total_sales"].sum().idxmax()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Units Sold", f"{total_sales:,.0f}")
    col2.metric("Avg Monthly Sales", f"{avg_monthly:,.0f}")
    col3.metric("Top Department", str(top_dept["dept_id"]))
    col4.metric("Top State", str(top_state))

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sales by Department")
        fig = px.bar(
            dept_sales.sort_values("total_sales", ascending=False),
            x="dept_id", y="total_sales", color="dept_id",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Sales by State & Category")
        fig2 = px.bar(state_cat_sales, x="state_id", y="total_sales", color="cat_id", barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------- SALES TRENDS --
elif page == "Sales Trends":
    st.header("Sales Trends")

    monthly_sales = load_csv("monthly_sales.csv")
    monthly_store_sales = load_csv("monthly_store_sales.csv")
    yearly_price_trend = load_csv("yearly_price_trend.csv")

    monthly_sales["period"] = (
        monthly_sales["year"].astype(str) + "-" + monthly_sales["month"].astype(str).str.zfill(2)
    )
    st.subheader("Monthly Sales by Category")
    fig = px.line(monthly_sales.sort_values(["year", "month"]), x="period", y="monthly_sales", color="cat_id")
    st.plotly_chart(fig, use_container_width=True)

    monthly_store_sales["period"] = (
        monthly_store_sales["year"].astype(str) + "-" + monthly_store_sales["month"].astype(str).str.zfill(2)
    )
    st.subheader("Monthly Sales by Store")
    fig2 = px.line(monthly_store_sales.sort_values(["year", "month"]), x="period", y="monthly_sales", color="store_id")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Yearly Price Trend by Category")
    fig3 = px.line(yearly_price_trend.sort_values("year"), x="year", y="sell_price", color="cat_id", markers=True)
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------- DEMAND FORECASTING --
elif page == "Demand Forecasting":
    st.header("Demand Forecasting")

    prophet_forecast = load_csv("prophet_forecast.csv")
    store_forecast_results = load_csv("store_forecast_results.csv")

    prophet_forecast["ds"] = pd.to_datetime(prophet_forecast["ds"])
    st.subheader("Prophet Forecast (with confidence interval)")
    fig = px.line(prophet_forecast, x="ds", y="yhat", labels={"yhat": "Predicted Sales"})
    fig.add_scatter(x=prophet_forecast["ds"], y=prophet_forecast["yhat_upper"],
                     mode="lines", name="Upper Bound", line=dict(dash="dot"))
    fig.add_scatter(x=prophet_forecast["ds"], y=prophet_forecast["yhat_lower"],
                     mode="lines", name="Lower Bound", line=dict(dash="dot"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("LightGBM Model Performance by Store")
    st.dataframe(store_forecast_results.sort_values("RMSE"), use_container_width=True)

    col1, col2 = st.columns(2)
    best_store = store_forecast_results.loc[store_forecast_results["RMSE"].idxmin(), "store_id"]
    col1.metric("Best Store (lowest RMSE)", str(best_store))
    col2.metric("Avg MAE Across Stores", f"{store_forecast_results['MAE'].mean():.2f}")

# ------------------------------------------------------- PRICE & PROMOTIONS --
elif page == "Price & Promotions":
    st.header("Price & Promotions Analysis")

    price_sensitivity = load_csv("price_sensitivity.csv")
    category_price_stats = load_csv("category_price_stats.csv")
    snap_impact = load_csv("snap_impact_analysis.csv")
    event_impact = load_csv("event_impact_analysis.csv")

    st.subheader("Price Sensitivity by Category")
    st.caption("Correlation between price and sales volume — negative values mean sales drop as price rises.")
    fig = px.bar(price_sensitivity, x="cat_id", y="price_sales_correlation", color="cat_id")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Category Price Stats")
    st.dataframe(category_price_stats, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("SNAP Day Impact on Sales")
        snap_impact["label"] = snap_impact["is_snap"].map({0: "Non-SNAP Day", 1: "SNAP Day"})
        fig2 = px.bar(snap_impact, x="label", y="avg_sales", color="label")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.subheader("Event Day Impact on Sales")
        event_impact["label"] = event_impact["has_event"].map({0: "No Event", 1: "Event Day"})
        fig3 = px.bar(event_impact, x="label", y="avg_sales", color="label")
        st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------- INVENTORY OPTIMIZATION --
elif page == "Inventory Optimization":
    st.header("Inventory Optimization")

    abc = load_csv("abc_inventory_analysis.csv")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total SKUs Analyzed", f"{len(abc):,}")
    high_risk_cutoff = abc["stockout_risk_score"].quantile(0.9)
    col2.metric("High Stockout-Risk Items", f"{(abc['stockout_risk_score'] > high_risk_cutoff).sum():,}")
    if "A" in abc["abc_class"].unique():
        col3.metric("Class A (high-value) Items", f"{(abc['abc_class'] == 'A').sum():,}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ABC Classification Distribution")
        fig = px.pie(abc, names="abc_class", title="Inventory Value Segmentation")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Stockout Risk Score Distribution")
        fig2 = px.histogram(abc, x="stockout_risk_score", nbins=30)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 20 Items by Projected Annual Value")
    top_items = abc.sort_values("projected_annual_value", ascending=False).head(20)
    st.dataframe(
        top_items[["item_id", "store_id", "cat_id", "abc_class", "projected_annual_value",
                   "reorder_point", "safety_stock", "order_quantity"]],
        use_container_width=True,
    )

    st.subheader("Reorder Point vs Avg Daily Demand")
    sample = abc.sample(min(2000, len(abc)), random_state=42)
    fig3 = px.scatter(sample, x="avg_daily_demand", y="reorder_point", color="abc_class")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------------------- REORDER ALERTS --
elif page == "Reorder Alerts":
    st.header("🚨 Reorder Alerts & Revenue Forecast")

    reorder_schedule = load_csv("reorder_schedule.csv")
    revenue_forecast = load_csv("revenue_forecast_30days.csv")

    col1, col2, col3 = st.columns(3)
    with col1:
        urgent = (reorder_schedule["priority"] == "Urgent").sum()
        st.metric("Urgent Reorders", urgent)
    with col2:
        soon = (reorder_schedule["priority"] == "Soon").sum()
        st.metric("Reorder Soon", soon)
    with col3:
        total_rev = revenue_forecast["total_monthly_revenue"].sum()
        st.metric("30-Day Revenue Forecast", f"${total_rev:,.0f}")

    st.divider()

    st.subheader("Top 15 Urgent Reorder Items")
    urgent_items = reorder_schedule[reorder_schedule["priority"] == "Urgent"].nsmallest(
        15, "days_until_reorder")[
        ["item_id", "store_id", "cat_id", "order_quantity", "days_until_reorder"]]
    st.dataframe(urgent_items, use_container_width=True)

    st.subheader("30-Day Revenue Forecast by Category")
    fig = px.bar(revenue_forecast, x="cat_id", y="total_monthly_revenue", color="cat_id")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Reorder Priority Distribution")
    priority_counts = reorder_schedule["priority"].value_counts().reset_index()
    priority_counts.columns = ["priority", "count"]
    fig2 = px.bar(priority_counts, x="priority", y="count", color="priority",
                  color_discrete_map={"Urgent": "#DC2626", "Soon": "#F59E0B", "Normal": "#059669"})
    st.plotly_chart(fig2, use_container_width=True)