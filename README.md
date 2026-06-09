# Retail Demand Forecasting & Inventory Optimization

## Project Overview
An automated analytics platform for retail and supply chain teams. 
It utilizes time-series forecasting to analyze historical sales data, 
promotional events and seasonality to accurately predict future product 
demand and generate optimal inventory restocking schedules.

##  Team Members
- Rajarapu Dhanalakshmi (Team Leader)
- Mounya Sai Sree
- Sathwika
- Divya Sri
- Sindhu

##  Dataset
- **Source:** M5 Forecasting Dataset (Walmart historical sales data — Kaggle)
- **Size:** 30,490 items × 1,913 days
- **Stores:** 10 stores across 3 states (CA, TX, WI)
- **Categories:** FOODS, HOUSEHOLD, HOBBIES

##  Key Insights
1. FOODS is top category with 45M total sales
2. Best sales month is March
3. Best sales year is 2015
4. SNAP days increase sales by 8.48%
5. California has highest sales of 28.6M
6. Average daily sales is 34,341 units

##  Tech Stack
- **Language:** Python, SQL
- **Forecasting:** Facebook Prophet, LightGBM
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Visualization:** Plotly, Streamlit
- **Database:** PostgreSQL (planned)

##  Project Structure
retail-demand-forecasting/
├── data/
│   ├── calendar.csv
│   ├── calendar_clean.csv
│   ├── featured_sample.csv
│   ├── prophet_forecast.csv
│   └── inventory_optimization.csv
├── models/
│   └── lightgbm_model.pkl
├── notebooks/
│   ├── eda.ipynb
│   ├── data_quality.ipynb
│   ├── feature_engineering.ipynb
│   ├── forecasting.ipynb
│   ├── lightgbm_model.ipynb
│   └── inventory_optimization.ipynb
└── README.md

## Weekly Progress
### Week 1 
- Day 1: EDA and sales analysis
- Day 2: Data quality checks and price analysis
- Day 3: Feature engineering and lag features
- Day 4: Time series forecasting with Prophet
- Day 5: LightGBM demand forecasting
- Day 6: Inventory optimization
- Day 7: Final cleanup and documentation

##  Model Performance
| Model | MAE | RMSE |
|-------|-----|------|
| Prophet | 12.0 | 17.04 |
| LightGBM | 1.2 | 2.49 |

##  How to Run
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run notebooks in order