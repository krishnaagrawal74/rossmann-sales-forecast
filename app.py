import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# ---------- Load model + reference data ----------
@st.cache_resource
def load_model():
    return joblib.load('models/xgb_model.pkl')

@st.cache_data
def load_store_summary():
    return joblib.load('models/store_summary.pkl')

@st.cache_data
def load_latest_state():
    return pd.read_csv('data/latest_store_state.csv')

model = load_model()
store_summary = load_store_summary()
latest_state = load_latest_state()

st.set_page_config(page_title="Rossmann Sales Forecaster", layout="centered")
st.title("Rossmann Store Sales Forecaster")
st.caption("Predicts daily store-level sales revenue using an XGBoost model trained on 2013–2015 historical data.")

# ---------- Sidebar inputs ----------
st.sidebar.header("Prediction Inputs")

store_id = st.sidebar.selectbox("Select Store", sorted(latest_state['Store'].unique()))
selected_date = st.sidebar.date_input("Select Date", datetime.date(2015, 8, 1))
promo = st.sidebar.selectbox("Promo running that day?", ["No", "Yes"])
school_holiday = st.sidebar.selectbox("School Holiday?", ["No", "Yes"])
state_holiday = st.sidebar.selectbox("State Holiday?", ["None", "Public holiday", "Easter", "Christmas"])

# ---------- Build feature row ----------
store_row = latest_state[latest_state['Store'] == store_id].iloc[0]

day_of_week = selected_date.isoweekday()  # 1=Mon ... 7=Sun
is_weekend = 1 if day_of_week >= 6 else 0
year = selected_date.year
month = selected_date.month
day = selected_date.day
week_of_year = selected_date.isocalendar()[1]

state_holiday_map = {"None": None, "Public holiday": "a", "Easter": "b", "Christmas": "c"}
sh = state_holiday_map[state_holiday]

input_dict = {
    'Store': store_id,
    'DayOfWeek': day_of_week,
    'Promo': 1 if promo == "Yes" else 0,
    'SchoolHoliday': 1 if school_holiday == "Yes" else 0,
    'CompetitionDistance': store_row['CompetitionDistance'],
    'Promo2': store_row['Promo2'],
    'Year': year,
    'Month': month,
    'Day': day,
    'WeekOfYear': week_of_year,
    'IsWeekend': is_weekend,
    'Sales_Lag_7': store_row['Sales_Lag_7'],
    'Sales_Lag_14': store_row['Sales_Lag_14'],
    'Sales_RollingMean_7': store_row['Sales_RollingMean_7'],
    'Sales_RollingMean_30': store_row['Sales_RollingMean_30'],
    'StoreType_b': store_row['StoreType_b'],
    'StoreType_c': store_row['StoreType_c'],
    'StoreType_d': store_row['StoreType_d'],
    'Assortment_b': store_row['Assortment_b'],
    'Assortment_c': store_row['Assortment_c'],
    'StateHoliday_a': 1 if sh == "a" else 0,
    'StateHoliday_b': 1 if sh == "b" else 0,
    'StateHoliday_c': 1 if sh == "c" else 0,
}

X_input = pd.DataFrame([input_dict])

# match training column order
X_input = X_input[model.get_booster().feature_names]

# ---------- Predict ----------
if st.sidebar.button("Predict Sales"):
    prediction = model.predict(X_input)[0]

    st.subheader(f"Predicted Sales for Store {store_id}")
    st.metric(label=f"Predicted revenue on {selected_date.strftime('%A, %d %b %Y')}", value=f"€{prediction:,.0f}")

    # Store cluster context
    cluster_info = store_summary[store_summary['Store'] == store_id]
    if not cluster_info.empty:
        cluster_num = cluster_info['Cluster'].values[0]
        avg_sales = cluster_info['avg_sales'].values[0]
        promo_lift = cluster_info['promo_lift'].values[0]

        cluster_names = {
            0: "Strong Performer, Promo-Sensitive",
            1: "Steady, Low-Volume Store",
            2: "Flagship / High-Volume Store",
            3: "Core / Average Store"
        }

        st.subheader("Store Segment")
        st.write(f"**Segment:** {cluster_names.get(cluster_num, cluster_num)}")
        st.write(f"**Historical average daily sales:** €{avg_sales:,.0f}")
        st.write(f"**Typical promo lift:** €{promo_lift:,.0f}")

    st.caption(
        "Note: Lag/rolling sales features use each store's most recent known values from the "
        "2013–2015 dataset (not live data) — in production these would be pulled from a live sales database."
    )
else:
    st.info("Set your inputs on the left and click **Predict Sales**.")

st.divider()
st.caption("Model: XGBoost Regressor | Test MAE: €668 (~9.6% of average daily sales) | Trained on Rossmann Store Sales dataset (Kaggle)")
