import os
import pickle
import urllib.request
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Page Configuration & Modern Dark Styling
st.set_page_config(
    page_title="Power Consumption Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for Modern Purple/Dark Theme
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #7C4DFF;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6200EE 0%, #7C4DFF 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #7C4DFF 0%, #B388FF 100%);
        box-shadow: 0 4px 12px rgba(124, 77, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Household Power Consumption Intelligence")
st.markdown("Interactive machine learning pipeline with real-time inference.")

# 2. Cached Model & Data Loader Functions
@st.cache_resource
def load_saved_models():
    model_paths = {
        "Random Forest": "models/random_forest.pkl",
        "XGBoost": "models/xgboost.pkl",
        "LightGBM": "models/lightgbm.pkl",
        "Linear Regression": "models/linear_regression.pkl"
    }
    loaded_models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                loaded_models[name] = pickle.load(f)
    return loaded_models

@st.cache_data
def load_and_preprocess():
    zip_path = 'household_power_consumption.zip'
    txt_path = 'household_power_consumption.txt'
    
    if not os.path.exists(zip_path) and not os.path.exists(txt_path):
        url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
        urllib.request.urlretrieve(url, zip_path)

    if os.path.exists(zip_path):
        df = pd.read_csv(zip_path, sep=';', compression='zip', low_memory=False)
    else:
        df = pd.read_csv(txt_path, sep=';', low_memory=False)

    numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
    
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    df = df.drop(columns=['Date', 'Time']).set_index('datetime').sort_index()
    
    hourly_df = df[['Global_active_power']].resample('h').mean()
    hourly_df['hour'] = hourly_df.index.hour
    hourly_df['day_of_week'] = hourly_df.index.dayofweek
    hourly_df['month'] = hourly_df.index.month
    
    hourly_df['lag_1'] = hourly_df['Global_active_power'].shift(1)
    hourly_df['lag_24'] = hourly_df['Global_active_power'].shift(24)
    hourly_df['lag_168'] = hourly_df['Global_active_power'].shift(168)
    hourly_df['rolling_mean_24'] = hourly_df['Global_active_power'].shift(1).rolling(24).mean()
    hourly_df['rolling_mean_168'] = hourly_df['Global_active_power'].shift(1).rolling(168).mean()
    
    return hourly_df.dropna()

hourly_df = load_and_preprocess()
saved_models = load_saved_models()

# 3. Sidebar Engine Selector
st.sidebar.header("⚙️ Model Settings")
selected_model = st.sidebar.selectbox(
    "Select ML Engine",
    list(saved_models.keys()) if saved_models else ["Linear Regression"]
)

# 4. Tab Layout
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Live Manual Prediction",
    "📊 Dataset Overview", 
    "📈 Exploratory Analysis", 
    "🤖 Historical Evaluation", 
    "🚨 Anomaly Diagnostics"
])

# --- TAB 1: MANUAL INPUT INFERENCE ---
with tab1:
    st.subheader("💡 Manual Parameter Input & Real-Time Forecast")
    st.write("Enter custom environmental and lagged parameters to generate an instant inference call.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        in_hour = st.slider("Hour of Day (0-23)", 0, 23, 14)
        in_day = st.selectbox("Day of Week", options=list(range(7)), format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
        in_month = st.slider("Month (1-12)", 1, 12, 6)
    with col2:
        in_lag1 = st.number_input("Previous Hour Usage (lag_1 kW)", value=1.2, step=0.1)
        in_lag24 = st.number_input("24-Hour Prior Usage (lag_24 kW)", value=1.5, step=0.1)
        in_lag168 = st.number_input("7-Day Prior Usage (lag_168 kW)", value=1.1, step=0.1)
    with col3:
        in_roll24 = st.number_input("24-Hour Rolling Mean (kW)", value=1.3, step=0.1)
        in_roll168 = st.number_input("7-Day Rolling Mean (kW)", value=1.25, step=0.1)

    if st.button("Calculate Prediction"):
        if selected_model in saved_models:
            model = saved_models[selected_model]
            input_features = np.array([[in_hour, in_day, in_month, in_roll24, in_roll168, in_lag1, in_lag24, in_lag168]])
            prediction = model.predict(input_features)[0]
            
            st.markdown("---")
            m_col, g_col = st.columns([1, 2])
            m_col.metric("Predicted Global Active Power", f"{prediction:.3f} kW")
            
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Active Load Index"},
                gauge={
                    'axis': {'range': [0, 8]},
                    'bar': {'color': "#7C4DFF"},
                    'steps': [
                        {'range': [0, 2], 'color': "#1E293B"},
                        {'range': [2, 5], 'color': "#334155"},
                        {'range': [5, 8], 'color': "#0F172A"}
                    ]
                }
            ))
            gauge_fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            g_col.plotly_chart(gauge_fig, use_container_width=True)
        else:
            st.error("No saved model found. Please execute 'train_and_save.py' to generate `.pkl` files in the `models/` directory.")

# --- TAB 2: DATASET OVERVIEW ---
with tab2:
    st.subheader("Data Summary & Inspection")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records (Hourly)", f"{len(hourly_df):,}")
    col2.metric("Date Range Start", str(hourly_df.index.min().date()))
    col3.metric("Date Range End", str(hourly_df.index.max().date()))
    
    st.dataframe(hourly_df.head(100), use_container_width=True)

# --- TAB 3: EXPLORATORY ANALYSIS ---
with tab3:
    st.subheader("Temporal Usage Distributions")
    col_left, col_right = st.columns(2)
    with col_left:
        diurnal = hourly_df.groupby('hour')['Global_active_power'].mean().reset_index()
        fig_hour = px.line(diurnal, x='hour', y='Global_active_power', title="Diurnal Pattern (Mean Power by Hour)", template="plotly_dark", color_discrete_sequence=['#7C4DFF'])
        st.plotly_chart(fig_hour, use_container_width=True)
        
    with col_right:
        weekly = hourly_df.groupby('day_of_week')['Global_active_power'].mean().reset_index()
        fig_week = px.bar(weekly, x='day_of_week', y='Global_active_power', title="Weekly Pattern (Mean Power by Day)", template="plotly_dark", color_discrete_sequence=['#00E676'])
        st.plotly_chart(fig_week, use_container_width=True)

# --- TAB 4: HISTORICAL EVALUATION ---
with tab4:
    st.subheader(f"Evaluation Matrix: {selected_model}")
    if selected_model in saved_models:
        feature_cols = ['hour', 'day_of_week', 'month', 'rolling_mean_24', 'rolling_mean_168', 'lag_1', 'lag_24', 'lag_168']
        X = hourly_df[feature_cols]
        y = hourly_df['Global_active_power']
        
        test_samples = 300
        y_test = y.tail(test_samples)
        X_test = X.tail(test_samples)
        
        preds = saved_models[selected_model].predict(X_test)
        
        m_mae = mean_absolute_error(y_test, preds)
        m_rmse = np.sqrt(mean_squared_error(y_test, preds))
        m_r2 = r2_score(y_test, preds)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("MAE", f"{m_mae:.4f}")
        m_col2.metric("RMSE", f"{m_rmse:.4f}")
        m_col3.metric("R² Score", f"{m_r2:.4f}")
        
        res_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds}, index=y_test.index)
        fig_res = px.line(res_df, title="Last 300 Hours: Actual vs Predicted", template="plotly_dark", color_discrete_map={'Actual': '#00E676', 'Predicted': '#7C4DFF'})
        st.plotly_chart(fig_res, use_container_width=True)

# --- TAB 5: ANOMALY DIAGNOSTICS ---
with tab5:
    st.subheader("Z-Score Anomaly Diagnostics")
    rolling_mean = hourly_df['Global_active_power'].rolling(24).mean()
    rolling_std = hourly_df['Global_active_power'].rolling(24).std()
    z_scores = (hourly_df['Global_active_power'] - rolling_mean) / rolling_std
    
    anomalies = hourly_df[z_scores.abs() > 3]
    st.write(f"Detected **{len(anomalies):,}** statistical load spikes (|Z| > 3).")
    
    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(x=hourly_df.index[-2000:], y=hourly_df['Global_active_power'].tail(2000), mode='lines', name='Power', line=dict(color='#38BDF8')))
    fig_anom.add_trace(go.Scatter(x=anomalies.index.intersection(hourly_df.index[-2000:]), y=anomalies['Global_active_power'].reindex(hourly_df.index[-2000:]).dropna(), mode='markers', name='Anomalies', marker=dict(color='#FF1744', size=6)))
    fig_anom.update_layout(template="plotly_dark")
    st.plotly_chart(fig_anom, use_container_width=True)
