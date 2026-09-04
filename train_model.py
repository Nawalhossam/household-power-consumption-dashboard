import os
import urllib.request
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Machine Learning Imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Time-Series Models Imports
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1. Page Configuration & Layout
st.set_page_config(
    page_title="Power Consumption Analytics & ML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Household Power Consumption Dashboard")
st.markdown("Interactive machine learning pipeline and time-series diagnostic engine.")

# 2. Sidebar Filters & Selections
st.sidebar.header("⚙️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select Forecasting Engine",
    [
        "Random Forest", 
        "XGBoost", 
        "LightGBM", 
        "Linear Regression", 
        "SARIMA (1,0,1)(1,1,1)24", 
        "Holt-Winters (ETS)", 
        "Seasonal Naive Baseline"
    ]
)

test_size = st.sidebar.slider(
    "Train/Test Split Ratio (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5
) / 100.0

lags_to_include = st.sidebar.multiselect(
    "Lag Features to Include (ML Models Only)",
    options=["lag_1", "lag_24", "lag_168"],
    default=["lag_1", "lag_24", "lag_168"]
)

# 3. Data Loading, Auto-Download & Caching Pipeline
@st.cache_data
def load_and_preprocess():
    zip_path = 'household_power_consumption.zip'
    txt_path = 'household_power_consumption.txt'
    
    # Auto-download from UCI repository if dataset is missing
    if not os.path.exists(zip_path) and not os.path.exists(txt_path):
        url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
        urllib.request.urlretrieve(url, zip_path)

    # Read zipped or unzipped file
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
    
    # Resample to Hourly Frequency
    hourly_df = df[['Global_active_power']].resample('h').mean()
    hourly_df['hour'] = hourly_df.index.hour
    hourly_df['day_of_week'] = hourly_df.index.dayofweek
    hourly_df['month'] = hourly_df.index.month
    
    # Feature Engineering for tabular ML
    hourly_df['lag_1'] = hourly_df['Global_active_power'].shift(1)
    hourly_df['lag_24'] = hourly_df['Global_active_power'].shift(24)
    hourly_df['lag_168'] = hourly_df['Global_active_power'].shift(168)
    hourly_df['rolling_mean_24'] = hourly_df['Global_active_power'].shift(1).rolling(24).mean()
    hourly_df['rolling_mean_168'] = hourly_df['Global_active_power'].shift(1).rolling(168).mean()
    
    return hourly_df.dropna()

data_load_state = st.text("Loading and preprocessing dataset...")
hourly_df = load_and_preprocess()
data_load_state.empty()

# 4. Tab Structure Definitions
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Overview", 
    "📈 Exploratory Analysis", 
    "🤖 Model Training", 
    "🏆 Full Leaderboard",
    "🚨 Anomaly Detection"
])

# --- TAB 1: DATASET OVERVIEW ---
with tab1:
    st.subheader("Data Summary & Raw Inspection")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records (Hourly)", f"{len(hourly_df):,}")
    col2.metric("Date Range Start", str(hourly_df.index.min().date()))
    col3.metric("Date Range End", str(hourly_df.index.max().date()))
    
    st.dataframe(hourly_df.head(100), use_container_width=True)

# --- TAB 2: EXPLORATORY ANALYSIS ---
with tab2:
    st.subheader("Temporal Patterns & Distributions")
    
    col_left, col_right = st.columns(2)
    with col_left:
        diurnal = hourly_df.groupby('hour')['Global_active_power'].mean().reset_index()
        fig_hour = px.line(diurnal, x='hour', y='Global_active_power', title="Diurnal Pattern (Mean Power by Hour)")
        st.plotly_chart(fig_hour, use_container_width=True)
        
    with col_right:
        weekly = hourly_df.groupby('day_of_week')['Global_active_power'].mean().reset_index()
        fig_week = px.bar(weekly, x='day_of_week', y='Global_active_power', title="Weekly Pattern (Mean Power by Day)")
        st.plotly_chart(fig_week, use_container_width=True)

# --- TAB 3: MODEL TRAINING & PREDICTIONS ---
with tab3:
    st.subheader(f"Predictive Engine: {model_choice}")
    
    # ML Features Setup
    feature_cols = ['hour', 'day_of_week', 'month', 'rolling_mean_24', 'rolling_mean_168'] + lags_to_include
    X = hourly_df[feature_cols]
    y = hourly_df['Global_active_power']
    
    split_idx = int(len(hourly_df) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    if st.button("Run Selected Model"):
        with st.spinner("Training model and executing forecast..."):
            
            # 1. ML Class Models
            if model_choice == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
            elif model_choice == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
            elif model_choice == "XGBoost":
                model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
            elif model_choice == "LightGBM":
                model = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbosity=-1)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
            # 2. Time-Series Class Models
            elif model_choice == "SARIMA (1,0,1)(1,1,1)24":
                sarima_model = SARIMAX(y_train, order=(1, 0, 1), seasonal_order=(1, 1, 1, 24))
                sarima_res = sarima_model.fit(disp=False)
                preds = sarima_res.forecast(steps=len(y_test))
                
            elif model_choice == "Holt-Winters (ETS)":
                ets_model = ExponentialSmoothing(y_train, trend="add", seasonal="add", seasonal_periods=24)
                ets_res = ets_model.fit(optimized=True)
                preds = ets_res.forecast(steps=len(y_test))
                
            elif model_choice == "Seasonal Naive Baseline":
                preds = y_test.shift(24)
                preds.iloc[:24] = y_train.iloc[-24:].values

            # Evaluation Metrics
            m_mae = mean_absolute_error(y_test, preds)
            m_rmse = np.sqrt(mean_squared_error(y_test, preds))
            m_r2 = r2_score(y_test, preds)
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("MAE", f"{m_mae:.4f}")
            m_col2.metric("RMSE", f"{m_rmse:.4f}")
            m_col3.metric("R² Score", f"{m_r2:.4f}")
            
            # Actual vs Predicted Visualization
            res_df = pd.DataFrame({'Actual': y_test[:300], 'Predicted': preds[:300]}, index=y_test.index[:300])
            fig_res = px.line(res_df, title="Actual vs Predicted (First 300 Hours in Test Set)")
            st.plotly_chart(fig_res, use_container_width=True)

# --- TAB 4: FULL LEADERBOARD & BENCHMARKS ---
with tab4:
    st.subheader("🏆 Model Benchmarking Matrix")
    st.write("Click the button below to train and evaluate all models across the current split.")
    
    if st.button("Benchmark All Models"):
        with st.spinner("Evaluating entire suite of models (ML & Time-Series)..."):
            
            # Features setup
            feature_cols = ['hour', 'day_of_week', 'month', 'rolling_mean_24', 'rolling_mean_168', 'lag_1', 'lag_24', 'lag_168']
            X = hourly_df[feature_cols]
            y = hourly_df['Global_active_power']
            
            split_idx = int(len(hourly_df) * (1 - test_size))
            X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
            y_tr, y_te = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # 1. Linear Regression
            lr = LinearRegression().fit(X_tr, y_tr)
            p_lr = lr.predict(X_te)
            
            # 2. Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
            p_rf = rf.predict(X_te)
            
            # 3. XGBoost
            xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
            p_xgb = xgb.predict(X_te)
            
            # 4. LightGBM
            lgb = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbosity=-1).fit(X_tr, y_tr)
            p_lgb = lgb.predict(X_te)
            
            # 5. Seasonal Naive
            p_snaive = y_te.shift(24)
            p_snaive.iloc[:24] = y_tr.iloc[-24:].values
            
            # 6. Holt-Winters ETS
            ets = ExponentialSmoothing(y_tr, trend="add", seasonal="add", seasonal_periods=24).fit(optimized=True)
            p_ets = ets.forecast(steps=len(y_te))

            # Metric Calculation Helper
            def get_metrics(y_true, y_pred):
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                return mae, rmse, r2

            results = {
                "Linear Regression": get_metrics(y_te, p_lr),
                "Random Forest": get_metrics(y_te, p_rf),
                "XGBoost": get_metrics(y_te, p_xgb),
                "LightGBM": get_metrics(y_te, p_lgb),
                "Seasonal Naive": get_metrics(y_te, p_snaive),
                "Holt-Winters (ETS)": get_metrics(y_te, p_ets)
            }
            
            bench_df = pd.DataFrame(results, index=["MAE", "RMSE", "R2"]).T.reset_index()
            bench_df.columns = ["Model", "MAE", "RMSE", "R² Score"]
            bench_df = bench_df.sort_values(by="RMSE").reset_index(drop=True)
            
            st.dataframe(bench_df.style.highlight_min(axis=0, subset=["MAE", "RMSE"], color="lightgreen").highlight_max(axis=0, subset=["R² Score"], color="lightgreen"), use_container_width=True)
            
            # Visual Metric Comparison
            fig_bench = px.bar(bench_df, x="Model", y="RMSE", color="R² Score", title="Model RMSE Comparison (Lower is Better)")
            st.plotly_chart(fig_bench, use_container_width=True)

# --- TAB 5: ANOMALY DETECTION ---
with tab5:
    st.subheader("Z-Score Anomaly Diagnostics")
    
    rolling_mean = hourly_df['Global_active_power'].rolling(24).mean()
    rolling_std = hourly_df['Global_active_power'].rolling(24).std()
    z_scores = (hourly_df['Global_active_power'] - rolling_mean) / rolling_std
    
    anomalies = hourly_df[z_scores.abs() > 3]
    st.write(f"Detected **{len(anomalies):,}** statistical spikes (|Z| > 3).")
    
    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(x=hourly_df.index, y=hourly_df['Global_active_power'], mode='lines', name='Consumption', opacity=0.6))
    fig_anom.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Global_active_power'], mode='markers', name='Anomalies', marker=dict(color='red', size=5)))
    st.plotly_chart(fig_anom, use_container_width=True)
