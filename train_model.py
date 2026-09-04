import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1. Page Configuration
st.set_page_config(
    page_title="Power Consumption Analytics & ML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Time Series Theme + Animated UI via Custom CSS
st.markdown("""
    <style>
    /* Time Series Dark Dashboard Background */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Smooth Keyframe Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 5px rgba(56, 189, 248, 0.2); }
        50% { box-shadow: 0 0 15px rgba(56, 189, 248, 0.5); }
        100% { box-shadow: 0 0 5px rgba(56, 189, 248, 0.2); }
    }

    /* Apply Fade In animation to main blocks */
    .stMainBlockContainer {
        animation: fadeIn 0.6s ease-out forwards;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    /* Animated Metric Cards with Glow Effect */
    div[data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid #1f2937;
        border-left: 4px solid #38bdf8;
        padding: 18px;
        border-radius: 10px;
        backdrop-filter: blur(8px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeIn 0.8s ease-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-left-color: #f59e0b;
        animation: pulseGlow 2s infinite;
    }

    /* Tab Customization with Hover Transition */
    button[data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 600;
        color: #94a3b8;
        transition: color 0.2s ease, border-color 0.2s ease;
    }
    button[data-baseweb="tab"]:hover {
        color: #f59e0b;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }

    /* Animated Glowing Primary Buttons */
    .stButton>button, .stFormSubmitButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #0f766e 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.6rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        background: linear-gradient(135deg, #38bdf8 0%, #14b8a6 100%);
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    }

    /* Input Fields Styling */
    input, select {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-radius: 6px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Time-Series Energy Forecasting Engine")
st.markdown("Interactive forecasting dashboard powered by pre-trained `.pkl` time series pipelines.")

# 3. Sidebar Setup
st.sidebar.header("⚙️ Model Configuration")

all_models_list = [
    "Random Forest", 
    "XGBoost", 
    "LightGBM", 
    "Linear Regression", 
    "SARIMA (1,0,1)(1,1,1)24", 
    "Holt-Winters (ETS)", 
    "Seasonal Naive Baseline"
]

model_choice = st.sidebar.selectbox("Select Forecasting Engine", all_models_list)

test_size = st.sidebar.slider(
    "Train/Test Split Ratio (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5
) / 100.0

MODEL_FILE_MAP = {
    "Linear Regression": "models/linear_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl",
    "LightGBM": "models/lightgbm.pkl"
}

# Plotly Time-Series Template Config
TS_PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        font=dict(color='#cbd5e1'),
        xaxis=dict(gridcolor='#1e293b', zerolinecolor='#1e293b', showgrid=True),
        yaxis=dict(gridcolor='#1e293b', zerolinecolor='#1e293b', showgrid=True),
    )
)

# 4. Fast Parquet & Dataset Loading
@st.cache_data
def load_and_preprocess():
    parquet_path = 'processed_hourly_power.parquet'
    local_zip = 'household_power_consumption.zip'
    local_txt = 'household_power_consumption.txt'

    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    if os.path.exists(local_zip):
        df = pd.read_csv(local_zip, sep=';', compression='zip', low_memory=False)
    elif os.path.exists(local_txt):
        df = pd.read_csv(local_txt, sep=';', low_memory=False)
    else:
        st.error("Missing data file! Ensure household_power_consumption.zip is present.")
        st.stop()

    numeric_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage', 
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
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

    processed_df = hourly_df.dropna()

    try:
        processed_df.to_parquet(parquet_path)
    except Exception:
        pass

    return processed_df

hourly_df = load_and_preprocess()

# 5. Load Pretrained Models
@st.cache_resource
def load_saved_model(model_name):
    file_path = MODEL_FILE_MAP.get(model_name)
    if file_path and os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        st.error(f"Model file '{file_path}' not found! Make sure it exists in 'models/' directory.")
        st.stop()

@st.cache_resource
def get_time_series_forecast(model_name, y_tr, test_len):
    eval_len = min(test_len, 168)
    if model_name == "SARIMA (1,0,1)(1,1,1)24":
        y_tr_sub = y_tr.iloc[-168:]
        sarima_model = SARIMAX(y_tr_sub, order=(1, 0, 1), seasonal_order=(1, 1, 1, 24))
        sarima_res = sarima_model.fit(disp=False)
        forecast = sarima_res.forecast(steps=eval_len)
        return forecast, eval_len
    elif model_name == "Holt-Winters (ETS)":
        y_tr_sub = y_tr.iloc[-500:]
        ets_model = ExponentialSmoothing(y_tr_sub, trend="add", seasonal="add", seasonal_periods=24)
        ets_res = ets_model.fit(optimized=True)
        forecast = ets_res.forecast(steps=eval_len)
        return forecast, eval_len

# Feature Engineering Setup
feature_cols = ['hour', 'day_of_week', 'month', 'rolling_mean_24', 'rolling_mean_168', 'lag_1', 'lag_24', 'lag_168']
X = hourly_df[feature_cols]
y = hourly_df['Global_active_power']

split_idx = int(len(hourly_df) * (1 - test_size))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dataset Overview", 
    "📈 Temporal Analysis", 
    "🤖 Model Inference", 
    "🏆 Leaderboard",
    "🚨 Anomaly Detection",
    "🎛️ Single Sample Predict"
])

with tab1:
    st.subheader("Time Series Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Hourly Records", f"{len(hourly_df):,}")
    col2.metric("Start Timestamp", str(hourly_df.index.min().date()))
    col3.metric("End Timestamp", str(hourly_df.index.max().date()))
    st.dataframe(hourly_df.head(100), use_container_width=True)

with tab2:
    st.subheader("Seasonal & Cyclic Trends")
    col_left, col_right = st.columns(2)
    with col_left:
        diurnal = hourly_df.groupby('hour')['Global_active_power'].mean().reset_index()
        fig_hour = px.line(
            diurnal, x='hour', y='Global_active_power', 
            title="Hourly Load Profile (Diurnal Cycle)",
            color_discrete_sequence=['#38bdf8']
        )
        fig_hour.update_layout(template=TS_PLOTLY_TEMPLATE)
        st.plotly_chart(fig_hour, use_container_width=True)
    with col_right:
        weekly = hourly_df.groupby('day_of_week')['Global_active_power'].mean().reset_index()
        fig_week = px.bar(
            weekly, x='day_of_week', y='Global_active_power', 
            title="Weekly Load Distribution",
            color_discrete_sequence=['#f59e0b']
        )
        fig_week.update_layout(template=TS_PLOTLY_TEMPLATE)
        st.plotly_chart(fig_week, use_container_width=True)

with tab3:
    st.subheader(f"Inference Pipeline: {model_choice}")
    if 1:
        with st.spinner("Calculating time series forecast..."):
            if model_choice in MODEL_FILE_MAP:
                model = load_saved_model(model_choice)
                preds = model.predict(X_test)
                y_eval = y_test
            elif model_choice in ["SARIMA (1,0,1)(1,1,1)24", "Holt-Winters (ETS)"]:
                preds, eval_len = get_time_series_forecast(model_choice, y_train, len(y_test))
                y_eval = y_test.iloc[:eval_len]
            elif model_choice == "Seasonal Naive Baseline":
                preds = y_test.shift(24)
                preds.iloc[:24] = y_train.iloc[-24:].values
                y_eval = y_test

            m_mae = mean_absolute_error(y_eval, preds)
            m_rmse = np.sqrt(mean_squared_error(y_eval, preds))
            m_r2 = r2_score(y_eval, preds)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{m_mae:.4f}")
            c2.metric("RMSE", f"{m_rmse:.4f}")
            c3.metric("R² Score", f"{m_r2:.4f}")
            
            res_df = pd.DataFrame({'Actual Load': y_eval[:168], 'Predicted Load': preds[:168]}, index=y_eval.index[:168])
            fig_res = px.line(
                res_df, 
                title="Actual vs Predicted Energy Consumption (168-Hour Horizon)",
                color_discrete_sequence=['#38bdf8', '#f43f5e']
            )
            fig_res.update_layout(template=TS_PLOTLY_TEMPLATE)
            st.plotly_chart(fig_res, use_container_width=True)

with tab4:
    st.subheader("🏆 Model Leaderboard")
    if st.button("Evaluate All Models"):
        with st.spinner("Benchmarking time series models..."):
            p_lr = load_saved_model("Linear Regression").predict(X_test)
            p_rf = load_saved_model("Random Forest").predict(X_test)
            p_xgb = load_saved_model("XGBoost").predict(X_test)
            p_lgb = load_saved_model("LightGBM").predict(X_test)
            
            p_snaive = y_test.shift(24)
            p_snaive.iloc[:24] = y_train.iloc[-24:].values
            
            p_ets, ets_len = get_time_series_forecast("Holt-Winters (ETS)", y_train, len(y_test))
            p_sarima, sarima_len = get_time_series_forecast("SARIMA (1,0,1)(1,1,1)24", y_train, len(y_test))

            def compute_metrics(y_true, y_pred):
                return mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred)

            results = {
                "Linear Regression": compute_metrics(y_test, p_lr),
                "Random Forest": compute_metrics(y_test, p_rf),
                "XGBoost": compute_metrics(y_test, p_xgb),
                "LightGBM": compute_metrics(y_test, p_lgb),
                "Seasonal Naive": compute_metrics(y_test, p_snaive),
                "Holt-Winters (ETS)": compute_metrics(y_test.iloc[:ets_len], p_ets),
                "SARIMA (1,0,1)(1,1,1)24": compute_metrics(y_test.iloc[:sarima_len], p_sarima)
            }
            
            bench_df = pd.DataFrame(results, index=["MAE", "RMSE", "R2"]).T.reset_index()
            bench_df.columns = ["Model", "MAE", "RMSE", "R² Score"]
            bench_df = bench_df.sort_values(by="RMSE").reset_index(drop=True)
            
            st.dataframe(
                bench_df.style.highlight_min(axis=0, subset=["MAE", "RMSE"], color="#065f46")
                              .highlight_max(axis=0, subset=["R² Score"], color="#065f46"),
                use_container_width=True
            )

with tab5:
    st.subheader("Rolling Z-Score Anomaly Detection")
    rolling_mean = hourly_df['Global_active_power'].rolling(24).mean()
    rolling_std = hourly_df['Global_active_power'].rolling(24).std()
    z_scores = (hourly_df['Global_active_power'] - rolling_mean) / rolling_std
    
    anomalies = hourly_df[z_scores.abs() > 3]
    st.write(f"Detected **{len(anomalies):,}** anomaly spikes (>3σ threshold).")
    
    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(x=hourly_df.index, y=hourly_df['Global_active_power'], mode='lines', name='Actual Power', opacity=0.4, line=dict(color='#38bdf8', width=1)))
    fig_anom.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Global_active_power'], mode='markers', name='Anomalies', marker=dict(color='#ef4444', size=6, symbol='x')))
    fig_anom.update_layout(template=TS_PLOTLY_TEMPLATE, title="Grid Anomaly Outliers")
    st.plotly_chart(fig_anom, use_container_width=True)

with tab6:
    st.subheader("🎛️ Manual Single-Sample Inference")
    st.write("Input temporal and lag features manually to test `.pkl` predictions:")
    
    ml_models = ["Random Forest", "XGBoost", "LightGBM", "Linear Regression"]
    selected_manual_model = st.selectbox("Select Model:", ml_models, key="manual_model_select")

    with st.form("manual_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            in_hour = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=12)
            in_day = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)
            in_month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=6)
            
        with col2:
            in_lag1 = st.number_input("Lag 1h Power (kW)", min_value=0.0, max_value=15.0, value=1.2, step=0.1)
            in_lag24 = st.number_input("Lag 24h Power (kW)", min_value=0.0, max_value=15.0, value=1.0, step=0.1)
            in_lag168 = st.number_input("Lag 168h Power (kW)", min_value=0.0, max_value=15.0, value=1.1, step=0.1)

        with col3:
            in_roll24 = st.number_input("Rolling Mean 24h (kW)", min_value=0.0, max_value=15.0, value=1.15, step=0.1)
            in_roll168 = st.number_input("Rolling Mean 168h (kW)", min_value=0.0, max_value=15.0, value=1.05, step=0.1)

        submit_btn = st.form_submit_button("⚡ Predict Active Power")

    if submit_btn:
        input_data = pd.DataFrame([{
            'hour': in_hour,
            'day_of_week': in_day,
            'month': in_month,
            'rolling_mean_24': in_roll24,
            'rolling_mean_168': in_roll168,
            'lag_1': in_lag1,
            'lag_24': in_lag24,
            'lag_168': in_lag168
        }])[feature_cols]

        model = load_saved_model(selected_manual_model)
        prediction = model.predict(input_data)[0]

        st.success(f"⚡ **Predicted Power Consumption:** `{prediction:.4f} kW`")
