import os
import pickle
import urllib.request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 1. Download & Preprocess Data
zip_path = 'household_power_consumption.zip'
if not os.path.exists(zip_path):
    print("Downloading dataset...")
    url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
    urllib.request.urlretrieve(url, zip_path)

print("Preprocessing dataset...")
df = pd.read_csv(zip_path, sep=';', compression='zip', low_memory=False)
numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
hourly_df = df[['Global_active_power']].set_index(df['datetime']).sort_index().resample('h').mean()

hourly_df['hour'] = hourly_df.index.hour
hourly_df['day_of_week'] = hourly_df.index.dayofweek
hourly_df['month'] = hourly_df.index.month
hourly_df['lag_1'] = hourly_df['Global_active_power'].shift(1)
hourly_df['lag_24'] = hourly_df['Global_active_power'].shift(24)
hourly_df['lag_168'] = hourly_df['Global_active_power'].shift(168)
hourly_df['rolling_mean_24'] = hourly_df['Global_active_power'].shift(1).rolling(24).mean()
hourly_df['rolling_mean_168'] = hourly_df['Global_active_power'].shift(1).rolling(168).mean()
hourly_df = hourly_df.dropna()

feature_cols = ['hour', 'day_of_week', 'month', 'rolling_mean_24', 'rolling_mean_168', 'lag_1', 'lag_24', 'lag_168']
X = hourly_df[feature_cols]
y = hourly_df['Global_active_power']

# 2. Train Models
print("Training models...")
models = {
    "linear_regression": LinearRegression().fit(X, y),
    "random_forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1).fit(X, y),
    "xgboost": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1).fit(X, y),
    "lightgbm": LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbosity=-1).fit(X, y)
}

# 3. Save Artifacts
os.makedirs('models', exist_ok=True)
for name, model in models.items():
    with open(f'models/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

print("Saved all trained models to 'models/' directory successfully!")
