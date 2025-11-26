"""Ultra-Fast XGBoost Training - Reduced complexity for quick results"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os, pickle, time

print("XGBOOST FAST TRAINING - COMMODITY PRICE PREDICTION")
print("="*70)

# Load
print("\nLoading data...")
df = pd.read_csv('Bengal_Prices_2014-25_final.csv')
print(f"Data: {df.shape[0]:,} rows")

# Quick preprocessing
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

for col in ['district', 'commodity_name']:
    le = LabelEncoder()
    df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))

# Simple features
feature_cols = ['year', 'month', 'district_enc', 'commodity_name_enc', 
                'temperature(celcius)', 'rainfall(mm)', 'CPI(base year2012=100)', 
                'MSP(per quintol)', 'Production(million tonnes)']

X = df[feature_cols].fillna(0)
y = df['modal_price(rs)']

# Small sample for speed (use 20% of data)
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

print(f"Training on: {X_train.shape[0]:,} samples (sampled for speed)")
print(f"Test: {X_test.shape[0]:,} samples")
print(f"Features: {len(feature_cols)}")

# Fast CPU training
print("\nTraining XGBoost (CPU optimized)...")
model = xgb.XGBRegressor(
    tree_method='hist',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,  # Reduced for speed
    random_state=42
)

start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start

print(f"Training time: {train_time:.2f}s")

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*70)
print("RESULTS (on sampled data):")
print(f"RMSE:     {rmse:.2f} Rs")
print(f"MAE:      {mae:.2f} Rs")  
print(f"R2 Score: {r2:.4f}")
print("="*70)

# Save
os.makedirs('models', exist_ok=True)
pickle.dump(model, open('models/xgboost_fast.pkl', 'wb'))
print("\nModel saved: models/xgboost_fast.pkl")
print("DONE!")
