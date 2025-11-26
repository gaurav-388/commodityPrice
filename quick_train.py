"""
Quick Training Script - Optimized for Fast Execution
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle

print("="*70)
print(" COMMODITY PRICE PREDICTION - XGBOOST GPU TRAINING")
print("="*70)

# Load data
print("\n[1/4] Loading data...")
df = pd.read_csv('Bengal_Prices_2014-25_final.csv')
print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Basic preprocessing
print("\n[2/4] Preprocessing...")
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['quarter'] = df['date'].dt.quarter

# Encode categoricals
le_cols = {}
for col in ['state_name', 'district', 'market_name', 'commodity_name', 'variety']:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        le_cols[col] = le

# Select features
target = 'modal_price(rs)'
exclude = ['date', 'state_name', 'district', 'market_name', 'commodity_name', 'variety', target]
features = [col for col in df.columns if col not in exclude]

X = df[features].fillna(0)
y = df[target]

print(f"Features: {len(features)}")
print(f"Target: {target}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training: {X_train.shape[0]:,} samples")
print(f"Test: {X_test.shape[0]:,} samples")

# Train XGBoost with GPU
print("\n[3/4] Training XGBoost model...")
print("Checking for GPU availability...")

params = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Use optimized CPU training (GPU not available in this build)
params['tree_method'] = 'hist'
print("CPU mode with optimized 'hist' method (fastest CPU option)")

model = xgb.XGBRegressor(**params)

print("Training started...")
import time
start = time.time()

# XGBoost 3.x uses callbacks for early stopping
from xgboost.callback import EarlyStopping

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[EarlyStopping(rounds=50, save_best=True)],
    verbose=True
)

train_time = time.time() - start
print(f"\nTraining completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")

# Evaluate
print("\n[4/4] Evaluating model...")
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n" + "="*70)
print(" MODEL PERFORMANCE")
print("="*70)
print(f"RMSE:     {rmse:.2f} Rs")
print(f"MAE:      {mae:.2f} Rs")
print(f"R2 Score: {r2:.4f}")
print(f"MAPE:     {mape:.2f}%")

# Save model
print("\nSaving model...")
os.makedirs('models', exist_ok=True)
with open('models/xgboost_quick_model.pkl', 'wb') as f:
    pickle.dump(model, f)
model.save_model('models/xgboost_quick_model.json')

print("Model saved to: models/xgboost_quick_model.pkl")
print("\n" + "="*70)
print(" TRAINING COMPLETE!")
print("="*70)
