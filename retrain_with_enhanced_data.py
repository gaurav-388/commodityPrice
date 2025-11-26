"""
Retrain Model with Enhanced 2024-2025 Dataset
==============================================
This script retrains the XGBoost model using the enhanced dataset
that includes complete 2024-2025 data for all districts and commodities.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("       RETRAINING MODEL WITH ENHANCED 2024-2025 DATASET")
print("="*70)

# ============================================================================
# STEP 1: LOAD ENHANCED DATASET
# ============================================================================
print("\n📂 STEP 1: Loading Enhanced Dataset...")

df = pd.read_csv('Bengal_Prices_2014-25_enhanced.csv')
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

print(f"   Total records: {len(df):,}")
print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"   Districts: {df['district'].nunique()}")
print(f"   Commodities: {df['commodity_name'].nunique()}")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("\n🔧 STEP 2: Feature Engineering...")

# Extract date features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week
df['quarter'] = df['date'].dt.quarter

# Seasonal features
df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
df['is_harvest'] = df['month'].isin([10, 11, 12, 1]).astype(int)
df['is_summer'] = df['month'].isin([3, 4, 5]).astype(int)

# Encode categorical variables
label_encoders = {}

# District encoding
le_district = LabelEncoder()
df['district_encoded'] = le_district.fit_transform(df['district'])
label_encoders['district'] = le_district

# Commodity encoding
le_commodity = LabelEncoder()
df['commodity_encoded'] = le_commodity.fit_transform(df['commodity_name'])
label_encoders['commodity'] = le_commodity

# Market encoding
le_market = LabelEncoder()
df['market_encoded'] = le_market.fit_transform(df['market_name'])
label_encoders['market'] = le_market

print(f"   Date features created: year, month, day, day_of_week, etc.")
print(f"   Seasonal features: is_monsoon, is_harvest, is_summer")
print(f"   Encoded: {len(label_encoders)} categorical columns")

# ============================================================================
# STEP 3: PREPARE FEATURES AND TARGET
# ============================================================================
print("\n📊 STEP 3: Preparing Features and Target...")

# Define feature columns
feature_columns = [
    'district_encoded', 'commodity_encoded', 'market_encoded',
    'year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter',
    'is_monsoon', 'is_harvest', 'is_summer',
    'temperature(celcius)', 'rainfall(mm)',
    'CPI(base year2012=100)', 'Per_Capita_Income(per capita nsdp,rs)',
    'Food_Subsidy(in thousand crores)', 'MSP(per quintol)',
    'Fertilizer_Consumption(kg/ha)', 'Area(million ha)',
    'Production(million tonnes)', 'Yield(kg/ha)'
]

# Check for missing columns and handle them
available_features = [col for col in feature_columns if col in df.columns]
print(f"   Using {len(available_features)} features")

# Target variable
target_column = 'modal_price(rs)'

# Remove rows with missing values
df_clean = df[available_features + [target_column]].dropna()
print(f"   Clean records: {len(df_clean):,}")

X = df_clean[available_features]
y = df_clean[target_column]

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================
print("\n✂️ STEP 4: Train-Test Split...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training samples: {len(X_train):,}")
print(f"   Testing samples: {len(X_test):,}")

# ============================================================================
# STEP 5: TRAIN XGBOOST MODEL
# ============================================================================
print("\n🚀 STEP 5: Training XGBoost Model...")

# XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 1
}

print(f"   Parameters: max_depth={params['max_depth']}, lr={params['learning_rate']}, n_estimators={params['n_estimators']}")

# Train model
model = xgb.XGBRegressor(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

print("   ✅ Model training complete!")

# ============================================================================
# STEP 6: EVALUATE MODEL
# ============================================================================
print("\n📈 STEP 6: Evaluating Model...")

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics
train_metrics = {
    'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
    'MAE': mean_absolute_error(y_train, y_pred_train),
    'R2': r2_score(y_train, y_pred_train),
    'MAPE': np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
}

test_metrics = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'MAE': mean_absolute_error(y_test, y_pred_test),
    'R2': r2_score(y_test, y_pred_test),
    'MAPE': np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
}

print("\n" + "-"*50)
print("                 TRAINING METRICS")
print("-"*50)
print(f"   RMSE:  {train_metrics['RMSE']:.2f}")
print(f"   MAE:   {train_metrics['MAE']:.2f}")
print(f"   R²:    {train_metrics['R2']:.4f}")
print(f"   MAPE:  {train_metrics['MAPE']:.2f}%")

print("\n" + "-"*50)
print("                 TEST METRICS")
print("-"*50)
print(f"   RMSE:  {test_metrics['RMSE']:.2f}")
print(f"   MAE:   {test_metrics['MAE']:.2f}")
print(f"   R²:    {test_metrics['R2']:.4f}")
print(f"   MAPE:  {test_metrics['MAPE']:.2f}%")

# ============================================================================
# STEP 7: SAVE MODEL AND ENCODERS
# ============================================================================
print("\n💾 STEP 7: Saving Model and Encoders...")

# Save XGBoost model in multiple formats
model.save_model('models/xgboost_final_model.json')
print("   ✅ Saved: models/xgboost_final_model.json")

# Save as pickle (for compatibility with existing app)
with open('models/xgboost_final_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✅ Saved: models/xgboost_final_model.pkl")

# Save label encoders
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("   ✅ Saved: models/label_encoders.pkl")

# Save feature columns list
with open('models/feature_columns.json', 'w') as f:
    json.dump(available_features, f)
print("   ✅ Saved: models/feature_columns.json")

# Save model performance metrics
metrics_report = {
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': 'Bengal_Prices_2014-25_enhanced.csv',
    'total_records': len(df),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'features_used': len(available_features),
    'train_metrics': train_metrics,
    'test_metrics': test_metrics
}

with open('models/model_metrics.json', 'w') as f:
    json.dump(metrics_report, f, indent=2)
print("   ✅ Saved: models/model_metrics.json")

# ============================================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================================
print("\n📊 STEP 8: Feature Importance Analysis...")

importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\n   Top 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

feature_importance.to_csv('models/feature_importance.csv', index=False)
print("\n   ✅ Saved: models/feature_importance.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("                    ✅ RETRAINING COMPLETE!")
print("="*70)
print(f"""
Summary:
--------
• Dataset: Bengal_Prices_2014-25_enhanced.csv
• Total Records: {len(df):,}
• Training Samples: {len(X_train):,}
• Test Samples: {len(X_test):,}
• Features Used: {len(available_features)}

Model Performance:
------------------
• Test RMSE: {test_metrics['RMSE']:.2f}
• Test MAE: {test_metrics['MAE']:.2f}
• Test R²: {test_metrics['R2']:.4f}
• Test MAPE: {test_metrics['MAPE']:.2f}%

Files Saved:
------------
• models/xgboost_final_model.json
• models/xgboost_final_model.pkl
• models/label_encoders.pkl
• models/feature_columns.json
• models/model_metrics.json
• models/feature_importance.csv

Next Steps:
-----------
1. Update app.py to use new dataset
2. Restart the server
""")
print("="*70)
