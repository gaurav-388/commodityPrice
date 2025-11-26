import warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, xgboost as xgb, pickle, os, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*70)
print("FULL DATASET XGBOOST TRAINING - COMMODITY PRICE PREDICTION")
print("="*70)

# Load full dataset
print("\n[1/4] Loading complete dataset...")
df = pd.read_csv('Bengal_Prices_2014-25_final.csv')
print(f"  Total rows: {len(df):,}")
print(f"  Total columns: {df.shape[1]}")

# Comprehensive preprocessing
print("\n[2/4] Feature engineering (full features)...")

# Parse dates and create time features
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['quarter'] = df['date'].dt.quarter
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_year'] = df['date'].dt.dayofyear

# Encode ALL categorical variables
cat_cols = ['state_name', 'district', 'market_name', 'commodity_name', 'variety']
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        print(f"  Encoded {col}: {df[col].nunique()} unique values")

# Create interaction features
df['temp_rainfall'] = df['temperature(celcius)'] * df['rainfall(mm)']
df['price_msp_ratio'] = df['modal_price(rs)'] / (df['MSP(per quintol)'] + 1)
df['production_per_area'] = df['Production(million tonnes)'] / (df['Area(million ha)'] + 0.001)

# Seasonal indicators
df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
df['is_winter'] = df['month'].isin([11, 12, 1, 2]).astype(int)

# Select all numeric features
target = 'modal_price(rs)'
exclude_cols = ['date', 'state_name', 'district', 'market_name', 'commodity_name', 'variety', target]
features = [col for col in df.columns if col not in exclude_cols]

X = df[features].fillna(0)
y = df[target]

print(f"\n  Total features: {len(features)}")
print(f"  Feature list: {features[:10]}... (showing first 10)")

# Split - using FULL dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n  Training samples: {len(X_train):,}")
print(f"  Test samples: {len(X_test):,}")

# Train with optimized parameters for full dataset
print("\n[3/4] Training XGBoost on FULL dataset...")
print("  Using optimized CPU training with 'hist' method")
print("  This may take several minutes...")

model = xgb.XGBRegressor(
    tree_method='hist',
    max_depth=8,
    learning_rate=0.1,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

print("\n  Training started...")
start_time = time.time()

model.fit(X_train, y_train, verbose=False)

train_time = time.time() - start_time
print(f"  Training completed in {train_time:.1f} seconds ({train_time/60:.2f} minutes)")

# Comprehensive evaluation
print("\n[4/4] Evaluating model performance...")
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n" + "="*70)
print("FINAL MODEL PERFORMANCE (FULL DATASET)")
print("="*70)
print(f"  RMSE (Root Mean Squared Error): {rmse:.2f} Rs")
print(f"  MAE (Mean Absolute Error):      {mae:.2f} Rs")
print(f"  R² Score:                       {r2:.4f} ({r2*100:.2f}% variance explained)")
print(f"  MAPE (Mean Abs % Error):        {mape:.2f}%")
print("="*70)

# Feature importance
print("\nTop 15 Most Important Features:")
importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': importance
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']:40s} {row['importance']:.6f}")

# Save model
print("\nSaving model...")
os.makedirs('models', exist_ok=True)

with open('models/xgboost_full_model.pkl', 'wb') as f:
    pickle.dump(model, f)

model.save_model('models/xgboost_full_model.json')

# Save feature importance
feature_importance.to_csv('models/feature_importance.csv', index=False)

print("\n  Model saved: models/xgboost_full_model.pkl")
print("  Model saved: models/xgboost_full_model.json")
print("  Feature importance saved: models/feature_importance.csv")

print("\n" + "="*70)
print("FULL TRAINING COMPLETE!")
print("="*70)
print(f"\nDataset: {len(df):,} rows")
print(f"Features: {len(features)}")
print(f"Training time: {train_time:.1f}s")
print(f"Final R² Score: {r2:.4f}")
print("\nModel is ready for production use!")
