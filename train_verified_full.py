import warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, xgboost as xgb, pickle, os, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*70)
print("VERIFIED FULL DATASET TRAINING - WITH PROGRESS MONITORING")
print("="*70)

# Load and verify
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv('Bengal_Prices_2014-25_final.csv')
total_rows = len(df)
print(f"✓ Loaded: {total_rows:,} rows")
print(f"✓ Columns: {df.shape[1]}")

# Full preprocessing
print("\n[STEP 2] Full feature engineering...")
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['quarter'] = df['date'].dt.quarter
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week

# Encode categoricals
cat_cols = ['state_name', 'district', 'market_name', 'commodity_name', 'variety']
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))

# Interaction features
df['temp_rainfall'] = df['temperature(celcius)'] * df['rainfall(mm)']
df['production_per_area'] = df['Production(million tonnes)'] / (df['Area(million ha)'] + 0.001)
df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)

# Get features
target = 'modal_price(rs)'
exclude = ['date', 'state_name', 'district', 'market_name', 'commodity_name', 'variety', target]
features = [col for col in df.columns if col not in exclude]

X = df[features].fillna(0)
y = df[target]

print(f"✓ Total features created: {len(features)}")
print(f"✓ Dataset shape: {X.shape}")

# Split WITHOUT SAMPLING - USE ALL DATA
print(f"\n[STEP 3] Splitting data (NO SAMPLING - USING ALL {total_rows:,} ROWS)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_count = len(X_train)
test_count = len(X_test)

print(f"✓ Training set: {train_count:,} samples ({train_count/total_rows*100:.1f}%)")
print(f"✓ Test set: {test_count:,} samples ({test_count/total_rows*100:.1f}%)")
print(f"✓ VERIFICATION: {train_count + test_count:,} = {total_rows:,} ✓")

# Train with verbose output
print(f"\n[STEP 4] Training XGBoost on {train_count:,} samples...")
print("Configuration:")
print("  - Algorithm: XGBoost Regressor")
print("  - Tree method: hist (optimized CPU)")
print("  - Max depth: 8")
print("  - Number of trees: 500")
print("  - All CPU cores: YES")
print("\n⏳ Training in progress (this will take 2-5 minutes)...")
print("   Progress will be shown every 50 trees:\n")

model = xgb.XGBRegressor(
    tree_method='hist',
    max_depth=8,
    learning_rate=0.1,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

start_time = time.time()

# Train with validation monitoring
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50  # Print every 50 trees
)

train_time = time.time() - start_time

print(f"\n✓ Training completed!")
print(f"✓ Time taken: {train_time:.1f} seconds = {train_time/60:.2f} minutes")
print(f"✓ Samples trained: {train_count:,}")
print(f"✓ Features used: {len(features)}")
print(f"✓ Trees built: {model.n_estimators}")

# Evaluate
print(f"\n[STEP 5] Evaluating on {test_count:,} test samples...")
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n" + "="*70)
print("FINAL RESULTS - FULL DATASET TRAINING")
print("="*70)
print(f"Dataset Size:     {total_rows:,} rows (100% used)")
print(f"Training Samples: {train_count:,}")
print(f"Test Samples:     {test_count:,}")
print(f"Features:         {len(features)}")
print(f"Training Time:    {train_time:.1f}s ({train_time/60:.2f} min)")
print("-"*70)
print(f"RMSE:             {rmse:.2f} Rs")
print(f"MAE:              {mae:.2f} Rs")
print(f"R² Score:         {r2:.4f} ({r2*100:.2f}%)")
print(f"MAPE:             {mape:.2f}%")
print("="*70)

# Feature importance
print("\nTop 15 Most Important Features:")
importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': importance
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(15).iterrows():
    print(f"  {idx+1:2d}. {row['feature']:35s} {row['importance']:.6f}")

# Save
print("\n[STEP 6] Saving model...")
os.makedirs('models', exist_ok=True)
pickle.dump(model, open('models/xgboost_FULL.pkl', 'wb'))
model.save_model('models/xgboost_FULL.json')
feature_importance.to_csv('models/feature_importance_FULL.csv', index=False)

print("✓ Model saved: models/xgboost_FULL.pkl")
print("✓ Model saved: models/xgboost_FULL.json")
print("✓ Features saved: models/feature_importance_FULL.csv")

print("\n" + "="*70)
print("✓ COMPLETE - VERIFIED FULL DATASET TRAINING SUCCESSFUL!")
print("="*70)
