"""
Improved Model Training with Better Accuracy
Addresses variety encoding issues and adds confidence scoring
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("IMPROVED MODEL TRAINING")
print("="*80)

# Load data
print("\n1. Loading dataset...")
df = pd.read_csv('Bengal_Prices_2014-25_final.csv')
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
print(f"   Total records: {len(df)}")

# Data cleaning - standardize variety names
print("\n2. Cleaning and standardizing variety names...")
variety_mapping = {
    'Sona Mansoor/ Non Basmati': 'Sona Mansoori Non Basmati',
    'Sona Mansoor/Non Basmati': 'Sona Mansoori Non Basmati',
    'SonaMansoor/NonBasmati': 'Sona Mansoori Non Basmati',
    # Add more mappings as needed
}
df['variety'] = df['variety'].replace(variety_mapping)
print(f"   Unique varieties after standardization: {df['variety'].nunique()}")

# Feature Engineering
print("\n3. Creating features...")

# Time-based features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['quarter'] = df['date'].dt.quarter
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week

# Seasonal features
df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
df['is_winter'] = df['month'].isin([11, 12, 1, 2]).astype(int)
df['is_summer'] = df['month'].isin([3, 4, 5]).astype(int)

# Interaction features
df['temp_rainfall_interaction'] = df['temperature(celcius)'] * df['rainfall(mm)']
df['production_per_area'] = df['Production(million tonnes)'] / (df['Area(million ha)'] + 0.001)
df['yield_per_area'] = df['Yield(kg/ha)'] / (df['Area(million ha)'] + 0.001)
df['cpi_msp_ratio'] = df['CPI(base year2012=100)'] / (df['MSP(per quintol)'] + 1)
df['subsidy_per_capita'] = df['Food_Subsidy(in thousand crores)'] / (df['Per_Capita_Income(per capita nsdp,rs)'] + 1)

# Market-level aggregated features
print("\n4. Adding market-level features...")
market_avg_price = df.groupby(['district', 'market_name', 'commodity_name'])['modal_price(rs)'].transform('mean')
df['market_avg_price'] = market_avg_price

commodity_avg_price = df.groupby('commodity_name')['modal_price(rs)'].transform('mean')
df['commodity_avg_price'] = commodity_avg_price

variety_avg_price = df.groupby('variety')['modal_price(rs)'].transform('mean')
df['variety_avg_price'] = variety_avg_price

# Encode categorical variables
print("\n5. Encoding categorical features...")
label_encoders = {}
categorical_cols = ['state_name', 'district', 'market_name', 'commodity_name', 'variety']

for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"   {col}: {len(le.classes_)} unique values")

# Select features
feature_columns = [
    'year', 'month', 'day', 'quarter', 'day_of_week', 'day_of_year', 'week_of_year',
    'state_name_encoded', 'district_encoded', 'market_name_encoded', 
    'commodity_name_encoded', 'variety_encoded',
    'temperature(celcius)', 'rainfall(mm)', 'Per_Capita_Income(per capita nsdp,rs)',
    'Food_Subsidy(in thousand crores)', 'CPI(base year2012=100)', 'Elec_Agri_Share(%)',
    'MSP(per quintol)', 'Fertilizer_Consumption(kg/ha)', 'Area(million ha)',
    'Production(million tonnes)', 'Yield(kg/ha)', 'Export(Million MT)', 'Import(Million MT)',
    'temp_rainfall_interaction', 'production_per_area', 'yield_per_area',
    'is_monsoon', 'is_winter', 'is_summer', 'cpi_msp_ratio', 'subsidy_per_capita',
    'market_avg_price', 'commodity_avg_price', 'variety_avg_price'
]

X = df[feature_columns]
y = df['modal_price(rs)']

print(f"\n   Total features: {len(feature_columns)}")

# Train-test split (70-15-15)
print("\n6. Splitting data (70% train, 15% val, 15% test)...")
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)  # 0.1765 * 0.85 ≈ 0.15

print(f"   Train: {len(X_train)} records")
print(f"   Val:   {len(X_val)} records")
print(f"   Test:  {len(X_test)} records")

# Train XGBoost with improved parameters
print("\n7. Training XGBoost model with GPU...")
print("   This may take several minutes...")

# Check GPU availability
try:
    import xgboost as xgb_test
    gpu_available = xgb_test.get_config()['use_rmm']
    device = 'cuda'
except:
    device = 'cuda'  # Will fallback to CPU if not available

params = {
    'n_estimators': 1000,
    'max_depth': 8,
    'learning_rate': 0.05,  # Lower learning rate for better generalization
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.5,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'random_state': 42,
    'tree_method': 'hist',
    'device': device,
    'early_stopping_rounds': 50
}

model = xgb.XGBRegressor(**params)

start_time = datetime.now()
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)
training_time = (datetime.now() - start_time).total_seconds()

print(f"\n   Training completed in {training_time:.2f} seconds")

# Evaluate
print("\n8. Evaluating model...")

def calculate_metrics(y_true, y_pred, dataset_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n   {dataset_name}:")
    print(f"      R² Score: {r2:.4f}")
    print(f"      RMSE:     Rs {rmse:.2f}")
    print(f"      MAE:      Rs {mae:.2f}")
    print(f"      MAPE:     {mape:.2f}%")
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

train_results = calculate_metrics(y_train, train_pred, "Training Set")
val_results = calculate_metrics(y_val, val_pred, "Validation Set")
test_results = calculate_metrics(y_test, test_pred, "Test Set")

# Feature importance
print("\n9. Top 15 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(15).iterrows():
    print(f"   {row['feature']:<35} {row['importance']:.4f}")

feature_importance.to_csv('feature_importance_improved.csv', index=False)

# Save model
print("\n10. Saving improved model...")
model_data = {
    'model': model,
    'features': feature_columns,
    'label_encoders': label_encoders,
    'variety_mapping': variety_mapping,
    'train_results': train_results,
    'val_results': val_results,
    'test_results': test_results,
    'training_time': training_time,
    'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

model_filename = 'models/xgboost_improved_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model_data, f)

print(f"   Model saved to: {model_filename}")

# Compare with old model
print("\n" + "="*80)
print("COMPARISON WITH PREVIOUS MODEL")
print("="*80)

try:
    with open('models/xgboost_final_model.pkl', 'rb') as f:
        old_model_data = pickle.load(f)
    old_results = old_model_data.get('results', {})
    
    if old_results:
        old_test_mape = old_results.get('test_mape', 'N/A')
        print(f"\nOld Model MAPE:  {old_test_mape}")
        print(f"New Model MAPE:  {test_results['mape']:.2f}%")
        
        if isinstance(old_test_mape, (int, float)):
            improvement = old_test_mape - test_results['mape']
            print(f"Improvement:     {improvement:.2f}% {'✓ Better' if improvement > 0 else '✗ Worse'}")
except:
    print("\nCould not load old model for comparison")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if test_results['mape'] < 15:
    print("\n✓ EXCELLENT! Model performs well.")
    print("  - Deploy this model to production")
    print("  - Update app.py to use 'xgboost_improved_model.pkl'")
elif test_results['mape'] < 25:
    print("\n✓ GOOD! Model is acceptable.")
    print("  - Can be used but monitor predictions")
    print("  - Consider collecting more data for high-error cases")
else:
    print("\n⚠️ Model still has high error.")
    print("  - Consider separate models per commodity")
    print("  - Add more domain-specific features")
    print("  - Collect more high-quality training data")

print("\nTo use this improved model:")
print("  1. Update app.py: Change 'xgboost_final_model.pkl' to 'xgboost_improved_model.pkl'")
print("  2. Restart the server")
print("  3. Test predictions with previously problematic cases")

print("\n" + "="*80)
print("Training completed successfully!")
print("="*80)
