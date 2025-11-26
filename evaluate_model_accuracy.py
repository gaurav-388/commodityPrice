"""
Comprehensive Model Accuracy Evaluation
Tests model performance across all markets, commodities, and time periods
"""
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE MODEL ACCURACY EVALUATION")
print("="*80)

# Load model and data
print("\n1. Loading model and dataset...")
with open('models/xgboost_final_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
features = model_data['features']
label_encoders = model_data['label_encoders']

df = pd.read_csv('Bengal_Prices_2014-25_final.csv')
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

print(f"   Model loaded: {type(model).__name__}")
print(f"   Dataset: {len(df)} records")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

# Test on recent data (2024-2025) that model might not have seen much
print("\n2. Preparing test data from 2024-2025...")
test_df = df[df['date'] >= '2024-01-01'].copy()
print(f"   Test records: {len(test_df)}")

if len(test_df) == 0:
    print("   No 2024-2025 data, using 2023 data instead...")
    test_df = df[df['date'] >= '2023-01-01'].copy()
    print(f"   Test records: {len(test_df)}")

# Sample for faster testing (use first 1000 records)
if len(test_df) > 1000:
    test_df = test_df.sample(n=1000, random_state=42)
    print(f"   Sampled to 1000 records for faster evaluation")

# Feature engineering function (same as in app.py)
def create_features(row):
    """Create features for a single row"""
    features_dict = {}
    
    date = pd.to_datetime(row['date'])
    
    # Time features
    features_dict['year'] = date.year
    features_dict['month'] = date.month
    features_dict['day'] = date.day
    features_dict['quarter'] = (date.month - 1) // 3 + 1
    features_dict['day_of_week'] = date.dayofweek
    features_dict['day_of_year'] = date.dayofyear
    features_dict['week_of_year'] = date.isocalendar()[1]
    
    # Encoded categorical features
    try:
        features_dict['state_name_encoded'] = label_encoders['state_name'].transform([row['state_name']])[0]
        features_dict['district_encoded'] = label_encoders['district'].transform([row['district']])[0]
        features_dict['market_name_encoded'] = label_encoders['market_name'].transform([row['market_name']])[0]
        features_dict['commodity_name_encoded'] = label_encoders['commodity_name'].transform([row['commodity_name']])[0]
        features_dict['variety_encoded'] = label_encoders['variety'].transform([row['variety']])[0]
    except:
        # If encoding fails, skip this row
        return None
    
    # Other features
    for col in ['temperature(celcius)', 'rainfall(mm)', 'Per_Capita_Income(per capita nsdp,rs)',
                'Food_Subsidy(in thousand crores)', 'CPI(base year2012=100)', 'Elec_Agri_Share(%)',
                'MSP(per quintol)', 'Fertilizer_Consumption(kg/ha)', 'Area(million ha)',
                'Production(million tonnes)', 'Yield(kg/ha)', 'Export(Million MT)', 'Import(Million MT)']:
        features_dict[col] = row[col]
    
    # Interaction features
    features_dict['temp_rainfall_interaction'] = features_dict['temperature(celcius)'] * features_dict['rainfall(mm)']
    features_dict['production_per_area'] = features_dict['Production(million tonnes)'] / (features_dict['Area(million ha)'] + 0.001)
    features_dict['yield_per_area'] = features_dict['Yield(kg/ha)'] / (features_dict['Area(million ha)'] + 0.001)
    
    # Seasonal indicators
    features_dict['is_monsoon'] = 1 if features_dict['month'] in [6, 7, 8, 9] else 0
    features_dict['is_winter'] = 1 if features_dict['month'] in [11, 12, 1, 2] else 0
    features_dict['is_summer'] = 1 if features_dict['month'] in [3, 4, 5] else 0
    
    # Economic ratios
    features_dict['cpi_msp_ratio'] = features_dict['CPI(base year2012=100)'] / (features_dict['MSP(per quintol)'] + 1)
    features_dict['subsidy_per_capita'] = features_dict['Food_Subsidy(in thousand crores)'] / (features_dict['Per_Capita_Income(per capita nsdp,rs)'] + 1)
    
    return features_dict

# Make predictions
print("\n3. Making predictions on test data...")
predictions = []
actuals = []
valid_rows = []

for idx, row in test_df.iterrows():
    feat_dict = create_features(row)
    if feat_dict is None:
        continue
    
    try:
        X = pd.DataFrame([feat_dict])[features]
        pred = model.predict(X)[0]
        predictions.append(pred)
        actuals.append(row['modal_price(rs)'])
        valid_rows.append(row)
    except Exception as e:
        continue

predictions = np.array(predictions)
actuals = np.array(actuals)

print(f"   Successfully predicted {len(predictions)} records")

# Calculate metrics
print("\n" + "="*80)
print("OVERALL MODEL PERFORMANCE")
print("="*80)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mae = mean_absolute_error(actuals, predictions)
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

print(f"\nR² Score:           {r2:.4f}  {'(Good)' if r2 > 0.8 else '(Needs Improvement)'}")
print(f"RMSE:               Rs {rmse:.2f}")
print(f"MAE:                Rs {mae:.2f}")
print(f"MAPE:               {mape:.2f}%  {'(Good)' if mape < 15 else '(High Error)'}")

# Error distribution
print("\n" + "-"*80)
print("ERROR DISTRIBUTION")
print("-"*80)

errors = np.abs(predictions - actuals)
percentage_errors = np.abs((predictions - actuals) / actuals) * 100

print(f"\nAbsolute Errors:")
print(f"  Min:              Rs {errors.min():.2f}")
print(f"  Max:              Rs {errors.max():.2f}")
print(f"  Median:           Rs {np.median(errors):.2f}")
print(f"  75th percentile:  Rs {np.percentile(errors, 75):.2f}")
print(f"  90th percentile:  Rs {np.percentile(errors, 90):.2f}")

print(f"\nPercentage Errors:")
print(f"  < 10%:            {(percentage_errors < 10).sum()} predictions ({(percentage_errors < 10).mean()*100:.1f}%)")
print(f"  10-20%:           {((percentage_errors >= 10) & (percentage_errors < 20)).sum()} predictions")
print(f"  20-50%:           {((percentage_errors >= 20) & (percentage_errors < 50)).sum()} predictions")
print(f"  > 50%:            {(percentage_errors >= 50).sum()} predictions ({(percentage_errors >= 50).mean()*100:.1f}%)")

# Performance by commodity
print("\n" + "="*80)
print("PERFORMANCE BY COMMODITY")
print("="*80)

test_df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
test_df_valid['prediction'] = predictions
test_df_valid['actual'] = actuals
test_df_valid['error'] = np.abs(predictions - actuals)
test_df_valid['pct_error'] = np.abs((predictions - actuals) / actuals) * 100

for commodity in test_df_valid['commodity_name'].unique():
    comm_data = test_df_valid[test_df_valid['commodity_name'] == commodity]
    if len(comm_data) > 5:  # Only if we have enough samples
        comm_mape = comm_data['pct_error'].mean()
        comm_mae = comm_data['error'].mean()
        print(f"\n{commodity}:")
        print(f"  Records:      {len(comm_data)}")
        print(f"  Avg Price:    Rs {comm_data['actual'].mean():.2f}")
        print(f"  MAE:          Rs {comm_mae:.2f}")
        print(f"  MAPE:         {comm_mape:.2f}%")

# Performance by district
print("\n" + "="*80)
print("PERFORMANCE BY TOP 5 DISTRICTS")
print("="*80)

district_counts = test_df_valid['district'].value_counts().head(5)
for district in district_counts.index:
    dist_data = test_df_valid[test_df_valid['district'] == district]
    dist_mape = dist_data['pct_error'].mean()
    dist_mae = dist_data['error'].mean()
    print(f"\n{district}:")
    print(f"  Records:      {len(dist_data)}")
    print(f"  MAE:          Rs {dist_mae:.2f}")
    print(f"  MAPE:         {dist_mape:.2f}%")

# Worst predictions
print("\n" + "="*80)
print("TOP 10 WORST PREDICTIONS (Highest % Error)")
print("="*80)

worst = test_df_valid.nlargest(10, 'pct_error')
print(f"\n{'Date':<12} {'District':<15} {'Market':<20} {'Commodity':<10} {'Actual':<10} {'Predicted':<10} {'Error %':<8}")
print("-"*100)
for _, row in worst.iterrows():
    print(f"{str(row['date'])[:10]:<12} {row['district'][:14]:<15} {row['market_name'][:19]:<20} "
          f"{row['commodity_name']:<10} {row['actual']:>9.2f} {row['prediction']:>10.2f} {row['pct_error']:>7.1f}%")

# Best predictions
print("\n" + "="*80)
print("TOP 10 BEST PREDICTIONS (Lowest % Error)")
print("="*80)

best = test_df_valid.nsmallest(10, 'pct_error')
print(f"\n{'Date':<12} {'District':<15} {'Market':<20} {'Commodity':<10} {'Actual':<10} {'Predicted':<10} {'Error %':<8}")
print("-"*100)
for _, row in best.iterrows():
    print(f"{str(row['date'])[:10]:<12} {row['district'][:14]:<15} {row['market_name'][:19]:<20} "
          f"{row['commodity_name']:<10} {row['actual']:>9.2f} {row['prediction']:>10.2f} {row['pct_error']:>7.1f}%")

# Save detailed results
print("\n" + "="*80)
print("SAVING DETAILED RESULTS")
print("="*80)

results_file = 'model_evaluation_results.csv'
test_df_valid[['date', 'district', 'market_name', 'commodity_name', 'variety', 
               'actual', 'prediction', 'error', 'pct_error']].to_csv(results_file, index=False)
print(f"\nDetailed results saved to: {results_file}")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if mape > 30:
    print("\n⚠️ HIGH ERROR - Model needs significant improvement:")
    print("  1. Retrain with more recent data")
    print("  2. Add more market-specific features")
    print("  3. Consider separate models per commodity")
    print("  4. Review feature engineering")
elif mape > 15:
    print("\n⚠️ MODERATE ERROR - Model can be improved:")
    print("  1. Fine-tune hyperparameters")
    print("  2. Add variety-specific features")
    print("  3. Balance training data across districts")
else:
    print("\n✓ GOOD PERFORMANCE - Model is working well")
    print("  Minor improvements possible through fine-tuning")

print("\n" + "="*80)
print(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
