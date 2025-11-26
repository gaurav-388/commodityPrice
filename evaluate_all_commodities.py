import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
print("Loading dataset...")
df = pd.read_csv('Bengal_Prices_2014-25_final.csv')
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

# Load improved model
print("Loading model...")
with open('models/xgboost_improved_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
features = model_data['features']
label_encoders = model_data['label_encoders']
variety_mapping = model_data.get('variety_mapping', {})

print("="*80)
print("EVALUATING MODEL ON ALL COMMODITIES (RICE, WHEAT, JUTE)")
print("="*80)

def create_features(row):
    """Create features for a single row"""
    features_dict = {}
    
    date = row['date']
    
    # Time features
    features_dict['year'] = date.year
    features_dict['month'] = date.month
    features_dict['day'] = date.day
    features_dict['quarter'] = (date.month - 1) // 3 + 1
    features_dict['day_of_week'] = date.dayofweek
    features_dict['day_of_year'] = date.dayofyear
    features_dict['week_of_year'] = date.isocalendar()[1]
    
    # Apply variety mapping
    variety = row['variety']
    variety_standardized = variety_mapping.get(variety, variety)
    
    # Encoded features
    try:
        features_dict['state_name_encoded'] = label_encoders['state_name'].transform([row['state_name']])[0]
        features_dict['district_encoded'] = label_encoders['district'].transform([row['district']])[0]
        features_dict['market_name_encoded'] = label_encoders['market_name'].transform([row['market_name']])[0]
        features_dict['commodity_name_encoded'] = label_encoders['commodity_name'].transform([row['commodity_name']])[0]
        features_dict['variety_encoded'] = label_encoders['variety'].transform([variety_standardized])[0]
    except:
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
    
    # Seasonal
    features_dict['is_monsoon'] = 1 if features_dict['month'] in [6, 7, 8, 9] else 0
    features_dict['is_winter'] = 1 if features_dict['month'] in [11, 12, 1, 2] else 0
    features_dict['is_summer'] = 1 if features_dict['month'] in [3, 4, 5] else 0
    
    # Economic ratios
    features_dict['cpi_msp_ratio'] = features_dict['CPI(base year2012=100)'] / (features_dict['MSP(per quintol)'] + 1)
    features_dict['subsidy_per_capita'] = features_dict['Food_Subsidy(in thousand crores)'] / (features_dict['Per_Capita_Income(per capita nsdp,rs)'] + 1)
    
    # Aggregated price features
    commodity = row['commodity_name']
    market = row['market_name']
    
    market_data = df[(df['market_name'] == market) & (df['commodity_name'] == commodity) & (df['date'] < date)]
    commodity_data = df[(df['commodity_name'] == commodity) & (df['date'] < date)]
    variety_data = df[(df['commodity_name'] == commodity) & (df['variety'] == variety_standardized) & (df['date'] < date)]
    
    features_dict['market_avg_price'] = market_data['modal_price(rs)'].mean() if len(market_data) > 0 else commodity_data['modal_price(rs)'].mean() if len(commodity_data) > 0 else 3000
    features_dict['commodity_avg_price'] = commodity_data['modal_price(rs)'].mean() if len(commodity_data) > 0 else 3000
    features_dict['variety_avg_price'] = variety_data['modal_price(rs)'].mean() if len(variety_data) > 0 else features_dict['commodity_avg_price']
    
    return features_dict

# Evaluate on each commodity
results_all = []

for commodity_name in ['Rice', 'Wheat', 'Jute']:
    print(f"\n[{commodity_name.upper()}]")
    print("-"*80)
    
    # Get commodity data
    comm_df = df[df['commodity_name'] == commodity_name].copy()
    
    # Sample 300 random records for testing (or all if less than 300)
    sample_size = min(300, len(comm_df))
    test_df = comm_df.sample(n=sample_size, random_state=42)
    
    print(f"Testing on {sample_size} samples...")
    
    predictions = []
    actuals = []
    errors = []
    valid_count = 0
    
    for idx, row in test_df.iterrows():
        features_dict = create_features(row)
        if features_dict is None:
            continue
        
        # Predict
        try:
            X_pred = pd.DataFrame([features_dict])[features]
            pred = model.predict(X_pred)[0]
            actual = row['modal_price(rs)']
            
            predictions.append(pred)
            actuals.append(actual)
            
            error_pct = abs(pred - actual) / actual * 100
            errors.append(error_pct)
            valid_count += 1
            
            results_all.append({
                'commodity': commodity_name,
                'actual': actual,
                'prediction': pred,
                'error_pct': error_pct
            })
        except Exception as e:
            continue
    
    # Calculate metrics
    if len(predictions) == 0:
        print("  No valid predictions!")
        continue
        
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    errors = np.array(errors)
    
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(errors)
    
    print(f"\nMetrics:")
    print(f"  Samples Tested:     {len(actuals)}")
    print(f"  R2 Score:           {r2:.4f}")
    print(f"  RMSE:               Rs {rmse:.2f}")
    print(f"  MAE:                Rs {mae:.2f}")
    print(f"  MAPE (Avg Error):   {mape:.2f}%")
    print(f"  Median Error:       {np.median(errors):.2f}%")
    print(f"  Best Prediction:    {np.min(errors):.4f}%")
    print(f"  Worst Prediction:   {np.max(errors):.2f}%")
    
    # Error distribution
    excellent = (errors < 5).sum()
    very_good = ((errors >= 5) & (errors < 10)).sum()
    good = ((errors >= 10) & (errors < 20)).sum()
    poor = (errors >= 20).sum()
    
    print(f"\nError Distribution:")
    print(f"  Excellent (<5%%):    {excellent:3d} ({excellent/len(errors)*100:5.1f}%%)")
    print(f"  Very Good (5-10%%):  {very_good:3d} ({very_good/len(errors)*100:5.1f}%%)")
    print(f"  Good (10-20%%):      {good:3d} ({good/len(errors)*100:5.1f}%%)")
    print(f"  Poor (>20%%):        {poor:3d} ({poor/len(errors)*100:5.1f}%%)")

# Summary comparison
print("\n" + "="*80)
print("COMMODITY COMPARISON SUMMARY")
print("="*80)

results_df = pd.DataFrame(results_all)

print(f"\n{'Commodity':<12} {'Samples':<10} {'Avg Error':<12} {'Median Error':<14} {'Best':<10} {'Worst':<10}")
print("-"*80)

for commodity in ['Rice', 'Wheat', 'Jute']:
    comm_results = results_df[results_df['commodity'] == commodity]
    if len(comm_results) == 0:
        continue
    print(f"{commodity:<12} {len(comm_results):<10} "
          f"{comm_results['error_pct'].mean():<12.2f}% "
          f"{comm_results['error_pct'].median():<14.2f}% "
          f"{comm_results['error_pct'].min():<10.4f}% "
          f"{comm_results['error_pct'].max():<10.2f}%")

print("\n" + "="*80)
print("ACCURACY RATING BY COMMODITY")
print("="*80)
for commodity in ['Rice', 'Wheat', 'Jute']:
    comm_results = results_df[results_df['commodity'] == commodity]
    if len(comm_results) == 0:
        continue
    avg_error = comm_results['error_pct'].mean()
    if avg_error < 3:
        status = "EXCELLENT"
        rating = "****"
    elif avg_error < 5:
        status = "VERY GOOD"
        rating = "***"
    elif avg_error < 10:
        status = "GOOD"
        rating = "**"
    else:
        status = "NEEDS IMPROVEMENT"
        rating = "*"
    print(f"  {commodity:<8}: {avg_error:5.2f}% avg error - {status} {rating}")
print("="*80)

print("\nDone! Results show model performance across all commodities.")
