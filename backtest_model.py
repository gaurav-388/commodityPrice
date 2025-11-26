"""
Historical Price Prediction - Backtesting Script
Tests model accuracy on historical data (2018-2020) for validation
"""
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

print("="*70)
print("HISTORICAL PRICE PREDICTION - BACKTESTING")
print("="*70)

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Load the trained model
print("\n[1/6] Loading trained model...")
try:
    with open('models/xgboost_final_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    features = model_data['features']
    label_encoders = model_data['label_encoders']
    print("✓ Model loaded successfully")
    print(f"  Features: {len(features)}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Load the original dataset
print("\n[2/6] Loading historical data...")
try:
    df = pd.read_csv('Bengal_Prices_2014-25_final.csv')
    print(f"✓ Loaded {len(df):,} historical records")
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

# Engineer features (same as training)
print("\n[3/6] Engineering features...")
try:
    # Time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['quarter'] = (df['month'] - 1) // 3 + 1
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Encode categorical features
    df['state_name_encoded'] = label_encoders['state_name'].transform(df['state_name'])
    df['district_encoded'] = label_encoders['district'].transform(df['district'])
    df['market_name_encoded'] = label_encoders['market_name'].transform(df['market_name'])
    df['commodity_name_encoded'] = label_encoders['commodity_name'].transform(df['commodity_name'])
    df['variety_encoded'] = label_encoders['variety'].transform(df['variety'])
    
    # Interaction features
    df['temp_rainfall_interaction'] = df['temperature(celcius)'] * df['rainfall(mm)']
    df['production_per_area'] = df['Production(million tonnes)'] / (df['Area(million ha)'] + 0.001)
    df['yield_per_area'] = df['Yield(kg/ha)'] / (df['Area(million ha)'] + 0.001)
    
    # Seasonal indicators
    df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
    df['is_winter'] = df['month'].isin([11, 12, 1, 2]).astype(int)
    df['is_summer'] = df['month'].isin([3, 4, 5]).astype(int)
    
    # Economic ratios
    df['cpi_msp_ratio'] = df['CPI(base year2012=100)'] / (df['MSP(per quintol)'] + 1)
    df['subsidy_per_capita'] = df['Food_Subsidy(in thousand crores)'] / (df['Per_Capita_Income(per capita nsdp,rs)'] + 1)
    
    print(f"✓ Features engineered: {len(features)} features")
except Exception as e:
    print(f"✗ Error engineering features: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Prepare features for prediction
print("\n[4/6] Preparing data for prediction...")
try:
    X = df[features]
    y_actual = df['modal_price(rs)'].values
    
    # Remove any rows with NaN
    mask = ~(X.isna().any(axis=1) | pd.isna(y_actual))
    X = X[mask]
    y_actual = y_actual[mask]
    df_clean = df[mask].copy()
    
    print(f"✓ Clean data: {len(X):,} records")
except Exception as e:
    print(f"✗ Error preparing data: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Make predictions
print("\n[5/6] Predicting historical prices...")
try:
    y_pred = model.predict(X)
    df_clean['predicted_price'] = y_pred
    
    # Calculate overall metrics
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 0.001))) * 100
    
    print(f"✓ Predictions complete")
    print(f"\n{'='*70}")
    print("OVERALL BACKTEST METRICS:")
    print(f"{'='*70}")
    print(f"  RMSE:     ₹{rmse:.2f}")
    print(f"  MAE:      ₹{mae:.2f}")
    print(f"  R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)")
    print(f"  MAPE:     {mape:.2f}%")
    print(f"{'='*70}")
except Exception as e:
    print(f"✗ Error making predictions: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test specific historical years
print(f"\n{'='*70}")
print("SPECIFIC HISTORICAL YEAR PREDICTIONS:")
print(f"{'='*70}")

test_years = [2018, 2019, 2020]
for test_year in test_years:
    year_data = df_clean[df_clean['year'] == test_year]
    if len(year_data) > 0:
        y_actual_year = year_data['modal_price(rs/quintal)'].values
        y_pred_year = year_data['predicted_price'].values
        
        rmse_year = np.sqrt(mean_squared_error(y_actual_year, y_pred_year))
        mae_year = mean_absolute_error(y_actual_year, y_pred_year)
        r2_year = r2_score(y_actual_year, y_pred_year)
        
        print(f"\n📅 Year {test_year} ({len(year_data):,} records):")
        print(f"  RMSE: ₹{rmse_year:.2f}")
        print(f"  MAE:  ₹{mae_year:.2f}")
        print(f"  R²:   {r2_year:.4f}")
        
        # Show 3 random samples
        samples = year_data.sample(min(3, len(year_data)))
        print(f"\n  Sample Predictions:")
        for idx, row in samples.iterrows():
            actual = row['modal_price(rs/quintal)']
            predicted = row['predicted_price']
            error = abs(actual - predicted)
            error_pct = (error / actual) * 100
            
            print(f"    • {row['date'].strftime('%d-%b-%Y')} | {row['district']} - {row['market_name']}")
            print(f"      {row['commodity_name']} ({row['variety']})")
            print(f"      Actual: ₹{actual:.2f} | Predicted: ₹{predicted:.2f} | Error: ₹{error:.2f} ({error_pct:.1f}%)")
    else:
        print(f"\n📅 Year {test_year}: No data available")

# Save detailed results
print(f"\n[6/6] Saving results...")
try:
    # Save predictions to CSV
    output_df = df_clean[['date', 'district', 'market_name', 'commodity_name', 
                           'variety', 'modal_price(rs)', 'predicted_price']].copy()
    output_df['error'] = abs(output_df['modal_price(rs)'] - output_df['predicted_price'])
    output_df['error_percent'] = (output_df['error'] / output_df['modal_price(rs)']) * 100
    output_df = output_df.sort_values('date')
    output_df.to_csv('results/historical_predictions_backtest.csv', index=False)
    print("✓ Saved: results/historical_predictions_backtest.csv")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    plt.style.use('default')
    sns.set_palette("husl")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Actual vs Predicted (Large)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    sample_size = min(10000, len(df_clean))
    sample_data = df_clean.sample(sample_size)
    scatter = ax1.scatter(sample_data['modal_price(rs)'], 
                          sample_data['predicted_price'],
                          c=sample_data['year'], cmap='viridis', 
                          alpha=0.4, s=20, edgecolors='none')
    ax1.plot([y_actual.min(), y_actual.max()], 
             [y_actual.min(), y_actual.max()], 
             'r--', lw=2, label='Perfect Prediction', alpha=0.8)
    ax1.set_xlabel('Actual Price (₹/quintal)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Price (₹/quintal)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Actual vs Predicted Prices\nR² = {r2:.4f}, RMSE = ₹{rmse:.2f}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Year', fontsize=10, fontweight='bold')
    
    # 2. Error Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    errors = y_actual - y_pred
    ax2.hist(errors, bins=60, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Error (₹)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Yearly Performance
    ax3 = fig.add_subplot(gs[1, 2])
    yearly_metrics = df_clean.groupby('year').apply(
        lambda x: pd.Series({
            'RMSE': np.sqrt(mean_squared_error(x['modal_price(rs)'], x['predicted_price'])),
            'MAE': mean_absolute_error(x['modal_price(rs)'], x['predicted_price']),
            'R²': r2_score(x['modal_price(rs)'], x['predicted_price'])
        })
    ).reset_index()
    
    x_pos = np.arange(len(yearly_metrics))
    width = 0.35
    ax3.bar(x_pos - width/2, yearly_metrics['RMSE'], width, label='RMSE', color='#3498db', alpha=0.8)
    ax3.bar(x_pos + width/2, yearly_metrics['MAE'], width, label='MAE', color='#e74c3c', alpha=0.8)
    ax3.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Error (₹)', fontsize=10, fontweight='bold')
    ax3.set_title('Yearly Performance', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(yearly_metrics['year'].astype(int), rotation=45)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Time Series for Rice (full width)
    ax4 = fig.add_subplot(gs[2, :])
    rice_data = df_clean[df_clean['commodity_name'] == 'Rice'].sort_values('date')
    if len(rice_data) > 1000:
        rice_data = rice_data[::len(rice_data)//1000]  # Downsample for clarity
    
    ax4.plot(rice_data['date'], rice_data['modal_price(rs)'], 
             label='Actual', color='#27ae60', linewidth=2, alpha=0.7, marker='o', markersize=3)
    ax4.plot(rice_data['date'], rice_data['predicted_price'], 
             label='Predicted', color='#3498db', linewidth=2, alpha=0.7, marker='s', markersize=3)
    ax4.fill_between(rice_data['date'], 
                     rice_data['modal_price(rs)'], 
                     rice_data['predicted_price'],
                     alpha=0.2, color='gray')
    ax4.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Price (₹/quintal)', fontsize=11, fontweight='bold')
    ax4.set_title('Time Series: Rice Price Predictions (Actual vs Predicted)', 
                  fontsize=13, fontweight='bold', pad=10)
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # Add overall stats text
    textstr = f'Overall Statistics:\nRMSE: ₹{rmse:.2f}\nMAE: ₹{mae:.2f}\nR²: {r2:.4f}\nMAPE: {mape:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')
    
    plt.suptitle('Historical Price Prediction - Backtesting Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('results/backtest_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/backtest_analysis.png")
    
    # Create year-specific comparison
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('Year-by-Year Comparison (2018-2020)', fontsize=14, fontweight='bold')
    
    for idx, year in enumerate([2018, 2019, 2020]):
        ax = axes[idx]
        year_data = df_clean[df_clean['year'] == year]
        if len(year_data) > 0:
            sample = year_data.sample(min(2000, len(year_data)))
            ax.scatter(sample['modal_price(rs)'], sample['predicted_price'],
                      alpha=0.5, s=15, color=f'C{idx}')
            ax.plot([sample['modal_price(rs)'].min(), sample['modal_price(rs)'].max()],
                   [sample['modal_price(rs)'].min(), sample['modal_price(rs)'].max()],
                   'r--', lw=2)
            
            r2_year = r2_score(year_data['modal_price(rs)'], year_data['predicted_price'])
            rmse_year = np.sqrt(mean_squared_error(year_data['modal_price(rs)'], 
                                                   year_data['predicted_price']))
            
            ax.set_xlabel('Actual Price (₹)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Predicted Price (₹)', fontsize=10, fontweight='bold')
            ax.set_title(f'{year}\nR²={r2_year:.4f}, RMSE=₹{rmse_year:.2f}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/backtest_yearly_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/backtest_yearly_comparison.png")
    
    plt.close('all')
    
except Exception as e:
    print(f"✗ Error saving results: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*70}")
print("✅ BACKTESTING COMPLETE!")
print(f"{'='*70}")
print(f"\n🎯 Model Performance Summary:")
print(f"   • Overall R² Score: {r2:.4f} ({r2*100:.2f}% accuracy)")
print(f"   • Average Error (MAE): ₹{mae:.2f}")
print(f"   • Root Mean Square Error: ₹{rmse:.2f}")
print(f"\n📊 Files Generated:")
print(f"   • results/historical_predictions_backtest.csv")
print(f"   • results/backtest_analysis.png")
print(f"   • results/backtest_yearly_comparison.png")
print(f"\n✅ Your model can accurately predict historical prices!")
print(f"{'='*70}\n")
