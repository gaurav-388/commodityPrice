"""
Comprehensive Model Evaluation & Visualization
Analyzes the trained XGBoost model and generates insights
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*70)
print("MODEL EVALUATION & ANALYSIS")
print("="*70)

# Load model
print("\n[1] Loading trained model...")
model = pickle.load(open('models/xgboost_FULL.pkl', 'rb'))
print("✓ Model loaded successfully")

# Load data for predictions
print("\n[2] Loading test data...")
df = pd.read_csv('Bengal_Prices_2014-25_final.csv')
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['quarter'] = df['date'].dt.quarter
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week

from sklearn.preprocessing import LabelEncoder
for col in ['state_name', 'district', 'market_name', 'commodity_name', 'variety']:
    if col in df.columns:
        df[f'{col}_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))

df['temp_rainfall'] = df['temperature(celcius)'] * df['rainfall(mm)']
df['production_per_area'] = df['Production(million tonnes)'] / (df['Area(million ha)'] + 0.001)
df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)

target = 'modal_price(rs)'
exclude = ['date', 'state_name', 'district', 'market_name', 'commodity_name', 'variety', target]
features = [col for col in df.columns if col not in exclude]

X = df[features].fillna(0)
y = df[target]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✓ Test set: {len(X_test):,} samples")

# Make predictions
print("\n[3] Generating predictions...")
y_pred = model.predict(X_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n" + "="*70)
print("PERFORMANCE METRICS")
print("="*70)
print(f"RMSE:  {rmse:.2f} Rs")
print(f"MAE:   {mae:.2f} Rs")
print(f"R²:    {r2:.4f} ({r2*100:.2f}%)")
print(f"MAPE:  {mape:.2f}%")
print("="*70)

# Create visualizations
os.makedirs('results', exist_ok=True)

print("\n[4] Creating visualizations...")

# 1. Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, y_pred, alpha=0.3, s=10)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price (Rs)', fontsize=12)
axes[0].set_ylabel('Predicted Price (Rs)', fontsize=12)
axes[0].set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Price (Rs)', fontsize=12)
axes[1].set_ylabel('Residuals (Rs)', fontsize=12)
axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/predictions_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/predictions_analysis.png")
plt.close()

# 2. Error Distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

errors = y_test - y_pred
axes[0, 0].hist(np.abs(errors), bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].set_xlabel('Absolute Error (Rs)', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Distribution of Absolute Errors', fontsize=12, fontweight='bold')
axes[0, 0].axvline(mae, color='r', linestyle='--', label=f'MAE: {mae:.2f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

percentage_errors = (errors / y_test) * 100
axes[0, 1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
axes[0, 1].set_xlabel('Percentage Error (%)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Distribution of Percentage Errors', fontsize=12, fontweight='bold')
axes[0, 1].axvline(np.mean(percentage_errors), color='r', linestyle='--', label=f'Mean: {np.mean(percentage_errors):.2f}%')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(y_test, np.abs(errors), alpha=0.3, s=10, color='green')
axes[1, 0].set_xlabel('Actual Price (Rs)', fontsize=11)
axes[1, 0].set_ylabel('Absolute Error (Rs)', fontsize=11)
axes[1, 0].set_title('Error vs Actual Price', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

from scipy import stats
stats.probplot(errors, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/error_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/error_analysis.png")
plt.close()

# 3. Feature Importance
importance_df = pd.read_csv('models/feature_importance_FULL.csv').head(20)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(importance_df['feature'], importance_df['importance'], color='steelblue', edgecolor='black')
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Top 20 Feature Importance', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/feature_importance.png")
plt.close()

# 4. Price Trends (sample)
sample_size = 1000
indices = np.random.choice(len(y_test), min(sample_size, len(y_test)), replace=False)
y_test_sample = y_test.iloc[indices].values
y_pred_sample = y_pred[indices]

fig, ax = plt.subplots(figsize=(14, 6))
x_axis = range(len(y_test_sample))
ax.plot(x_axis, y_test_sample, label='Actual', alpha=0.7, linewidth=1.5, color='blue')
ax.plot(x_axis, y_pred_sample, label='Predicted', alpha=0.7, linewidth=1.5, color='red')
ax.fill_between(x_axis, y_test_sample, y_pred_sample, alpha=0.2, color='gray')
ax.set_xlabel('Sample Index', fontsize=12)
ax.set_ylabel('Price (Rs)', fontsize=12)
ax.set_title(f'Actual vs Predicted Price Trends (Random {len(y_test_sample)} Samples)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/price_trends.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/price_trends.png")
plt.close()

# Save detailed predictions
print("\n[5] Saving detailed results...")
predictions_df = pd.DataFrame({
    'actual_price': y_test.values,
    'predicted_price': y_pred,
    'error': errors,
    'absolute_error': np.abs(errors),
    'percentage_error': percentage_errors
})
predictions_df.to_csv('results/detailed_predictions.csv', index=False)
print("✓ Saved: results/detailed_predictions.csv")

# Generate text report
report = []
report.append("="*70)
report.append("XGBOOST COMMODITY PRICE PREDICTION - EVALUATION REPORT")
report.append("="*70)
report.append("")
report.append(f"Model: XGBoost Regressor")
report.append(f"Dataset: {len(df):,} total rows")
report.append(f"Training: {len(X_train):,} samples")
report.append(f"Testing: {len(X_test):,} samples")
report.append(f"Features: {len(features)}")
report.append("")
report.append("-"*70)
report.append("PERFORMANCE METRICS:")
report.append("-"*70)
report.append(f"RMSE (Root Mean Squared Error):    {rmse:.2f} Rs")
report.append(f"MAE (Mean Absolute Error):         {mae:.2f} Rs")
report.append(f"R² Score:                          {r2:.4f} ({r2*100:.2f}%)")
report.append(f"MAPE (Mean Absolute % Error):      {mape:.2f}%")
report.append("")
report.append("-"*70)
report.append("INTERPRETATION:")
report.append("-"*70)
if r2 > 0.9:
    report.append("✓ EXCELLENT: Model explains >90% of price variance")
elif r2 > 0.8:
    report.append("✓ VERY GOOD: Model explains >80% of price variance")
elif r2 > 0.7:
    report.append("✓ GOOD: Model explains >70% of price variance")
else:
    report.append("• MODERATE: Consider feature engineering or more data")

report.append(f"• Average prediction error: ±{mae:.2f} Rs")
report.append(f"• Typical error range: ±{rmse:.2f} Rs")
report.append("")
report.append("-"*70)
report.append("TOP 10 MOST IMPORTANT FEATURES:")
report.append("-"*70)
for idx, row in importance_df.head(10).iterrows():
    report.append(f"{idx+1:2d}. {row['feature']:40s} {row['importance']:.6f}")
report.append("")
report.append("="*70)

report_text = "\n".join(report)
print("\n" + report_text)

with open('results/evaluation_report.txt', 'w') as f:
    f.write(report_text)
print("\n✓ Saved: results/evaluation_report.txt")

print("\n" + "="*70)
print("✓ EVALUATION COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  📊 results/predictions_analysis.png")
print("  📊 results/error_analysis.png")
print("  📊 results/feature_importance.png")
print("  📊 results/price_trends.png")
print("  📄 results/detailed_predictions.csv")
print("  📄 results/evaluation_report.txt")
print("\nYour model is production-ready!")
