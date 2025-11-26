import warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, xgboost as xgb, pickle, os, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*60)
print("XGBOOST COMMODITY PRICE PREDICTION")
print("="*60)

# Load
print("\n[1/3] Loading...")
df = pd.read_csv('Bengal_Prices_2014-25_final.csv')
print(f"  {len(df):,} rows loaded")

# Fast preprocess - skip date parsing, use direct encoding
print("\n[2/3] Preprocessing...")
for col in ['district', 'commodity_name', 'variety']:
    if col in df.columns:
        df[f'{col}_e'] = LabelEncoder().fit_transform(df[col].astype(str))

features = [c for c in df.columns if c.endswith('_e') or c in [
    'temperature(celcius)', 'rainfall(mm)', 'CPI(base year2012=100)',
    'MSP(per quintol)', 'Production(million tonnes)', 'Yield(kg/ha)']]

X = df[features].fillna(0)
y = df['modal_price(rs)']

print(f"  Features: {len(features)}")

# Sample 25% for speed
X_sub, _, y_sub, _ = train_test_split(X, y, train_size=0.25, random_state=42, shuffle=True)
X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)

print(f"  Train: {len(X_tr):,}, Test: {len(X_te):,}")

# Train
print("\n[3/3] Training XGBoost (CPU-optimized)...")
m = xgb.XGBRegressor(tree_method='hist', max_depth=7, n_estimators=150, learning_rate=0.1)

t0 = time.time()
m.fit(X_tr, y_tr)
t1 = time.time()

print(f"  Time: {t1-t0:.1f}s")

# Evaluate
y_p = m.predict(X_te)
print(f"\n{'='*60}")
print("RESULTS:")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_te, y_p)):.2f} Rs")
print(f"  MAE:  {mean_absolute_error(y_te, y_p):.2f} Rs")
print(f"  R2:   {r2_score(y_te, y_p):.4f}")
print(f"{'='*60}")

# Save
os.makedirs('models', exist_ok=True)
pickle.dump(m, open('models/xgboost_trained.pkl', 'wb'))
print("\nModel saved: models/xgboost_trained.pkl")
print("COMPLETE!")
