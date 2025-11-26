"""
Neural Network Model Training with GPU Support
==============================================
Retrains the neural network with enhanced 2024-2025 data
Uses GPU acceleration for faster training
"""
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# TensorFlow setup with GPU
import tensorflow as tf

# Check GPU availability
print("="*80)
print("GPU CONFIGURATION")
print("="*80)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU(s) detected: {len(gpus)}")
    for gpu in gpus:
        print(f"   • {gpu}")
    # Enable memory growth to avoid OOM errors
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("   Memory growth enabled for GPU")
else:
    print("⚠️ No GPU detected. Using CPU (slower)")

print(f"TensorFlow version: {tf.__version__}")

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

print("\n" + "="*80)
print("NEURAL NETWORK TRAINING WITH ENHANCED 2024-2025 DATA")
print("="*80)

# ============================================================
# 1. LOAD ENHANCED DATASET
# ============================================================
print("\n📂 [1/8] Loading Enhanced Dataset...")
df = pd.read_csv('Bengal_Prices_2014-25_enhanced.csv')
print(f"   Total records: {len(df):,}")

# Rename columns for consistency
df = df.rename(columns={
    'market_name': 'market',
    'commodity_name': 'commodity',
    'modal_price(rs)': 'modal_price'
})

# Convert date
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df = df.sort_values('date').reset_index(drop=True)

print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"   Districts: {df['district'].nunique()}")
print(f"   Commodities: {df['commodity'].unique().tolist()}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n🔧 [2/8] Feature Engineering...")

# Time-based features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Cyclical encoding for periodic features
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

# Seasonal features
df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
df['is_harvest'] = df['month'].isin([10, 11, 12, 1]).astype(int)
df['is_summer'] = df['month'].isin([3, 4, 5]).astype(int)

# Price statistics
print("   Computing price statistics...")
df['commodity_avg_price'] = df.groupby('commodity')['modal_price'].transform('mean')
df['market_avg_price'] = df.groupby('market')['modal_price'].transform('mean')
df['district_avg_price'] = df.groupby('district')['modal_price'].transform('mean')
df['month_commodity_avg'] = df.groupby(['month', 'commodity'])['modal_price'].transform('mean')
df['district_commodity_avg'] = df.groupby(['district', 'commodity'])['modal_price'].transform('mean')

print("   ✅ Features created")

# ============================================================
# 3. ENCODE CATEGORICAL VARIABLES
# ============================================================
print("\n🏷️ [3/8] Encoding Categorical Variables...")

label_encoders = {}

# Encode categorical columns
categorical_cols = ['district', 'market', 'commodity', 'variety', 'state_name']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"   Encoded {col}: {len(le.classes_)} unique values")

# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================
print("\n📊 [4/8] Preparing Features and Target...")

# Define feature columns
feature_cols = [
    'district_encoded', 'market_encoded', 'commodity_encoded', 'variety_encoded',
    'year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter',
    'is_weekend', 'is_monsoon', 'is_harvest', 'is_summer',
    'month_sin', 'month_cos', 'day_sin', 'day_cos',
    'commodity_avg_price', 'market_avg_price', 'district_avg_price',
    'month_commodity_avg', 'district_commodity_avg',
    'temperature(celcius)', 'rainfall(mm)',
    'CPI(base year2012=100)', 'Per_Capita_Income(per capita nsdp,rs)',
    'MSP(per quintol)', 'Fertilizer_Consumption(kg/ha)'
]

# Filter available features
available_features = [col for col in feature_cols if col in df.columns]
print(f"   Using {len(available_features)} features")

# Prepare data
X = df[available_features].fillna(0)
y = df['modal_price']

print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")

# ============================================================
# 5. SCALE FEATURES
# ============================================================
print("\n⚖️ [5/8] Scaling Features...")

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

print("   ✅ Features and target scaled")

# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================
print("\n✂️ [6/8] Train-Test Split...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

print(f"   Training samples: {len(X_train):,}")
print(f"   Testing samples: {len(X_test):,}")

# ============================================================
# 7. BUILD AND TRAIN NEURAL NETWORK
# ============================================================
print("\n🧠 [7/8] Building and Training Neural Network...")

# Model architecture
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    
    Dense(1)  # Output layer
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\n   Model Architecture:")
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'models/nn_best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train model
print("\n   Training started...")
print("   " + "-"*50)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)

print("   " + "-"*50)
print("   ✅ Training complete!")

# ============================================================
# 8. EVALUATE MODEL
# ============================================================
print("\n📈 [8/8] Evaluating Model...")

# Predictions
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
mae = mean_absolute_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)
mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100

print("\n" + "-"*50)
print("                 TEST METRICS")
print("-"*50)
print(f"   RMSE:  {rmse:.2f}")
print(f"   MAE:   {mae:.2f}")
print(f"   R²:    {r2:.4f}")
print(f"   MAPE:  {mape:.2f}%")

# ============================================================
# 9. SAVE MODEL AND SCALERS
# ============================================================
print("\n💾 Saving Model and Scalers...")

# Save model in keras format
model.save('models/neural_network_model.keras')
print("   ✅ Saved: models/neural_network_model.keras")

# Save scalers and encoders
nn_data = {
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'label_encoders': label_encoders,
    'feature_columns': available_features,
    'training_date': datetime.now().isoformat(),
    'metrics': {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }
}

with open('models/nn_scalers.pkl', 'wb') as f:
    pickle.dump(nn_data, f)
print("   ✅ Saved: models/nn_scalers.pkl")

# Save training history
history_dict = {
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'mae': [float(x) for x in history.history['mae']],
    'val_mae': [float(x) for x in history.history['val_mae']]
}

with open('models/nn_training_history.json', 'w') as f:
    json.dump(history_dict, f, indent=2)
print("   ✅ Saved: models/nn_training_history.json")

# ============================================================
# TEST ON 2025 DATA
# ============================================================
print("\n" + "="*80)
print("TESTING ON 2025 DATA")
print("="*80)

# Filter 2025 data
df_2025 = df[df['date'].dt.year == 2025]
print(f"\n   2025 records: {len(df_2025):,}")

if len(df_2025) > 0:
    X_2025 = df_2025[available_features].fillna(0)
    X_2025_scaled = scaler_X.transform(X_2025)
    y_2025_actual = df_2025['modal_price'].values
    
    y_2025_pred_scaled = model.predict(X_2025_scaled, verbose=0)
    y_2025_pred = scaler_y.inverse_transform(y_2025_pred_scaled).flatten()
    
    rmse_2025 = np.sqrt(mean_squared_error(y_2025_actual, y_2025_pred))
    mae_2025 = mean_absolute_error(y_2025_actual, y_2025_pred)
    r2_2025 = r2_score(y_2025_actual, y_2025_pred)
    mape_2025 = np.mean(np.abs((y_2025_actual - y_2025_pred) / y_2025_actual)) * 100
    
    print("\n   2025 Prediction Metrics:")
    print(f"   • RMSE:  {rmse_2025:.2f}")
    print(f"   • MAE:   {mae_2025:.2f}")
    print(f"   • R²:    {r2_2025:.4f}")
    print(f"   • MAPE:  {mape_2025:.2f}%")
    
    # Sample predictions
    print("\n   Sample 2025 Predictions:")
    print(f"   {'District':<18} {'Commodity':<10} {'Actual':<12} {'Predicted':<12} {'Error %':<10}")
    print("   " + "-"*70)
    
    sample_idx = np.random.choice(len(df_2025), min(10, len(df_2025)), replace=False)
    for idx in sample_idx:
        row = df_2025.iloc[idx]
        actual = y_2025_actual[idx]
        pred = y_2025_pred[idx]
        error_pct = abs(actual - pred) / actual * 100
        print(f"   {row['district']:<18} {row['commodity']:<10} ₹{actual:<10.0f} ₹{pred:<10.0f} {error_pct:.1f}%")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*80)
print("                    ✅ NEURAL NETWORK TRAINING COMPLETE!")
print("="*80)
print(f"""
Model Performance:
------------------
• Test RMSE: {rmse:.2f}
• Test MAE: {mae:.2f}
• Test R²: {r2:.4f}
• Test MAPE: {mape:.2f}%

Files Saved:
------------
• models/neural_network_model.keras
• models/nn_scalers.pkl
• models/nn_training_history.json
• models/nn_best_model.h5

The Neural Network is now trained with 2024-2025 data!
Restart the server to use the updated model.
""")
print("="*80)
