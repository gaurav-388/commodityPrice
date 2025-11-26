"""
Retrain Neural Network Model with Enhanced 2024-2025 Dataset
=============================================================
This script retrains the Neural Network model using the enhanced dataset
that includes complete 2024-2025 data for all districts and commodities.
"""
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

print("="*80)
print("RETRAINING NEURAL NETWORK WITH ENHANCED 2024-2025 DATASET")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")

# ============================================================
# 1. LOAD ENHANCED DATASET
# ============================================================
print("\n📂 [1/7] Loading Enhanced Dataset...")
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
print(f"   Commodities: {df['commodity'].unique()}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n🔧 [2/7] Feature Engineering...")

# Time-based features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['month_start'] = (df['day'] <= 5).astype(int)
df['month_end'] = (df['day'] >= 25).astype(int)

# Cyclical encoding for periodic features
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

# Price statistics (aggregated features)
print("   Computing price statistics...")
df['commodity_avg_price'] = df.groupby('commodity')['modal_price'].transform('mean')
df['market_avg_price'] = df.groupby('market')['modal_price'].transform('mean')
df['district_avg_price'] = df.groupby('district')['modal_price'].transform('mean')
df['variety_avg_price'] = df.groupby('variety')['modal_price'].transform('mean')

# Commodity-specific features
df['month_commodity_avg'] = df.groupby(['month', 'commodity'])['modal_price'].transform('mean')
df['district_commodity_avg'] = df.groupby(['district', 'commodity'])['modal_price'].transform('mean')

# Seasonal features
df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
df['is_harvest'] = df['month'].isin([10, 11, 12, 1]).astype(int)
df['is_summer'] = df['month'].isin([3, 4, 5]).astype(int)

print("   ✅ Created time and price features")

# ============================================================
# 3. ENCODE CATEGORICAL VARIABLES
# ============================================================
print("\n🏷️ [3/7] Encoding Categorical Variables...")

label_encoders = {}
categorical_cols = ['district', 'market', 'commodity', 'variety', 'state_name']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"   • {col}: {len(le.classes_)} unique values")

# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================
print("\n📊 [4/7] Preparing Features and Target...")

# Define feature columns
feature_cols = [
    # Encoded categorical
    'district_encoded', 'market_encoded', 'commodity_encoded', 'variety_encoded',
    # Time features
    'year', 'month', 'day', 'day_of_week', 'quarter',
    'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_sin', 'week_cos',
    'is_weekend', 'month_start', 'month_end',
    # Seasonal
    'is_monsoon', 'is_harvest', 'is_summer',
    # Price aggregates
    'commodity_avg_price', 'market_avg_price', 'district_avg_price',
    'variety_avg_price', 'month_commodity_avg', 'district_commodity_avg',
    # Economic indicators
    'CPI(base year2012=100)', 'Per_Capita_Income(per capita nsdp,rs)',
    'Food_Subsidy(in thousand crores)', 'MSP(per quintol)',
    'temperature(celcius)', 'rainfall(mm)'
]

# Filter available features
available_features = [col for col in feature_cols if col in df.columns]
print(f"   Using {len(available_features)} features")

# Prepare data
X = df[available_features].copy()
y = df['modal_price'].copy()

# Handle missing values
X = X.fillna(X.median())

print(f"   Feature matrix shape: {X.shape}")
print(f"   Target shape: {y.shape}")

# ============================================================
# 5. SCALE FEATURES
# ============================================================
print("\n⚖️ [5/7] Scaling Features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"   Training samples: {len(X_train):,}")
print(f"   Testing samples: {len(X_test):,}")

# ============================================================
# 6. BUILD AND TRAIN NEURAL NETWORK
# ============================================================
print("\n🧠 [6/7] Building and Training Neural Network...")

# Model architecture
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    
    # First hidden layer
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Second hidden layer
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Third hidden layer
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    
    # Fourth hidden layer
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    
    # Output layer
    Dense(1, activation='linear')
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
        min_lr=0.00001,
        verbose=1
    ),
    ModelCheckpoint(
        'models/nn_best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
]

# Train model
print("\n   Training started...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

print("   ✅ Training complete!")

# ============================================================
# 7. EVALUATE MODEL
# ============================================================
print("\n📈 [7/7] Evaluating Model...")

# Predictions
y_pred_train = model.predict(X_train, verbose=0).flatten()
y_pred_test = model.predict(X_test, verbose=0).flatten()

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

# ============================================================
# SAVE MODEL AND COMPONENTS
# ============================================================
print("\n💾 Saving Model and Components...")

# Save model in .keras format (recommended)
model.save('models/neural_network_model.keras')
print("   ✅ Saved: models/neural_network_model.keras")

# Save model in .h5 format (backup)
model.save('models/neural_network_model.h5')
print("   ✅ Saved: models/neural_network_model.h5")

# Save scaler
with open('models/nn_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✅ Saved: models/nn_scaler.pkl")

# Save label encoders
with open('models/nn_label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("   ✅ Saved: models/nn_label_encoders.pkl")

# Save feature columns
with open('models/nn_feature_columns.pkl', 'wb') as f:
    pickle.dump(available_features, f)
print("   ✅ Saved: models/nn_feature_columns.pkl")

# Save training metrics
import json
metrics_report = {
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': 'Bengal_Prices_2014-25_enhanced.csv',
    'total_records': len(df),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'features_used': len(available_features),
    'epochs_trained': len(history.history['loss']),
    'train_metrics': train_metrics,
    'test_metrics': test_metrics
}

with open('models/nn_model_metrics.json', 'w') as f:
    json.dump(metrics_report, f, indent=2)
print("   ✅ Saved: models/nn_model_metrics.json")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*80)
print("                    ✅ NEURAL NETWORK RETRAINING COMPLETE!")
print("="*80)
print(f"""
Summary:
--------
• Dataset: Bengal_Prices_2014-25_enhanced.csv
• Total Records: {len(df):,}
• Training Samples: {len(X_train):,}
• Test Samples: {len(X_test):,}
• Features Used: {len(available_features)}
• Epochs Trained: {len(history.history['loss'])}

Model Performance:
------------------
• Test RMSE: {test_metrics['RMSE']:.2f}
• Test MAE: {test_metrics['MAE']:.2f}
• Test R²: {test_metrics['R2']:.4f}
• Test MAPE: {test_metrics['MAPE']:.2f}%

Files Saved:
------------
• models/neural_network_model.keras
• models/neural_network_model.h5
• models/nn_scaler.pkl
• models/nn_label_encoders.pkl
• models/nn_feature_columns.pkl
• models/nn_model_metrics.json

Next Steps:
-----------
Restart the server to use the updated Neural Network model!
""")
print("="*80)
