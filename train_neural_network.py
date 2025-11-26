"""
Neural Network Model for Commodity Price Prediction
Uses backpropagation with multiple training epochs to minimize error
"""
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.regularizers import l2
    print(f"TensorFlow version: {tf.__version__}")
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow'])
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.regularizers import l2
    TF_AVAILABLE = True

print("="*80)
print("NEURAL NETWORK MODEL TRAINING WITH BACKPROPAGATION")
print("="*80)

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================
print("\n[1/7] Loading dataset...")
df = pd.read_csv('Bengal_Prices_2014-25_final.csv')
print(f"Total records: {len(df):,}")

# Rename columns for consistency
df = df.rename(columns={
    'market_name': 'market',
    'commodity_name': 'commodity',
    'modal_price(rs)': 'modal_price'
})

# Convert date
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df = df.sort_values('date').reset_index(drop=True)

print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Commodities: {df['commodity'].unique()}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n[2/7] Feature engineering...")

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
df['market_commodity_avg'] = df.groupby(['market', 'commodity'])['modal_price'].transform('mean')

# Lag features (historical prices)
print("   Computing lag features...")
df = df.sort_values(['district', 'market', 'commodity', 'variety', 'date'])
for lag in [1, 3, 7, 14, 30]:
    df[f'price_lag_{lag}'] = df.groupby(['district', 'market', 'commodity', 'variety'])['modal_price'].shift(lag)

# Rolling statistics
print("   Computing rolling statistics...")
for window in [7, 14, 30]:
    df[f'rolling_mean_{window}'] = df.groupby(['district', 'market', 'commodity', 'variety'])['modal_price'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    df[f'rolling_std_{window}'] = df.groupby(['district', 'market', 'commodity', 'variety'])['modal_price'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std()
    )

# Fill NaN values
df = df.fillna(method='bfill').fillna(method='ffill')

# For remaining NaNs, fill with column mean
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mean())

print(f"   Total features created: {len(df.columns)}")

# ============================================================
# 3. ENCODE CATEGORICAL VARIABLES
# ============================================================
print("\n[3/7] Encoding categorical variables...")

label_encoders = {}
categorical_cols = ['district', 'market', 'commodity', 'variety']

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"   {col}: {len(le.classes_)} unique values")

# ============================================================
# 4. PREPARE FEATURES FOR NEURAL NETWORK
# ============================================================
print("\n[4/7] Preparing features for Neural Network...")

# Select features
feature_cols = [
    # Encoded categorical
    'district_encoded', 'market_encoded', 'commodity_encoded', 'variety_encoded',
    # Time features
    'year', 'month', 'day', 'day_of_week', 'quarter',
    'is_weekend', 'month_start', 'month_end',
    # Cyclical features
    'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_sin', 'week_cos',
    # Price statistics
    'commodity_avg_price', 'market_avg_price', 'district_avg_price', 'variety_avg_price',
    'month_commodity_avg', 'district_commodity_avg', 'market_commodity_avg',
    # Lag features
    'price_lag_1', 'price_lag_3', 'price_lag_7', 'price_lag_14', 'price_lag_30',
    # Rolling statistics
    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14',
    'rolling_mean_30', 'rolling_std_30'
]

# Ensure all features exist
feature_cols = [col for col in feature_cols if col in df.columns]
print(f"   Using {len(feature_cols)} features")

X = df[feature_cols].values
y = df['modal_price'].values

# Split data (time-based split for time series)
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")

# Scale features
print("   Scaling features...")
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale target (helps with training)
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# ============================================================
# 5. BUILD NEURAL NETWORK MODEL
# ============================================================
print("\n[5/7] Building Neural Network model...")

def build_model(input_dim):
    """Build a deep neural network with backpropagation"""
    model = Sequential([
        # Input layer
        Input(shape=(input_dim,)),
        
        # Hidden layer 1
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layer 2
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layer 3
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Hidden layer 4
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Hidden layer 5
        Dense(16, activation='relu'),
        
        # Output layer
        Dense(1, activation='linear')
    ])
    
    # Compile with Adam optimizer (uses backpropagation)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

model = build_model(X_train_scaled.shape[1])
model.summary()

# ============================================================
# 6. TRAIN WITH BACKPROPAGATION
# ============================================================
print("\n[6/7] Training with backpropagation...")
print("   This uses gradient descent to minimize error iteratively")

# Callbacks for better training
callbacks = [
    # Early stopping to prevent overfitting
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ),
    # Save best model (use .h5 format for checkpoint - more reliable)
    ModelCheckpoint(
        'models/nn_best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
]

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Train the model
print("\n   Starting training (backpropagation with gradient descent)...")
print("   Each epoch = one pass through entire dataset")
print("   Backpropagation adjusts weights to minimize error\n")

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=200,  # Maximum epochs (will stop early if converged)
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)

print(f"\n   Training completed in {len(history.history['loss'])} epochs")

# ============================================================
# 7. EVALUATE MODEL
# ============================================================
print("\n[7/7] Evaluating model...")

# Make predictions
y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n" + "="*60)
print("NEURAL NETWORK MODEL RESULTS")
print("="*60)
print(f"Mean Absolute Error (MAE):     Rs {mae:.2f}")
print(f"Root Mean Square Error (RMSE): Rs {rmse:.2f}")
print(f"R-squared Score:               {r2:.4f}")
print(f"Mean Absolute % Error (MAPE):  {mape:.2f}%")
print("="*60)

# Accuracy by error threshold
print("\nAccuracy by Error Threshold:")
for threshold in [5, 10, 15, 20]:
    pct_error = np.abs((y_test - y_pred) / y_test) * 100
    accuracy = (pct_error <= threshold).mean() * 100
    print(f"   Within {threshold}% error: {accuracy:.1f}%")

# ============================================================
# SAVE MODEL AND ARTIFACTS
# ============================================================
print("\n" + "="*60)
print("SAVING MODEL...")
print("="*60)

# Delete old model files if they exist (to avoid format conflicts)
import shutil
for old_file in ['models/neural_network_model.keras', 'models/nn_best_model.keras']:
    if os.path.exists(old_file):
        os.remove(old_file)
        print(f"   Removed old: {old_file}")

# Save Keras model in the new .keras format (zip-based)
# Using save_format='keras' explicitly ensures proper format
model.save('models/neural_network_model.keras', save_format='keras')
print("   Saved: models/neural_network_model.keras")

# Verify the model was saved correctly
try:
    import zipfile
    if zipfile.is_zipfile('models/neural_network_model.keras'):
        print("   ✓ Model format verified (keras zip format)")
    else:
        # Fallback: save in H5 format if keras format fails
        print("   Warning: .keras format failed, saving as .h5")
        model.save('models/neural_network_model.h5', save_format='h5')
        print("   Saved: models/neural_network_model.h5")
except Exception as e:
    print(f"   Format check error: {e}")

# Save model artifacts (scalers, encoders, feature names)
model_artifacts = {
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'label_encoders': label_encoders,
    'features': feature_cols,
    'training_date': datetime.now().isoformat(),
    'metrics': {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
}

with open('models/nn_model_artifacts.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)
print("   Saved: models/nn_model_artifacts.pkl")

# ============================================================
# COMPARE SAMPLE PREDICTIONS
# ============================================================
print("\n" + "="*60)
print("SAMPLE PREDICTIONS vs ACTUAL")
print("="*60)

# Get some test samples
sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
test_df = df.iloc[split_idx:].reset_index(drop=True)

print(f"{'Date':<12} {'District':<15} {'Commodity':<10} {'Actual':>10} {'Predicted':>10} {'Error%':>8}")
print("-" * 70)

for idx in sample_indices:
    row = test_df.iloc[idx]
    actual = y_test[idx]
    predicted = y_pred[idx]
    error_pct = abs(actual - predicted) / actual * 100
    
    print(f"{str(row['date'])[:10]:<12} {row['district'][:15]:<15} {row['commodity']:<10} "
          f"{actual:>10.2f} {predicted:>10.2f} {error_pct:>7.1f}%")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nNeural Network model saved to: models/neural_network_model.keras")
print("Model artifacts saved to: models/nn_model_artifacts.pkl")
print("\nThis model uses:")
print("  - Deep Neural Network (5 hidden layers)")
print("  - Backpropagation with gradient descent")
print("  - Adam optimizer")
print("  - Early stopping to prevent overfitting")
print("  - Batch normalization and dropout for regularization")
print("="*60)
