"""
Flask Web Application for Commodity Price Prediction - STABLE VERSION
Uses Waitress WSGI server for production-grade stability
Supports both XGBoost and Neural Network models
"""
from flask import Flask, render_template, request, jsonify
import logging
from logging.handlers import RotatingFileHandler
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import sqlite3
import os
import gc
import warnings
import threading

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['DEBUG'] = False
app.config['JSON_AS_ASCII'] = False

# Database configuration
DB_PATH = 'commodity_prices.db'

# Thread lock for model predictions
model_lock = threading.Lock()

# Configure logging
logger = logging.getLogger('commodity_app')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = RotatingFileHandler('app.log', maxBytes=5_000_000, backupCount=3, encoding='utf-8')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# ============================================================
# LOAD MODELS - XGBoost and Neural Network
# ============================================================
logger.info('Loading models...')

# XGBoost Model
xgb_model = None
xgb_features = None
xgb_label_encoders = None

try:
    with open('models/xgboost_improved_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    xgb_model = model_data['model']
    xgb_features = model_data['features']
    xgb_label_encoders = model_data['label_encoders']
    
    try:
        xgb_model.set_param({'device': 'cpu', 'predictor': 'cpu_predictor'})
    except:
        pass
    
    logger.info('[OK] XGBoost model loaded successfully')
except Exception as e:
    logger.error(f'[ERROR] Failed to load XGBoost model: {e}')

# Neural Network Model
nn_model = None
nn_scaler_X = None
nn_scaler_y = None
nn_features = None
nn_label_encoders = None

try:
    # Load TensorFlow
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    # Try multiple model file locations and formats
    nn_model_paths = [
        os.path.join(BASE_DIR, 'models', 'neural_network_model.keras'),
        os.path.join(BASE_DIR, 'models', 'neural_network_model.h5'),
        os.path.join(BASE_DIR, 'models', 'nn_best_model.h5'),
    ]
    # Try new scalers first, then fall back to old artifacts
    nn_scalers_paths = [
        os.path.join(BASE_DIR, 'models', 'nn_scalers.pkl'),  # NEW from retrain_nn_gpu.py
        os.path.join(BASE_DIR, 'models', 'nn_model_artifacts.pkl'),  # OLD format
    ]
    
    nn_model = None
    for nn_model_path in nn_model_paths:
        if not os.path.exists(nn_model_path):
            continue
            
        logger.info(f'Trying to load NN model from: {nn_model_path}')
        
        try:
            # Check if file is valid (zip for .keras, HDF5 for .h5)
            import zipfile
            
            if nn_model_path.endswith('.keras') and zipfile.is_zipfile(nn_model_path):
                # Valid .keras format
                nn_model = tf.keras.models.load_model(nn_model_path)
                logger.info('Loaded model in .keras format')
                break
            elif nn_model_path.endswith('.h5'):
                # H5 format
                nn_model = tf.keras.models.load_model(nn_model_path, compile=False)
                logger.info('Loaded model in .h5 format')
                break
            else:
                # Try generic loading
                nn_model = tf.keras.models.load_model(nn_model_path, compile=False)
                logger.info('Loaded model with generic loader')
                break
        except Exception as load_err:
            logger.warning(f'Failed to load {nn_model_path}: {load_err}')
            continue
    
    if nn_model is None:
        raise FileNotFoundError('No valid Neural Network model file found')
    
    # Load scalers/artifacts - try new format first
    nn_artifacts = None
    for scalers_path in nn_scalers_paths:
        if os.path.exists(scalers_path):
            logger.info(f'Loading NN scalers from: {scalers_path}')
            with open(scalers_path, 'rb') as f:
                nn_artifacts = pickle.load(f)
            break
    
    if nn_artifacts is None:
        raise FileNotFoundError('No NN scalers file found')
    
    nn_scaler_X = nn_artifacts['scaler_X']
    nn_scaler_y = nn_artifacts['scaler_y']
    # Handle both 'features' (old) and 'feature_columns' (new) keys
    nn_features = nn_artifacts.get('feature_columns', nn_artifacts.get('features', []))
    nn_label_encoders = nn_artifacts['label_encoders']
    
    logger.info(f'[OK] Neural Network model loaded with {len(nn_features)} features')
except Exception as e:
    logger.warning(f'[WARN] Neural Network model not available: {e}')
    nn_model = None

# Use XGBoost as default
model = xgb_model
features = xgb_features
label_encoders = xgb_label_encoders

if model is None:
    logger.error('[ERROR] No models available!')
    sys.exit(1)

logger.info('[OK] Models loaded successfully')

# ============================================================
# DATABASE FUNCTIONS
# ============================================================
def get_db_connection():
    """Get a fresh database connection for each request"""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f'Database connection error: {e}')
        return None

# Load unique values from database
def load_unique_values():
    """Load districts and commodities from database"""
    conn = get_db_connection()
    if not conn:
        return {'districts': [], 'commodities': []}
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT district FROM lookup_districts ORDER BY district")
        districts = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("SELECT DISTINCT commodity_name FROM lookup_commodities ORDER BY commodity_name")
        commodities = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return {'districts': districts, 'commodities': commodities}
    except Exception as e:
        logger.error(f'Error loading unique values: {e}')
        if conn:
            conn.close()
        return {'districts': [], 'commodities': []}

unique_values = load_unique_values()
logger.info(f'Loaded {len(unique_values["districts"])} districts, {len(unique_values["commodities"])} commodities')

# ============================================================
# PREDICTION FUNCTIONS - Support for both XGBoost and Neural Network
# ============================================================

def get_feature_data(district, market, commodity, variety, date_str=None):
    """Get real feature data from database for prediction"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        
        # Get the most recent record for this combination to get base features
        if date_str:
            cursor.execute("""
                SELECT * FROM prices 
                WHERE district = ? AND market_name = ? AND commodity_name = ? AND variety = ?
                AND date_str <= ?
                ORDER BY date_str DESC
                LIMIT 1
            """, (district, market, commodity, variety, date_str))
        else:
            cursor.execute("""
                SELECT * FROM prices 
                WHERE district = ? AND market_name = ? AND commodity_name = ? AND variety = ?
                ORDER BY date_str DESC
                LIMIT 1
            """, (district, market, commodity, variety))
        
        row = cursor.fetchone()
        
        if not row:
            # Try with just district/commodity if exact match not found
            cursor.execute("""
                SELECT * FROM prices 
                WHERE district = ? AND commodity_name = ?
                ORDER BY date_str DESC
                LIMIT 1
            """, (district, commodity))
            row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            data = dict(zip(columns, row))
            
            # Get aggregate statistics
            cursor.execute("""
                SELECT 
                    AVG("modal_price(rs)") as commodity_avg_price
                FROM prices WHERE commodity_name = ?
            """, (commodity,))
            stat = cursor.fetchone()
            data['commodity_avg_price'] = stat[0] if stat and stat[0] else 3000.0
            
            cursor.execute("""
                SELECT 
                    AVG("modal_price(rs)") as market_avg_price
                FROM prices WHERE market_name = ?
            """, (market,))
            stat = cursor.fetchone()
            data['market_avg_price'] = stat[0] if stat and stat[0] else 3000.0
            
            cursor.execute("""
                SELECT 
                    AVG("modal_price(rs)") as variety_avg_price
                FROM prices WHERE variety = ?
            """, (variety,))
            stat = cursor.fetchone()
            data['variety_avg_price'] = stat[0] if stat and stat[0] else 3000.0
            
            cursor.execute("""
                SELECT 
                    AVG("modal_price(rs)") as district_avg_price
                FROM prices WHERE district = ?
            """, (district,))
            stat = cursor.fetchone()
            data['district_avg_price'] = stat[0] if stat and stat[0] else 3000.0
            
            cursor.execute("""
                SELECT 
                    AVG("modal_price(rs)") as market_commodity_avg
                FROM prices WHERE market_name = ? AND commodity_name = ?
            """, (market, commodity))
            stat = cursor.fetchone()
            data['market_commodity_avg'] = stat[0] if stat and stat[0] else 3000.0
            
            cursor.execute("""
                SELECT 
                    AVG("modal_price(rs)") as district_commodity_avg
                FROM prices WHERE district = ? AND commodity_name = ?
            """, (district, commodity))
            stat = cursor.fetchone()
            data['district_commodity_avg'] = stat[0] if stat and stat[0] else 3000.0
            
            cursor.execute("""
                SELECT 
                    AVG("modal_price(rs)") as month_commodity_avg
                FROM prices WHERE commodity_name = ? AND strftime('%m', date_str) = ?
            """, (commodity, date_str[5:7] if date_str else '01'))
            stat = cursor.fetchone()
            data['month_commodity_avg'] = stat[0] if stat and stat[0] else 3000.0
            
            conn.close()
            return data
        
        conn.close()
        return None
        
    except Exception as e:
        logger.error(f'Error getting feature data: {e}')
        if conn:
            conn.close()
        return None

def get_base_price(district, commodity):
    """Get base price from database"""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT AVG("modal_price(rs)") as avg_price 
                FROM prices 
                WHERE commodity_name = ? AND district = ?
            """, (commodity, district))
            row = cursor.fetchone()
            base_price = row[0] if row and row[0] else 3000.0
            conn.close()
            return base_price
        except:
            if conn:
                conn.close()
    return 3000.0

def predict_with_xgboost(date_str, district, market, commodity, variety):
    """Prediction using XGBoost model with REAL features from database"""
    predictions = []
    
    try:
        start_date = datetime.strptime(date_str, '%Y-%m-%d')
    except:
        start_date = datetime.now()
    
    # Get REAL feature data from database
    feature_data = get_feature_data(district, market, commodity, variety, date_str)
    base_price = get_base_price(district, commodity)
    
    if feature_data is None:
        logger.warning(f'No feature data found for {district}/{market}/{commodity}/{variety}')
        feature_data = {}
    
    for day_offset in range(7):
        pred_date = start_date + timedelta(days=day_offset)
        
        try:
            # Date features
            features_dict = {
                'year': pred_date.year,
                'month': pred_date.month,
                'day': pred_date.day,
                'day_of_week': pred_date.weekday(),
                'day_of_year': pred_date.timetuple().tm_yday,
                'week_of_year': pred_date.isocalendar()[1],
                'quarter': (pred_date.month - 1) // 3 + 1,
                'is_weekend': 1 if pred_date.weekday() >= 5 else 0,
                'month_start': 1 if pred_date.day <= 5 else 0,
                'month_end': 1 if pred_date.day >= 25 else 0,
            }
            
            # Encode categorical variables using correct encoder names
            encoder_map = {
                'state_name_encoded': ('state_name', 'West Bengal'),
                'district_encoded': ('district', district),
                'market_name_encoded': ('market_name', market),
                'commodity_name_encoded': ('commodity_name', commodity),
                'variety_encoded': ('variety', variety),
            }
            
            for feature_name, (encoder_name, value) in encoder_map.items():
                if encoder_name in xgb_label_encoders:
                    try:
                        features_dict[feature_name] = xgb_label_encoders[encoder_name].transform([value])[0]
                    except:
                        features_dict[feature_name] = 0
                else:
                    features_dict[feature_name] = 0
            
            # Use REAL economic/agricultural features from database
            features_dict['temperature(celcius)'] = feature_data.get('temperature(celcius)', 28.0)
            features_dict['rainfall(mm)'] = feature_data.get('rainfall(mm)', 1.0)
            features_dict['Per_Capita_Income(per capita nsdp,rs)'] = feature_data.get('Per_Capita_Income(per capita nsdp,rs)', 50000)
            features_dict['Food_Subsidy(in thousand crores)'] = feature_data.get('Food_Subsidy(in thousand crores)', 180.0)
            features_dict['CPI(base year2012=100)'] = feature_data.get('CPI(base year2012=100)', 125.0)
            features_dict['Elec_Agri_Share(%)'] = feature_data.get('Elec_Agri_Share(%)', 0.2)
            features_dict['MSP(per quintol)'] = feature_data.get('MSP(per quintol)', 3500)
            features_dict['Fertilizer_Consumption(kg/ha)'] = feature_data.get('Fertilizer_Consumption(kg/ha)', 130)
            features_dict['Area(million ha)'] = feature_data.get('Area(million ha)', 0.5)
            features_dict['Production(million tonnes)'] = feature_data.get('Production(million tonnes)', 1.5)
            features_dict['Yield(kg/ha)'] = feature_data.get('Yield(kg/ha)', 2700)
            features_dict['Export(Million MT)'] = feature_data.get('Export(Million MT)', 2.0)
            features_dict['Import(Million MT)'] = feature_data.get('Import(Million MT)', 0.3)
            
            # Derived features
            temp = features_dict['temperature(celcius)']
            rain = features_dict['rainfall(mm)']
            features_dict['temp_rainfall_interaction'] = temp * rain
            
            prod = features_dict['Production(million tonnes)']
            area = features_dict['Area(million ha)']
            features_dict['production_per_area'] = prod / area if area > 0 else 0
            features_dict['yield_per_area'] = features_dict['Yield(kg/ha)'] / (area * 1000) if area > 0 else 0
            
            # Season flags
            month = pred_date.month
            features_dict['is_monsoon'] = 1 if month in [6, 7, 8, 9] else 0
            features_dict['is_winter'] = 1 if month in [11, 12, 1, 2] else 0
            features_dict['is_summer'] = 1 if month in [3, 4, 5] else 0
            
            # Economic ratios
            cpi = features_dict['CPI(base year2012=100)']
            msp = features_dict['MSP(per quintol)']
            features_dict['cpi_msp_ratio'] = cpi / msp if msp > 0 else 0
            
            subsidy = features_dict['Food_Subsidy(in thousand crores)']
            income = features_dict['Per_Capita_Income(per capita nsdp,rs)']
            features_dict['subsidy_per_capita'] = (subsidy * 10000) / income if income > 0 else 0
            
            # Price statistics from database
            features_dict['commodity_avg_price'] = feature_data.get('commodity_avg_price', base_price)
            features_dict['market_avg_price'] = feature_data.get('market_avg_price', base_price)
            features_dict['variety_avg_price'] = feature_data.get('variety_avg_price', base_price)
            
            # Fill any remaining missing features
            for feat in xgb_features:
                if feat not in features_dict:
                    features_dict[feat] = 0
            
            # Create feature array in correct order
            X = np.array([[features_dict.get(f, 0) for f in xgb_features]], dtype=np.float32)
            
            with model_lock:
                pred_price = float(xgb_model.predict(X)[0])
            
            if pred_price < 100 or pred_price > 100000:
                pred_price = base_price
            
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'day_name': pred_date.strftime('%A'),
                'price': round(pred_price, 2),
                'day_offset': day_offset
            })
            
        except Exception as e:
            logger.error(f'XGBoost prediction error day {day_offset}: {e}')
            import traceback
            logger.error(traceback.format_exc())
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'day_name': pred_date.strftime('%A'),
                'price': round(base_price, 2),
                'day_offset': day_offset
            })
    
    gc.collect()
    return predictions

def predict_with_neural_network(date_str, district, market, commodity, variety):
    """Prediction using Neural Network model with REAL features from database"""
    if nn_model is None:
        logger.error('Neural Network model not available')
        return predict_with_xgboost(date_str, district, market, commodity, variety)
    
    predictions = []
    
    try:
        start_date = datetime.strptime(date_str, '%Y-%m-%d')
    except:
        start_date = datetime.now()
    
    # Get REAL feature data from database
    feature_data = get_feature_data(district, market, commodity, variety, date_str)
    base_price = get_base_price(district, commodity)
    
    if feature_data is None:
        logger.warning(f'No feature data found for {district}/{market}/{commodity}/{variety}')
        feature_data = {}
    
    # Get the last known price for lag features
    last_price = feature_data.get('modal_price(rs)', base_price)
    
    for day_offset in range(7):
        pred_date = start_date + timedelta(days=day_offset)
        
        try:
            # Basic date features
            features_dict = {
                'year': pred_date.year,
                'month': pred_date.month,
                'day': pred_date.day,
                'day_of_week': pred_date.weekday(),
                'day_of_year': pred_date.timetuple().tm_yday,
                'week_of_year': pred_date.isocalendar()[1],
                'quarter': (pred_date.month - 1) // 3 + 1,
                'is_weekend': 1 if pred_date.weekday() >= 5 else 0,
                'month_start': 1 if pred_date.day <= 5 else 0,
                'month_end': 1 if pred_date.day >= 25 else 0,
            }
            
            # Cyclical features
            features_dict['month_sin'] = np.sin(2 * np.pi * pred_date.month / 12)
            features_dict['month_cos'] = np.cos(2 * np.pi * pred_date.month / 12)
            day_of_year = pred_date.timetuple().tm_yday
            features_dict['day_sin'] = np.sin(2 * np.pi * day_of_year / 365)
            features_dict['day_cos'] = np.cos(2 * np.pi * day_of_year / 365)
            week_of_year = pred_date.isocalendar()[1]
            features_dict['week_sin'] = np.sin(2 * np.pi * week_of_year / 52)
            features_dict['week_cos'] = np.cos(2 * np.pi * week_of_year / 52)
            
            # Seasonal features
            month = pred_date.month
            features_dict['is_monsoon'] = 1 if month in [6, 7, 8, 9] else 0
            features_dict['is_harvest'] = 1 if month in [10, 11, 12, 1] else 0
            features_dict['is_summer'] = 1 if month in [3, 4, 5] else 0
            features_dict['is_winter'] = 1 if month in [11, 12, 1, 2] else 0
            
            # Encode categorical variables (NN uses different naming)
            encoder_map = {
                'district_encoded': ('district', district),
                'market_encoded': ('market', market),
                'commodity_encoded': ('commodity', commodity),
                'variety_encoded': ('variety', variety),
            }
            
            for feature_name, (encoder_name, value) in encoder_map.items():
                if encoder_name in nn_label_encoders:
                    try:
                        features_dict[feature_name] = nn_label_encoders[encoder_name].transform([value])[0]
                    except:
                        features_dict[feature_name] = 0
                else:
                    features_dict[feature_name] = 0
            
            # Use REAL price statistics from database
            features_dict['commodity_avg_price'] = feature_data.get('commodity_avg_price', base_price)
            features_dict['market_avg_price'] = feature_data.get('market_avg_price', base_price)
            features_dict['district_avg_price'] = feature_data.get('district_avg_price', base_price)
            features_dict['variety_avg_price'] = feature_data.get('variety_avg_price', base_price)
            features_dict['month_commodity_avg'] = feature_data.get('month_commodity_avg', base_price)
            features_dict['district_commodity_avg'] = feature_data.get('district_commodity_avg', base_price)
            features_dict['market_commodity_avg'] = feature_data.get('market_commodity_avg', base_price)
            
            # Economic/agricultural features from database
            features_dict['temperature(celcius)'] = feature_data.get('temperature(celcius)', 28.0)
            features_dict['rainfall(mm)'] = feature_data.get('rainfall(mm)', 1.0)
            features_dict['CPI(base year2012=100)'] = feature_data.get('CPI(base year2012=100)', 125.0)
            features_dict['Per_Capita_Income(per capita nsdp,rs)'] = feature_data.get('Per_Capita_Income(per capita nsdp,rs)', 50000)
            features_dict['MSP(per quintol)'] = feature_data.get('MSP(per quintol)', 3500)
            features_dict['Fertilizer_Consumption(kg/ha)'] = feature_data.get('Fertilizer_Consumption(kg/ha)', 130)
            
            # Lag features - use last known price
            for lag in [1, 3, 7, 14, 30]:
                features_dict[f'price_lag_{lag}'] = last_price
            
            # Rolling statistics based on last known price
            for window in [7, 14, 30]:
                features_dict[f'rolling_mean_{window}'] = last_price
                features_dict[f'rolling_std_{window}'] = last_price * 0.05  # Assume 5% std
            
            # Fill any missing features with 0
            for feat in nn_features:
                if feat not in features_dict:
                    features_dict[feat] = 0
            
            # Create feature array in correct order
            X = np.array([[features_dict.get(f, 0) for f in nn_features]], dtype=np.float32)
            
            # Scale features
            X_scaled = nn_scaler_X.transform(X)
            
            with model_lock:
                pred_scaled = nn_model.predict(X_scaled, verbose=0)[0][0]
                pred_price = float(nn_scaler_y.inverse_transform([[pred_scaled]])[0][0])
            
            if pred_price < 100 or pred_price > 100000:
                pred_price = base_price
            
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'day_name': pred_date.strftime('%A'),
                'price': round(pred_price, 2),
                'day_offset': day_offset
            })
            
            # Update last_price for subsequent lag features
            last_price = pred_price
            
        except Exception as e:
            logger.error(f'Neural Network prediction error day {day_offset}: {e}')
            import traceback
            logger.error(traceback.format_exc())
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'day_name': pred_date.strftime('%A'),
                'price': round(base_price, 2),
                'day_offset': day_offset
            })
    
    gc.collect()
    return predictions

def safe_predict(date_str, district, market, commodity, variety, model_type='xgboost'):
    """Thread-safe prediction function - routes to appropriate model"""
    if model_type == 'neural_network' and nn_model is not None:
        logger.info(f'Using Neural Network model')
        return predict_with_neural_network(date_str, district, market, commodity, variety)
    else:
        logger.info(f'Using XGBoost model')
        return predict_with_xgboost(date_str, district, market, commodity, variety)

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    """Render main page with React frontend"""
    # Pass data as an object with districts and commodities to match template
    data = {
        'districts': unique_values['districts'],
        'commodities': unique_values['commodities']
    }
    return render_template('index_react.html', data=data)

@app.route('/get_markets', methods=['POST'])
def get_markets():
    """Get markets for a district"""
    try:
        data = request.get_json(force=True, silent=True) or {}
        district = data.get('district', '').strip()
        
        if not district:
            return jsonify({'markets': []}), 200
        
        logger.info(f'Fetching markets for: {district}')
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'markets': [], 'error': 'Database unavailable'}), 200
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT market_name FROM lookup_markets 
                WHERE district = ? ORDER BY market_name
            """, (district,))
            markets = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            logger.info(f'Found {len(markets)} markets for {district}')
            return jsonify({'markets': markets}), 200
            
        except Exception as e:
            logger.error(f'Query error: {e}')
            if conn:
                conn.close()
            return jsonify({'markets': [], 'error': str(e)}), 200
            
    except Exception as e:
        logger.exception(f'Error in get_markets: {e}')
        return jsonify({'markets': []}), 200

@app.route('/get_varieties', methods=['POST'])
def get_varieties():
    """Get varieties for a commodity"""
    try:
        data = request.get_json(force=True, silent=True) or {}
        commodity = data.get('commodity', '').strip()
        
        if not commodity:
            return jsonify({'varieties': []}), 200
        
        logger.info(f'Fetching varieties for: {commodity}')
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'varieties': [], 'error': 'Database unavailable'}), 200
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT variety FROM lookup_varieties 
                WHERE commodity_name = ? ORDER BY variety
            """, (commodity,))
            varieties = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            logger.info(f'Found {len(varieties)} varieties for {commodity}')
            return jsonify({'varieties': varieties}), 200
            
        except Exception as e:
            logger.error(f'Query error: {e}')
            if conn:
                conn.close()
            return jsonify({'varieties': [], 'error': str(e)}), 200
            
    except Exception as e:
        logger.exception(f'Error in get_varieties: {e}')
        return jsonify({'varieties': []}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        data = request.get_json(force=True, silent=True) or {}
        
        date = data.get('date', '').strip()
        district = data.get('district', '').strip()
        market = data.get('market', '').strip()
        commodity = data.get('commodity', '').strip()
        variety = data.get('variety', '').strip()
        model_type = data.get('model', 'xgboost').strip()  # Get selected model
        
        if not all([date, district, market, commodity, variety]):
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        
        logger.info(f'Prediction: {district}/{market}/{commodity}/{variety} on {date} using {model_type}')
        
        predictions = safe_predict(date, district, market, commodity, variety, model_type)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'inputs': {
                'date': date,
                'district': district,
                'market': market,
                'commodity': commodity,
                'variety': variety,
                'model': model_type
            }
        }), 200
        
    except Exception as e:
        logger.exception(f'Prediction error: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'}), 200

@app.route('/favicon.ico')
def favicon():
    """Return empty favicon to prevent 500 errors"""
    return '', 204

# ============================================================
# MAIN - Use Waitress for production stability
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Commodity Price Prediction System - DUAL MODEL VERSION")
    print("=" * 60)
    print(f"Districts: {len(unique_values['districts'])}")
    print(f"Commodities: {len(unique_values['commodities'])}")
    print("-" * 60)
    print("Available Models:")
    print(f"  [{'OK' if xgb_model else 'X'}] XGBoost (Gradient Boosting)")
    print(f"  [{'OK' if nn_model else 'X'}] Neural Network (Deep Learning)")
    print("=" * 60)
    
    # Try to use Waitress (production server), fallback to Flask dev server
    try:
        from waitress import serve
        print("\nStarting PRODUCTION server (Waitress) on http://localhost:5000")
        print("This server is more stable and won't crash easily!")
        print("Press Ctrl+C to stop\n")
        serve(app, host='0.0.0.0', port=5000, threads=4)
    except ImportError:
        print("\nWaitress not installed. Installing it now...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'waitress'])
        from waitress import serve
        print("\nStarting PRODUCTION server (Waitress) on http://localhost:5000")
        print("Press Ctrl+C to stop\n")
        serve(app, host='0.0.0.0', port=5000, threads=4)
    except Exception as e:
        print(f"\nFalling back to Flask dev server: {e}")
        print("Starting on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
