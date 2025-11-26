"""
Flask Web Application for Commodity Price Prediction
Predicts current day price and next 6 days forecast
Uses SQLite database for reliable and fast data access
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
import gc  # Garbage collection for memory management
import warnings

# Suppress XGBoost device mismatch warnings (harmless but noisy)
warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')

app = Flask(__name__)

# Configure Flask for stability
app.config['DEBUG'] = False  # Disable debug mode to prevent crashes
app.config['JSON_AS_ASCII'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

# Database configuration
DB_PATH = 'commodity_prices.db'

# Load the trained model and data
# Configure logging to file and console
logger = logging.getLogger('commodity_app')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler('app.log', maxBytes=5_000_000, backupCount=3, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
# Configure stream handler for Windows console
import sys
if sys.platform == 'win32':
    import codecs
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

logger.info('Loading model...')
try:
    with open('models/xgboost_improved_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    features = model_data['features']
    label_encoders = model_data['label_encoders']
    variety_mapping = model_data.get('variety_mapping', {})
    
    # Force CPU prediction to avoid GPU/CPU mismatch warnings
    # This saves and reloads the model in CPU mode
    try:
        # Method 1: Try set_param
        model.set_param({'device': 'cpu'})
        logger.info('[OK] Model set to CPU mode (set_param)')
    except:
        pass
    
    try:
        # Method 2: Save to JSON and reload with CPU config
        import tempfile
        temp_path = tempfile.mktemp(suffix='.json')
        model.save_model(temp_path)
        model.load_model(temp_path)
        model.set_param({'device': 'cpu'})
        os.remove(temp_path)
        logger.info('[OK] Model reloaded in CPU mode')
    except Exception as reload_err:
        logger.warning(f'Could not reload model in CPU mode: {reload_err}')
        logger.info('Model will use fallback DMatrix prediction (slower but works)')
    
    logger.info('[OK] Improved model loaded successfully')
except Exception as e:
    logger.exception(f'[ERROR] Error loading model: {e}')
    sys.exit(1)

# Database helper functions
def get_db_connection():
    """Get database connection with error handling"""
    try:
        if not os.path.exists(DB_PATH):
            logger.error(f'[ERROR] Database not found: {DB_PATH}')
            logger.error('Please run: python init_database.py')
            return None
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f'Database connection error: {e}')
        return None

def query_db(query, args=(), one=False):
    """Execute database query"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(query, args)
        rv = cur.fetchall()
        conn.close()
        return (rv[0] if rv else None) if one else rv
    except Exception as e:
        logger.error(f'Database query error: {e}')
        return None if one else []

# Load dataset for feature calculation (cached in memory)
logger.info('Loading dataset for feature engineering...')
try:
    # Load full dataset into memory for feature calculations
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM prices", conn)
    df['date'] = pd.to_datetime(df['date_str'])
    conn.close()
    logger.info(f'[OK] Dataset loaded: {len(df)} rows')
except Exception as e:
    logger.exception(f'[ERROR] Error loading dataset: {e}')
    sys.exit(1)

# Get unique values for dropdowns from database lookup tables
logger.info('Loading dropdown values from database...')
try:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT district FROM lookup_districts ORDER BY district")
    districts = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT commodity_name FROM lookup_commodities ORDER BY commodity_name")
    commodities = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    
    unique_values = {
        'districts': districts,
        'commodities': commodities
    }
    logger.info(f'[OK] Loaded {len(districts)} districts, {len(commodities)} commodities')
except Exception as e:
    logger.exception(f'[ERROR] Error loading dropdown values: {e}')
    sys.exit(1)

# Get latest values for features (used for forecasting)
latest_data = df.iloc[-1].to_dict()

def create_features_for_prediction(date, district, market, commodity, variety):
    """Create feature vector for prediction"""
    features_dict = {}
    
    # Parse date
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    # Apply variety name standardization if mapping exists
    variety_standardized = variety_mapping.get(variety, variety)
    
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
        features_dict['state_name_encoded'] = label_encoders['state_name'].transform(['West Bengal'])[0]
        features_dict['district_encoded'] = label_encoders['district'].transform([district])[0]
        features_dict['market_name_encoded'] = label_encoders['market_name'].transform([market])[0]
        features_dict['commodity_name_encoded'] = label_encoders['commodity_name'].transform([commodity])[0]
        features_dict['variety_encoded'] = label_encoders['variety'].transform([variety_standardized])[0]
    except:
        # If encoding fails, use default values
        features_dict['state_name_encoded'] = 0
        features_dict['district_encoded'] = 0
        features_dict['market_name_encoded'] = 0
        features_dict['commodity_name_encoded'] = 0
        features_dict['variety_encoded'] = 0
    
    # Use latest available values for other features
    features_dict['temperature(celcius)'] = latest_data.get('temperature(celcius)', 28)
    features_dict['rainfall(mm)'] = latest_data.get('rainfall(mm)', 0)
    features_dict['Per_Capita_Income(per capita nsdp,rs)'] = latest_data.get('Per_Capita_Income(per capita nsdp,rs)', 50000)
    features_dict['Food_Subsidy(in thousand crores)'] = latest_data.get('Food_Subsidy(in thousand crores)', 120)
    features_dict['CPI(base year2012=100)'] = latest_data.get('CPI(base year2012=100)', 150)
    features_dict['Elec_Agri_Share(%)'] = latest_data.get('Elec_Agri_Share(%)', 0.2)
    features_dict['MSP(per quintol)'] = latest_data.get('MSP(per quintol)', 2000)
    features_dict['Fertilizer_Consumption(kg/ha)'] = latest_data.get('Fertilizer_Consumption(kg/ha)', 150)
    features_dict['Area(million ha)'] = latest_data.get('Area(million ha)', 5.5)
    features_dict['Production(million tonnes)'] = latest_data.get('Production(million tonnes)', 15)
    features_dict['Yield(kg/ha)'] = latest_data.get('Yield(kg/ha)', 2800)
    features_dict['Export(Million MT)'] = latest_data.get('Export(Million MT)', 10)
    features_dict['Import(Million MT)'] = latest_data.get('Import(Million MT)', 0)
    
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
    
    # Calculate aggregated price features (market_avg_price, commodity_avg_price, variety_avg_price)
    # These are calculated from historical data in the dataset
    try:
        # Filter relevant historical data
        market_data = df[(df['market_name'] == market) & (df['commodity_name'] == commodity)]
        commodity_data = df[df['commodity_name'] == commodity]
        variety_data = df[(df['commodity_name'] == commodity) & (df['variety'] == variety_standardized)]
        
        # Calculate averages
        features_dict['market_avg_price'] = market_data['modal_price(rs)'].mean() if len(market_data) > 0 else commodity_data['modal_price(rs)'].mean()
        features_dict['commodity_avg_price'] = commodity_data['modal_price(rs)'].mean() if len(commodity_data) > 0 else 3000
        features_dict['variety_avg_price'] = variety_data['modal_price(rs)'].mean() if len(variety_data) > 0 else features_dict['commodity_avg_price']
    except Exception as e:
        # Fallback values if calculation fails
        logger.warning(f'Could not calculate aggregated features: {e}')
        features_dict['market_avg_price'] = 3000
        features_dict['commodity_avg_price'] = 3000
        features_dict['variety_avg_price'] = 3000
    
    return features_dict

def predict_prices(date, district, market, commodity, variety, days=7):
    """Predict prices for current day and next N days"""
    predictions = []
    
    try:
        start_date = pd.to_datetime(date)
    except Exception as e:
        logger.error(f'Invalid date format: {date}, error: {e}')
        raise ValueError(f'Invalid date format: {date}')
    
    for i in range(days):
        try:
            pred_date = start_date + timedelta(days=i)
            
            # Create features
            features_dict = create_features_for_prediction(pred_date, district, market, commodity, variety)
            
            # Create DataFrame with correct feature order
            X_pred = pd.DataFrame([features_dict])[features]
            
            # Predict with error handling
            try:
                price = model.predict(X_pred)[0]
            except Exception as pred_error:
                logger.error(f'Model prediction error: {pred_error}')
                # Use fallback price
                price = features_dict.get('commodity_avg_price', 3000)
            
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'day_name': pred_date.strftime('%A'),
                'price': round(float(price), 2)
            })
        except Exception as day_error:
            logger.error(f'Error predicting day {i}: {day_error}')
            # Add placeholder for this day
            predictions.append({
                'date': (start_date + timedelta(days=i)).strftime('%Y-%m-%d'),
                'day_name': (start_date + timedelta(days=i)).strftime('%A'),
                'price': 0.0,
                'error': str(day_error)
            })
    
    return predictions

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html', data=unique_values)

@app.route('/debug/districts', methods=['GET'])
def debug_districts():
    """Debug endpoint to check districts"""
    return jsonify({
        'districts': unique_values['districts'],
        'total': len(unique_values['districts'])
    })

@app.route('/debug/markets/<district>', methods=['GET'])
def debug_markets(district):
    """Debug endpoint to check markets for a district"""
    markets = df[df['district'] == district]['market_name'].unique().tolist()
    return jsonify({
        'district': district,
        'markets': sorted(markets),
        'total': len(markets)
    })

@app.route('/get_markets', methods=['POST'])
def get_markets():
    """Get markets for selected district from database"""
    try:
        # Safely get JSON data
        data = None
        try:
            data = request.get_json(force=True)
        except Exception as json_err:
            logger.error(f'Failed to parse JSON: {json_err}')
            return jsonify({'markets': [], 'error': 'Invalid JSON data'}), 400
        
        logger.info(f"Received request: {data}")
        
        if not data or not isinstance(data, dict):
            logger.error('No data provided or invalid format')
            return jsonify({'markets': [], 'error': 'No data provided'}), 400
            
        district = data.get('district', '').strip()
        if not district:
            logger.error('District is required for /get_markets')
            return jsonify({'markets': [], 'error': 'District is required'}), 400
        
        logger.info(f"Fetching markets for district: '{district}'")
        
        # Query database for markets
        try:
            conn = get_db_connection()
            if not conn:
                logger.error('Failed to connect to database')
                return jsonify({'markets': [], 'error': 'Database connection failed'}), 500
                
            cursor = conn.cursor()
            cursor.execute(
                "SELECT market_name FROM lookup_markets WHERE district = ? ORDER BY market_name",
                (district,)
            )
            markets = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            logger.info(f"[OK] Found {len(markets)} markets for {district}")
            return jsonify({'markets': markets}), 200
            
        except Exception as db_error:
            logger.exception(f'Database error: {db_error}')
            return jsonify({'markets': [], 'error': 'Database query failed'}), 500
            
    except Exception as e:
        logger.exception(f'CRITICAL ERROR in get_markets: {str(e)}')
        return jsonify({'markets': [], 'error': 'Server error, please try again'}), 500

@app.route('/get_varieties', methods=['POST'])
def get_varieties():
    """Get varieties for selected commodity from database"""
    try:
        # Safely get JSON data
        data = None
        try:
            data = request.get_json(force=True)
        except Exception as json_err:
            logger.error(f'Failed to parse JSON: {json_err}')
            return jsonify({'varieties': [], 'error': 'Invalid JSON data'}), 400
            
        if not data or not isinstance(data, dict):
            logger.error('No data provided to /get_varieties')
            return jsonify({'varieties': [], 'error': 'No data provided'}), 400
            
        commodity = data.get('commodity', '').strip()
        if not commodity:
            logger.error('Commodity is required for /get_varieties')
            return jsonify({'varieties': [], 'error': 'Commodity is required'}), 400
        
        logger.info(f"Fetching varieties for commodity: {commodity}")
        
        try:
            # Query database for varieties
            conn = get_db_connection()
            if not conn:
                logger.error('Failed to connect to database')
                return jsonify({'varieties': [], 'error': 'Database connection failed'}), 500
                
            cursor = conn.cursor()
            cursor.execute(
                "SELECT variety FROM lookup_varieties WHERE commodity_name = ? ORDER BY variety",
                (commodity,)
            )
            varieties = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            logger.info(f"[OK] Found {len(varieties)} varieties for {commodity}")
            return jsonify({'varieties': varieties}), 200
            
        except Exception as db_error:
            logger.exception(f'Database error: {db_error}')
            return jsonify({'varieties': [], 'error': 'Database query failed'}), 500
            
    except Exception as e:
        logger.exception(f'CRITICAL ERROR in get_varieties: {str(e)}')
        return jsonify({'varieties': [], 'error': 'Server error, please try again'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Safely get JSON data
        try:
            data = request.get_json(force=True)
        except Exception as json_err:
            logger.error(f'Failed to parse JSON in predict: {json_err}')
            return jsonify({'success': False, 'error': 'Invalid JSON data'}), 400
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        date = data.get('date', '').strip()
        district = data.get('district', '').strip()
        market = data.get('market', '').strip()
        commodity = data.get('commodity', '').strip()
        variety = data.get('variety', '').strip()
        
        # Validate inputs
        if not all([date, district, market, commodity, variety]):
            logger.error('Missing fields in /predict request')
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        
        logger.info(f'Prediction request: {district}/{market}/{commodity}/{variety} on {date}')
        
        # Get predictions
        try:
            predictions = predict_prices(date, district, market, commodity, variety)
            
            # Clean up memory after prediction
            gc.collect()
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'inputs': {
                    'date': date,
                    'district': district,
                    'market': market,
                    'commodity': commodity,
                    'variety': variety
                }
            }), 200
        except Exception as pred_err:
            gc.collect()  # Clean up even on error
            logger.exception(f'Prediction error: {pred_err}')
            return jsonify({'success': False, 'error': f'Prediction failed: {str(pred_err)}'}), 500
    
    except Exception as e:
        gc.collect()  # Clean up even on error
        logger.exception(f'CRITICAL ERROR in /predict: {str(e)}')
        return jsonify({'success': False, 'error': 'Server error, please try again'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health endpoint to check server status"""
    return jsonify({'status': 'ok', 'model_loaded': True}), 200


@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler that logs exceptions and returns JSON"""
    logger.exception(f'Unhandled exception: {str(e)}')
    return jsonify({'error': 'Internal server error'}), 500

# Signal handler to prevent crashes
import signal
def handle_signal(signum, frame):
    logger.info(f'Received signal {signum}, ignoring to keep server running')
    
# Ignore certain signals on Windows
if sys.platform == 'win32':
    try:
        signal.signal(signal.SIGBREAK, handle_signal)
    except:
        pass

if __name__ == '__main__':
    print("="*70)
    print("Commodity Price Prediction System")
    print("="*70)
    print("Model loaded successfully!")
    print(f"Available districts: {len(unique_values['districts'])}")
    print(f"Available commodities: {len(unique_values['commodities'])}")
    
    # Get total markets and varieties from database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM lookup_markets")
        total_markets = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM lookup_varieties")
        total_varieties = cursor.fetchone()[0]
        conn.close()
        print(f"Total markets in database: {total_markets}")
        print(f"Total varieties in database: {total_varieties}")
    except:
        pass
    
    print("="*70)
    print("\nStarting server on http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*70)
    
    # Run with threaded=True for better stability and use_reloader=False to prevent double loading
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except Exception as server_error:
        logger.exception(f'Server error: {server_error}')
        print(f'\nServer crashed: {server_error}')
        print('Restarting is recommended.')
