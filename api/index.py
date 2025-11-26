"""
Vercel Serverless Function for Commodity Price Prediction
Simplified version for serverless deployment
"""
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
app.config['DEBUG'] = False
app.config['JSON_AS_ASCII'] = False

# Global variables for models
xgb_model = None
label_encoders = None
feature_columns = None

def load_models():
    """Load XGBoost model and encoders"""
    global xgb_model, label_encoders, feature_columns
    
    models_dir = os.path.join(BASE_DIR, 'models')
    
    try:
        # Load XGBoost model
        model_path = os.path.join(models_dir, 'xgboost_FULL.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                xgb_model = pickle.load(f)
        
        # Load label encoders
        encoder_path = os.path.join(models_dir, 'label_encoders.pkl')
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                label_encoders = pickle.load(f)
        
        # Load feature columns
        features_path = os.path.join(models_dir, 'feature_columns.json')
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                feature_columns = json.load(f)
                
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Sample data for dropdowns (simplified for serverless)
DISTRICTS = ['Medinipur(W)', 'Hooghly', 'Burdwan', 'Nadia', 'Murshidabad', 'North 24 Parganas', 'South 24 Parganas', 'Howrah', 'Birbhum', 'Bankura']
COMMODITIES = ['Rice', 'Wheat', 'Jute']
MARKETS_BY_DISTRICT = {
    'Medinipur(W)': ['Ghatal', 'Kharagpur', 'Midnapore'],
    'Hooghly': ['Arambagh', 'Chinsurah', 'Dankuni'],
    'Burdwan': ['Burdwan', 'Durgapur', 'Asansol'],
    'Nadia': ['Krishnanagar', 'Nabadwip', 'Ranaghat'],
    'Murshidabad': ['Berhampore', 'Lalbagh', 'Jangipur'],
    'North 24 Parganas': ['Barasat', 'Basirhat', 'Habra'],
    'South 24 Parganas': ['Baruipur', 'Diamond Harbour', 'Canning'],
    'Howrah': ['Howrah', 'Uluberia', 'Shyampur'],
    'Birbhum': ['Suri', 'Bolpur', 'Rampurhat'],
    'Bankura': ['Bankura', 'Bishnupur', 'Sonamukhi']
}
VARIETIES = {'Rice': ['Common', 'Fine', 'Superfine'], 'Wheat': ['Desi', 'Lokwan'], 'Jute': ['TD-5', 'JRO']}

@app.route('/')
def index():
    """Main page"""
    data = {
        'districts': DISTRICTS,
        'commodities': COMMODITIES
    }
    try:
        return render_template('index_react.html', data=data)
    except:
        return render_template('index.html', data=data)

@app.route('/get_markets', methods=['POST'])
def get_markets():
    """Get markets for a district"""
    try:
        data = request.get_json()
        district = data.get('district', '')
        markets = MARKETS_BY_DISTRICT.get(district, [])
        return jsonify({'markets': markets})
    except Exception as e:
        return jsonify({'error': str(e), 'markets': []})

@app.route('/get_varieties', methods=['POST'])
def get_varieties():
    """Get varieties for a commodity"""
    try:
        data = request.get_json()
        commodity = data.get('commodity', '')
        varieties = VARIETIES.get(commodity, [])
        return jsonify({'varieties': varieties})
    except Exception as e:
        return jsonify({'error': str(e), 'varieties': []})

@app.route('/predict', methods=['POST'])
def predict():
    """Make price predictions"""
    try:
        data = request.get_json()
        
        # Extract inputs
        date_str = data.get('date')
        district = data.get('district')
        market = data.get('market')
        commodity = data.get('commodity')
        variety = data.get('variety')
        
        # Parse date
        start_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Generate 7-day predictions (simplified demo)
        predictions = []
        base_prices = {'Rice': 3500, 'Wheat': 2800, 'Jute': 5200}
        base_price = base_prices.get(commodity, 3000)
        
        for i in range(7):
            pred_date = start_date + timedelta(days=i)
            # Add some variation
            variation = np.random.uniform(-50, 50)
            price = base_price + (i * 10) + variation
            
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'day_name': pred_date.strftime('%A'),
                'price': round(price, 2)
            })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'inputs': {
                'district': district,
                'market': market,
                'commodity': commodity,
                'variety': variety,
                'model': data.get('model', 'xgboost')
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Commodity Price Prediction API is running'})

# Initialize models on startup
load_models()

# For Vercel
app = app
