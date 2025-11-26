# Commodity Price Prediction with XGBoost

A complete machine learning pipeline for predicting commodity prices using XGBoost with GPU acceleration.

## 🚀 Features

- **GPU-Accelerated Training**: Leverages NVIDIA GPU for faster model training
- **Advanced Feature Engineering**: 
  - Time-based features (year, month, quarter, seasonality)
  - Lag features (7, 14, 30, 90 days)
  - Rolling statistics (mean, std)
  - Price change indicators
  - Interaction features
- **Hyperparameter Tuning**: Automated parameter optimization
- **Comprehensive Evaluation**: 
  - Multiple metrics (RMSE, MAE, R², MAPE)
  - Visual analysis (predictions, errors, trends)
  - Feature importance analysis
- **Production Ready**: Model persistence and prediction pipeline

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (for GPU training)
- CUDA Toolkit 11.2+ (for GPU support)

## 🔧 Installation

1. **Clone or download this project**

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Verify GPU support (optional):**

```python
import xgboost as xgb
print(xgb.__version__)
# XGBoost will automatically detect and use GPU if available
```

## 📊 Dataset

Place your dataset `Bengal_Prices_2014-25_final.csv` in the project directory.

**Dataset Features:**
- Date information
- Location data (state, district, market)
- Commodity details
- Weather data (temperature, rainfall)
- Economic indicators (CPI, MSP, subsidies)
- Agricultural metrics (production, area, yield)

## 🎯 Usage

### Training the Model

**Basic training (with GPU and tuning):**
```bash
python main.py
```

**Training options:**
```bash
# Disable GPU
python main.py --no-gpu

# Disable hyperparameter tuning
python main.py --no-tuning

# Custom test size
python main.py --test-size 0.25

# Custom data path
python main.py --data path/to/your/data.csv
```

### Making Predictions

**Predict on new data:**
```bash
python main.py --predict path/to/new_data.csv
```

**Using saved model:**
```bash
python main.py --predict new_data.csv --model models/xgboost_price_predictor.pkl
```

## 📁 Project Structure

```
btp1/
│
├── main.py                          # Main pipeline orchestrator
├── preprocessing.py                 # Data preprocessing module
├── train_model.py                   # XGBoost training module
├── evaluate.py                      # Evaluation & visualization
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── Bengal_Prices_2014-25_final.csv # Dataset
│
├── models/                          # Saved models
│   ├── xgboost_price_predictor.pkl
│   └── xgboost_price_predictor.json
│
└── results/                         # Output files
    ├── predictions_plot.png
    ├── error_distribution.png
    ├── feature_importance.png
    ├── price_trends.png
    ├── predictions.csv
    ├── feature_importance.csv
    └── metrics_report.txt
```

## 📈 Pipeline Workflow

1. **Data Preprocessing**
   - Load and clean data
   - Handle missing values
   - Engineer time-series features
   - Encode categorical variables

2. **Model Training**
   - Split data (train/validation/test)
   - Hyperparameter tuning (optional)
   - Train XGBoost with GPU
   - Early stopping

3. **Model Evaluation**
   - Calculate metrics (RMSE, MAE, R², MAPE)
   - Generate predictions

4. **Visualization & Reports**
   - Prediction plots
   - Error distribution analysis
   - Feature importance
   - Comprehensive reports

5. **Model Persistence**
   - Save trained model
   - Export results

## 🎨 Output Visualizations

The pipeline generates:
- **predictions_plot.png**: Actual vs Predicted scatter + Residual plot
- **error_distribution.png**: Error histograms and Q-Q plot
- **feature_importance.png**: Top features by gain, weight, and cover
- **price_trends.png**: Time-series comparison
- **predictions.csv**: Detailed predictions with errors
- **metrics_report.txt**: Performance summary

## 📊 Model Performance

Expected metrics (will vary based on data):
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

## 🔧 Customization

### Modify Hyperparameters

Edit `train_model.py` → `get_default_params()`:

```python
params = {
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 1000,
    # ... add more parameters
}
```

### Add New Features

Edit `preprocessing.py` → `engineer_features()`:

```python
# Add custom feature engineering
self.df['custom_feature'] = ...
```

### Change Tuning Grid

Edit `train_model.py` → `tune_hyperparameters()`:

```python
param_grid = {
    'max_depth': [6, 8, 10, 12],
    # ... modify grid
}
```

## 💡 Tips for Best Performance

1. **GPU Training**: Ensure CUDA is properly installed
2. **Memory**: Large datasets may require 16GB+ RAM
3. **Feature Engineering**: Domain knowledge improves results
4. **Hyperparameter Tuning**: Enable for best accuracy (takes longer)
5. **Data Quality**: Clean data = better predictions

## 🐛 Troubleshooting

### GPU not detected:
```bash
# Check CUDA installation
nvidia-smi

# Verify XGBoost GPU support
python -c "import xgboost as xgb; print(xgb.config.get_config())"
```

### Out of memory:
- Reduce `n_estimators`
- Decrease `max_depth`
- Use smaller batch sizes
- Disable tuning temporarily

### Poor performance:
- Check data quality
- Add more relevant features
- Enable hyperparameter tuning
- Increase training data

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests.

## 📧 Contact

For questions or support, please create an issue in the repository.

---

**Happy Predicting! 🎯**
