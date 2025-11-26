"""
XGBoost Model Training Module with GPU Support
Includes hyperparameter tuning and model persistence
"""

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import pickle
import json
import time
from datetime import datetime


class XGBoostTrainer:
    def __init__(self, use_gpu=True, enable_tuning=True):
        """
        Initialize XGBoost trainer
        
        Args:
            use_gpu (bool): Whether to use GPU for training
            enable_tuning (bool): Whether to perform hyperparameter tuning
        """
        self.use_gpu = use_gpu
        self.enable_tuning = enable_tuning
        self.model = None
        self.best_params = None
        self.training_history = {}
        
        # Check GPU availability
        if self.use_gpu:
            print("GPU Training ENABLED")
            print(f"XGBoost version: {xgb.__version__}")
        else:
            print("CPU Training Mode")
    
    def get_default_params(self):
        """Get default parameters for XGBoost"""
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1
        }
        
        # Add GPU-specific parameters
        if self.use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['predictor'] = 'gpu_predictor'
            # Remove n_jobs for GPU mode
            del params['n_jobs']
        else:
            params['tree_method'] = 'hist'
        
        return params
    
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        """
        Perform hyperparameter tuning using validation set
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features  
            y_val: Validation target
        
        Returns:
            best_params: Dictionary of best parameters
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        # Parameter grid for tuning
        param_grid = {
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
        }
        
        base_params = self.get_default_params()
        best_score = float('inf')
        best_params = base_params.copy()
        
        # Simple grid search with early stopping
        print("\nTesting parameter combinations...")
        iteration = 0
        total_combinations = 3  # Test 3 key combinations for efficiency
        
        # Key combinations to test
        test_configs = [
            {'max_depth': 8, 'learning_rate': 0.1, 'min_child_weight': 3},
            {'max_depth': 10, 'learning_rate': 0.05, 'min_child_weight': 1},
            {'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 5}
        ]
        
        for config in test_configs:
            iteration += 1
            print(f"\n[{iteration}/{total_combinations}] Testing: {config}")
            
            params = base_params.copy()
            params.update(config)
            
            # Remove n_estimators for training, we'll use early stopping
            n_est = params.pop('n_estimators')
            
            # Train model
            model = xgb.XGBRegressor(**params, n_estimators=n_est)
            
            start_time = time.time()
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            train_time = time.time() - start_time
            
            # Evaluate
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            print(f"  RMSE: {rmse:.2f} | Time: {train_time:.2f}s | Best iteration: {model.best_iteration}")
            
            if rmse < best_score:
                best_score = rmse
                best_params = params.copy()
                best_params['n_estimators'] = model.best_iteration
                print(f"  ✓ New best RMSE: {best_score:.2f}")
        
        print(f"\n{'='*60}")
        print("TUNING COMPLETE")
        print(f"Best RMSE: {best_score:.2f}")
        print(f"Best parameters: {best_params}")
        print(f"{'='*60}")
        
        self.best_params = best_params
        return best_params
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        
        Returns:
            model: Trained XGBoost model
        """
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)
        
        # Perform hyperparameter tuning if enabled
        if self.enable_tuning and X_val is not None and y_val is not None:
            params = self.tune_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            params = self.get_default_params()
        
        # Remove n_estimators for training with early stopping
        n_estimators = params.pop('n_estimators', 1000)
        
        print(f"\nTraining final model...")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        print(f"GPU Mode: {self.use_gpu}")
        
        # Initialize model
        self.model = xgb.XGBRegressor(**params, n_estimators=n_estimators)
        
        # Training
        start_time = time.time()
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_metric=['rmse', 'mae'],
                early_stopping_rounds=50,
                verbose=100
            )
        else:
            self.model.fit(X_train, y_train, verbose=100)
        
        training_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Best iteration: {getattr(self.model, 'best_iteration', n_estimators)}")
        print(f"{'='*60}")
        
        # Store training history
        self.training_history = {
            'training_time': training_time,
            'best_iteration': getattr(self.model, 'best_iteration', n_estimators),
            'params': params,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test target
        
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2,
            'MAPE': mape
        }
        
        print(f"\nTest Set Performance:")
        print(f"  RMSE:     {rmse:.2f}")
        print(f"  MAE:      {mae:.2f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAPE:     {mape:.2f}%")
        print(f"\n{'='*60}")
        
        return metrics, y_pred
    
    def save_model(self, filepath='models/xgboost_model.pkl'):
        """Save the trained model"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'training_history': self.training_history,
            'best_params': self.best_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to: {filepath}")
        
        # Also save as XGBoost native format for better compatibility
        xgb_filepath = filepath.replace('.pkl', '.json')
        self.model.save_model(xgb_filepath)
        print(f"XGBoost native format saved to: {xgb_filepath}")
    
    def load_model(self, filepath='models/xgboost_model.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.training_history = model_data.get('training_history', {})
        self.best_params = model_data.get('best_params', {})
        
        print(f"Model loaded from: {filepath}")
        return self.model
    
    def get_feature_importance(self, feature_names=None, top_n=20):
        """
        Get feature importance scores
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        importance_dict = {
            'gain': self.model.get_booster().get_score(importance_type='gain'),
            'weight': self.model.get_booster().get_score(importance_type='weight'),
            'cover': self.model.get_booster().get_score(importance_type='cover')
        }
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance_dict['gain'].keys()),
            'gain': list(importance_dict['gain'].values()),
            'weight': [importance_dict['weight'].get(f, 0) for f in importance_dict['gain'].keys()],
            'cover': [importance_dict['cover'].get(f, 0) for f in importance_dict['cover'].keys()]
        })
        
        # Map feature names if provided
        if feature_names:
            feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}
            importance_df['feature'] = importance_df['feature'].map(lambda x: feature_map.get(x, x))
        
        # Sort by gain
        importance_df = importance_df.sort_values('gain', ascending=False).head(top_n)
        
        return importance_df


if __name__ == "__main__":
    print("XGBoost Trainer Module")
    print("This module should be imported and used with preprocessed data")
