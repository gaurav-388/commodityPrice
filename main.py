"""
Main Pipeline Script
Orchestrates the complete commodity price prediction workflow
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from datetime import datetime
import argparse

from preprocessing import DataPreprocessor
from train_model import XGBoostTrainer
from evaluate import ModelEvaluator


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def main(data_path='Bengal_Prices_2014-25_final.csv', 
         use_gpu=True, 
         enable_tuning=True,
         test_size=0.2):
    """
    Main pipeline for commodity price prediction
    
    Args:
        data_path (str): Path to the dataset
        use_gpu (bool): Whether to use GPU for training
        enable_tuning (bool): Whether to perform hyperparameter tuning
        test_size (float): Proportion of data for testing
    """
    
    # Start timestamp
    start_time = datetime.now()
    
    print_banner("COMMODITY PRICE PREDICTION PIPELINE")
    print(f"\n🚀 Starting pipeline at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 Configuration:")
    print(f"   - Data: {data_path}")
    print(f"   - GPU Training: {'ENABLED ✓' if use_gpu else 'DISABLED'}")
    print(f"   - Hyperparameter Tuning: {'ENABLED ✓' if enable_tuning else 'DISABLED'}")
    print(f"   - Test Size: {test_size*100}%")
    
    try:
        # ==================== STEP 1: DATA PREPROCESSING ====================
        print_banner("STEP 1: DATA PREPROCESSING")
        
        preprocessor = DataPreprocessor(data_path)
        X_train, X_test, y_train, y_test, feature_names = preprocessor.run_pipeline(test_size=test_size)
        
        print(f"\n✓ Preprocessing complete")
        print(f"   - Training samples: {X_train.shape[0]:,}")
        print(f"   - Test samples: {X_test.shape[0]:,}")
        print(f"   - Total features: {X_train.shape[1]}")
        
        # ==================== STEP 2: MODEL TRAINING ====================
        print_banner("STEP 2: MODEL TRAINING WITH XGBOOST")
        
        # Split training data for validation (80-20 split of training data)
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 Data splits:")
        print(f"   - Training: {X_train_split.shape[0]:,} samples")
        print(f"   - Validation: {X_val_split.shape[0]:,} samples")
        print(f"   - Test: {X_test.shape[0]:,} samples")
        
        trainer = XGBoostTrainer(use_gpu=use_gpu, enable_tuning=enable_tuning)
        model = trainer.train(X_train_split, y_train_split, X_val_split, y_val_split)
        
        print(f"\n✓ Training complete")
        
        # ==================== STEP 3: MODEL EVALUATION ====================
        print_banner("STEP 3: MODEL EVALUATION")
        
        metrics, y_pred = trainer.evaluate(X_test, y_test)
        
        print(f"\n✓ Evaluation complete")
        
        # ==================== STEP 4: GENERATE VISUALIZATIONS ====================
        print_banner("STEP 4: GENERATING VISUALIZATIONS & REPORTS")
        
        evaluator = ModelEvaluator(output_dir='results')
        importance_df = trainer.get_feature_importance(feature_names=feature_names, top_n=20)
        evaluator.create_comprehensive_report(y_test, y_pred, metrics, importance_df)
        
        print(f"\n✓ Visualizations and reports generated")
        
        # ==================== STEP 5: SAVE MODEL ====================
        print_banner("STEP 5: SAVING MODEL")
        
        trainer.save_model('models/xgboost_price_predictor.pkl')
        
        print(f"\n✓ Model saved successfully")
        
        # ==================== PIPELINE SUMMARY ====================
        print_banner("PIPELINE SUMMARY")
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print(f"\n✓ Pipeline completed successfully!")
        print(f"\n⏱️  Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"\n📈 Final Model Performance:")
        print(f"   - RMSE:     {metrics['RMSE']:.2f} Rs")
        print(f"   - MAE:      {metrics['MAE']:.2f} Rs")
        print(f"   - R² Score: {metrics['R2_Score']:.4f}")
        print(f"   - MAPE:     {metrics['MAPE']:.2f}%")
        
        print(f"\n📁 Outputs:")
        print(f"   - Model:          models/xgboost_price_predictor.pkl")
        print(f"   - Model (JSON):   models/xgboost_price_predictor.json")
        print(f"   - Results:        results/")
        print(f"   - Visualizations: results/*.png")
        print(f"   - Predictions:    results/predictions.csv")
        
        print(f"\n🎉 All done! Model is ready for predictions.")
        print("="*70 + "\n")
        
        return model, metrics
        
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed!")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def predict_new_data(model_path='models/xgboost_price_predictor.pkl', 
                     new_data_path=None):
    """
    Make predictions on new data using trained model
    
    Args:
        model_path (str): Path to saved model
        new_data_path (str): Path to new data CSV
    """
    print_banner("PREDICTION MODE")
    
    trainer = XGBoostTrainer()
    trainer.load_model(model_path)
    
    print(f"✓ Model loaded from: {model_path}")
    
    if new_data_path:
        # Load and preprocess new data
        preprocessor = DataPreprocessor(new_data_path)
        preprocessor.load_data()
        preprocessor.clean_data()
        preprocessor.engineer_features()
        preprocessor.encode_categorical()
        X_new, _ = preprocessor.prepare_features()
        
        # Make predictions
        predictions = trainer.predict(X_new)
        
        print(f"\n✓ Predictions complete!")
        print(f"   - Total predictions: {len(predictions)}")
        print(f"   - Price range: {predictions.min():.2f} - {predictions.max():.2f} Rs")
        print(f"   - Average price: {predictions.mean():.2f} Rs")
        
        # Save predictions
        output_df = preprocessor.df.copy()
        output_df['predicted_price'] = predictions
        output_df.to_csv('predictions_output.csv', index=False)
        
        print(f"\n📁 Predictions saved to: predictions_output.csv")
    else:
        print("⚠️  No new data provided for prediction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Commodity Price Prediction Pipeline')
    
    parser.add_argument('--data', type=str, default='Bengal_Prices_2014-25_final.csv',
                       help='Path to the dataset')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU training')
    parser.add_argument('--no-tuning', action='store_true',
                       help='Disable hyperparameter tuning')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--predict', type=str, default=None,
                       help='Path to new data for prediction')
    parser.add_argument('--model', type=str, default='models/xgboost_price_predictor.pkl',
                       help='Path to saved model for prediction')
    
    args = parser.parse_args()
    
    if args.predict:
        # Prediction mode
        predict_new_data(model_path=args.model, new_data_path=args.predict)
    else:
        # Training mode
        main(
            data_path=args.data,
            use_gpu=not args.no_gpu,
            enable_tuning=not args.no_tuning,
            test_size=args.test_size
        )
