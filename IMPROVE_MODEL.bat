@echo off
echo ================================================================================
echo MODEL IMPROVEMENT PIPELINE
echo ================================================================================
echo.
echo This will:
echo   1. Evaluate current model accuracy
echo   2. Train improved model with better features
echo   3. Compare old vs new model performance
echo.
echo Estimated time: 15-20 minutes
echo ================================================================================
echo.

cd /d "%~dp0"

echo Step 1: Activating conda environment...
call C:\Users\acer\anaconda3\Scripts\activate.bat tf_env
echo.

echo ================================================================================
echo STEP 1: EVALUATING CURRENT MODEL
echo ================================================================================
echo.
python evaluate_model_accuracy.py
echo.
echo Evaluation complete! Results saved to: model_evaluation_results.csv
echo.
pause
echo.

echo ================================================================================
echo STEP 2: TRAINING IMPROVED MODEL
echo ================================================================================
echo.
echo This will take 10-15 minutes. Please wait...
echo.
python retrain_improved_model.py
echo.
echo Training complete! New model saved to: models/xgboost_improved_model.pkl
echo.
pause
echo.

echo ================================================================================
echo IMPROVEMENT COMPLETE!
echo ================================================================================
echo.
echo Next steps:
echo   1. Check model_evaluation_results.csv for detailed analysis
echo   2. Check feature_importance_improved.csv for feature rankings
echo   3. Update app.py to use the improved model
echo   4. Restart the server and test predictions
echo.
echo To use the improved model:
echo   - Open app.py
echo   - Change line 42 from 'xgboost_final_model.pkl' to 'xgboost_improved_model.pkl'
echo   - Restart server: python run_server.py
echo.
echo ================================================================================
pause
