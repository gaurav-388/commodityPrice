@echo off
echo ================================================
echo NEURAL NETWORK TRAINING WITH ENHANCED DATA
echo ================================================

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', len(tf.config.list_physical_devices('GPU')))"

python retrain_nn_gpu.py

echo.
echo Training complete!
pause
