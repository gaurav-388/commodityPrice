"""
SIMPLE Flask Server Launcher
Run this to start the server - keeps running until you press Ctrl+C
"""
import sys
import os

print("="*70)
print("COMMODITY PRICE PREDICTION SERVER")
print("="*70)
print()

# Check files first
required_files = [
    'models/xgboost_final_model.pkl',
    'Bengal_Prices_2014-25_final.csv',
    'templates/index.html'
]

print("Checking required files...")
all_good = True
for fname in required_files:
    exists = os.path.exists(fname)
    status = "OK" if exists else "MISSING"
    print(f"  {fname}: {status}")
    if not exists:
        all_good = False

if not all_good:
    print("\nERROR: Some required files are missing!")
    sys.exit(1)

print("\nAll files OK. Starting server...\n")
print("="*70)

# Now import and run Flask
if __name__ == '__main__':
    # Import app from app.py
    from app import app
    
    print("\nSERVER STARTED!")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*70)
    print()
    
    # Run server
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\nERROR: Server crashed: {e}")
        import traceback
        traceback.print_exc()
