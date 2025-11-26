# Testing Guide - Commodity Price Prediction System

## Current Status
✅ Backend updated with:
- File logging to `app.log` (rotating logs, 5MB max)
- Health endpoint: `/health`
- Global exception handler
- Enhanced error logging throughout

## Quick Start - Testing Steps

### Step 1: Start the Server

Open **Command Prompt** and run:

```cmd
cd /d C:\Users\acer\Desktop\btp1
conda activate tf_env
python app.py
```

**Or use the batch file:**
```cmd
cd /d C:\Users\acer\Desktop\btp1
restart_server.bat
```

**What to look for:**
- Should print: "Loading model..."
- Should print: "✓ Model loaded successfully"
- Should print: "✓ Dataset loaded: 173094 rows"
- Should print: "Starting server on http://localhost:5000"
- Keep this terminal window OPEN (don't close it)

---

### Step 2: Run Automated Tests

Open a **NEW Command Prompt** window (keep server running in first window):

```cmd
cd /d C:\Users\acer\Desktop\btp1
conda activate tf_env
python run_tests.py
```

**This will test:**
- ✓ Health endpoint
- ✓ Home page loading
- ✓ Debug districts endpoint
- ✓ GET /get_markets for multiple districts
- ✓ GET /get_varieties for multiple commodities
- ✓ POST /predict with real data
- ✓ Error handling (invalid inputs)

**Expected output:**
- All tests should show green checkmarks ✓
- Summary should say "All X tests PASSED!"

---

### Step 3: Manual UI Testing

1. **Open browser:** http://localhost:5000

2. **Test District → Markets:**
   - Select "Burdwan" from District dropdown
   - Markets dropdown should populate with: Asansol, Burdwan, Durgapur, etc.
   - Select "Birbhum" 
   - Markets should change to: Birbhum, Bolpur, Rampurhat, Sainthia

3. **Test Commodity → Varieties:**
   - Select "Rice" from Commodity dropdown
   - Varieties should populate (multiple rice varieties)
   - Select "Wheat"
   - Varieties should change to wheat varieties

4. **Make a Prediction:**
   - District: Burdwan
   - Market: Burdwan
   - Commodity: Rice
   - Variety: Swarna
   - Date: 2025-11-26 (or any recent date)
   - Click "Predict Prices"
   - Should show: Current day price + next 6 days forecast

5. **Test After Prediction (Critical Test):**
   - After prediction results appear
   - Change District to "Birbhum"
   - **Markets dropdown should reload** with Birbhum markets
   - If you see "Error loading markets" → report this

6. **Test Hard Refresh:**
   - Press **Ctrl + Shift + R** (hard refresh)
   - Page should reload
   - Select any district
   - Markets should load (not show error)

---

### Step 4: Check Logs

**Server Console Output:**
- Look in the terminal where `python app.py` is running
- Should see log messages like:
  ```
  2025-11-26 12:34:56 - INFO - Received request: {'district': 'Burdwan'}
  2025-11-26 12:34:56 - INFO - Found 7 markets for Burdwan
  ```

**app.log File:**
```cmd
type app.log
```
- Should contain same logs as console
- Any ERROR or exception will be logged here with full stack traces

---

## Troubleshooting

### Server Won't Start

**Error: "Address already in use"**
```cmd
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F
```
Then restart server.

**Error: "Model not found"**
- Check that `models/xgboost_final_model.pkl` exists
- If not, retrain model by running notebook

**Error: "Dataset not found"**
- Check that `Bengal_Prices_2014-25_final.csv` exists in project root

---

### Tests Fail

**"Server is NOT RUNNING"**
- Make sure server is running in another terminal
- Check Step 1 above

**"Connection refused"**
- Server crashed - check server terminal for errors
- Check `app.log` for exception traces

**Specific endpoint fails:**
- Check server logs for the failing endpoint
- Look for ERROR messages in console or app.log

---

### UI Issues

**"Error loading markets"**
1. Open browser Developer Tools (F12)
2. Go to Console tab
3. Look for red error messages
4. Go to Network tab
5. Find the failed request (will be red)
6. Click it and check:
   - Request payload
   - Response (status code, error message)
7. Share screenshots of Console + Network tabs

**Markets don't reload after prediction:**
1. Check browser console for errors
2. Check Network tab - is `/get_markets` being called?
3. If called but fails, check Response
4. If not called, there's a JS issue

**Dropdowns empty:**
- Check that server is returning data
- In browser, go to: http://localhost:5000/debug/districts
- Should see JSON with list of districts
- If empty, dataset may not be loaded correctly

---

## What to Report

If tests fail or UI has issues, please provide:

1. **Server console output** (last 50 lines):
   ```cmd
   # In the terminal where server is running, copy last ~50 lines
   ```

2. **app.log contents**:
   ```cmd
   type app.log
   ```

3. **Browser console errors** (if UI issue):
   - F12 → Console tab → screenshot or copy errors

4. **Network tab** (if API issue):
   - F12 → Network tab → find failed request → screenshot

5. **Exact steps to reproduce:**
   - "I selected Burdwan, then clicked Predict, then changed to Birbhum, then markets showed error"

---

## Expected Behavior (All Working)

✅ Server starts without errors  
✅ `run_tests.py` - all tests pass  
✅ Browser UI loads at http://localhost:5000  
✅ District selection → Markets populate  
✅ Commodity selection → Varieties populate  
✅ After prediction → Changing district reloads markets  
✅ Hard refresh (Ctrl+Shift+R) → Everything still works  
✅ No errors in browser console  
✅ No errors in `app.log`  

---

## Files Updated

- `app.py` - Added logging, /health endpoint, global error handler
- `run_tests.py` - Comprehensive test suite (NEW)
- `TESTING_GUIDE.md` - This file (NEW)

## Previous Files (Still Available)

- `start_server.bat` - Simple server start
- `restart_server.bat` - Kill & restart server
- `test_api.py` - Basic API tests
- `quick_backtest.py` - Quick model validation
- `backtest_model.py` - Full backtesting with plots
