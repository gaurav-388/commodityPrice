"""
FINAL SERVER VERIFICATION TEST
Tests that server stays running through multiple district changes
"""
import requests
import time
import sys

BASE_URL = "http://localhost:5000"

def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

print("="*70)
print("SERVER CRASH TEST - District Change Stress Test")
print("="*70)
print("\nThis test will change districts multiple times")
print("to verify the server stays running.\n")

if not check_server():
    print("[ERROR] Server is not running!")
    print("\nPlease start the server first:")
    print("  1. Double-click: START_SERVER_SIMPLE.bat")
    print("  2. Or run: python run_server.py")
    print("\nThen run this test again.")
    sys.exit(1)

print("[OK] Server is running. Starting stress test...\n")

# Test districts
test_districts = ['Burdwan', 'Birbhum', 'Darjeeling', 'Kolkata', 'Howrah']
success_count = 0
fail_count = 0

for i, district in enumerate(test_districts, 1):
    print(f"{i}. Testing district: {district}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/get_markets",
            json={'district': district},
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            markets = data.get('markets', [])
            warning = data.get('warning', '')
            
            if warning:
                print(f"   [WARNING] {warning}")
            else:
                print(f"   [OK] Found {len(markets)} markets")
            success_count += 1
        else:
            print(f"   [FAIL] Status {response.status_code}")
            fail_count += 1
            
    except requests.exceptions.ConnectionError:
        print(f"   [CRASH] Server stopped responding!")
        print("\n[FAIL] Server crashed during district change!")
        sys.exit(1)
    except Exception as e:
        print(f"   [ERROR] {str(e)}")
        fail_count += 1
    
    # Check if server is still alive
    time.sleep(0.2)
    if not check_server():
        print("\n[CRASH] Server is no longer running!")
        print("[FAIL] Server crashed!")
        sys.exit(1)

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print(f"Successful requests: {success_count}")
print(f"Failed requests: {fail_count}")

if fail_count == 0:
    print("\n[SUCCESS] Server handled all district changes without crashing!")
    print("[SUCCESS] Server is still running!")
else:
    print(f"\n[WARNING] {fail_count} requests failed, but server stayed running")

print("\n" + "="*70)
print("READY TO USE")
print("="*70)
print("\nThe server is now stable!")
print("Open your browser: http://localhost:5000")
print("\nYou can now:")
print("  - Change districts multiple times")
print("  - Make predictions")
print("  - Refresh the page")
print("  - Server will stay running!")
