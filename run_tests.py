"""
Comprehensive API Test Suite for Commodity Price Prediction System
Tests all endpoints systematically with detailed reporting
"""
import requests
import json
from datetime import datetime
import time
import sys

BASE_URL = "http://localhost:5000"
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(title):
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{title.center(70)}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

def print_success(message):
    print(f"{GREEN}✓ {message}{RESET}")

def print_error(message):
    print(f"{RED}✗ {message}{RESET}")

def print_warning(message):
    print(f"{YELLOW}⚠ {message}{RESET}")

def print_info(message):
    print(f"  {message}")

def check_server():
    """Check if server is running"""
    print_header("SERVER STATUS CHECK")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Server is RUNNING")
            print_info(f"Health check response: {data}")
            return True
        else:
            print_error(f"Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Server is NOT RUNNING (Connection refused)")
        print_warning("\nTo start the server:")
        print_info("1. Open Command Prompt")
        print_info("2. cd /d C:\\Users\\acer\\Desktop\\btp1")
        print_info("3. conda activate tf_env")
        print_info("4. python app.py")
        print_info("\nOr simply run: restart_server.bat")
        return False
    except Exception as e:
        print_error(f"Error checking server: {str(e)}")
        return False

def test_home_page():
    """Test home page loads"""
    print_header("HOME PAGE TEST")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print_success("Home page loaded successfully")
            print_info(f"Response length: {len(response.text)} bytes")
            # Check for key elements
            if 'Commodity Price Prediction' in response.text:
                print_success("Page contains expected title")
            if 'district' in response.text.lower():
                print_success("Page contains district field")
            return True
        else:
            print_error(f"Home page returned status code: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error loading home page: {str(e)}")
        return False

def test_debug_districts():
    """Test debug districts endpoint"""
    print_header("DEBUG DISTRICTS ENDPOINT TEST")
    try:
        response = requests.get(f"{BASE_URL}/debug/districts", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Retrieved {data['total']} districts")
            print_info(f"Sample districts: {data['districts'][:5]}")
            return True, data['districts']
        else:
            print_error(f"Districts endpoint returned: {response.status_code}")
            return False, []
    except Exception as e:
        print_error(f"Error fetching districts: {str(e)}")
        return False, []

def test_get_markets():
    """Test get_markets endpoint"""
    print_header("GET MARKETS ENDPOINT TEST")
    
    test_districts = ['Burdwan', 'Birbhum', 'Darjeeling']
    all_passed = True
    
    for district in test_districts:
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
                print_success(f"{district}: {len(markets)} markets found")
                print_info(f"Markets: {', '.join(markets[:3])}{'...' if len(markets) > 3 else ''}")
            elif response.status_code == 404:
                print_warning(f"{district}: Not found in dataset")
                all_passed = False
            else:
                print_error(f"{district}: Status code {response.status_code}")
                print_info(f"Response: {response.text}")
                all_passed = False
        except Exception as e:
            print_error(f"{district}: {str(e)}")
            all_passed = False
        
        time.sleep(0.1)
    
    return all_passed

def test_get_varieties():
    """Test get_varieties endpoint"""
    print_header("GET VARIETIES ENDPOINT TEST")
    
    test_commodities = ['Rice', 'Wheat', 'Potato']
    all_passed = True
    
    for commodity in test_commodities:
        try:
            response = requests.post(
                f"{BASE_URL}/get_varieties",
                json={'commodity': commodity},
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                varieties = data.get('varieties', [])
                print_success(f"{commodity}: {len(varieties)} varieties found")
                print_info(f"Varieties: {', '.join(varieties[:5])}{'...' if len(varieties) > 5 else ''}")
            elif response.status_code == 404:
                print_warning(f"{commodity}: Not found in dataset")
                all_passed = False
            else:
                print_error(f"{commodity}: Status code {response.status_code}")
                print_info(f"Response: {response.text}")
                all_passed = False
        except Exception as e:
            print_error(f"{commodity}: {str(e)}")
            all_passed = False
        
        time.sleep(0.1)
    
    return all_passed

def test_predict():
    """Test prediction endpoint"""
    print_header("PREDICTION ENDPOINT TEST")
    
    # Test prediction with valid data
    test_payload = {
        'date': '2025-11-26',
        'district': 'Burdwan',
        'market': 'Burdwan',
        'commodity': 'Rice',
        'variety': 'Swarna'
    }
    
    try:
        print_info(f"Testing prediction with: {test_payload}")
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                predictions = data.get('predictions', [])
                print_success(f"Prediction successful! Got {len(predictions)} day forecasts")
                
                # Display first 3 predictions
                for i, pred in enumerate(predictions[:3]):
                    print_info(f"Day {i+1} ({pred['date']} - {pred['day_name']}): Rs {pred['price']}")
                
                if len(predictions) > 3:
                    print_info(f"... and {len(predictions)-3} more days")
                
                return True
            else:
                print_error("Prediction returned success=False")
                print_info(f"Response: {json.dumps(data, indent=2)}")
                return False
        else:
            print_error(f"Prediction failed with status code: {response.status_code}")
            print_info(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Error during prediction: {str(e)}")
        return False

def test_error_handling():
    """Test error handling with invalid inputs"""
    print_header("ERROR HANDLING TEST")
    
    tests = [
        {
            'name': 'Missing district in get_markets',
            'url': f"{BASE_URL}/get_markets",
            'data': {},
            'expected': 400
        },
        {
            'name': 'Missing commodity in get_varieties',
            'url': f"{BASE_URL}/get_varieties",
            'data': {},
            'expected': 400
        },
        {
            'name': 'Invalid district in get_markets',
            'url': f"{BASE_URL}/get_markets",
            'data': {'district': 'InvalidDistrict123'},
            'expected': 404
        },
        {
            'name': 'Missing fields in predict',
            'url': f"{BASE_URL}/predict",
            'data': {'date': '2025-11-26'},
            'expected': 400
        }
    ]
    
    all_passed = True
    for test in tests:
        try:
            response = requests.post(
                test['url'],
                json=test['data'],
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == test['expected']:
                print_success(f"{test['name']}: Correctly returned {response.status_code}")
            else:
                print_error(f"{test['name']}: Expected {test['expected']}, got {response.status_code}")
                all_passed = False
        except Exception as e:
            print_error(f"{test['name']}: {str(e)}")
            all_passed = False
        
        time.sleep(0.1)
    
    return all_passed

def main():
    """Run all tests"""
    print_header("COMMODITY PRICE PREDICTION - API TEST SUITE")
    print_info(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"Base URL: {BASE_URL}")
    
    results = {}
    
    # Check if server is running
    if not check_server():
        print_error("\nTests cannot proceed - server is not running!")
        sys.exit(1)
    
    # Run all tests
    results['home_page'] = test_home_page()
    results['debug_districts'] = test_debug_districts()[0]
    results['get_markets'] = test_get_markets()
    results['get_varieties'] = test_get_varieties()
    results['predict'] = test_predict()
    results['error_handling'] = test_error_handling()
    
    # Print summary
    print_header("TEST SUMMARY")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n{BLUE}{'='*70}{RESET}")
    if passed == total:
        print(f"{GREEN}All {total} tests PASSED! ✓{RESET}")
    else:
        print(f"{YELLOW}{passed}/{total} tests passed ({total-passed} failed){RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    # Additional info
    print_info("Next steps:")
    print_info("1. Open browser: http://localhost:5000")
    print_info("2. Test the UI manually")
    print_info("3. Check app.log for any errors")
    print_info("4. Try hard refresh (Ctrl+Shift+R) and test dropdowns")

if __name__ == "__main__":
    main()
