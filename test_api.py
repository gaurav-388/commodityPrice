"""
Test script to verify the Flask API endpoints
"""
import requests
import json

BASE_URL = 'http://localhost:5000'

def test_get_markets():
    """Test the get_markets endpoint"""
    print("Testing /get_markets endpoint...")
    
    districts = ['Darjeeling', 'Kolkata', 'Bankura']
    
    for district in districts:
        print(f"\n--- Testing district: {district} ---")
        
        try:
            response = requests.post(
                f'{BASE_URL}/get_markets',
                json={'district': district},
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Markets found: {len(data['markets'])}")
                print(f"Markets: {data['markets']}")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Exception: {str(e)}")

def test_debug_endpoints():
    """Test debug endpoints"""
    print("\n\n=== Testing Debug Endpoints ===")
    
    # Test districts
    try:
        response = requests.get(f'{BASE_URL}/debug/districts')
        print(f"\nDistricts endpoint:")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Total districts: {data['total']}")
        print(f"Districts: {data['districts']}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test markets for Darjeeling
    try:
        response = requests.get(f'{BASE_URL}/debug/markets/Darjeeling')
        print(f"\nMarkets for Darjeeling:")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Total markets: {data['total']}")
        print(f"Markets: {data['markets']}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    print("="*60)
    print("API Testing Script")
    print("="*60)
    print("\nMake sure the Flask server is running on localhost:5000")
    print("Press Enter to continue...")
    input()
    
    test_debug_endpoints()
    test_get_markets()
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)
