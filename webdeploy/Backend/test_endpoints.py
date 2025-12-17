#!/usr/bin/env python3
"""
Quick test script to verify all backend endpoints are working
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(method, path, data=None, params=None):
    """Test an endpoint"""
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{path}", params=params, timeout=5)
        elif method == "POST":
            response = requests.post(f"{BASE_URL}{path}", json=data, params=params, timeout=10)
        else:
            return False, f"Unknown method: {method}"
        
        return response.status_code == 200, f"Status: {response.status_code}, Response: {response.text[:200]}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused - is the backend server running?"
    except Exception as e:
        return False, f"Error: {str(e)}"

print("=" * 60)
print("Testing Backend Endpoints")
print("=" * 60)
print()

# Test root endpoint
print("1. Testing GET /")
success, msg = test_endpoint("GET", "/")
print(f"   {'✅' if success else '❌'} {msg}")
print()

# Test health endpoint
print("2. Testing GET /health")
success, msg = test_endpoint("GET", "/health")
print(f"   {'✅' if success else '❌'} {msg}")
print()

# Test predict endpoint with sample data
print("3. Testing POST /predict")
sample_data = {
    "title": "Test Video",
    "hashtags": "#test",
    "category": "Comedy",
    "title_length": 2,
    "hashtag_count": 1,
    "duration_sec": 300,
    "log_duration": 5.7,
    "has_description": 1,
    "category_id": 0
}
success, msg = test_endpoint("POST", "/predict", data=sample_data)
print(f"   {'✅' if success else '❌'} {msg}")
print()

print("=" * 60)
print("If all tests pass, the backend is working correctly!")
print("If you see 'Connection refused', start the backend with:")
print("  cd webdeploy/Backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000")
print("=" * 60)
