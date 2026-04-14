import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

TEST_CASES = [
    {
        "text": "This is a safe investment analysis of Apple stocks.",
        "expected_hint": "Safe"
    },
    {
        "text": "URGENT: 500% returns guaranteed in 2 days! Click this link to invest now!",
        "expected_hint": "Scam"
    },
    {
        "text": "Elon Musk says this new crypto will moon tomorrow. Buy now or miss out!",
        "expected_hint": "Misleading"
    },
    {
        "text": "Highly leveraged options trading carries significant risk of total capital loss.",
        "expected_hint": "High Risk"
    }
]

def test_health():
    print("\n[Test] Checking Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}, Response: {response.json()}")
    except Exception as e:
        print(f"Health Check Failed: {e}")

def test_classification():
    print("\n[Test] Checking Classification Endpoint...")
    for i, case in enumerate(TEST_CASES):
        print(f"\n--- Test Case {i+1} ---")
        payload = {"text": case["text"]}
        
        start = time.time()
        try:
            response = requests.post(f"{BASE_URL}/v1/classify", json=payload)
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                res = response.json()
                print(f"Input   : {case['text'][:50]}...")
                print(f"Label   : {res['label']}")
                print(f"Score   : {res['risk_score']}")
                print(f"Latency : {res['latency_ms']}ms (API) / {latency:.1f}ms (Total)")
                print(f"Probs   : {res['probabilities']}")
            else:
                print(f"Failed : {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error  : {e}")

if __name__ == "__main__":
    print("Ensure the FastAPI server is running (uvicorn app:app) before testing.")
    test_health()
    test_classification()
