#!/usr/bin/env python3
"""
A/B Testing Test Script

This script sends requests to the A/B testing setup and verifies the responses.
"""
import requests
import json
import time
import random
from typing import Dict, Any, Optional

# Configuration
BASE_URL = "http://localhost"  # NGINX is exposed on port 80
TEST_ENDPOINT = "/api/v1/predict"
SUMMARY_ENDPOINT = "/api/v1/management/ab-test/summary"

# Sample customer data for testing
SAMPLE_CUSTOMER = {
    "customer_id": "test_customer",
    "credit_score": 700,
    "country": "France",
    "gender": "Male",
    "age": 40,
    "tenure": 3,
    "balance": 50000.0,
    "products_number": 3,
    "credit_card": 1,
    "active_member": 1,
    "estimated_salary": 100000.0
}

def send_prediction_request(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send a prediction request to the API."""
    url = f"{BASE_URL}{TEST_ENDPOINT}"
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=customer_data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending prediction request: {e}")
        return {}

def get_ab_test_summary() -> Dict[str, Any]:
    """Get the A/B test summary."""
    url = f"{BASE_URL}{SUMMARY_ENDPOINT}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting A/B test summary: {e}")
        return {}

def update_ground_truth(request_id: str, ground_truth: int) -> bool:
    """Update the ground truth for a prediction."""
    url = f"{BASE_URL}{TEST_ENDPOINT}/update-ground-truth"
    params = {"request_id": request_id, "ground_truth": ground_truth}
    
    try:
        response = requests.post(url, params=params)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error updating ground truth: {e}")
        return False

def run_test_cycle(num_requests: int = 10) -> None:
    """Run a test cycle with the specified number of requests."""
    print(f"Starting test cycle with {num_requests} requests...\n")
    
    # Make prediction requests
    for i in range(num_requests):
        # Randomize customer data slightly
        customer = SAMPLE_CUSTOMER.copy()
        customer["credit_score"] = random.randint(600, 800)
        customer["age"] = random.randint(20, 70)
        customer["balance"] = random.uniform(0, 200000)
        
        # Send prediction request
        print(f"Sending request {i+1}/{num_requests}...")
        response = send_prediction_request(customer)
        
        # Print response details
        if response:
            print(f"  Request ID: {response.get('request_id')}")
            print(f"  Prediction: {response.get('prediction')}")
            print(f"  Model: {response.get('model_name')}")
            print(f"  A/B Group: {response.get('ab_test_group')}")
            
            # Randomly update ground truth for some predictions
            if random.random() < 0.5:  # 50% chance to update ground truth
                ground_truth = random.randint(0, 1)
                if update_ground_truth(response['request_id'], ground_truth):
                    print(f"  Updated ground truth to: {ground_truth}")
        
        print("-" * 50)
        time.sleep(0.5)  # Small delay between requests
    
    # Get and display A/B test summary
    print("\nTest complete. Getting A/B test summary...\n")
    summary = get_ab_test_summary()
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    # Run a test cycle with 10 requests
    run_test_cycle(10)
