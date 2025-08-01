import requests
import json
import time

def make_prediction():
    """Make a prediction request to the API."""
    url = "http://localhost:8000/api/v1/predict"
    
    # Sample customer data
    customer_data = {
        'SeniorCitizen': 0,
        'tenure': 1,
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85,
        'gender': 'Female',
        'Partner': 'Yes',
        'Dependents': 'No',
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check'
    }
    
    try:
        response = requests.post(
            url,
            json={"input_data": customer_data},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error making prediction: {e}")
        if hasattr(e, 'response'):
            print(e.response.text)
        return None

def submit_ground_truth(request_id, ground_truth):
    """Submit ground truth for a prediction."""
    url = "http://localhost:8000/api/v1/predict/update-ground-truth"
    
    try:
        # Send as JSON in the request body
        data = {
            'request_id': request_id,
            'ground_truth': ground_truth
        }
        
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error submitting ground truth: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        return None

def get_metrics():
    """Get current metrics from Prometheus."""
    try:
        response = requests.get("http://localhost:8000/metrics")
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error getting metrics: {e}")
        return None

def test_ground_truth_flow():
    print("1. Making prediction...")
    prediction = make_prediction()
    if not prediction:
        print("Failed to make prediction")
        return
        
    print("\n2. Prediction result:")
    print(json.dumps(prediction, indent=2))
    
    # Get the request ID and prediction
    request_id = prediction.get('request_id')
    model_type = prediction['predictions'][0]['model_type']
    predicted_value = prediction['predictions'][0]['prediction']
    
    print(f"\n3. Submitting ground truth (opposite of prediction to test accuracy)...")
    ground_truth = 1 - predicted_value  # Use opposite to test accuracy update
    
    time.sleep(1)  # Small delay to ensure logs are written
    
    result = submit_ground_truth(request_id, ground_truth)
    if not result:
        print("Failed to submit ground truth")
        return
        
    print("\n4. Ground truth submission result:")
    print(json.dumps(result, indent=2))
    
    print("\n5. Current metrics:")
    metrics = get_metrics()
    if metrics:
        # Filter for relevant metrics
        for line in metrics.split('\n'):
            if 'ab_test_accuracy' in line or 'prediction_total' in line or 'prediction_correct' in line:
                print(line)
    
    print("\nTest complete! Check the Grafana dashboard to see the updated metrics.")

if __name__ == "__main__":
    test_ground_truth_flow()
