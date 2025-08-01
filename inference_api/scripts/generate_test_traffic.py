import requests
import random
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# API endpoints
PRODUCTION_API = "http://localhost:8000/api/v1/predict"
SHADOW_API = "http://localhost:8001/api/v1/predict"

# Sample customer data for predictions
SAMPLE_CUSTOMERS = [
    {
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
    },
    # Add more sample customers as needed
]

def preprocess_input(customer_data):
    """Preprocess input data to match the model's expected format."""
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame([customer_data])
    
    # Define numeric and categorical features
    numeric_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    # Create transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform the data
    preprocessor.fit(df)
    
    # Get feature names after one-hot encoding
    numeric_feature_names = numeric_features
    categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([numeric_feature_names, categorical_feature_names])
    
    # Transform the data
    processed_data = preprocessor.transform(df)
    
    # Convert to dictionary with the expected feature names
    processed_dict = {}
    for i, name in enumerate(feature_names):
        # Convert numpy types to Python native types for JSON serialization
        value = processed_data[0, i]
        if hasattr(value, 'item'):  # For numpy types
            value = value.item()
        processed_dict[name] = value
    
    return processed_dict

def make_prediction(api_url, customer_data, ground_truth=None, ab_test_group=None):
    """Make a prediction request to the specified API endpoint."""
    headers = {"Content-Type": "application/json"}
    if ab_test_group:
        headers["X-AB-Test"] = ab_test_group
    
    try:
        # Preprocess the input data
        processed_data = preprocess_input(customer_data)
        
        # Add ground truth if provided
        if ground_truth is not None:
            processed_data["ground_truth"] = ground_truth
        
        # Make the prediction request
        response = requests.post(
            api_url,
            json={"input_data": processed_data},
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction successful - Model: {result['predictions'][0]['model_name']}")
            print(f"  Prediction: {result['predictions'][0]['prediction']}")
            print(f"  Probability: {result['predictions'][0]['probability']:.2f}")
            print(f"  Model Type: {result['predictions'][0]['model_type']}")
            if 'ab_test_group' in result['predictions'][0]:
                print(f"  A/B Test Group: {result['predictions'][0]['ab_test_group']}")
            return result
        else:
            print(f"Prediction failed with status code {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None

def submit_ground_truth(request_id, ground_truth, api_url=PRODUCTION_API):
    """Submit ground truth for a prediction."""
    try:
        response = requests.post(
            f"{api_url}/{request_id}/ground_truth",
            json={"ground_truth": ground_truth}
        )
        
        if response.status_code == 200:
            print(f"Ground truth submitted successfully for request {request_id}")
            return True
        else:
            print(f"Failed to submit ground truth: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Error submitting ground truth: {str(e)}")
        return False

def generate_test_traffic(num_requests=10):
    """Generate test traffic with ground truth submission."""
    for i in range(num_requests):
        print(f"\n--- Test Request {i+1}/{num_requests} ---")
        
        # Randomly select a customer and ground truth
        customer = random.choice(SAMPLE_CUSTOMERS).copy()
        ground_truth = random.choice([0, 1])  # Random ground truth (0 or 1)
        
        # Randomly decide whether to specify A/B test group
        ab_test_group = random.choice(['A', 'B'])  # Always use A/B test group
        
        # Make prediction to the production API (which will route based on A/B test group)
        print(f"\nMaking prediction (A/B Test Group: {ab_test_group}):")
        result = make_prediction(
            PRODUCTION_API,  # This should be your load balancer URL in production
            customer,
            ground_truth=ground_truth,
            ab_test_group=ab_test_group
        )
        
        # If we have a result with a request ID, submit ground truth
        if result and 'request_id' in result:
            # Wait a bit before submitting ground truth
            time.sleep(0.5)
            submit_ground_truth(result['request_id'], ground_truth)
        
        # Wait a bit between requests
        time.sleep(1)

if __name__ == "__main__":
    print("Generating test traffic for A/B testing...")
    generate_test_traffic(num_requests=5)
    print("\nTest traffic generation complete.")
    print("Check your Prometheus and Grafana dashboards for metrics.")
