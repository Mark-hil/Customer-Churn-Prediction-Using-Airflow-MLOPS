import requests
import json
import concurrent.futures
import time
from datetime import datetime
import random
from collections import defaultdict
from threading import Lock

def make_prediction(i, base_url, results, use_ab_test):
    """Send a prediction request to the API with optional A/B test header."""
    headers = {}
    group = None
    
    # Add A/B test header if specified
    if use_ab_test:
        group = random.choice(['A', 'B'])
        headers['X-AB-Test'] = group
    
    # Sample input data
    data = {
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
    
    start_time = time.time()
    try:
        response = requests.post(base_url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        # Extract model type and other info from response
        model_type = None
        response_group = None
        
        if 'predictions' in result and len(result['predictions']) > 0:
            model_type = result['predictions'][0].get('model_type')
            response_group = result['predictions'][0].get('ab_test_group')
        
        # Store the complete response for analysis
        with results['lock']:
            results['all_responses'].append({
                'status': 'success',
                'ab_test_group': group,
                'model_used': model_type,
                'response_group': response_group,
                'response': result
            })
            
            # Update metrics
            results['success'] += 1
            results['response_times'].append(time.time() - start_time)
            
            if model_type:
                results['model_types'][model_type] += 1
                
            if group:
                results['ab_test_groups'][group] += 1
                
            if response_group:
                results['response_groups'][response_group] += 1
        
        return {
            'status': 'success',
            'model_type': model_type,
            'response_time': time.time() - start_time,
            'ab_test_group': group,
            'response_group': response_group
        }
    except requests.exceptions.HTTPError as errh:
        return {
            "status": "error",
            "status_code": errh.response.status_code,
            "response_time": time.time() - start_time,
            "error": str(errh)
        }
    except requests.exceptions.RequestException as err:
        return {
            "status": "exception",
            "response_time": time.time() - start_time,
            "error": str(err)
        }

def run_concurrent_requests(num_requests=100, max_workers=20, base_url="http://localhost:8083/api/v1/predict"):
    """Run multiple prediction requests concurrently."""
    print(f"Starting {num_requests} concurrent requests with {max_workers} workers...")
    start_time = datetime.now()
    
    # Initialize results with thread-safe lock
    results = {
        "success": 0,
        "errors": 0,
        "exceptions": 0,
        "model_types": defaultdict(int),
        "ab_test_groups": defaultdict(int),
        "response_groups": defaultdict(int),
        "response_times": [],
        "all_responses": [],  # Store all responses for analysis
        "lock": Lock()  # Thread lock for thread-safe updates
    }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_req = {executor.submit(make_prediction, i, base_url, results, random.random() < 0.5): i for i in range(num_requests)}
        for future in concurrent.futures.as_completed(future_to_req):
            result = future.result()
            
            # Track success/error counts
            if result["status"] == "success":
                results["success"] += 1
                # Track model types
                model_type = result["model_type"]
                results["model_types"][model_type] = results["model_types"].get(model_type, 0) + 1
                
                # Track AB test groups
                ab_group = result["ab_test_group"]
                results["ab_test_groups"][ab_group] = results["ab_test_groups"].get(ab_group, 0) + 1
                
                # Track response groups
                resp_group = result["response_group"]
                results["response_groups"][resp_group] = results["response_groups"].get(resp_group, 0) + 1
                
                # Track response times
                results["response_times"].append(result["response_time"])
                
            elif result["status"] == "error":
                results["errors"] += 1
            else:
                results["exceptions"] += 1
    
    # Calculate statistics
    total_time = (datetime.now() - start_time).total_seconds()
    avg_response_time = sum(results["response_times"]) / len(results["response_times"]) if results["response_times"] else 0
    requests_per_second = results["success"] / total_time if total_time > 0 else 0
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Total requests: {num_requests}")
    print(f"Successful: {results['success']}")
    print(f"Errors: {results['errors']}")
    print(f"Exceptions: {results['exceptions']}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Requests per second: {requests_per_second:.2f}")
    print(f"Average response time: {avg_response_time*1000:.2f} ms")
    
    print("\n=== Model Types ===")
    for model, count in results["model_types"].items():
        print(f"{model}: {count} ({(count/num_requests)*100:.1f}%)")
    
    print("\n=== AB Test Groups (request headers) ===")
    # Sort groups handling None values
    sorted_ab_groups = sorted(
        results["ab_test_groups"].items(),
        key=lambda x: (x[0] is None, str(x[0]))
    )
    for group, count in sorted_ab_groups:
        group_str = str(group) if group is not None else 'None (No A/B Test Header)'
        print(f"{group_str}: {count} ({(count/num_requests)*100:.1f}%)")
    
    print("\n=== Response Groups (actual model used) ===")
    # Track model usage for requests without A/B test group
    no_group_models = defaultdict(int)
    
    # Count model usage for requests without A/B test group
    for result in results.get('all_responses', []):
        if result.get('ab_test_group') is None and 'model_used' in result:
            model = result['model_used']
            if model:  # Only count if model_used is not None
                no_group_models[model] += 1
    
    # Convert None to 'none' for sorting
    sorted_groups = sorted(
        results["response_groups"].items(),
        key=lambda x: str(x[0]) if x[0] is not None else 'none'
    )
    
    for group, count in sorted_groups:
        group_str = str(group) if group is not None else 'None (No A/B Test Header)'
        print(f"{group_str}: {count} ({(count/num_requests)*100:.1f}%)")
    
    # Print model distribution for requests without A/B test group
    if no_group_models:
        print("\n=== Model Usage for Requests Without A/B Test Group ===")
        total_no_group = sum(no_group_models.values())
        for model, count in sorted(no_group_models.items()):
            print(f"{model}: {count} ({(count/total_no_group)*100:.1f}% of no-group requests)")
        
        # Print the actual distribution
        print("\n=== A/B Test Group Distribution by Model Type ===")
        total_requests = results['success'] + results['errors']
        
        # Initialize counters
        group_a = 0  # production
        group_b = 0   # shadow
        
        # Count model usage and map to A/B test groups
        for result in results.get('all_responses', []):
            if 'model_used' in result and result['model_used']:
                if result['model_used'] == 'production':
                    group_a += 1
                elif result['model_used'] == 'shadow':
                    group_b += 1
        
        print(f"Total requests: {total_requests}")
        print(f"Group A (production model): {group_a} ({(group_a/total_requests)*100:.1f}%)")
        print(f"Group B (shadow model): {group_b} ({(group_b/total_requests)*100:.1f}%)")
        
        # Show original model distribution for reference
        print("\n=== Original Model Distribution ===")
        all_models = defaultdict(int)
        for result in results.get('all_responses', []):
            if 'model_used' in result and result['model_used']:
                all_models[result['model_used']] += 1
        
        for model, count in sorted(all_models.items()):
            print(f"{model}: {count} ({(count/total_requests)*100:.1f}%)")

if __name__ == "__main__":
    run_concurrent_requests(num_requests=100, max_workers=20)
