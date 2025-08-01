#!/usr/bin/env python3
"""
NGINX Traffic Splitting Test Script

This script tests the NGINX routing configuration by sending requests with and without
the X-AB-Test header and analyzing the routing behavior.
"""
import requests
import json
import time
import random
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

def make_request(url, headers=None, request_id=None):
    """Make a single request to the API endpoint."""
    if headers is None:
        headers = {}
    
    # Add request ID if provided
    if request_id:
        headers['X-Request-ID'] = request_id
    
    # Sample input data
    data = {
        "SeniorCitizen": 0,
        "tenure": random.randint(1, 72),
        "MonthlyCharges": round(random.uniform(20, 120), 2),
        "TotalCharges": round(random.uniform(20, 1000), 2),
        "gender": random.choice(["Male", "Female"]),
        "Partner": random.choice(["Yes", "No"]),
        "Dependents": random.choice(["Yes", "No"]),
        "PhoneService": random.choice(["Yes", "No"]),
        "MultipleLines": random.choice(["No phone service", "No", "Yes"]),
        "InternetService": random.choice(["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": random.choice(["No", "Yes", "No internet service"]),
        "OnlineBackup": random.choice(["No", "Yes", "No internet service"]),
        "DeviceProtection": random.choice(["No", "Yes", "No internet service"]),
        "TechSupport": random.choice(["No", "Yes", "No internet service"]),
        "StreamingTV": random.choice(["No", "Yes", "No internet service"]),
        "StreamingMovies": random.choice(["No", "Yes", "No internet service"]),
        "Contract": random.choice(["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": random.choice(["Yes", "No"]),
        "PaymentMethod": random.choice([
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=10
        )
        response_time = (time.time() - start_time) * 1000  # in milliseconds
        
        result = {
            'status': 'success',
            'status_code': response.status_code,
            'response_time_ms': response_time,
            'request_id': request_id,
            'headers': dict(response.request.headers)
        }
        
        if response.status_code == 200:
            try:
                # Get the response data
                resp_data = response.json()
                predictions = resp_data.get('predictions', [{}])
                first_pred = predictions[0] if predictions else {}
                
                # Get the server port from the response headers
                server_port = response.headers.get('X-Forwarded-Port')
                if not server_port and 'x-forwarded-port' in response.headers:
                    server_port = response.headers['x-forwarded-port']
                
                # Also try to get the port from the response body if available
                if not server_port and 'server' in response.headers:
                    server_info = response.headers['server']
                    if 'port' in server_info.lower():
                        import re
                        port_match = re.search(r'port[\s:=]*(\d+)', server_info.lower())
                        if port_match:
                            server_port = port_match.group(1)
                
                result.update({
                    'model_type': first_pred.get('model_type'),
                    'ab_test_group': first_pred.get('ab_test_group'),
                    'response_port': server_port,
                    'response_headers': dict(response.headers),
                    'response_body': resp_data
                })
            except (json.JSONDecodeError, IndexError, AttributeError) as e:
                result.update({
                    'error': f'Response parsing error: {str(e)}',
                    'response_text': response.text[:500]  # First 500 chars of response
                })
        else:
            result.update({
                'error': f'Request failed with status {response.status_code}',
                'response_text': response.text[:500]  # First 500 chars of response
            })
            
    except Exception as e:
        result = {
            'status': 'error',
            'error': str(e),
            'request_id': request_id,
            'headers': dict(headers) if headers else {}
        }
    
    return result

def run_test(url, num_requests=100, max_workers=10, ab_test_ratio=0.5):
    """Run the NGINX routing test."""
    results = {
        'total_requests': 0,
        'successful': 0,
        'errors': 0,
        'response_times': [],
        'status_codes': defaultdict(int),
        'model_types': defaultdict(int),
        'ab_test_groups': defaultdict(int),
        'ports': defaultdict(int),
        'header_usage': {
            'with_ab_test': 0,
            'without_ab_test': 0
        },
        'details': []
    }
    
    print(f"Starting {num_requests} requests to {url} with {max_workers} workers...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures with different header configurations
        futures = []
        for i in range(num_requests):
            request_id = f"test-{i:04d}-{int(time.time() * 1000)}"
            
            # Initialize headers
            headers = {}
            
            # Add request ID header
            headers['X-Request-ID'] = request_id
            
            # Handle header configuration based on arguments
            if not args.no_headers and random.random() < ab_test_ratio:
                # Use A/B test header
                group = random.choice(['A', 'B'])
                headers['X-AB-Test'] = group
                results['header_usage']['with_ab_test'] += 1
            else:
                # No A/B test header
                results['header_usage']['without_ab_test'] += 1
            
            # Submit the request
            future = executor.submit(
                make_request,
                url=url,
                headers=headers,
                request_id=request_id
            )
            futures.append(future)
        
        # Process results as they complete
        for future in as_completed(futures):
            result = future.result()
            results['total_requests'] += 1
            results['details'].append(result)
            
            if result['status'] == 'success':
                results['successful'] += 1
                results['status_codes'][result['status_code']] += 1
                results['response_times'].append(result['response_time_ms'])
                
                # Track model types and AB test groups
                if 'model_type' in result and result['model_type']:
                    results['model_types'][result['model_type']] += 1
                
                if 'ab_test_group' in result:
                    group = result['ab_test_group']
                    if group is None:
                        group = 'None'
                    results['ab_test_groups'][group] += 1
                
                if 'response_port' in result and result['response_port']:
                    results['ports'][result['response_port']] += 1
                
                # Print progress
                if results['successful'] % 10 == 0:
                    print(f"  Completed {results['successful']} successful requests...")
            else:
                results['errors'] += 1
                print(f"  Error on request: {result.get('error', 'Unknown error')}")
    
    # Calculate summary statistics
    total_time = time.time() - start_time
    avg_response_time = sum(results['response_times']) / len(results['response_times']) if results['response_times'] else 0
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Total requests: {results['total_requests']}")
    print(f"Successful: {results['successful']} ({results['successful']/results['total_requests']*100:.1f}%)")
    print(f"Errors: {results['errors']} ({results['errors']/results['total_requests']*100:.1f}%)")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Requests per second: {results['successful']/total_time:.2f}")
    print(f"Average response time: {avg_response_time:.2f} ms")
    
    print("\n=== Status Codes ===")
    for code, count in sorted(results['status_codes'].items()):
        print(f"  {code}: {count} ({count/results['total_requests']*100:.1f}%)")
    
    print("\n=== Model Types ===")
    for model, count in sorted(results['model_types'].items()):
        print(f"  {model}: {count} ({count/results['successful']*100:.1f}%)")
    
    print("\n=== A/B Test Groups ===")
    print(f"  Requests with A/B test header: {results['header_usage']['with_ab_test']} ({results['header_usage']['with_ab_test']/results['total_requests']*100:.1f}%)")
    print(f"  Requests without A/B test header: {results['header_usage']['without_ab_test']} ({results['header_usage']['without_ab_test']/results['total_requests']*100:.1f}%)")
    
    if results['ab_test_groups']:
        print("\n  A/B Test Group Distribution:")
        for group, count in sorted(results['ab_test_groups'].items()):
            print(f"    {group}: {count} ({count/sum(results['ab_test_groups'].values())*100:.1f}%)")
    
    print("\n=== Ports ===")
    for port, count in sorted(results['ports'].items()):
        print(f"  Port {port}: {count} ({count/results['successful']*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test NGINX A/B Testing Configuration')
    parser.add_argument('--url', type=str, default='http://localhost:8083/api/v1/predict',
                       help='Base URL of the API endpoint')
    parser.add_argument('--requests', type=int, default=100,
                       help='Number of requests to send')
    parser.add_argument('--workers', type=int, default=10,
                       help='Number of concurrent workers')
    parser.add_argument('--ab-ratio', type=float, default=0.0,
                       help='Ratio of requests to send with A/B test headers (0.0 to 1.0). Set to 0 for no headers.')
    parser.add_argument('--no-headers', action='store_true',
                       help='Send requests without any headers (overrides --ab-ratio)')
    
    args = parser.parse_args()
    
    print(f"Testing NGINX A/B Testing Configuration")
    print(f"Endpoint: {args.url}")
    print(f"Total Requests: {args.requests}")
    print(f"Concurrent Workers: {args.workers}")
    print(f"A/B Test Header Ratio: {args.ab_ratio*100:.0f}%\n")
    
    run_test(
        url=args.url,
        num_requests=args.requests,
        max_workers=args.workers,
        ab_test_ratio=args.ab_ratio
    )
