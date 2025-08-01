from fastapi import FastAPI, Request, Response, status
"""
The above function is a middleware in a FastAPI application that logs all incoming requests and
responses.
:return: In the provided code snippet, the `log_requests` middleware function is logging information
about incoming requests before processing them. After logging the request details, the middleware
function then proceeds to process the request by calling the `call_next` function, which represents
the next middleware or the actual endpoint handler.
"""
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from .config import API_PREFIX, API_TITLE, API_DESCRIPTION, API_VERSION, DEBUG
from .api.endpoints import predict, management
import logging
import time
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json"
)

# Include routers
app.include_router(predict.router, prefix=API_PREFIX)
app.include_router(management.router)  # Management endpoints don't use API_PREFIX

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Churn Prediction API",
        "version": API_VERSION,
        "docs": "/api/v1/docs"
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests and responses."""
    # Log request
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
    
    # Calculate processing time
    process_time = (time.time() - start_time) * 1000
    process_time = round(process_time, 2)
    
    # Log response
    logger.info(
        f"Response: {request.method} {request.url} - "
        f"Status: {response.status_code} - "
        f"Process Time: {process_time}ms"
    )
    
    # Add headers
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Add GZip compression for responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize Prometheus metrics with safe registration
def register_metrics():
    """Register Prometheus metrics, handling potential duplicate registration."""
    from prometheus_client import Counter, Histogram, Gauge, REGISTRY, start_http_server
    
    # Unregister any existing metrics with our prefix
    for name, metric in list(REGISTRY._names_to_collectors.items()):
        if name.startswith('ab_test_'):
            try:
                REGISTRY.unregister(metric)
            except KeyError:
                pass  # Already unregistered
    
    # Start HTTP server for metrics (if not already started)
    try:
        start_http_server(8002)  # Use a different port than the main app
    except OSError:
        pass  # Server already started
    
    # Create metrics dictionary
    metrics = {}
    
    # Define metrics with unique names
    metrics['REQUEST_COUNT'] = Counter(
        'ab_test_request_count',
        'Total number of requests',
        ['method', 'endpoint', 'status_code', 'model']
    )
    
    metrics['REQUEST_LATENCY'] = Histogram(
        'ab_test_request_duration_seconds',
        'Request latency in seconds',
        ['method', 'endpoint', 'model']
    )
    
    metrics['MODEL_ACCURACY'] = Gauge(
        'ab_test_accuracy',
        'Model accuracy (updated when ground truth is available)',
        ['model_type']
    )
    
    metrics['PREDICTION_COUNTER'] = Counter(
        'ab_test_predictions_total',
        'Total number of predictions',
        ['model_type', 'status']
    )
    
    metrics['PREDICTION_PROBABILITY'] = Histogram(
        'ab_test_prediction_probability',
        'Prediction probability distribution',
        ['model_type', 'prediction_class'],
        buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    )
    
    metrics['PREDICTION_TIME'] = Histogram(
        'ab_test_prediction_time_seconds',
        'Prediction processing time in seconds',
        ['model_type'],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf'))
    )
    
    # Initialize counters and gauges with default values
    for model_type in ['production', 'shadow']:
        metrics['PREDICTION_COUNTER'].labels(model_type=model_type, status='total')._value.set(0.0)
        metrics['PREDICTION_COUNTER'].labels(model_type=model_type, status='correct')._value.set(0.0)
        metrics['PREDICTION_COUNTER'].labels(model_type=model_type, status='success')._value.set(0.0)
        metrics['PREDICTION_COUNTER'].labels(model_type=model_type, status='error')._value.set(0.0)
        metrics['MODEL_ACCURACY'].labels(model_type=model_type).set(0.0)
    
    # Add metrics to module globals
    globals().update(metrics)
    
    return metrics

# Initialize Prometheus Instrumentator
instrumentator = Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=[".*admin.*"],
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    inprogress_name="inprogress_requests",
    inprogress_labels=True,
)

# Register and get metrics
global_metrics = register_metrics()

# Make metrics available in app state for other modules to access
app.state.metrics = global_metrics

# Export metrics for this module
PREDICTION_COUNTER = global_metrics['PREDICTION_COUNTER']
PREDICTION_TIME = global_metrics['PREDICTION_TIME']
PREDICTION_ACCURACY = global_metrics['MODEL_ACCURACY']
PREDICTION_PROBABILITY = global_metrics['PREDICTION_PROBABILITY']

# Import predict router after metrics are defined to avoid circular imports
# This import is now safe because we're not using the metrics in the router module level
from .api.endpoints import predict

# Add metrics endpoint
@app.get("/metrics", include_in_schema=False)
async def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
    
    # Collect metrics from the registry
    return Response(
        content=generate_latest(REGISTRY), 
        media_type=CONTENT_TYPE_LATEST,
        headers={"Access-Control-Allow-Origin": "*"}
    )

# Initialize metrics in the predict module when the app starts
@app.on_event("startup")
async def startup_event():
    """Initialize metrics in the predict module."""
    from .api.endpoints import predict
    predict.init_metrics(global_metrics)
    logger.info("Initialized metrics in predict module")

# Add a middleware to track request metrics
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    
    # Skip metrics endpoint
    if endpoint == "/metrics":
        return await call_next(request)
    
    # Set model based on endpoint
    if "/shadow/" in endpoint:
        model = "shadow"
    elif "/predict" in endpoint and "/shadow/" not in endpoint:
        model = "production"
    else:
        model = "unknown"
    
    # Track request start time for latency calculation
    start_time = time.time()
    
    try:
        response = await call_next(request)
        status_code = response.status_code
        
        # Calculate request duration
        request_duration = time.time() - start_time
        
        # Record metrics
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code, model=model).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint, model=model).observe(request_duration)
        
        return response
    except Exception as e:
        status_code = 500
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code, model=model).inc()
        raise e

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=DEBUG,
        log_level="info" if not DEBUG else "debug"
    )
