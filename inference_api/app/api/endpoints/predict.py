from fastapi import APIRouter, HTTPException, Request, Header, Depends, status
from fastapi import status as http_status
from fastapi.params import Header as HeaderParam
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from ...services.model_loader import ModelLoader
from ...services.ab_testing import ABTestingService
from ...config import MODEL_DIR, API_PREFIX
import logging
import time
import uuid


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Predictions"])

# These will be initialized in the FastAPI app startup event
PREDICTION_COUNTER = None
PREDICTION_TIME = None
PREDICTION_ACCURACY = None
PREDICTION_PROBABILITY = None

def init_metrics(metrics):
    """Initialize metrics from the app state.
    
    This function should be called during app startup to set up the metrics.
    """
    global PREDICTION_COUNTER, PREDICTION_TIME, PREDICTION_ACCURACY, PREDICTION_PROBABILITY
    
    PREDICTION_COUNTER = metrics['PREDICTION_COUNTER']
    PREDICTION_TIME = metrics['PREDICTION_TIME']
    PREDICTION_ACCURACY = metrics['MODEL_ACCURACY']
    PREDICTION_PROBABILITY = metrics['PREDICTION_PROBABILITY']
    
    # Initialize metrics for all model types and statuses
    for model_type in ['production', 'shadow']:
        # Initialize counter labels
        for status in ['total', 'correct', 'success', 'error']:
            PREDICTION_COUNTER.labels(model_type=model_type, status=status)
        
        # Initialize accuracy gauge
        PREDICTION_ACCURACY.labels(model_type=model_type)
        
        # Initialize probability histogram
        for pred_class in ['0', '1']:  # Binary classification
            PREDICTION_PROBABILITY.labels(model_type=model_type, prediction_class=pred_class)

# Custom header to make x-request-id truly optional
def get_request_id(
    x_request_id: Optional[str] = HeaderParam(
        None,
        alias="x-request-id",
        description="Optional request ID. If not provided, a UUID will be generated automatically.",
        include_in_schema=False
    )
) -> str:
    return x_request_id or str(uuid.uuid4())

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Predictions"])

# Initialize model loader and A/B testing service
model_loader = ModelLoader(MODEL_DIR)
ab_testing = ABTestingService(model_loader)
model_loader.load_models()

class PredictionInput(BaseModel):
    """Input data model for prediction."""
    # Add your feature fields here based on your model's input requirements
    # Example:
    # age: int
    # gender: str
    # account_balance: float
    # ... other features
    pass

class PredictionResult(BaseModel):
    """Prediction result with essential information."""
    model_type: str
    prediction: int
    probability: float
    prediction_time: str
    model_name: str
    model_version: str
    interpretation: str
    ab_test_group: Optional[str] = None

class PredictionResponse(BaseModel):
    """Simplified response model for prediction."""
    predictions: List[PredictionResult]
    production_model: str
    shadow_mode: bool = False

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "models_loaded": {
            "production": model_loader.models['production'] is not None,
            "shadow": model_loader.models['shadow'] is not None,
            "previous": model_loader.models['previous'] is not None
        },
        "current_production_model": model_loader.current_production_model
    }
    return status

@router.post("", response_model=PredictionResponse)
async def predict(
    request: Request,
    input_data: Dict[str, Any],
    shadow_mode: bool = False,
    include_previous: bool = False,
    request_id: str = Depends(get_request_id),
    ab_test: bool = False,
    x_ab_test: Optional[str] = Header(None),
    ground_truth: Optional[int] = None
):
    """
    Make predictions using the loaded models.
    
    Args:
        request: FastAPI request object
        input_data: Dictionary containing input features
        shadow_mode: Whether to run in shadow mode
        include_previous: Whether to include previous model predictions
        x_request_id: Request ID for tracking
        ab_test: Whether to run in A/B test mode
        ground_truth: Optional ground truth value (0 or 1)
    """
    try:
        start_time = time.time()
        
        # Log the prediction request
        logger.info(f"Prediction request received. Request ID: {request_id}")
        logger.info(f"Input data keys: {list(input_data.keys())}")
        logger.info(f"Model loader models: {list(model_loader.models.keys())}")
        
        # Check if models are loaded
        if not model_loader.models.get('production'):
            logger.error("No production model loaded")
            if not model_loader.models.get('shadow'):
                logger.error("No shadow model loaded either")
            
        # Log model info if available
        if model_loader.models.get('production'):
            logger.info(f"Production model loaded from: {model_loader.models['production'].get('model_dir', 'unknown')}")
        if model_loader.models.get('shadow'):
            logger.info(f"Shadow model loaded from: {model_loader.models['shadow'].get('model_dir', 'unknown')}")
            
        # Initialize A/B test group
        ab_test_group = None
        
        # Handle A/B test header
        if x_ab_test and x_ab_test.upper() in ['A', 'B']:
            ab_test = True
            ab_test_group = x_ab_test.upper()
            logger.info(f"A/B testing enabled via header: {ab_test_group}")
        
        # Log A/B testing status
        logger.info(f"A/B testing enabled: {ab_test}")
        if ab_test:
            logger.info(f"A/B test traffic split: {ab_testing.traffic_split}")
            logger.info(f"A/B test group: {ab_test_group}")
        
        # Track ground truth updates and update accuracy metrics
        if ground_truth is not None:
            try:
                # Find or create log entry for this request
                log_entry = next((log for log in ab_testing.logs if log.get('request_id') == request_id), None)
                if not log_entry:
                    # Create a new log entry if none exists (for direct prediction with ground truth)
                    log_entry = {
                        'request_id': request_id,
                        'timestamp': datetime.utcnow().isoformat(),
                        'model_type': 'production',  # Default to production if not in A/B test
                        'prediction': None,
                        'probability': None,
                        'features': {},
                        'ground_truth': ground_truth,
                        'ground_truth_updated_at': datetime.utcnow().isoformat()
                    }
                    ab_testing.logs.append(log_entry)
                else:
                    # Update existing log entry
                    log_entry['ground_truth'] = ground_truth
                    log_entry['ground_truth_updated_at'] = datetime.utcnow().isoformat()
                
                # Save logs to disk
                ab_testing._save_logs_to_disk()
                logger.info(f"Ground truth updated for request {request_id}: {ground_truth}")
                
                # Update metrics if we have a prediction
                if 'model_type' in log_entry and 'prediction' in log_entry and log_entry['prediction'] is not None:
                    model_type = log_entry['model_type']
                    prediction = log_entry['prediction']
                    
                    # Increment total prediction counter
                    PREDICTION_COUNTER.labels(
                        model_type=model_type,
                        status='total'
                    ).inc()
                    
                    # Check if prediction was correct
                    is_correct = int(prediction == ground_truth)
                    
                    # Increment correct predictions counter if correct
                    if is_correct:
                        PREDICTION_COUNTER.labels(
                            model_type=model_type,
                            status='correct'
                        ).inc()
                    
                    # Calculate and update accuracy
                    total = PREDICTION_COUNTER.labels(
                        model_type=model_type,
                        status='total'
                    )._value.get()
                    
                    correct = PREDICTION_COUNTER.labels(
                        model_type=model_type,
                        status='correct'
                    )._value.get()
                    
                    accuracy = correct / total if total > 0 else 0.0
                    
                    # Update accuracy gauge
                    PREDICTION_ACCURACY.labels(
                        model_type=model_type
                    ).set(accuracy)
                    
                    # Update prediction time if available
                    if 'prediction_time' in log_entry and log_entry['prediction_time'] is not None:
                        PREDICTION_TIME.labels(
                            model_type=model_type
                        ).observe(log_entry['prediction_time'])
                    
                    logger.info(f"Updated metrics for {model_type}: accuracy={accuracy:.2f}, total={total}, correct={correct}")
            except Exception as e:
                logger.error(f"Error updating ground truth and metrics: {str(e)}", exc_info=True)
                # Don't fail the request if ground truth update fails
                pass

        if not input_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input data is required"
            )

        if 'input_data' in input_data:
            features = input_data['input_data']
        else:
            features = input_data

        # Get model data based on mode
        if x_ab_test:
            # Use explicit A/B test group from header
            if x_ab_test.upper() == 'A':
                model_type = 'production'
                model_data = model_loader.models.get('production')
                group = 'A'
            elif x_ab_test.upper() == 'B':
                model_type = 'shadow'
                model_data = model_loader.models.get('shadow')
                group = 'B'
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid A/B test group: {x_ab_test}. Must be 'A' or 'B'"
                )
        elif shadow_mode:
            model_type = "shadow"
            model_data = model_loader.models['shadow']
            group = "shadow"
            if not model_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No shadow model available"
                )
        else:
            # No header provided, use NGINX's automatic split
            # First check for X-AB-Test-Group header set by NGINX
            ab_test_group = request.headers.get('X-AB-Test-Group')
            
            if ab_test_group == 'production':
                model_type = 'production'
                model_data = model_loader.models.get('production')
                group = 'A'
                logger.info("Routed to production model via X-AB-Test-Group header")
            elif ab_test_group == 'shadow':
                model_type = 'shadow'
                model_data = model_loader.models.get('shadow')
                group = 'B'
                logger.info("Routed to shadow model via X-AB-Test-Group header")
            else:
                # Fall back to port-based routing if X-AB-Test-Group is not available
                forwarded_port = request.headers.get('X-Forwarded-Port')
                if forwarded_port is None:
                    forwarded_port = request.url.port
                
                logger.info(f"Port-based routing - X-Forwarded-Port: {forwarded_port}")
                
                try:
                    port = int(forwarded_port) if forwarded_port is not None else None
                    
                    if port == 8000 or (port is None and not shadow_mode):
                        model_type = 'production'
                        model_data = model_loader.models.get('production')
                        group = 'A'
                        logger.info("Routed to production model via port")
                    elif port == 8001 or (port is None and shadow_mode):
                        model_type = 'shadow'
                        model_data = model_loader.models.get('shadow')
                        group = 'B'
                        logger.info("Routed to shadow model via port")
                    else:
                        logger.warning(f"Unexpected port received: {port}, defaulting to production model")
                        model_type = 'production'
                        model_data = model_loader.models.get('production')
                        group = 'A'
                    
                except (ValueError, TypeError) as e:
                    # If there's an error parsing the port, default to production but log the error
                    logger.error(f"Error parsing port '{forwarded_port}': {str(e)}")
                    model_type = 'production'
                    model_data = model_loader.models.get('production')
                    group = 'A'

            if not model_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No {model_type} model available"
                )

        # Process input data
        try:
            processed_input = model_loader.preprocess_input(
                input_data=features,
                preprocessor=model_data['preprocessor'],
                feature_names=model_data['feature_names']
            )
        except Exception as e:
            logger.error(f"Error in preprocessor: {str(e)}")
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to preprocess input: {str(e)}"
            )

        # Make predictions
        results = []
        try:
            if ab_test and ab_test_group:
                # Use the specified A/B test group
                if ab_test_group == 'A':
                    model_type = 'production'
                else:
                    model_type = 'shadow' if 'shadow' in model_loader.models else 'production'
                
                # Get the appropriate model
                model_info = model_loader.get_model_info(model_type)
                if not model_info or 'model' not in model_info or model_info['model'] is None:
                    if PREDICTION_COUNTER:
                        PREDICTION_COUNTER.labels(model_type=model_type, status='error').inc()
                    raise HTTPException(
                        status_code=500,
                        detail=f"{model_type.capitalize()} model not available or not loaded"
                    )
                
                # Get the model and preprocessor
                model = model_info['model']
                preprocessor = model_info.get('preprocessor')
                
                # Make prediction with timing
                prediction_start = time.time()
                try:
                    # Preprocess input if preprocessor is available
                    if preprocessor is not None:
                        processed_data = preprocessor.transform(processed_input)
                    else:
                        processed_data = processed_input
                    
                    # Get prediction from the model
                    prediction = model.predict(processed_data)
                    probability = model.predict_proba(processed_data)
                    prediction_time = time.time() - prediction_start
                    
                    # Get prediction values
                    pred = int(prediction[0])
                    prob = float(probability[0][1])
                    
                    # Increment total predictions counter
                    PREDICTION_COUNTER.labels(model_type=model_type, status='total').inc()
                    
                    # If ground truth is provided, check if prediction was correct
                    if ground_truth is not None:
                        is_correct = int(pred == ground_truth)
                        if is_correct:
                            PREDICTION_COUNTER.labels(model_type=model_type, status='correct').inc()
                    
                    # Record metrics for successful prediction
                    PREDICTION_TIME.labels(model_type=model_type).observe(prediction_time)
                    PREDICTION_COUNTER.labels(model_type=model_type, status='success').inc()
                    
                    # Log probability distribution
                    pred_class = 'churn' if pred == 1 else 'no_churn'
                    PREDICTION_PROBABILITY.labels(
                        model_type=model_type,
                        prediction_class=pred_class
                    ).observe(prob)
                    
                    # Update accuracy metric if ground truth is available
                    if ground_truth is not None:
                        total = PREDICTION_COUNTER.labels(model_type=model_type, status='total')._value.get()
                        correct = PREDICTION_COUNTER.labels(model_type=model_type, status='correct')._value.get()
                        if total > 0:
                            accuracy = correct / total
                            PREDICTION_ACCURACY.labels(model_type=model_type).set(accuracy)
                    
                    # Create interpretation based on prediction and probability
                    if pred == 1:
                        if prob >= 0.7:
                            interpretation = "Customer is highly likely to churn"
                        elif prob >= 0.55:
                            interpretation = "Customer is likely to churn"
                        else:
                            interpretation = "Customer shows some signs of potential churn"
                    else:
                        if prob >= 0.7:
                            interpretation = "Customer is very unlikely to churn"
                        elif prob >= 0.55:
                            interpretation = "Customer is unlikely to churn"
                        else:
                            interpretation = "Customer shows some signs of potential churn"
                    
                    # Format results
                    results.append({
                        'model_type': model_type,
                        'prediction': pred,
                        'probability': prob,
                        'prediction_time': f"{prediction_time:.4f}s",
                        'model_name': model_info.get('name', 'unknown'),
                        'model_version': model_info.get('version', '1.0'),
                        'interpretation': interpretation
                    })
                    
                except Exception as e:
                    # Record prediction error in metrics
                    if PREDICTION_COUNTER:
                        PREDICTION_COUNTER.labels(model_type=model_type, status='error').inc()
                    logger.error(f"Prediction error for model {model_type}: {str(e)}", exc_info=True)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error making prediction with {model_type} model: {str(e)}"
                    )
            else:
                # Make prediction for non-A/B test case
                prediction_start = time.time()
                try:
                    prediction = model_data['model'].predict(processed_input)
                    probability = model_data['model'].predict_proba(processed_input)
                    prediction_time = time.time() - prediction_start
                    
                    # Get prediction values
                    pred = int(prediction[0])
                    prob = float(probability[0][1])
                    
                    # Increment total predictions counter
                    PREDICTION_COUNTER.labels(model_type='production', status='total').inc()
                    
                    # If ground truth is provided, check if prediction was correct
                    if ground_truth is not None:
                        is_correct = int(pred == ground_truth)
                        if is_correct:
                            PREDICTION_COUNTER.labels(model_type='production', status='correct').inc()
                    
                    # Record metrics for successful prediction
                    PREDICTION_TIME.labels(model_type='production').observe(prediction_time)
                    PREDICTION_COUNTER.labels(model_type='production', status='success').inc()
                    
                    # Log probability distribution
                    pred_class = 'churn' if pred == 1 else 'no_churn'
                    PREDICTION_PROBABILITY.labels(
                        model_type='production',
                        prediction_class=pred_class
                    ).observe(prob)
                    
                    # Update accuracy metric if ground truth is available
                    if ground_truth is not None:
                        total = PREDICTION_COUNTER.labels(model_type='production', status='total')._value.get()
                        correct = PREDICTION_COUNTER.labels(model_type='production', status='correct')._value.get()
                        if total > 0:
                            accuracy = correct / total
                            PREDICTION_ACCURACY.labels(model_type='production').set(accuracy)
                    
                    # Create interpretation
                    if pred == 1:
                        if prob >= 0.7:
                            interpretation = "Customer is highly likely to churn"
                        elif prob >= 0.55:
                            interpretation = "Customer is likely to churn"
                        else:
                            interpretation = "Customer shows some signs of potential churn"
                    else:
                        if prob >= 0.7:
                            interpretation = "Customer is very unlikely to churn"
                        elif prob >= 0.55:
                            interpretation = "Customer is unlikely to churn"
                        else:
                            interpretation = "Customer shows some signs of potential churn"
                    
                    # Format results
                    results.append({
                        'model_type': 'production',
                        'prediction': pred,
                        'probability': prob,
                        'prediction_time': f"{prediction_time:.4f}s",
                        'model_name': model_data.get('metadata', {}).get('model_name', 'unknown'),
                        'model_version': model_data.get('metadata', {}).get('model_version', '1.0'),
                        'interpretation': interpretation
                    })
                    
                except Exception as e:
                    # Record prediction error in metrics
                    PREDICTION_COUNTER.labels(model_type='production', status='error').inc()
                    logger.error(f"Prediction error: {str(e)}")
                    raise HTTPException(
                        status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Error making prediction: {str(e)}"
                    )
            # Create prediction result dictionary
            prediction_result = {
                'model_type': model_type,
                'prediction': pred,
                'probability': prob,
                'prediction_time': prediction_time,
                'model_name': model_data.get('metadata', {}).get('model_name', 'unknown'),
                'model_version': model_data.get('metadata', {}).get('model_version', '1.0'),
                'interpretation': interpretation,
                'ab_test_group': group if ab_test else None
            }
            
            # Ensure we have a valid A/B test group
            final_ab_test_group = group if group in ['A', 'B'] else ab_test_group
            
            # Log the prediction with A/B test information
            ab_testing.log_prediction(
                request_id=request_id,
                input_data=features,
                prediction=prediction_result,
                model_type=model_type,
                model_name=model_data.get('metadata', {}).get('model_name', 'unknown'),
                model_version=model_data.get('metadata', {}).get('model_version', '1.0'),
                ground_truth=ground_truth,
                ab_test_group=final_ab_test_group if final_ab_test_group in ['A', 'B'] else 'A',  # Default to 'A' if no group specified
                prediction_time=prediction_time
            )

            # Generate the response with the prediction
            response_data = {
                "request_id": request_id,
                "predictions": [{
                    **prediction_result,
                    "prediction_time": f"{prediction_time:.2f}s",  # Format for response only
                    "ab_test_group": group if ab_test else None  # Include AB test group in response
                }],
                "production_model": model_loader.current_production_model,
                "shadow_mode": shadow_mode
            }

            return response_data
        except HTTPException as http_exc:
            # Re-raise HTTP exceptions as they are already properly formatted
            logger.error(f"HTTP Exception in prediction: {str(http_exc.detail)}")
            raise http_exc
        except Exception as e:
            # Log the full error with traceback for debugging
            logger.error(f"Unexpected error in prediction: {str(e)}", exc_info=True)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error args: {e.args}")
            
            # Provide a more user-friendly error message
            error_detail = "An unexpected error occurred during prediction"
            if hasattr(e, '__module__') and 'pydantic' in e.__module__:
                error_detail = f"Validation error: {str(e)}"
            
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_detail
            )

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/update-ground-truth")
async def update_ground_truth(
    request_id: str,
    ground_truth: int,
    current_request_id: str = Depends(get_request_id)
):
    try:
        # First, ensure we have the latest logs from disk
        ab_testing._load_logs_from_disk(force_reload=True)
        
        # Find the log entry
        log_entry = next((log for log in ab_testing.logs if log.get('request_id') == request_id), None)
        if not log_entry:
            # Try one more time with a fresh load in case of sync issues
            ab_testing._load_logs_from_disk(force_reload=True)
            log_entry = next((log for log in ab_testing.logs if log.get('request_id') == request_id), None)
            if not log_entry:
                logger.warning(f"Prediction not found for request_id: {request_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "status": "error",
                        "error": "Prediction not found",
                        "request_id": request_id,
                        "current_request_id": current_request_id,
                        "message": f"No prediction found with request ID: {request_id}",
                        "available_ids": [log.get('request_id') for log in ab_testing.logs]
                    }
                )

        # Update ground truth using the service method
        ab_testing.update_ground_truth(request_id, ground_truth)
        
        # Get the updated entry
        updated_entry = next((log for log in ab_testing.logs if log.get('request_id') == request_id), {})
        
        # Update Prometheus metrics
        model_type = updated_entry.get('model_type', 'unknown')
        prediction = updated_entry.get('prediction')
        is_correct = None
        
        if prediction is not None and ground_truth is not None:
            # Increment total predictions counter for this model type
            PREDICTION_COUNTER.labels(model_type=model_type, status='total').inc()
            
            # Check if prediction was correct and update metrics
            is_correct = int(prediction == ground_truth)
            if is_correct:
                PREDICTION_COUNTER.labels(model_type=model_type, status='correct').inc()
            
            # Update accuracy metric
            total = PREDICTION_COUNTER.labels(model_type=model_type, status='total')._value.get()
            correct = PREDICTION_COUNTER.labels(model_type=model_type, status='correct')._value.get()
            if total > 0:
                accuracy = correct / total
                PREDICTION_ACCURACY.labels(model_type=model_type).set(accuracy)
                logger.info(f"Updated accuracy for {model_type} model: {accuracy:.2f} ({correct}/{total} correct)")
        
        logger.info(f"Updated metrics for request {request_id} - Model: {model_type}, "
                  f"Prediction: {prediction}, Ground Truth: {ground_truth}, "
                  f"Correct: {is_correct if is_correct is not None else 'N/A'}")
        
        return {
            "status": "success",
            "request_id": request_id,
            "ground_truth": ground_truth,
            "updated_at": updated_entry.get('ground_truth_updated_at'),
            "current_request_id": current_request_id,
            "model_type": model_type,
            "ab_test_group": updated_entry.get('ab_test_group'),
            "metrics_updated": True
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        logger.error(f"Error updating ground truth: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "error": str(e),
                "request_id": request_id,
                "message": "An error occurred while updating ground truth"
            }
        )
