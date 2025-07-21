from fastapi import APIRouter, HTTPException, status, Request, Header, Depends
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
            
        # Log A/B testing status
        logger.info(f"A/B testing enabled: {ab_test}")
        if ab_test:
            logger.info(f"A/B test traffic split: {ab_testing.traffic_split}")
        
        if ground_truth is not None:
            try:
                log_entry = next((log for log in ab_testing.logs if log['request_id'] == request_id), None)
                if log_entry:
                    log_entry['ground_truth'] = ground_truth
                    log_entry['ground_truth_updated_at'] = datetime.utcnow().isoformat()
                    ab_testing._save_logs_to_disk()
                    logger.info(f"Ground truth updated for request {request_id}: {ground_truth}")
                else:
                    logger.warning(f"No log entry found for request {request_id} when updating ground truth")
            except Exception as e:
                logger.error(f"Error updating ground truth: {str(e)}")
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
        if ab_test:
            model_type, model_data, group = ab_testing.get_model_for_request(request_id)
            if not model_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No {model_type} model available"
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
            model_type = "production"
            model_data = model_loader.models['production']
            group = "production"
            if not model_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No production model available"
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
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to preprocess input: {str(e)}"
            )

        # Make prediction
        try:
            start_time = time.time()
            prediction = model_data['model'].predict(processed_input)
            probability = model_data['model'].predict_proba(processed_input)
            prediction_time = time.time() - start_time

            # Create interpretation based on prediction and probability
            pred = int(prediction[0])
            prob = float(probability[0][1])
            
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
            
            # Log the prediction
            prediction_result = {
                "model_type": model_type,
                "prediction": pred,
                "probability": prob,
                "prediction_time": prediction_time,
                "model_name": model_data.get('metadata', {}).get('model_name', 'unknown'),
                "model_version": model_data.get('metadata', {}).get('model_version', 'unknown'),
                "interpretation": interpretation,
                "ab_test_group": group if ab_test else None
            }
            
            # Log the prediction with ground truth if provided
            ab_testing.log_prediction(
                request_id=request_id,
                input_data=input_data,
                prediction=prediction_result,
                model_type=model_type,
                model_name=model_data.get('metadata', {}).get('model_name', 'unknown'),
                model_version=model_data.get('metadata', {}).get('model_version', 'unknown'),
                ground_truth=ground_truth,
                ab_test_group=group if ab_test else None,
                prediction_time=prediction_time
            )

            # Generate the response with the prediction
            response_data = {
                "request_id": request_id,
                "predictions": [{
                    **prediction_result,
                    "prediction_time": f"{prediction_result['prediction_time']:.2f}s"  # Format for response only
                }],
                "production_model": model_loader.current_production_model,
                "shadow_mode": shadow_mode
            }

            return response_data
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
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
        
        return {
            "status": "success",
            "request_id": request_id,
            "ground_truth": ground_truth,
            "updated_at": updated_entry.get('ground_truth_updated_at'),
            "current_request_id": current_request_id,
            "model_type": updated_entry.get('model_type'),
            "ab_test_group": updated_entry.get('ab_test_group')
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        logger.error(f"Error updating ground truth: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "error": str(e),
                "request_id": request_id,
                "message": "An error occurred while updating ground truth"
            }
        )
