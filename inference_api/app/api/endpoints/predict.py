from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Union
from ...services.model_loader import ModelLoader
from ...config import MODEL_DIR
import logging
import json

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize model loader
model_loader = ModelLoader(MODEL_DIR)
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

class SimpleModelPrediction(BaseModel):
    """Simplified model prediction result."""
    model_type: str
    prediction: int
    probability: float
    prediction_time: str
    model_name: str
    model_version: str
    interpretation: str

class ModelPrediction(SimpleModelPrediction):
    """Full model prediction result with metadata."""
    model_metadata: Dict[str, Any] = {}

class SimplePredictionResponse(BaseModel):
    """Simplified response model for prediction."""
    predictions: List[SimpleModelPrediction]
    production_model: str
    shadow_mode: bool = False

class PredictionResponse(SimplePredictionResponse):
    """Full response model with detailed metadata."""
    predictions: List[ModelPrediction]

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

@router.post("/predict", response_model=SimplePredictionResponse)
async def predict(
    request: Request,
    input_data: Dict[str, Any],
    shadow_mode: bool = False,
    include_previous: bool = False
):
    """
    Make predictions using the loaded models.
    
    Args:
        request: FastAPI request object
        input_data: Dictionary containing the input features for prediction
        shadow_mode: If True, returns predictions from both production and shadow models
        include_previous: If True, includes the previous production model in the response
        
    Returns:
        Prediction results with probabilities and model metadata
    """
    import datetime
    
    if not model_loader.models['production']:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No production model loaded"
        )
    
    try:
        predictions = []
        
        # Get client IP for logging
        client_ip = request.client.host if request.client else "unknown"
        logger.info(f"Prediction request from {client_ip}: {json.dumps(input_data, indent=2)}")
        
        # Make predictions with active models
        for model_type in ['production', 'shadow', 'previous']:
            # Skip if model not loaded or if it's previous model and not explicitly requested
            if not model_loader.models[model_type] or (model_type == 'previous' and not include_previous):
                continue
                
            # Skip shadow model if not in shadow mode
            if model_type == 'shadow' and not shadow_mode and not include_previous:
                continue
                
            model_data = model_loader.models[model_type]
            
            try:
                # Make prediction
                start_time = datetime.datetime.now()
                
                # Preprocess input
                processed_input = model_loader.preprocess_input(
                    input_data=input_data['input_data'],  # Only pass the actual input features
                    preprocessor=model_data['preprocessor'],
                    feature_names=model_data['feature_names']
                )
                
                # Predict
                prediction = model_data['model'].predict(processed_input)
                prediction_proba = model_data['model'].predict_proba(processed_input)
                
                end_time = datetime.datetime.now()
                prediction_time_ms = (end_time - start_time).total_seconds() * 1000
                
                # Get prediction and probability
                pred = int(prediction[0])
                prob = float(prediction_proba[0][1])
                
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
                
                # Format simplified response
                predictions.append({
                    "model_type": model_type,
                    "prediction": pred,
                    "probability": prob,
                    "prediction_time": f"{prediction_time_ms:.2f}ms",
                    "model_name": model_data['metadata'].get('model_name', 'unknown'),
                    "model_version": model_data['metadata'].get('model_version', 'unknown'),
                    "interpretation": interpretation
                })
                
                logger.info(f"{model_type.upper()} prediction: {predictions[-1]}")
                
            except Exception as e:
                logger.error(f"{model_type.upper()} prediction error: {str(e)}")
                # Don't fail the whole request if one model fails
                continue
        
        if not predictions:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No models were able to process the request"
            )
        
        return {
            "predictions": predictions,
            "production_model": model_loader.current_production_model,
            "shadow_mode": shadow_mode
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}"
        )
