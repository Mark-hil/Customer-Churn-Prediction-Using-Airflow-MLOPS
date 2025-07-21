from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from ...services.model_loader import ModelLoader
from ...services.ab_testing import ABTestingService
from ...config import MODEL_DIR, API_PREFIX
import logging
import uuid

logger = logging.getLogger(__name__)
router = APIRouter(prefix=f"{API_PREFIX}/management", tags=["Model Management"])

# Initialize model loader and A/B testing service
model_loader = ModelLoader(MODEL_DIR)
ab_testing = ABTestingService(model_loader, traffic_split=0.5)  # Set default traffic split
model_loader.load_models()

class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    models: Dict[str, Dict[str, Any]]
    current_production: str
    available_models: List[str]

class SwitchModelRequest(BaseModel):
    """Request model for switching models."""
    model_type: str  # 'best' or 'second_best'

@router.get("/models", response_model=ModelInfoResponse)
async def get_models():
    """
    Get information about all loaded models.
    
    Returns:
        Information about all loaded models and their status
    """
    try:
        models_info = {}
        available_models = []
        
        # Check for available models in the model directory
        model_dirs = [d for d in MODEL_DIR.glob("*") if d.is_dir()]
        for model_dir in model_dirs:
            if "_best_" in model_dir.name.lower():
                available_models.append("best")
            elif "_second_best_" in model_dir.name.lower():
                available_models.append("second_best")
        
        # Get info for each model type
        for model_type in ['production', 'shadow', 'previous']:
            models_info[model_type] = model_loader.get_model_info(model_type)
        
        return {
            "models": models_info,
            "current_production": model_loader.current_production_model or "unknown",
            "available_models": list(set(available_models))  # Remove duplicates
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )

@router.get("/ab-test", response_model=Dict[str, Any])
async def get_ab_test_info():
    """
    Get information about the current A/B test configuration.
    
    Returns:
        Information about the A/B test configuration
    """
    try:
        return model_loader.get_ab_test_info()
    except Exception as e:
        logger.error(f"Error getting A/B test info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get A/B test info: {str(e)}"
        )

@router.get("/ab-test/summary", response_model=Dict[str, Any])
async def get_ab_test_summary():
    """
    Get a summary of the A/B test results.
    
    Returns:
        Summary of A/B test results
    """
    try:
        return ab_testing.get_test_summary()
    except Exception as e:
        logger.error(f"Error getting A/B test summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get A/B test summary: {str(e)}"
        )

@router.get("/ab-test/logs", response_model=List[Dict[str, Any]])
async def get_ab_test_logs(limit: int = 10):
    """
    Get recent A/B test logs.
    
    Args:
        limit: Number of recent logs to return (default: 10)
        
    Returns:
        List of recent A/B test logs
    """
    try:
        return ab_testing.get_recent_logs(limit)
    except Exception as e:
        logger.error(f"Error getting A/B test logs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get A/B test logs: {str(e)}"
        )

@router.post("/ab-test/traffic-split", response_model=Dict[str, Any])
async def set_traffic_split(split: float = 0.5):
    """
    Set the traffic split for the A/B test.
    
    Args:
        split: Percentage of traffic to send to the candidate model (0.0 to 1.0)
        
    Returns:
        Status of the operation
    """
    try:
        # Validate split value
        if not 0 <= split <= 1:
            raise ValueError("Split must be between 0.0 and 1.0")
            
        ab_testing.traffic_split = split
        return {
            "status": "success",
            "message": f"Traffic split set to {split*100:.0f}% for candidate model",
            "traffic_split": split
        }
    except Exception as e:
        logger.error(f"Error setting traffic split: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to set traffic split: {str(e)}"
        )

@router.post("/models/switch", response_model=Dict[str, Any])
async def switch_models(request: SwitchModelRequest):
    """
    Switch the production model to a different model.
    
    Args:
        request: Contains the model type to switch to ('best' or 'second_best')
        
    Returns:
        Status of the model switch operation
    """
    try:
        if request.model_type not in ['best', 'second_best']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="model_type must be 'best' or 'second_best'"
            )
        
        success = model_loader.switch_models(request.model_type)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to switch to {request.model_type} model"
            )
        
        return {
            "status": "success",
            "message": f"Switched to {request.model_type} model",
            "current_production": model_loader.current_production_model
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to switch models: {str(e)}"
        )

@router.post("/models/reload", response_model=Dict[str, Any])
async def reload_models():
    """
    Reload all models from disk.
    
    Returns:
        Status of the model reload operation
    """
    try:
        success = model_loader.load_models()
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reload models"
            )
        
        return {
            "status": "success",
            "message": "Successfully reloaded all models",
            "current_production": model_loader.current_production_model
        }
        
    except Exception as e:
        logger.error(f"Error reloading models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload models: {str(e)}"
        )
