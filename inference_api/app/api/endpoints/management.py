from fastapi import APIRouter, HTTPException, status, Depends, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from ...services.model_loader import ModelLoader
from ...services.ab_testing import ABTestingService
from ...config import MODEL_DIR, API_PREFIX
import logging
import uuid
import json
from pathlib import Path

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
    model_type: str = Field(..., description="Type of model to switch to ('best' or 'second_best')")
    reason: Optional[str] = Field(None, description="Reason for the model switch")

class RollbackRequest(BaseModel):
    """Request model for rolling back to the previous model."""
    reason: str = Field(..., description="Reason for the rollback")
    force: bool = Field(False, description="Force rollback even if it's the same model")

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
async def switch_models(
    request: SwitchModelRequest,
    background_tasks: BackgroundTasks
):
    """
    Switch the production model to a different model.
    
    Args:
        request: Contains the model type to switch to ('best' or 'second_best') and reason
        background_tasks: FastAPI background tasks
        
    Returns:
        Status of the model switch operation
    """
    try:
        model_type = request.model_type.lower()
        reason = request.reason or "No reason provided"
        
        # Log the model switch attempt
        logger.info(f"Attempting to switch to {model_type} model. Reason: {reason}")
        
        # Validate model type
        if model_type not in ['best', 'second_best']:
            error_msg = f"Invalid model type: {model_type}. Must be 'best' or 'second_best'"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # Get current model info before switching
        current_model = model_loader.current_production_model
        
        # Switch models in the background
        background_tasks.add_task(
            _perform_model_switch,
            model_type=model_type,
            reason=reason,
            current_model=current_model
        )
        
        return {
            "status": "pending",
            "message": f"Model switch to {model_type} initiated. The system will update shortly.",
            "current_model": current_model,
            "target_model": model_type,
            "reason": reason
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error initiating model switch: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

def _perform_model_switch(model_type: str, reason: str, current_model: str):
    """
    Helper function to perform the actual model switch.
    
    Args:
        model_type: Type of model to switch to ('best' or 'second_best')
        reason: Reason for the model switch
        current_model: Name of the current production model
    """
    try:
        logger.info(f"Starting model switch from {current_model} to {model_type}")
        
        # Perform the switch
        success = model_loader.switch_models(model_type)
        
        if success:
            # Log the successful switch
            logger.info(
                f"Successfully switched from {current_model} to {model_type} model. "
                f"Reason: {reason}"
            )
            
            # Log the model switch event
            ab_testing.log_event(
                event_type="model_switch",
                details={
                    "from_model": current_model,
                    "to_model": model_loader.current_production_model,
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "completed"
                }
            )
        else:
            error_msg = f"Failed to switch from {current_model} to {model_type} model"
            logger.error(error_msg)
            
            # Log the failed switch
            ab_testing.log_event(
                event_type="model_switch_error",
                details={
                    "from_model": current_model,
                    "to_model": model_type,
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "failed",
                    "error": error_msg
                }
            )
            
    except Exception as e:
        error_msg = f"Error during model switch: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Log the error
        ab_testing.log_event(
            event_type="model_switch_error",
            details={
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "current_model": current_model,
                "target_model": model_type,
                "reason": reason
            }
        )

@router.post("/models/rollback", response_model=Dict[str, Any])
async def rollback_to_previous_model(
    request: RollbackRequest,
    background_tasks: BackgroundTasks
):
    """
    Roll back to the previous model version.
    
    This endpoint will switch the production model back to the previous version
    and update the shadow model accordingly.
    
    Args:
        request: Contains the reason for rollback and force flag
        background_tasks: FastAPI background tasks
        
    Returns:
        Status of the rollback operation
    """
    try:
        current_model = model_loader.current_production_model
        previous_model = model_loader.get_previous_model_info().get("name", "unknown")
        
        # Check if rollback is needed
        if not request.force and current_model == previous_model:
            return {
                "status": "no_change",
                "message": "Current production model is already the same as the previous model. Use force=True to force rollback.",
                "current_model": current_model,
                "previous_model": previous_model
            }
        
        # Log the rollback attempt
        logger.warning(
            f"Initiating rollback from {current_model} to {previous_model}. "
            f"Reason: {request.reason or 'No reason provided'}"
        )
        
        # Determine which model type to switch to
        model_type = "best" if "best" in previous_model.lower() else "second_best"
        
        # Perform the rollback in the background
        background_tasks.add_task(
            _perform_model_switch,
            model_type=model_type,
            reason=f"ROLLBACK: {request.reason or 'No reason provided'}",
            current_model=current_model
        )
        
        return {
            "status": "pending",
            "message": "Rollback initiated. The system will switch to the previous model shortly.",
            "current_model": current_model,
            "target_model": previous_model,
            "reason": request.reason
        }
        
    except Exception as e:
        error_msg = f"Error initiating rollback: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

@router.post("/models/reload", response_model=Dict[str, Any])
async def reload_models():
    """
    Reload all models from disk.
    
    Returns:
        Status of the model reload operation
    """
    try:
        # Log the reload attempt
        logger.info("Attempting to reload all models from disk")
        
        # Reload models
        success = model_loader.load_models()
        
        if not success:
            error_msg = "Failed to reload models"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )
        
        # Log the successful reload
        logger.info("Successfully reloaded all models")
        
        # Log the model reload event
        ab_testing.log_event(
            event_type="model_reload",
            details={
                "models": {
                    "production": model_loader.get_model_info("production").get("name"),
                    "shadow": model_loader.get_model_info("shadow").get("name"),
                    "previous": model_loader.get_previous_model_info().get("name")
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return {
            "status": "success",
            "message": "Successfully reloaded all models",
            "models": {
                "production": model_loader.get_model_info("production").get("name"),
                "shadow": model_loader.get_model_info("shadow").get("name"),
                "previous": model_loader.get_previous_model_info().get("name")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error reloading models: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )
