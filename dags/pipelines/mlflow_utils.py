"""MLflow utilities for model tracking and management."""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator


def setup_mlflow():
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("telco-churn-prediction")


def log_model_metadata(
    model: BaseEstimator,
    model_name: str,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    feature_importance: Dict[str, float],
    data_hash: str,
    training_time: float,
):
    """Log model and metadata to MLflow.
    
    Args:
        model: Trained model
        model_name: Name of the model
        metrics: Dictionary of evaluation metrics
        params: Model parameters
        feature_importance: Dictionary of feature importances
        data_hash: Hash of the training data
        training_time: Training time in seconds
    """
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        
        # Log feature importance as artifact
        if feature_importance:
            importance_path = "feature_importance.json"
            with open(importance_path, "w") as f:
                json.dump(feature_importance, f)
            mlflow.log_artifact(importance_path)
        
        # Log additional metadata
        mlflow.set_tag("data_hash", data_hash)
        mlflow.set_tag("training_time_seconds", training_time)
        mlflow.set_tag("model_type", model.__class__.__name__)
        
        logger.info(f"Logged model {model_name} to MLflow")


def get_feature_importance(model: BaseEstimator, feature_names: list) -> Dict[str, float]:
    """Extract feature importance from model."""
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, "coef_"):
        # For linear models
        return dict(zip(feature_names, np.abs(model.coef_[0])))
    return {}


def get_best_model(metric: str = "roc_auc", ascending: bool = False) -> Optional[Dict]:
    """Retrieve the best model based on a metric.
    
    Args:
        metric: Metric to sort by
        ascending: Sort order
        
    Returns:
        Dictionary with model metadata or None if no models found
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("telco-churn-prediction")
    
    if not experiment:
        return None
        
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC}'}"],
        max_results=1,
    )
    
    if not runs:
        return None
        
    best_run = runs[0]
    return {
        "run_id": best_run.info.run_id,
        "metrics": best_run.data.metrics,
        "params": best_run.data.params,
        "model_uri": f"runs:/{best_run.info.run_id}/model",
        "artifact_uri": best_run.info.artifact_uri,
    }
