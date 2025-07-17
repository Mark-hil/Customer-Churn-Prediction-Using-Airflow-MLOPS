"""Airflow DAG for automated daily training & validation.

This mirrors the Prefect `training_flow.py` but uses Airflow 2.x primitives.
Place this file inside your `$AIRFLOW_HOME/dags` folder or add the current
repository path to `AIRFLOW__CORE__DAGS_FOLDER`.

Schedule: daily at 02:00 UTC.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
import glob
import os
import traceback
from pathlib import Path

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import joblib
import matplotlib
# Set the backend to 'Agg' to avoid GUI-related issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from airflow import DAG
from airflow.models.xcom_arg import XComArg
from airflow.operators.python import PythonOperator
from loguru import logger

# Import MLflow utilities
from pipelines.mlflow_utils import setup_mlflow, log_model_metadata, get_feature_importance

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("/opt/airflow/data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
PLOTS_DIR = DATA_DIR / "plots"

def ensure_directory(directory: Path) -> None:
    """Safely create a directory with appropriate permissions."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Try to set permissions (0o777 & ~umask, typically 0o755)
        try:
            os.chmod(directory, 0o755)
        except Exception as e:
            logger.warning(f"Could not set permissions for {directory}: {e}")
        logger.info(f"Ensured directory exists: {directory}")
    except PermissionError as e:
        logger.error(f"Permission denied when creating directory {directory}. "
                   f"Please ensure the Airflow user has write access to {directory.parent}. "
                   f"Error: {e}")
        # Don't raise here to allow the DAG to load, but it will likely fail at runtime
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        # Don't raise here to allow the DAG to load

# Ensure all required directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR, PLOTS_DIR]:
    ensure_directory(directory)

def setup_mlflow():
    """Set up MLflow tracking with MLflow server."""
    try:
        # Set tracking URI to use MLflow server
        tracking_uri = "http://mlflow:5001"  # Using service name in docker-compose network
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        
        # Set environment variable for artifact storage
        os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
        
        # Ensure the experiment exists
        experiment_name = "churn-prediction"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.info(f"Creating new MLflow experiment: {experiment_name}")
                # Create experiment with default artifact location
                experiment_id = mlflow.create_experiment(experiment_name)
                experiment = mlflow.get_experiment(experiment_id)
            
            # Set the experiment
            mlflow.set_experiment(experiment_name)
            logger.info(f"Using MLflow experiment: {experiment_name} (ID: {experiment.experiment_id})")
            
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            logger.error(traceback.format_exc())
            raise
            
        return {
            'tracking_uri': tracking_uri,
            'experiment_name': experiment_name,
            'experiment_id': experiment.experiment_id
        }
        
        # Configure MLflow to use local file system for artifacts
        os.environ['MLFLOW_ARTIFACT_URI'] = str(artifacts_dir / experiment_name)
        
        # Disable MLflow autologging to have more control over what gets logged
        mlflow.sklearn.autolog(
            log_input_examples=False,
            log_model_signatures=True,
            log_models=True,
            log_datasets=False
        )
        
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"MLflow experiment ID: {experiment.experiment_id}")
        logger.info(f"MLflow artifact URI: {os.environ.get('MLFLOW_ARTIFACT_URI')}")
        
        return {
            'tracking_uri': tracking_uri,
            'experiment_id': experiment.experiment_id,
            'artifact_uri': os.environ.get('MLFLOW_ARTIFACT_URI')
        }
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    logger.info(f"MLflow artifact URI: {os.environ.get('MLFLOW_ARTIFACT_URI')}")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = Path("/opt/airflow")
RAW_DATA_PATH = Path("/opt/airflow/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DIR = Path("/opt/airflow/data/processed")
MODEL_REGISTRY = Path("/opt/airflow/models")

# ---------------------------------------------------------------------------
# HELPERS (re-usable across tasks)
# ---------------------------------------------------------------------------

def _validate_data(df: pd.DataFrame) -> None:
    expected_cols = {
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.isna().sum().sum() > 0:
        logger.warning("Raw data contains NaNs - handled in preprocessing.")


def _preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    y = df["Churn"].map({"Yes": 1, "No": 0}).values
    X = df.drop(columns=["Churn"])
    cat_cols = X.select_dtypes("object").columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    preprocessor: ColumnTransformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
            ("num", MinMaxScaler(), num_cols),
        ]
    )
    X_processed = preprocessor.fit_transform(X)
    feature_names = (
        preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
        + num_cols
    )
    return X_processed, y, feature_names


def _train_model(X_train: np.ndarray, y_train: np.ndarray, model_params: Optional[dict] = None) -> LogisticRegression:
    """Train a model with optional hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_params: Dictionary of model parameters
        
    Returns:
        Trained model
    """
    params = {
        'max_iter': 1000,
        'solver': 'lbfgs',
        'random_state': 42,
        **(model_params or {})
    }
    
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model


def _evaluate(model: LogisticRegression, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds),
        "f1": f1_score(y_val, preds),
        "roc_auc": roc_auc_score(y_val, proba),
    }


def _persist(model: LogisticRegression, metrics: Dict[str, Any], feature_names: List[str], data_hash: str, duration: float):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_name = f"logreg_{timestamp}.pkl"
    joblib.dump(model, MODEL_REGISTRY / model_name)

    record = {
        "model_file": model_name,
        "created_at": timestamp,
        "data_hash": data_hash,
        "metrics": metrics,
        "features": feature_names,
        "model_type": "LogisticRegression",
        "training_time_sec": duration,
    }
    with open(MODEL_REGISTRY / "metadata.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("Model and metadata saved for {}", model_name)


# ---------------------------------------------------------------------------
# DAG DEFINITION
# ---------------------------------------------------------------------------

def ensure_directories():
    """Ensure all required directories exist."""
    try:
        # Create directories with default permissions
        for directory in [RAW_DATA_PATH.parent, PROCESSED_DIR, MODEL_REGISTRY]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except PermissionError as pe:
                logger.warning(f"Could not set permissions for {directory}: {pe}")
                continue
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        raise

def load_data(**context):
    ensure_directories()
    logger.info("Loading data from {}", RAW_DATA_PATH)
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        context["ti"].xcom_push(key="raw_df", value=df)
    except FileNotFoundError:
        logger.error(f"Data file not found at {RAW_DATA_PATH}")
        raise


def validate_data(**context):
    df = context["ti"].xcom_pull(key="raw_df", task_ids="load_data")
    _validate_data(df)


def preprocess(**context):
    import numpy as np
    import joblib
    from pathlib import Path
    
    # Ensure processed directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    df = context["ti"].xcom_pull(key="raw_df", task_ids="load_data")
    X, y, feature_names = _preprocess(df)
    
    # Save processed data to disk
    timestamp = context["ts_nodash"]
    processed_data = {
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'timestamp': timestamp
    }
    
    # Save as joblib file for efficient storage
    processed_file = PROCESSED_DIR / f"processed_data_{timestamp}.joblib"
    joblib.dump(processed_data, processed_file)
    logger.info(f"Saved processed data to {processed_file}")
    
    # Convert numpy arrays to lists for XCom serialization
    X_list = X.tolist() if hasattr(X, 'tolist') else X
    y_list = y.tolist() if hasattr(y, 'tolist') else y
    
    # Push preprocessed data to XCom
    context["ti"].xcom_push(key="X", value=X_list)
    context["ti"].xcom_push(key="y", value=y_list)
    context["ti"].xcom_push(key="feature_names", value=feature_names)
    context["ti"].xcom_push(key="processed_data_path", value=str(processed_file))


def serialize_data(data):
    """Helper function to serialize data for XCom."""
    import numpy as np
    import pickle
    import base64
    
    if data is None:
        return None
    if isinstance(data, (np.ndarray, np.generic)):
        return {
            '__type__': 'numpy_array',
            'data': base64.b64encode(pickle.dumps(data)).decode('utf-8')
        }
    elif hasattr(data, 'tolist'):
        return data.tolist()
    return data

def deserialize_data(data):
    """Helper function to deserialize data from XCom."""
    import numpy as np
    import pickle
    import base64
    
    if data is None:
        return None
    if isinstance(data, dict) and data.get('__type__') == 'numpy_array':
        return pickle.loads(base64.b64decode(data['data'].encode('utf-8')))
    return np.array(data) if isinstance(data, list) else data

def split(**context):
    import numpy as np
    import joblib
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    
    try:
        # Get the path to the processed data file from XCom
        processed_data_path = context["ti"].xcom_pull(key="processed_data_path", task_ids="preprocess")
        
        if not processed_data_path:
            raise ValueError("No processed_data_path found in XCom. Check if preprocess task completed successfully.")
            
        logger.info(f"Loading processed data from {processed_data_path}")
        
        # Check if file exists
        if not Path(processed_data_path).exists():
            raise FileNotFoundError(f"Processed data file not found at {processed_data_path}")
        
        # Load the processed data
        processed_data = joblib.load(processed_data_path)
        
        if not all(key in processed_data for key in ['X', 'y', 'feature_names']):
            raise KeyError("Processed data is missing required keys: 'X', 'y', or 'feature_names'")
            
        X = processed_data['X']
        y = processed_data['y']
        feature_names = processed_data['feature_names']
        
        # Log data shapes for debugging
        logger.info(f"Loaded X shape: {X.shape if hasattr(X, 'shape') else len(X)}, y shape: {y.shape if hasattr(y, 'shape') else len(y)}")
        logger.info(f"Feature names: {feature_names}")
        
        # Ensure X and y have the same number of samples
        if len(X) != len(y):
            raise ValueError(f"Mismatched number of samples: X has {len(X)} samples, y has {len(y)} samples")
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Log split shapes
        logger.info(f"Split data - X_train: {X_train.shape}, X_val: {X_val.shape}, y_train: {y_train.shape}, y_val: {y_val.shape}")
        
        # Log data types before serialization
        logger.info(f"Data types before serialization - X_train: {type(X_train).__name__}, y_train: {type(y_train).__name__}, X_val: {type(X_val).__name__}, y_val: {type(y_val).__name__}")
        
        # Ensure data is numpy arrays before serialization
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train, dtype=np.float32)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train, dtype=np.float32)
        if not isinstance(X_val, np.ndarray):
            X_val = np.array(X_val, dtype=np.float32)
        if not isinstance(y_val, np.ndarray):
            y_val = np.array(y_val, dtype=np.float32)
            
        # Serialize data
        serialized_X_train = serialize_data(X_train)
        serialized_y_train = serialize_data(y_train)
        serialized_X_val = serialize_data(X_val)
        serialized_y_val = serialize_data(y_val)
        
        # Log serialized data types
        logger.info(f"Serialized data types - X_train: {type(serialized_X_train).__name__}, y_train: {type(serialized_y_train).__name__}, X_val: {type(serialized_X_val).__name__}, y_val: {type(serialized_y_val).__name__}")
        
        # Push split data to XCom with serialization
        context["ti"].xcom_push(key="X_train", value=serialized_X_train)
        context["ti"].xcom_push(key="X_val", value=serialized_X_val)
        context["ti"].xcom_push(key="y_train", value=serialized_y_train)
        context["ti"].xcom_push(key="y_val", value=serialized_y_val)
        context["ti"].xcom_push(key="feature_names", value=feature_names)
        
        logger.info("Successfully pushed split data to XCom")
        
    except Exception as e:
        logger.error(f"Error in split task: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def train_model(model, model_name, X_train, X_val, y_train, y_val, feature_names, context):
    """
    Helper function to train and evaluate a single model.
    
    Args:
        model: The model to train
        model_name: Name of the model (for logging)
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        feature_names: List of feature names
        context: Airflow context
        
    Returns:
        dict: Dictionary containing model, metrics, and other information
    """
    import time
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    logger.info(f"Training {model_name}...")
    
    # Start timer
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = {
        'training_time_seconds': training_time,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'val_accuracy': accuracy_score(y_val, y_val_pred),
        'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
        'val_precision': precision_score(y_val, y_val_pred, zero_division=0),
        'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
        'val_recall': recall_score(y_val, y_val_pred, zero_division=0),
        'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
        'val_f1': f1_score(y_val, y_val_pred, zero_division=0),
        'train_auc_roc': roc_auc_score(y_train, y_train_proba),
        'val_auc_roc': roc_auc_score(y_val, y_val_proba)
    }
    
    # Get feature importance
    feature_importance = get_feature_importance(model, feature_names)
    
    return {
        'model': model,
        'model_name': model_name,
        'metrics': metrics,
        'training_time': training_time,
        'feature_importance': feature_importance
    }

class WeightedEnsemble:
    """Weighted ensemble model that combines predictions from multiple models."""
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
        
    def predict_proba(self, X):
        if X.size == 0:
            raise ValueError("Cannot make predictions on empty input")
            
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        n_samples = X.shape[0]
        preds = np.zeros((n_samples, 2))
        
        for (name, model), weight in zip(self.models.items(), self.weights):
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    if proba.shape[1] == 2:  # Binary classification
                        preds += proba * weight
                    elif proba.shape[1] == 1:  # Handle single class output
                        preds += np.column_stack((1 - proba, proba)) * weight
                    else:
                        logger.warning(f"Unexpected predict_proba shape {proba.shape} from {name}")
                else:
                    proba = model.predict(X)
                    preds += np.column_stack((1 - proba, proba)) * weight
            except Exception as e:
                logger.error(f"Error in {name} prediction: {str(e)}")
                logger.error(traceback.format_exc())
                raise
                
        # Normalize by sum of weights
        weight_sum = sum(self.weights)
        if weight_sum > 0:
            return preds / weight_sum
        return preds  # Fallback if all weights are zero
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
    
    def __getstate__(self):
        # Ensure the object can be pickled
        state = self.__dict__.copy()
        return state
        
    def __setstate__(self, state):
        # Ensure the object can be unpickled
        self.__dict__.update(state)


def train(**context):
    """Train multiple models and log to MLflow."""
    import os
    import json
    import joblib
    import numpy as np
    import pandas as pd
    import traceback
    from pathlib import Path
    
    logger.info("Starting train task...")
    
    # Ensure models directory exists
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating models directory: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    # Get data from previous tasks with error handling
    def get_xcom_data(key, task_id):
        data = context["ti"].xcom_pull(key=key, task_ids=task_id)
        if data is None:
            raise ValueError(f"No data found in XCom for {key} from {task_id}")
        return deserialize_data(data)
    
    logger.info("Loading data from XCom...")
    X_train = get_xcom_data("X_train", "split")
    X_val = get_xcom_data("X_val", "split")
    y_train = get_xcom_data("y_train", "split")
    y_val = get_xcom_data("y_val", "split")
    feature_names = context["ti"].xcom_pull(key="feature_names", task_ids="preprocess")
    
    if feature_names is None:
        raise ValueError("No feature_names found in XCom. Check if preprocess task completed successfully.")
    
    # Log data types and shapes
    logger.info(f"X_train type: {type(X_train)}, shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}")
    logger.info(f"X_val type: {type(X_val)}, shape: {X_val.shape if hasattr(X_val, 'shape') else 'N/A'}")
    logger.info(f"y_train type: {type(y_train)}, shape: {y_train.shape if hasattr(y_train, 'shape') else 'N/A'}")
    logger.info(f"y_val type: {type(y_val)}, shape: {y_val.shape if hasattr(y_val, 'shape') else 'N/A'}")
    
    # Ensure data is in the correct format
    if not all(isinstance(x, np.ndarray) for x in [X_train, X_val, y_train, y_val]):
        logger.warning("Converting data to numpy arrays...")
        X_train = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        X_val = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val
        y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
        y_val = np.array(y_val) if not isinstance(y_val, np.ndarray) else y_val
    
    # Ensure y is 1D
    if len(y_train.shape) > 1 and y_train.shape[1] == 1:
        y_train = y_train.ravel()
    if len(y_val.shape) > 1 and y_val.shape[1] == 1:
        y_val = y_val.ravel()
        
    logger.info(f"Final shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    # Initialize MLflow with error handling
    try:
        mlflow_config = setup_mlflow()
        logger.info(f"MLflow configuration: {mlflow_config}")
    except Exception as e:
        logger.error(f"Error initializing MLflow: {str(e)}")
        logger.error(traceback.format_exc())
        raise# Continue without MLflow if initialization fails
    
    # Define models to train with better error handling
    models = []
    try:
        models.append((
            LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                solver='lbfgs',
                random_state=42,
                n_jobs=-1
            ), 
            "LogisticRegression"
        ))
    except Exception as e:
        logger.error(f"Error initializing LogisticRegression: {str(e)}")
        logger.error(traceback.format_exc())
    
    try:
        models.append((
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ), 
            "RandomForest"
        ))
    except Exception as e:
        logger.error(f"Error initializing RandomForest: {str(e)}")
        logger.error(traceback.format_exc())
    
    try:
        models.append((
            XGBClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]) if sum(y_train) > 0 else 1
            ), 
            "XGBoost"
        ))
    except Exception as e:
        logger.error(f"Error initializing XGBoost: {str(e)}")
        logger.error(traceback.format_exc())
    
    try:
        models.append((
            LGBMClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ), 
            "LightGBM"
        ))
    except Exception as e:
        logger.error(f"Error initializing LightGBM: {str(e)}")
        logger.error(traceback.format_exc())
    
    try:
        models.append((
            CatBoostClassifier(
                iterations=100,
                random_seed=42,
                verbose=0,
                thread_count=-1,
                class_weights=[1, len(y_train[y_train==0])/len(y_train[y_train==1])] if sum(y_train) > 0 else [1, 1]
            ), 
            "CatBoost"
        ))
    except Exception as e:
        logger.error(f"Error initializing CatBoost: {str(e)}")
        logger.error(traceback.format_exc())
    
    if not models:
        raise ValueError("Failed to initialize any models. Check logs for initialization errors.")
    
    logger.info(f"Successfully initialized {len(models)} models for training")
    
    best_model = None
    best_metric = -1
    best_model_path = None
    results = {}
    trained_models = {}
    total_models = len(models)
    successful_models = 0
    
    logger.info("\n" + "="*80)
    logger.info(f"STARTING MODEL TRAINING - {total_models} MODELS TO TRAIN")
    logger.info("="*80 + "\n")
    
    for idx, (model, model_name) in enumerate(models, 1):
        logger.info("\n" + "-"*60)
        logger.info(f"STAGE {idx}/{total_models}: TRAINING {model_name.upper()}")
        logger.info("-"*60)
        try:
            with mlflow.start_run(run_name=model_name, nested=True) as run:
                run_id = run.info.run_id
                # Train and evaluate the model
                result = train_model(model, model_name, X_train, X_val, y_train, y_val, feature_names, context)
                results[model_name] = result
                trained_models[model_name] = result['model']
                
                # Log parameters and metrics
                mlflow.log_params({
                    'model_type': model_name,
                    'training_samples': len(X_train),
                    'validation_samples': len(X_val),
                    'features_count': X_train.shape[1],
                    'positive_class_ratio': np.mean(y_train)
                })
                
                # Log metrics
                mlflow.log_metrics({
                    k: v for k, v in result['metrics'].items() 
                    if not k.startswith('train_')
                })
                
                # Log feature importance with error handling
                try:
                    if result['feature_importance']:
                        # Save feature importance locally first
                        feature_importance_path = str(MODELS_DIR / f"{model_name}_feature_importance.json")
                        with open(feature_importance_path, 'w') as f:
                            json.dump(result['feature_importance'], f)
                        # Log the saved file
                        mlflow.log_artifact(feature_importance_path, "feature_importance")
                except Exception as e:
                    logger.error(f"Error logging feature importance: {str(e)}")
                    logger.error(traceback.format_exc())
                
                # Save the model to a temporary file
                model_path = str(MODELS_DIR / f"{model_name}_{run_id}.joblib")
                joblib.dump(model, model_path)
                logger.info(f"Model saved to {model_path}")
                
                # Log the model to MLflow
                try:
                    # Log the model with MLflow
                    model_info = mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path=f"models/{model_name}",
                        registered_model_name=f"churn-prediction-{model_name}"
                    )
                    logger.info(f"Model logged to MLflow with run_id: {mlflow.active_run().info.run_id}")
                    logger.info(f"Model artifact path: {model_info.model_uri}")
                    
                    # Store the model URI in XCom for the persist task
                    context['ti'].xcom_push(key='model_uri', value=model_info.model_uri)
                    context['ti'].xcom_push(key='model_path', value=model_path)
                    
                except Exception as e:
                    logger.error(f"Failed to log model to MLflow: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Still save the model path even if MLflow logging fails
                    context['ti'].xcom_push(key='model_path', value=model_path)
                    raise
                
                # Track best model based on validation AUC-ROC
                if result['metrics']['val_auc_roc'] > best_metric:
                    best_metric = result['metrics']['val_auc_roc']
                    best_model = result.copy()  # Create a copy to avoid reference issues
                    best_model['model_name'] = model_name  # Ensure model_name is set
                    best_model['model_path'] = model_path
                    best_model_path = model_path
                
                logger.info(f"✓ {model_name} training completed in {result['training_time']:.2f} seconds")
                logger.info(f"✓ {model_name} validation AUC-ROC: {result['metrics']['val_auc_roc']:.4f}")
                logger.info(f"✓ Model saved to: {model_path}")
                successful_models += 1
                
        except Exception as e:
            logger.error(f"✗ Error training {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            logger.info(f"COMPLETED STAGE {idx}/{total_models}: {model_name.upper()}")
            logger.info("-"*60 + "\n")
    
    if best_model is None:
        error_msg = f"No models were successfully trained; {successful_models} models attempted"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    # Create and evaluate ensembles
    if len(trained_models) > 1:
        logger.info("\n" + "="*80)
        logger.info("STARTING ENSEMBLE TRAINING")
        logger.info("="*80 + "\n")
        
        try:
            logger.info("STAGE 1/2: CREATING VOTING CLASSIFIER")
            
            # 1. Voting Classifier
            voting_clf = VotingClassifier(
                estimators=[(name, model['model']) for name, model in results.items()],
                voting='soft',
                n_jobs=-1
            )
            
            # Train voting classifier
            start_time = time.time()
            voting_clf.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate voting classifier
            y_val_proba = voting_clf.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, y_val_proba)
            
            # Save voting classifier
            voting_path = str(MODELS_DIR / f"voting_classifier_{datetime.now().strftime('%Y%m%dT%H%M%S')}.joblib")
            joblib.dump(voting_clf, voting_path)
            
            # Log to MLflow
            with mlflow.start_run(run_name="VotingClassifier", nested=True):
                mlflow.log_metrics({
                    'val_auc_roc': val_auc,
                    'training_time': training_time
                })
                mlflow.sklearn.log_model(
                    sk_model=voting_clf,
                    artifact_path="models/voting_classifier",
                    registered_model_name="churn_prediction_voting"
                )
                
                # Track if this is the best model
                if val_auc > best_metric:
                    best_metric = val_auc
                    best_model = {
                        'model': voting_clf,
                        'metrics': {'val_auc_roc': val_auc},
                        'training_time': training_time,
                        'model_path': voting_path,
                        'model_type': 'VotingClassifier'
                    }
                    best_model_path = voting_path
            
            logger.info(f"✓ VotingClassifier training completed in {training_time:.2f} seconds")
            logger.info(f"✓ VotingClassifier validation AUC-ROC: {val_auc:.4f}")
            
            logger.info("\nSTAGE 2/2: CREATING WEIGHTED AVERAGE ENSEMBLE")
            
            # 2. Weighted Average Ensemble using the module-level class
            
            # Create and evaluate weighted ensemble
            weights = [result['metrics']['val_auc_roc'] for result in results.values()]
            weighted_ensemble = WeightedEnsemble(trained_models, weights)
            
            # Evaluate weighted ensemble
            start_time = time.time()
            y_val_proba = weighted_ensemble.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, y_val_proba)
            training_time = time.time() - start_time
            
            # Save weighted ensemble
            weighted_path = str(MODELS_DIR / f"weighted_ensemble_{datetime.now().strftime('%Y%m%dT%H%M%S')}.joblib")
            joblib.dump(weighted_ensemble, weighted_path)
            
            # Log to MLflow
            with mlflow.start_run(run_name="WeightedEnsemble", nested=True):
                mlflow.log_metrics({
                    'val_auc_roc': val_auc,
                    'training_time': training_time
                })
                mlflow.sklearn.log_model(
                    sk_model=weighted_ensemble,
                    artifact_path="models/weighted_ensemble",
                    registered_model_name="churn_prediction_weighted"
                )
                
                # Track if this is the best model
                if val_auc > best_metric:
                    best_metric = val_auc
                    best_model = {
                        'model': weighted_ensemble,
                        'metrics': {'val_auc_roc': val_auc},
                        'training_time': training_time,
                        'model_path': weighted_path,
                        'model_name': 'WeightedEnsemble'
                    }
                    best_model_path = weighted_path
            
            logger.info(f"✓ WeightedEnsemble training completed in {training_time:.2f} seconds")
            logger.info(f"✓ WeightedEnsemble validation AUC-ROC: {val_auc:.4f}")
            
        except Exception as e:
            logger.error(f"Error creating ensembles: {str(e)}")
            logger.error(traceback.format_exc())
            if best_model is None:
                raise
    
    # Prepare best model info for XCom
    best_model_info = {
        'best_model_path': best_model_path or 'unknown_path',
        'best_model_type': best_model.get('model_name', 'unknown_model') if isinstance(best_model, dict) else 'unknown_model',
        'best_metric': best_metric,
        'model_name': best_model.get('model_name', 'unknown_model') if isinstance(best_model, dict) else 'unknown_model'
    }
    
    # Push all values to XCom
    for key, value in best_model_info.items():
        if key != 'model_name':  # Don't push model_name separately as it's included in the dict
            context['ti'].xcom_push(key=key, value=value)
    
    # Log completion with safe dictionary access
    logger.info("\n" + "="*80)
    logger.info(f"TRAINING COMPLETED - BEST MODEL: {best_model_info['model_name']} (AUC-ROC: {best_metric:.4f})")
    logger.info("="*80 + "\n")
    
    return best_model_info


def evaluate(**context):
    """Evaluate the best model and log metrics to MLflow."""
    import os
    import joblib
    import numpy as np
    import pandas as pd
    import traceback
    import json
    from pathlib import Path
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    import mlflow
    
    logger.info("Starting model evaluation...")
    
    # Ensure directories exist with proper error handling
    ensure_directory(PLOTS_DIR)
    
    # Get data from previous tasks with error handling
    def get_xcom_data(key, task_id, required=True):
        data = context["ti"].xcom_pull(key=key, task_ids=task_id)
        if data is None and required:
            raise ValueError(f"No data found in XCom for {key} from {task_id}")
        return data
    
    try:
        # Get validation data (using validation split since we don't have a separate test set)
        logger.info("Fetching X_val from XCom...")
        X_test = get_xcom_data("X_val", "split")
        logger.info(f"X_val raw type: {type(X_test)}")
        
        logger.info("Fetching y_val from XCom...")
        y_test = get_xcom_data("y_val", "split")
        logger.info(f"y_val raw type: {type(y_test)}")
        
        # Get model info from training task
        logger.info("Fetching model info from XCom...")
        model_path = get_xcom_data("best_model_path", "train")
        logger.info(f"Model path: {model_path}")
        model_name = get_xcom_data("best_model_type", "train")
        
        # Optional data
        model_params = get_xcom_data("model_params", "train", required=False)
        training_time = get_xcom_data("training_time", "train", required=False)
        feature_importance = get_xcom_data("feature_importance", "train", required=False)
        
        # Deserialize the data if it's in serialized format
        if isinstance(X_test, dict) and X_test.get('__type__') == 'numpy_array':
            logger.info("Deserializing X_test from base64")
            X_test = deserialize_data(X_test)
            
        if isinstance(y_test, dict) and y_test.get('__type__') == 'numpy_array':
            logger.info("Deserializing y_test from base64")
            y_test = deserialize_data(y_test)
            
        logger.info(f"After deserialization - X_test type: {type(X_test)}, y_test type: {type(y_test)}")
        
        # Convert to numpy arrays if they're not already
        if not isinstance(X_test, np.ndarray):
            logger.info("Converting X_test to numpy array")
            X_test = np.array(X_test, dtype=np.float32)
            
        if not isinstance(y_test, np.ndarray):
            logger.info("Converting y_test to numpy array")
            y_test = np.array(y_test, dtype=np.float32)
            
        logger.info(f"After conversion - X_test shape: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}, y_test shape: {y_test.shape if hasattr(y_test, 'shape') else 'N/A'}")
        
        # Debug logging after conversion
        logger.info(f"After conversion - X_test shape: {X_test.shape if hasattr(X_test, 'shape') else 'N/A'}, y_test shape: {y_test.shape if hasattr(y_test, 'shape') else 'N/A'}")
        
        # Ensure y is 1D
        if hasattr(y_test, 'shape') and len(y_test.shape) > 1 and y_test.shape[1] == 1:
            logger.info("Reshaping y_test to 1D")
            y_test = y_test.ravel()
            logger.info(f"Reshaped y_test shape: {y_test.shape}")
            
        # Validate data shapes
        x_shape = X_test.shape if hasattr(X_test, 'shape') else 'N/A'
        y_shape = y_test.shape if hasattr(y_test, 'shape') else 'N/A'
        
        logger.info(f"Validating shapes - X_test: {x_shape}, y_test: {y_shape}")
        
        if not hasattr(X_test, 'size') or not hasattr(y_test, 'size') or X_test.size == 0 or y_test.size == 0:
            raise ValueError(f"Empty or invalid data received - X_test: {x_shape}, y_test: {y_shape}")
            
        if not hasattr(X_test, 'shape') or not hasattr(y_test, 'shape') or len(X_test.shape) == 0 or len(y_test.shape) == 0:
            raise ValueError(f"Invalid shape dimensions - X_test: {x_shape}, y_test: {y_shape}")
            
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError(f"Mismatched number of samples - X_test: {X_test.shape[0]}, y_test: {y_test.shape[0]}")
            
        logger.info(f"Successfully loaded test data - X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Load the model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        logger.info(f"Loading {model_name} model from {model_path}")
        model = joblib.load(model_path)
        
        # Initialize MLflow with server configuration
        mlflow_config = setup_mlflow()
        
        # Start MLflow run with the model name and link to the training run if available
        with mlflow.start_run(run_name=f"evaluate_{model_name}", 
                           experiment_id=mlflow_config['experiment_id'],
                           nested=True) as run:
            # Log model parameters if available
            if model_params:
                try:
                    mlflow.log_params({
                        'model_type': model_name,
                        **{k: str(v) for k, v in model_params.items() if len(str(v)) < 100}  # Avoid logging large params
                    })
                except Exception as e:
                    logger.warning(f"Failed to log model parameters: {str(e)}")
            else:
                logger.warning("No model parameters found to log")
                
            # Make predictions
            logger.info("Making predictions...")
            try:
                preds = model.predict(X_test)
                
                # Safely handle predict_proba
                pred_proba = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)
                    # Check if we have at least 2 classes
                    if proba.shape[1] >= 2:
                        pred_proba = proba[:, 1]
                    else:
                        logger.warning("Model's predict_proba returned less than 2 classes. ROC-AUC will not be calculated.")
                        
            except Exception as e:
                logger.error(f"Error making predictions: {str(e)}")
                logger.error(traceback.format_exc())
                raise
                
            # Calculate metrics
            metrics = {
                'test_accuracy': accuracy_score(y_test, preds),
                'test_precision': precision_score(y_test, preds, zero_division=0),
                'test_recall': recall_score(y_test, preds, zero_division=0),
                'test_f1': f1_score(y_test, preds, zero_division=0),
            }
            
            # Add AUC-ROC if we have probability predictions
            if pred_proba is not None:
                metrics['test_auc_roc'] = roc_auc_score(y_test, pred_proba)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log training time if available and valid
            if training_time is not None and isinstance(training_time, (int, float)) and training_time > 0:
                try:
                    mlflow.log_metric("training_time_seconds", float(training_time))
                except Exception as e:
                    logger.warning(f"Failed to log training time: {str(e)}")
            else:
                logger.warning(f"Invalid or missing training time: {training_time}")
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=f"churn_prediction_{model_name.lower()}"
            )
            
            # Log feature importance if available
            if feature_importance:
                try:
                    importance_path = "feature_importance.json"
                    with open(importance_path, "w") as f:
                        json.dump(feature_importance, f, indent=2)
                    mlflow.log_artifact(importance_path)
                    logger.info("Logged feature importance to MLflow")
                except Exception as e:
                    logger.warning(f"Failed to log feature importance: {str(e)}")
            
            # Log confusion matrix
            try:
                cm = confusion_matrix(y_test, preds)
                cm_path = "confusion_matrix.png"
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(cm_path)
                plt.close()
                mlflow.log_artifact(cm_path)
                logger.info("Logged confusion matrix to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log confusion matrix: {str(e)}")
            
            # Log classification report
            try:
                report = classification_report(y_test, preds, output_dict=True)
                report_path = "classification_report.json"
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)
                mlflow.log_artifact(report_path)
                logger.info("Logged classification report to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log classification report: {str(e)}")
        
        # Push metrics to XCom for downstream tasks
        context["ti"].xcom_push(key="metrics", value=metrics)
        context["ti"].xcom_push(key="model_path", value=model_path)
        context["ti"].xcom_push(key="best_model_name", value=model_name)
        
        logger.info(f"Evaluation completed for {model_name}. Test metrics: {json.dumps(metrics, indent=2)}")
        
        return {
            'status': 'success',
            'model': model_name,
            'run_id': run.info.run_id,
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def cleanup_old_files(directory: Path, pattern: str, max_files_to_keep: int = 2):
    """
    Clean up old files in the specified directory matching the given pattern.
    Keeps the most recent max_files_to_keep files.
    
    Args:
        directory: Directory to search for files
        pattern: File pattern to match (e.g., "*.joblib")
        max_files_to_keep: Number of most recent files to keep
    """
    try:
        # Ensure directory exists
        if not directory.exists():
            logger.info(f"Directory {directory} does not exist, nothing to clean up")
            return
            
        # Get all files matching the pattern
        files = list(directory.glob(pattern))
        
        if not files:
            logger.info(f"No files matching {pattern} found in {directory}")
            return
            
        # Sort by modification time (newest first)
        files.sort(key=os.path.getmtime, reverse=True)
        
        # Log what we found
        logger.info(f"Found {len(files)} files matching {pattern} in {directory}")
        for i, f in enumerate(files[:max_files_to_keep], 1):
            logger.info(f"  Keeping ({i}/{max_files_to_keep}): {f.name} (modified: {datetime.fromtimestamp(f.stat().st_mtime)})")
        
        # Remove old files
        removed_count = 0
        for old_file in files[max_files_to_keep:]:
            try:
                file_size = old_file.stat().st_size / (1024 * 1024)  # Size in MB
                old_file.unlink()
                logger.info(f"Removed old file: {old_file.name} (size: {file_size:.2f}MB)")
                removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove {old_file}: {e}")
                
        logger.info(f"Cleanup complete: Kept {min(len(files), max_files_to_keep)} files, removed {removed_count} files")
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_files for {directory}/{pattern}: {str(e)}")
        logger.error(traceback.format_exc())


def cleanup_processed_data(**context):
    """Clean up old processed data and model files, keeping only the 2 most recent."""
    try:
        # Clean up old processed data files, keeping the 2 most recent
        cleanup_old_files(PROCESSED_DIR, "processed_data_*.joblib", max_files_to_keep=2)
        
        # Clean up old model files, keeping the 2 most recent
        models_dir = Path("/opt/airflow/data/models")
        cleanup_old_files(MODELS_DIR, "model_*.joblib", max_files_to_keep=2)
        
        # Also clean up old metrics and feature importance files
        cleanup_old_files(MODELS_DIR, "metrics_*.json", max_files_to_keep=2)
        cleanup_old_files(MODELS_DIR, "feature_importance_*.json", max_files_to_keep=2)
        
        logger.info("Cleanup completed successfully - kept 2 most recent files")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise


def persist(**context):
    """Persist the model and related artifacts."""
    import joblib
    import os
    import mlflow
    from mlflow.tracking import MlflowClient
    from pathlib import Path
    import traceback
    
    # Get data from previous tasks
    model_path = context["ti"].xcom_pull(key="best_model_path", task_ids="train")
    model_uri = context["ti"].xcom_pull(key="model_uri", task_ids="train")
    metrics = context["ti"].xcom_pull(key="metrics", task_ids="evaluate")
    feature_importance = context["ti"].xcom_pull(key="feature_importance", task_ids="train")
    
    # Log the retrieved values for debugging
    logger.info(f"Retrieved from XCom - model_path: {model_path}")
    logger.info(f"Retrieved from XCom - model_uri: {model_uri}")
    logger.info(f"Retrieved from XCom - metrics: {metrics is not None}")
    logger.info(f"Retrieved from XCom - feature_importance: {feature_importance is not None}")
    
    # Create output directory if it doesn't exist
    output_dir = Path("/opt/airflow/data/models").absolute()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o755)  # Ensure proper permissions
    
    # Define output paths with timestamp
    timestamp = context['ts_nodash']
    final_model_path = output_dir / f"model_{timestamp}.joblib"
    metrics_path = output_dir / f"metrics_{timestamp}.json"
    
    # Set up MLflow with the server
    mlflow_config = setup_mlflow()
    
    # Clear any S3-related environment variables if they exist
    for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'MLFLOW_S3_ENDPOINT_URL']:
        os.environ.pop(key, None)
    
    # Handle model file persistence
    model_file = Path(model_path)
    final_model_file = Path(final_model_path)
    
    # If source and destination are the same, no need to copy
    if str(model_file.absolute()) == str(final_model_file.absolute()):
        logger.info(f"Source and destination paths are the same: {model_file}. No need to copy.")
    elif not model_file.exists():
        logger.error(f"Model file not found at {model_path}. Cannot persist model.")
        # Try to load the model from MLflow if available
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            client = MlflowClient()
            # Get the latest version of the model
            model_versions = client.search_model_versions(f"name='telco-churn-lr'")
            if model_versions:
                # Sort by version number (descending) and get the latest
                latest_version = sorted(model_versions, key=lambda x: int(x.version), reverse=True)[0]
                model_uri = f"models:/telco-churn-lr/{latest_version.version}"
                
                # Download the model
                logger.info(f"Downloading model from MLflow: {model_uri}")
                model = mlflow.sklearn.load_model(model_uri)
                
                # Save the model locally
                joblib.dump(model, final_model_path)
                logger.info(f"Model downloaded from MLflow and saved to {final_model_path}")
            else:
                raise FileNotFoundError("No model versions found in MLflow")
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            raise
    else:
        # Copy the model file to the final location
        import shutil
        shutil.copy2(str(model_file), str(final_model_file))
        logger.info(f"Model copied from {model_file} to {final_model_file}")
    
    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save feature importance if available
    if feature_importance:
        importance_path = output_dir / f"feature_importance_{timestamp}.json"
        with open(importance_path, 'w') as f:
            json.dump(feature_importance, f)
        logger.info(f"Feature importance saved to {importance_path}")
    
    # Configure MLflow to use local filesystem and disable S3
    mlflow_dir = Path("/opt/airflow/mlruns").absolute()
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    
    # Set tracking URI to local filesystem
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    
    # Clear any S3-related environment variables
    os.environ.pop('AWS_ACCESS_KEY_ID', None)
    os.environ.pop('AWS_SECRET_ACCESS_KEY', None)
    os.environ.pop('MLFLOW_S3_ENDPOINT_URL', None)
    
    # Ensure the experiment exists
    experiment_name = "churn-prediction"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.info(f"Creating new MLflow experiment: {experiment_name}")
            experiment_id = mlflow.create_experiment(experiment_name, str(mlflow_dir / experiment_name))
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
    except Exception as e:
        logger.error(f"Error setting up MLflow experiment: {e}")
        logger.error(traceback.format_exc())
        experiment_id = "0"  # Fallback to default experiment
    
    # Log to MLflow before cleaning up the model file
    with mlflow.start_run(experiment_id=experiment_id):
        # Get model parameters and training time from XCom
        model_params = context["ti"].xcom_pull(key="model_params", task_ids="train")
        training_time = context["ti"].xcom_pull(key="training_time", task_ids="train")
        
        # Log parameters and metrics
        if model_params:
            mlflow.log_params(model_params)
        
        if metrics:
            mlflow.log_metrics(metrics)
        
        # Log feature importance if available
        if feature_importance:
            importance_path = output_dir / "feature_importance_mlflow.json"
            with open(importance_path, "w") as f:
                json.dump(feature_importance, f)
            mlflow.log_artifact(str(importance_path))
            logger.info(f"Feature importance logged to MLflow: {importance_path}")
        
        # Load the model and log it to MLflow
        try:
            if final_model_file.exists():
                model = joblib.load(final_model_file)
                # Create a local directory for MLflow artifacts
                artifact_path = str(output_dir / "mlflow_artifacts")
                os.makedirs(artifact_path, exist_ok=True)
                
                # Log the model to the local filesystem
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=None  # Don't register in model registry
                )
                logger.info(f"Model successfully logged to MLflow at {mlflow.get_artifact_uri()}")
            else:
                logger.warning(f"Model file not found at {final_model_file}. Skipping MLflow model logging.")
        except Exception as e:
            logger.error(f"Failed to log model to MLflow: {e}")
            logger.error(traceback.format_exc())  # Log full traceback for debugging
            # Don't fail the task if MLflow logging fails
        
        # Log additional metadata
        mlflow.set_tag("model_type", "LogisticRegression")
    
    # Clean up the temporary model file if it still exists
    try:
        if model_file.exists() and model_file != final_model_file:
            model_file.unlink()
            logger.info(f"Removed temp model file: {model_file}")
    except Exception as e:
        logger.warning(f"Failed to remove temporary model file: {e}")


default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2025, 7, 1),
}

with DAG(
    dag_id="daily_training_pipeline",
    schedule_interval="0 2 * * *",
    default_args=default_args,
    catchup=False,
    tags=["training", "automation"],
) as dag:
    load = PythonOperator(task_id="load_data", python_callable=load_data, provide_context=True)
    validate = PythonOperator(task_id="validate_data", python_callable=validate_data, provide_context=True)
    prep = PythonOperator(task_id="preprocess", python_callable=preprocess, provide_context=True)
    split_data = PythonOperator(task_id="split", python_callable=split, provide_context=True)
    train_task = PythonOperator(task_id="train", python_callable=train, provide_context=True)
    eval_task = PythonOperator(task_id="evaluate", python_callable=evaluate, provide_context=True)
    persist_task = PythonOperator(task_id="persist", python_callable=persist, provide_context=True)

    # Add cleanup task
    cleanup_task = PythonOperator(
        task_id="cleanup",
        python_callable=cleanup_processed_data,
        provide_context=True,
    )
    
    # Task dependencies
    load >> validate >> prep >> split_data >> train_task >> eval_task >> persist_task >> cleanup_task
    
    # Create directories at DAG load time
    ensure_directories()
    
    # Add MLflow UI URL to task documentation
    dag.doc_md = """
    # Telco Churn Prediction Training Pipeline
    
    This DAG automates the training of a churn prediction model.
    
    ## MLflow UI
    Access the MLflow tracking UI at: http://localhost:5001
    
    ## Model Registry
    Models are registered in MLflow and can be promoted to production via the MLflow UI.
    """
