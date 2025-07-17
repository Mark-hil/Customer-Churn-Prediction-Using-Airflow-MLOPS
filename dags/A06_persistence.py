"""Persistence module for saving models, metrics, and artifacts."""
import os
import json
import joblib
import logging
import mlflow
import tempfile
import boto3
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple
from datetime import datetime
from botocore.exceptions import ClientError

# Import utility functions
from A05_utility import ensure_directory, cleanup_old_files

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("/opt/airflow/data")
MODELS_DIR = DATA_DIR / "models"
MLFLOW_DIR = Path("/opt/airflow/mlruns").absolute()

# Ensure directories exist
for directory in [MODELS_DIR, MLFLOW_DIR]:
    ensure_directory(directory)

class ModelPersistence:
    """Class for handling model persistence and MLflow integration."""
    
    def __init__(self, experiment_name: str = "churn-prediction", use_s3: bool = False, s3_bucket: Optional[str] = None):
        """
        Initialize the ModelPersistence class.
        
        Args:
            experiment_name: Name of the MLflow experiment
            use_s3: Whether to store models in S3
            s3_bucket: Name of the S3 bucket to use for storage
        """
        self.experiment_name = experiment_name
        self.experiment_id = self._setup_mlflow()
        self.use_s3 = use_s3
        self.s3_bucket = s3_bucket
        self.s3_client = None
        self.models_dir = MODELS_DIR  # Expose the models directory
        
        if self.use_s3 and self.s3_bucket:
            try:
                self.s3_client = boto3.client('s3')
                # Verify S3 connection
                self.s3_client.head_bucket(Bucket=self.s3_bucket)
                logger.info(f"Successfully connected to S3 bucket: {self.s3_bucket}")
            except ClientError as e:
                logger.error(f"Error connecting to S3: {e}")
                self.use_s3 = False
    
    def _setup_mlflow(self) -> str:
        """Set up MLflow tracking with the MLflow service."""
        try:
            # Configure MLflow to use the MLflow service
            mlflow.set_tracking_uri("http://mlflow:5000")
            
            # # Clear any S3-related environment variables
            # for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'MLFLOW_S3_ENDPOINT_URL']:
            #     os.environ.pop(key, None)
            
            # Set MLflow to use SQLite for better reliability
            os.environ['MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE'] = '3600'
            os.environ['MLFLOW_SQLALCHEMYSTORE_POOL_PRE_PING'] = 'True'
            
            # Create or get the experiment
            try:
                # Try to get the experiment first
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    logger.info(f"Creating new MLflow experiment: {self.experiment_name}")
                    experiment_id = mlflow.create_experiment(self.experiment_name)
                    logger.info(f"Created MLflow experiment with ID: {experiment_id}")
                else:
                    experiment_id = experiment.experiment_id
                    logger.info(f"Using existing MLflow experiment: {self.experiment_name} (ID: {experiment_id})")
                
                return experiment_id
                
            except Exception as e:
                logger.error(f"Error in MLflow experiment setup: {e}")
                # Fall back to default experiment
                return "0"
            
        except Exception as e:
            logger.error(f"Critical error setting up MLflow: {e}")
            # Fall back to default experiment
            return "0"
    
    def _get_s3_key(self, model_name: str, filename: str, base_path: str = None) -> str:
        """
        Generate a clean S3 key path for model artifacts.
        
        Args:
            model_name: Name of the model
            filename: Name of the file to store
            base_path: Base path in S3 (default: 'trained_models')
            
        Returns:
            str: S3 key path in format: {base_path}/{model_name}/{timestamp}/{filename}
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        base = base_path or 'trained_models'
        return f"{base}/{model_name}/{timestamp}/{filename}"
    
    def _upload_to_s3(self, local_path: str, s3_key: str) -> bool:
        """
        Upload a file to S3.
        
        Args:
            local_path: Path to the local file or directory
            s3_key: S3 object key (path in the bucket)
            
        Returns:
            bool: True if upload was successful, False otherwise
        """
        if not self.use_s3 or not self.s3_client:
            return False
            
        try:
            local_path = Path(local_path)
            
            # If it's a directory, upload all files recursively
            if local_path.is_dir():
                success = True
                for file_path in local_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path)
                        file_s3_key = str(Path(s3_key) / relative_path).replace('\\', '/')
                        
                        try:
                            self.s3_client.upload_file(
                                str(file_path),
                                self.s3_bucket,
                                file_s3_key
                            )
                            logger.info(f"Uploaded {file_path} to s3://{self.s3_bucket}/{file_s3_key}")
                        except Exception as e:
                            logger.error(f"Error uploading {file_path} to S3: {e}")
                            success = False
                return success
            
            # If it's a file, upload it directly
            else:
                self.s3_client.upload_file(
                    str(local_path),
                    self.s3_bucket,
                    s3_key
                )
                logger.info(f"Uploaded {local_path} to s3://{self.s3_bucket}/{s3_key}")
                return True
                
        except Exception as e:
            logger.error(f"Error in S3 upload operation for {local_path}: {e}")
            return False
    
    def _save_model_locally(self, model: Any, model_path: Path) -> bool:
        """Save model to local filesystem."""
        try:
            joblib.dump(model, model_path)
            return True
        except Exception as e:
            logger.error(f"Error saving model locally: {e}")
            return False
            
    def save_model(
        self,
        model: Any,
        model_name: str,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        artifacts: Optional[Dict[str, str]] = None,
        registered_model_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save a model and related artifacts to local filesystem and optionally to S3.
        
        Args:
            model: The trained model to save
            model_name: Name of the model
            metrics: Dictionary of evaluation metrics
            params: Dictionary of model parameters
            feature_importance: Dictionary of feature importances
            artifacts: Dictionary of additional artifacts to save
            registered_model_name: If provided, register the model with this name in MLflow
            
        Returns:
            Dictionary with paths to saved artifacts
        """
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        model_dir_name = f"{model_name}_{timestamp}"
        model_dir = MODELS_DIR / model_dir_name
        ensure_directory(model_dir)
        
        saved_paths = {}
        s3_paths = {}
        
        try:
            # Save model locally first
            model_path = model_dir / "model.joblib"
            if not self._save_model_locally(model, model_path):
                return {}
            
            # Initialize saved paths dictionary
            saved_paths = {
                'model': str(model_path),
                'model_dir': str(model_dir)
            }
            
            # Save metrics
            if metrics is not None:
                metrics_path = model_dir / "metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                saved_paths['metrics'] = str(metrics_path)
            
            # Save feature importance
            if feature_importance is not None:
                fi_path = model_dir / "feature_importance.json"
                with open(fi_path, 'w') as f:
                    json.dump(feature_importance, f, indent=2)
                saved_paths['feature_importance'] = str(fi_path)
            
            # Save model parameters
            if params is not None:
                params_path = model_dir / "params.json"
                with open(params_path, 'w') as f:
                    json.dump(params, f, indent=2)
                saved_paths['params'] = str(params_path)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'timestamp': timestamp,
                'model_type': type(model).__name__,
                'metrics': metrics if metrics is not None else {},
                'params': params if params is not None else {},
                'feature_importance': feature_importance if feature_importance is not None else {},
                'registered_model_name': registered_model_name
            }
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            saved_paths['metadata'] = str(metadata_path)
            
            # Save preprocessing pipeline if provided in artifacts
            if artifacts and 'preprocessor' in artifacts:
                preprocessor_path = model_dir / "preprocessor.joblib"
                joblib.dump(artifacts['preprocessor'], preprocessor_path)
                saved_paths['preprocessor'] = str(preprocessor_path)
            
            # Save feature names if provided in artifacts
            if artifacts and 'feature_names' in artifacts:
                feature_names_path = model_dir / "feature_names.json"
                with open(feature_names_path, 'w') as f:
                    json.dump(artifacts['feature_names'], f, indent=2)
                saved_paths['feature_names'] = str(feature_names_path)
            
            # Save parameters
            if params is not None:
                params_path = model_dir / "params.json"
                with open(params_path, 'w') as f:
                    json.dump(params, f, indent=2)
                saved_paths['params'] = str(params_path)
            
            # Save additional artifacts
            if artifacts is not None:
                artifacts_dir = model_dir / "artifacts"
                artifacts_dir.mkdir(exist_ok=True)
                
                for name, content in artifacts.items():
                    artifact_path = artifacts_dir / name
                    with open(artifact_path, 'w') as f:
                        if hasattr(content, 'savefig'):
                            content.savefig(artifact_path, bbox_inches='tight')
                        elif isinstance(content, (str, bytes)):
                            f.write(content)
                        else:
                            json.dump(content, f, indent=2)
                    saved_paths[f"artifact_{name}"] = str(artifact_path)
            
            # If S3 is enabled, upload the entire model directory
            if self.use_s3 and self.s3_bucket:
                # For non-production models, use 'trained_models' as base path
                base_path = 'trained_models' if 'best' not in model_name.lower() else 'production_models'
                timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
                base_s3_path = f"{base_path}/{model_name}/{timestamp}"
                s3_paths = {}
                
                # Upload the entire model directory
                if self._upload_to_s3(model_dir, base_s3_path):
                    # Update S3 paths for all saved files
                    for key, local_path in saved_paths.items():
                        if key in ['model_dir', 's3_paths']:
                            continue
                            
                        # Convert local path to relative path
                        rel_path = Path(local_path).relative_to(model_dir)
                        s3_key = f"{base_s3_path}/{rel_path}"
                        s3_paths[key] = f"s3://{self.s3_bucket}/{s3_key}"
                
                saved_paths['s3_paths'] = s3_paths
            
            # Log to MLflow
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=model_name):
                # Log parameters
                if params:
                    mlflow.log_params(params)
                
                # Log metrics
                if metrics:
                    mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=registered_model_name or model_name
                )
                
                # Log artifacts
                for name, path in saved_paths.items():
                    if name not in ['model', 'model_dir', 's3_paths'] and Path(path).exists():
                        mlflow.log_artifact(path)
                
                # Log feature importance as a plot if available
                if feature_importance:
                    try:
                        import matplotlib.pyplot as plt
                        
                        # Sort features by importance
                        features = list(feature_importance.keys())
                        importances = [feature_importance[f] for f in features]
                        
                        # Create and save feature importance plot
                        plt.figure(figsize=(10, 6))
                        plt.barh(features[:20][::-1], importances[:20][::-1])  # Show top 20 features
                        plt.xlabel("Importance")
                        plt.title("Feature Importance")
                        
                        # Log the plot to MLflow
                        temp_dir = tempfile.mkdtemp()
                        plot_path = os.path.join(temp_dir, "feature_importance.png")
                        plt.savefig(plot_path, bbox_inches='tight')
                        plt.close()
                        
                        mlflow.log_artifact(plot_path, "plots")
                        
                    except Exception as e:
                        logger.warning(f"Could not log feature importance plot: {e}")
            
            return saved_paths
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
            logger.error(f"Traceback: {e.__traceback__}")
            raise
            
    def load_model(self, model_path: Union[str, Path]) -> Any:
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Loaded model
        """
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def load_latest_model(self, model_name: str) -> Any:
        """
        Load the most recent model with the given name.
        
        Args:
            model_name: Base name of the model to load
            
        Returns:
            Loaded model and metadata
        """
        try:
            # Find all model directories matching the pattern
            model_dirs = sorted(
                MODELS_DIR.glob(f"{model_name}_*"),
                key=os.path.getmtime,
                reverse=True
            )
            
            if not model_dirs:
                raise FileNotFoundError(f"No models found matching pattern: {model_name}_*")
            
            # Use the most recent model
            latest_dir = model_dirs[0]
            model_path = latest_dir / "model.joblib"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load the model
            model = self.load_model(model_path)
            
            # Load metadata if available
            metadata = {'model_path': str(model_path)}
            
            # Load metrics
            metrics_path = latest_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metadata['metrics'] = json.load(f)
            
            # Load feature importance
            fi_path = latest_dir / "feature_importance.json"
            if fi_path.exists():
                with open(fi_path, 'r') as f:
                    metadata['feature_importance'] = json.load(f)
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading latest model {model_name}: {e}")
            raise


def persist_model(
    model: Any,
    model_name: str,
    metrics: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    feature_importance: Optional[Dict[str, float]] = None,
    artifacts: Optional[Dict[str, str]] = None,
    registered_model_name: Optional[str] = None
) -> Dict[str, str]:
    """
    Helper function to persist a model and related artifacts.
    
    Args:
        model: The trained model to save
        model_name: Name of the model
        metrics: Dictionary of evaluation metrics
        params: Dictionary of model parameters
        feature_importance: Dictionary of feature importances
        artifacts: Dictionary of additional artifacts to save
        registered_model_name: If provided, register the model with this name in MLflow
        
    Returns:
        Dictionary with paths to saved artifacts
    """
    persister = ModelPersistence()
    return persister.save_model(
        model=model,
        model_name=model_name,
        metrics=metrics,
        params=params,
        feature_importance=feature_importance,
        artifacts=artifacts,
        registered_model_name=registered_model_name
    )


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Create a sample model
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Sample metrics and feature importance
    metrics = {
        'accuracy': 0.95,
        'precision': 0.94,
        'recall': 0.93,
        'f1': 0.935,
        'roc_auc': 0.98
    }
    
    params = {
        'n_estimators': 10,
        'max_depth': 5,
        'random_state': 42
    }
    
    feature_importance = {f"feature_{i}": float(imp) 
                         for i, imp in enumerate(model.feature_importances_, 1)}
    
    # Save the model
    print("Saving model...")
    paths = persist_model(
        model=model,
        model_name="random_forest_example",
        metrics=metrics,
        params=params,
        feature_importance=feature_importance,
        registered_model_name="churn_prediction_rf"
    )
    
    print("\nSaved paths:")
    for name, path in paths.items():
        print(f"{name}: {path}")
    
    # Load the model
    print("\nLoading model...")
    loaded_model, metadata = ModelPersistence().load_latest_model("random_forest_example")
    print(f"Loaded model type: {type(loaded_model)}")
    print(f"Metrics: {metadata.get('metrics')}")
