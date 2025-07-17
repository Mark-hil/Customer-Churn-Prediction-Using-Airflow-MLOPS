"""Model experimentation module for the churn prediction pipeline."""
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Set the backend to 'Agg' to avoid GUI-related issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define directories
BASE_DIR = Path("/opt/airflow")
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
PLOTS_DIR = DATA_DIR / "plots"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class ModelTrainer:
    """Class for training and evaluating machine learning models."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the model trainer."""
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
        
    def get_models(self) -> Dict[str, Any]:
        """Get a dictionary of models to evaluate."""
        models = {
            'logistic': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    class_weight='balanced',
                    max_iter=1000,
                    solver='liblinear'
                ),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'class_weight': ['balanced', None]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', 'balanced_subsample', None]
                }
            },
            'xgboost': {
                'model': XGBClassifier(
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 6, 9],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'scale_pos_weight': [1, (sum(self.y_train == 0) / sum(self.y_train == 1)) if sum(self.y_train == 1) > 0 else 1]
                }
            },
            'lightgbm': {
                'model': LGBMClassifier(
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 6, 9],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'class_weight': ['balanced', None]
                }
            },
            'catboost': {
                'model': CatBoostClassifier(
                    random_seed=self.random_state,
                    verbose=0,
                    thread_count=-1
                ),
                'params': {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'depth': [3, 6, 9],
                    'l2_leaf_reg': [1, 3, 5],
                    'scale_pos_weight': [1, (sum(self.y_train == 0) / sum(self.y_train == 1)) if sum(self.y_train == 1) > 0 else 1]
                }
            }
        }
        return models
    
    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cv: int = 5,
        n_iter: int = 10,
        scoring: str = 'roc_auc',
        use_smote: bool = False,
        models_to_train: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train multiple models with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            cv: Number of cross-validation folds
            n_iter: Number of iterations for randomized search
            scoring: Scoring metric to optimize
            use_smote: Whether to use SMOTE for handling class imbalance
            models_to_train: List of model names to train (if None, train all)
            
        Returns:
            Dictionary containing trained models and results
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # Get models to train
        all_models = self.get_models()
        if models_to_train is None:
            models_to_train = list(all_models.keys())
        
        # Train each model
        for model_name in models_to_train:
            if model_name not in all_models:
                logger.warning(f"Model {model_name} not found. Skipping...")
                continue
                
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name}...")
            logger.info(f"{'='*50}")
            
            try:
                # Get model and parameters
                model_info = all_models[model_name]
                model = model_info['model']
                param_dist = model_info['params']
                
                # Create pipeline with SMOTE if needed
                if use_smote:
                    pipeline = ImbPipeline([
                        ('smote', SMOTE(random_state=self.random_state)),
                        ('model', model)
                    ])
                    # Update parameter names for the pipeline
                    param_dist = {f'model__{k}': v for k, v in param_dist.items()}
                else:
                    pipeline = model
                
                # Perform randomized search with cross-validation
                start_time = time.time()
                
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_dist,
                    n_iter=n_iter,
                    cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
                    scoring=scoring,
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=1
                )
                
                # Fit the model
                search.fit(X_train, y_train)
                
                # Get the best model
                best_model = search.best_estimator_
                
                # Extract feature importances if available
                feature_importances = None
                if hasattr(best_model, 'feature_importances_'):
                    feature_importances = best_model.feature_importances_
                elif hasattr(best_model, 'coef_'):
                    feature_importances = best_model.coef_[0]
                
                # Make predictions on validation set if available
                val_metrics = {}
                if X_val is not None and y_val is not None:
                    y_pred = best_model.predict(X_val)
                    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
                    
                    val_metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
                
                # Store results
                self.models[model_name] = {
                    'model': best_model,
                    'best_params': search.best_params_,
                    'cv_results': search.cv_results_,
                    'feature_importances': feature_importances,
                    'train_time': time.time() - start_time,
                    'val_metrics': val_metrics
                }
                
                # Update best model
                if X_val is not None and y_val is not None:
                    score = val_metrics.get('roc_auc', 0)
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = best_model
                        self.best_model_name = model_name
                
                logger.info(f"{model_name} training completed in {self.models[model_name]['train_time']:.2f} seconds")
                logger.info(f"Best parameters: {search.best_params_}")
                if val_metrics:
                    logger.info(f"Validation ROC-AUC: {val_metrics.get('roc_auc', 0):.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        return self.models
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba)
        }
    
    def evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on the given dataset.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        results = {}
        
        for model_name, model_info in self.models.items():
            logger.info(f"\nEvaluating {model_name}...")
            
            try:
                # Make predictions
                y_pred = model_info['model'].predict(X)
                y_pred_proba = model_info['model'].predict_proba(X)[:, 1]
                
                # Calculate metrics
                metrics = self._calculate_metrics(y, y_pred, y_pred_proba)
                results[model_name] = metrics
                
                # Log metrics
                logger.info(f"{model_name} Metrics:")
                for metric, value in metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")
                
                # Log classification report
                logger.info("\nClassification Report:")
                logger.info(classification_report(y, y_pred, target_names=['No Churn', 'Churn']))
                
                # Log confusion matrix
                cm = confusion_matrix(y, y_pred)
                logger.info("Confusion Matrix:")
                logger.info(f"True Negatives: {cm[0, 0]}")
                logger.info(f"False Positives: {cm[0, 1]}")
                logger.info(f"False Negatives: {cm[1, 0]}")
                logger.info(f"True Positives: {cm[1, 1]}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Store results in XCom
        if 'ti' in context:
            ti = context['ti']
            # Store each model individually
            for model_name, model_info in results.items():
                ti.xcom_push(key=f'model_{model_name}', value=model_info)
            # Store best and second best model names
            ti.xcom_push(key='best_model_name', value=max(results.items(), key=lambda x: x[1]['metrics']['roc_auc'])[0])
            ti.xcom_push(key='second_best_model_name', value=sorted(results.items(), key=lambda x: x[1]['metrics']['roc_auc'], reverse=True)[1][0])
        
        return results
    
    def save_models(self, output_dir: Union[str, Path] = None) -> Dict[str, str]:
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models (default: MODELS_DIR)
            
        Returns:
            Dictionary mapping model names to their file paths
        """
        if not self.models:
            logger.warning("No models to save. Train models first.")
            return {}
        
        output_dir = Path(output_dir) if output_dir else MODELS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_models = {}
        
        # Save each model
        for model_name, model_info in self.models.items():
            try:
                model = model_info['model']
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = output_dir / f"{model_name}_{timestamp}.joblib"
                
                # Save model
                joblib.dump(model, model_path)
                saved_models[model_name] = str(model_path)
                
                # Save model metadata
                metadata = {
                    'model_name': model_name,
                    'timestamp': timestamp,
                    'best_params': model_info.get('best_params', {}),
                    'val_metrics': model_info.get('val_metrics', {}),
                    'feature_importances': model_info.get('feature_importances', []).tolist() if model_info.get('feature_importances') is not None else []
                }
                
                metadata_path = output_dir / f"{model_name}_{timestamp}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Saved {model_name} to {model_path}")
                
            except Exception as e:
                logger.error(f"Error saving {model_name}: {str(e)}")
        
        # Save best model separately
        if self.best_model is not None:
            best_model_path = output_dir / f"best_model_{self.best_model_name}.joblib"
            joblib.dump(self.best_model, best_model_path)
            saved_models['best_model'] = str(best_model_path)
            logger.info(f"Saved best model ({self.best_model_name}) to {best_model_path}")
        
        return saved_models


def load_model(model_path: Union[str, Path]) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        raise


def create_ensemble(
    models: Dict[str, Any], 
    X: pd.DataFrame, 
    y: pd.Series,
    voting: str = 'soft',
    weights: Optional[List[float]] = None
) -> VotingClassifier:
    """
    Create an ensemble of trained models.
    
    Args:
        models: Dictionary of trained models
        X: Features for weight estimation (if weights=None)
        y: Labels for weight estimation (if weights=None)
        voting: 'hard' or 'soft' voting
        weights: Optional list of weights for each model
        
    Returns:
        Fitted VotingClassifier
    """
    if not models:
        raise ValueError("No models provided for ensemble")
    
    # Create list of (name, model) tuples
    estimators = [(name, model) for name, model in models.items()]
    
    # If weights not provided, estimate weights based on validation performance
    if weights is None and X is not None and y is not None:
        logger.info("Estimating model weights based on validation performance...")
        weights = []
        
        for name, model in models.items():
            try:
                y_pred_proba = model.predict_proba(X)[:, 1]
                score = roc_auc_score(y, y_pred_proba)
                weights.append(score)
                logger.info(f"  {name}: ROC-AUC = {score:.4f}")
            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}")
                weights.append(1.0)  # Default weight if evaluation fails
    
    # Create and fit the ensemble
    ensemble = VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights,
        n_jobs=-1
    )
    
    # Fit the ensemble (this will refit all models unless they support warm_start)
    ensemble.fit(X, y)
    
    return ensemble


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
    logger.info(f"Training {model_name}...")
    
    # Start timer
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
    y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
    
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
    }
    
    # Add AUC-ROC if we have probabilities
    if y_train_proba is not None and y_val_proba is not None:
        metrics.update({
            'train_auc_roc': roc_auc_score(y_train, y_train_proba),
            'val_auc_roc': roc_auc_score(y_val, y_val_proba)
        })
    
    # Get feature importance if available
    feature_importance = {}
    try:
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            feature_importance = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # For linear models like Logistic Regression
            coef = model.coef_
            # Handle both binary and multi-class classification
            if len(coef.shape) == 1:
                importance_values = np.abs(coef[0])
            else:
                # For multi-class, take mean absolute coefficient across classes
                importance_values = np.mean(np.abs(coef), axis=0)
            
            feature_importance = dict(zip(feature_names, importance_values))
            
            # Normalize to sum to 1 for consistency with tree-based models
            if feature_importance:
                total = sum(feature_importance.values())
                if total > 0:
                    feature_importance = {k: v/total for k, v in feature_importance.items()}
    except Exception as e:
        logger.warning(f"Error extracting feature importance for {model_name}: {str(e)}")
        feature_importance = {}
    
    return {
        'model': model,
        'model_name': model_name,
        'metrics': metrics,
        'training_time': training_time,
        'feature_importance': feature_importance
    }

def train_models(data: Dict[str, Any], models_to_train: List[str], **context) -> Dict[str, Any]:
    """
    Train multiple models and return their results with MLflow logging.
    
    Args:
        data: Dictionary containing X_train, y_train, X_val, y_val
        models_to_train: List of model names to train
        **context: Airflow context (contains task instance for XCom)
        
    Returns:
        Dictionary containing trained models and their metadata
    """
    # Extract data
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    categorical_columns = data.get('categorical_columns', [])
    numerical_columns = data.get('numerical_columns', [])
    feature_names = categorical_columns + numerical_columns
    
    # Initialize MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("churn-prediction-training")
    
    # Model configurations
    model_configs = {
        'logistic_regression': {
            'model': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000,
                n_jobs=-1
            ),
            'params': {
                'C': 1.0,
                'solver': 'lbfgs'
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        },
        'xgboost': {
            'model': XGBClassifier(
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8
            }
        },
        'lightgbm': {
            'model': LGBMClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8
            }
        },
        'catboost': {
            'model': CatBoostClassifier(
                random_state=42,
                verbose=0,
                thread_count=-1
            ),
            'params': {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3
            }
        },
        'hist_gradient_boosting': {
            'model': HistGradientBoostingClassifier(
                random_state=42,
                max_iter=100
            ),
            'params': {
                'max_iter': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'min_samples_leaf': 20
            }
        }
    }
    
    results = {}
    
    # Train each model
    for model_name in models_to_train:
        if model_name not in model_configs:
            logger.warning(f"Model {model_name} not found in configurations, skipping...")
            continue
            
        logger.info(f"Starting MLflow run for {model_name}...")
        
        # Create a descriptive run name with model name and timestamp
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start MLflow run with a descriptive name (let MLflow handle the run_id)
        with mlflow.start_run(run_name=run_name):
            try:
                # Get model configuration
                config = model_configs[model_name]
                model = config['model']
                
                # Set model parameters
                if hasattr(model, 'set_params'):
                    model.set_params(**config.get('params', {}))
                
                # Log parameters
                mlflow.log_params(config.get('params', {}))
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("training_date", datetime.now().strftime("%Y-%m-%d"))
                
                # Train the model
                logger.info(f"Training {model_name}...")
                result = train_model(
                    model=model,
                    model_name=model_name,
                    X_train=X_train,
                    X_val=X_val,
                    y_train=y_train,
                    y_val=y_val,
                    feature_names=feature_names,
                    context=context
                )
                
                # Add preprocessing pipeline and feature names
                result['preprocessor'] = data.get('preprocessor', None)
                result['feature_names'] = feature_names
                
                # Store results
                results[model_name] = result
                
                # Log metrics to MLflow
                if 'metrics' in result:
                    for metric_name, metric_value in result['metrics'].items():
                        if isinstance(metric_value, (int, float)):
                            mlflow.log_metric(metric_name, metric_value)
                
                # Log feature importance if available
                try:
                    if 'feature_importance' in result and result['feature_importance']:
                        # Create feature importance plot
                        fi = result['feature_importance']
                        plt.figure(figsize=(10, 6))
                        pd.Series(fi).sort_values(ascending=False).head(20).plot(kind='bar')
                        plt.title(f'Feature Importance - {model_name}')
                        plt.tight_layout()
                        
                        # Save and log the plot directly under model name
                        plot_path = f'/tmp/feature_importance_{model_name}.png'
                        plt.savefig(plot_path)
                        mlflow.log_artifact(plot_path, artifact_path=f'models/{model_name}')
                        
                        # Clean up temporary file
                        os.remove(plot_path)
                except Exception as e:
                    logger.error(f"Error logging feature importance for {model_name}: {str(e)}")
                    mlflow.log_param("feature_importance_error", str(e))
                
                # Log the model under the model name directly
                mlflow.sklearn.log_model(
                    sk_model=result['model'],
                    artifact_path=model_name,  # This will be under the run's artifacts
                    registered_model_name=f'churn_prediction_{model_name}'
                )
                
                # Log confusion matrix under model name
                if 'confusion_matrix' in result:
                    cm = result['confusion_matrix']
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f'Confusion Matrix - {model_name}')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    cm_path = f'/tmp/confusion_matrix_{model_name}.png'
                    plt.savefig(cm_path)
                    mlflow.log_artifact(cm_path, model_name)
                    plt.close()
                
                logger.info(f"Successfully trained and logged {model_name} to MLflow")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Log the error to MLflow
                mlflow.log_param('error', str(e))
                mlflow.set_tag('status', 'failed')
                
                # Re-raise the exception to fail the task
                raise
            finally:
                # End the MLflow run
                if 'mlflow' in locals() and mlflow.active_run():
                    mlflow.end_run()
    
    return results
