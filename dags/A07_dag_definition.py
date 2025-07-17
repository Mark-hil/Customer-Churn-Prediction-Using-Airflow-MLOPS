"""
Airflow DAG definition for the churn prediction pipeline.

This DAG orchestrates the entire churn prediction workflow, including:
1. Data loading and validation
2. Exploratory data analysis
3. Feature engineering and preprocessing
4. Model training and hyperparameter tuning
5. Model evaluation and selection
6. Model persistence
7. Cleanup of temporary files
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import os
import sys

# Add the dags directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago

# Import functions from our modules
from A00_data_understanding import load_and_validate_data
from A01_exploratory_data_analysis import run_eda, generate_eda_report
from A02_feature_engineering import preprocess_data, FeatureEngineer
from A03_experimentation import train_models as train_models_func
from A04_model_selection import evaluate_models as evaluate_models_func
from A05_utility import cleanup_old_files
from A06_persistence import ModelPersistence

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": days_ago(1),
}

# Define the DAG
dag = DAG(
    dag_id="churn_prediction_pipeline",
    default_args=default_args,
    description="End-to-end pipeline for churn prediction",
    schedule_interval="@weekly",  # Run weekly, adjust as needed
    catchup=False,
    tags=["mlops", "churn-prediction"],
)

def load_and_validate_data_with_context(**context):
    """Wrapper function to load and validate data with Airflow context."""
    # Import here to avoid circular imports
    from A00_data_understanding import load_and_validate_data as _load_and_validate_data
    
    # Call the actual function
    df = _load_and_validate_data(**context)
    
    # Push the dataframe to XCom for downstream tasks
    context['ti'].xcom_push(key='raw_data', value=df)
    
    return "Data loaded and validated successfully"

# Task 1: Load and Validate Data
task_load_validate_data = PythonOperator(
    task_id="load_validate_data",
    python_callable=load_and_validate_data_with_context,
    provide_context=True,
    dag=dag,
)

def run_eda_with_context(**context):
    """Wrapper function to run EDA with Airflow context."""
    # Import here to avoid circular imports
    from A01_exploratory_data_analysis import run_eda as _run_eda
    
    # Get the data from the previous task
    ti = context['ti']
    df = ti.xcom_pull(task_ids='load_validate_data', key='raw_data')
    
    if df is None:
        raise ValueError("No data found for EDA")
    
    # Run EDA and generate reports
    report = _run_eda(df)
    
    # Push report paths to XCom if needed
    if report and isinstance(report, dict):
        for name, path in report.items():
            ti.xcom_push(key=f'eda_{name}', value=str(path))
    
    return "EDA completed successfully"

# Task 3: Run EDA
task_run_eda = PythonOperator(
    task_id="run_eda",
    python_callable=run_eda_with_context,
    provide_context=True,
    dag=dag,
)

def preprocess_with_context(**context):
    """Wrapper function to preprocess data with Airflow context."""
    # Import here to avoid circular imports
    from A02_feature_engineering import preprocess_data as _preprocess_data
    
    ti = context['ti']
    
    # Get the data from the previous task
    df = ti.xcom_pull(task_ids='load_validate_data', key='raw_data')
    
    if df is None:
        raise ValueError("No data found for preprocessing")
    
    # Preprocess the data (this includes splitting)
    result = _preprocess_data(
        df,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        context=context
    )
    
    # Store column information for later use
    ti.xcom_push(key='categorical_columns', value=result['categorical_columns'])
    ti.xcom_push(key='numerical_columns', value=result['numerical_columns'])
    
    # Push the results to XCom
    ti.xcom_push(key='X_train', value=result['X_train'])
    ti.xcom_push(key='y_train', value=result['y_train'])
    ti.xcom_push(key='X_val', value=result['X_val'])
    ti.xcom_push(key='y_val', value=result['y_val'])
    ti.xcom_push(key='X_test', value=result['X_test'])
    ti.xcom_push(key='y_test', value=result['y_test'])
    
    return "Data preprocessed and split successfully"

# Task 4: Preprocess Data
task_preprocess_data = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_with_context,
    provide_context=True,
    dag=dag,
)

def train_models_with_context(**context):
    """Wrapper function to train models with data from XCom."""
    # Import here to avoid circular imports
    from A03_experimentation import train_models as _train_models
    import logging
    
    logger = logging.getLogger(__name__)
    ti = context['ti']
    
    # Get data from XCom
    X_train = ti.xcom_pull(task_ids='preprocess_data', key='X_train')
    y_train = ti.xcom_pull(task_ids='preprocess_data', key='y_train')
    X_val = ti.xcom_pull(task_ids='preprocess_data', key='X_val')
    y_val = ti.xcom_pull(task_ids='preprocess_data', key='y_val')
    categorical_cols = ti.xcom_pull(task_ids='preprocess_data', key='categorical_columns')
    numerical_cols = ti.xcom_pull(task_ids='preprocess_data', key='numerical_columns')
    
    # Log the shapes of the data for debugging
    logger.info(f"X_train shape: {X_train.shape if X_train is not None else 'None'}")
    logger.info(f"y_train shape: {y_train.shape if y_train is not None else 'None'}")
    logger.info(f"X_val shape: {X_val.shape if X_val is not None else 'None'}")
    logger.info(f"y_val shape: {y_val.shape if y_val is not None else 'None'}")
    
    if any(v is None for v in [X_train, y_train, X_val, y_val, categorical_cols, numerical_cols]):
        missing = []
        if X_train is None: missing.append('X_train')
        if y_train is None: missing.append('y_train')
        if X_val is None: missing.append('X_val')
        if y_val is None: missing.append('y_val')
        if categorical_cols is None: missing.append('categorical_columns')
        if numerical_cols is None: missing.append('numerical_columns')
        raise ValueError(f"Missing required data for model training: {', '.join(missing)}")
    
    # Define the models to train
    models_to_train = [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "hist_gradient_boosting"
    ]
    
    # Prepare the data dictionary
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'categorical_columns': categorical_cols,
        'numerical_columns': numerical_cols,
        'feature_names': list(X_train.columns) if hasattr(X_train, 'columns') else []
    }
    
    logger.info(f"Starting model training with models: {models_to_train}")
    
    # Train the models
    results = _train_models(
        data=data,
        models_to_train=models_to_train,
        **context  # Pass the context to the training function
    )
    
    if not results:
        raise ValueError("No models were successfully trained")
    
    logger.info(f"Successfully trained {len(results)} models")
    
    # Push results to XCom for evaluation and persistence
    for model_name, model_info in results.items():
        if model_info and 'model' in model_info:
            ti.xcom_push(key=f'model_{model_name}', value=model_info)
            logger.info(f"Pushed model {model_name} to XCom")
        else:
            logger.error(f"Skipping invalid model info for {model_name}")
    
    # Also push the complete results dictionary for reference
    ti.xcom_push(key='all_models', value=list(results.keys()))
    
    return results
    
    return f"Trained {len(models_to_train)} models successfully"

# Task 5: Train Models
task_train_models = PythonOperator(
    task_id="train_models",
    python_callable=train_models_with_context,
    provide_context=True,
    dag=dag,
)

def evaluate_models_with_context(**context):
    """Wrapper function to evaluate models with data from XCom."""
    # Import here to avoid circular imports
    from A04_model_selection import evaluate_models as _evaluate_models
    from pathlib import Path
    
    ti = context['ti']
    
    # Get data from XCom
    X_test = ti.xcom_pull(task_ids='preprocess_data', key='X_test')
    y_test = ti.xcom_pull(task_ids='preprocess_data', key='y_test')
    
    if X_test is None or y_test is None:
        raise ValueError("Test data not found in XCom")
    
    # Define all models to evaluate
    all_models = [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "hist_gradient_boosting"
    ]
    
    # Get trained models from XCom
    models_info = {}
    for model_name in all_models:
        model_info = ti.xcom_pull(task_ids='train_models', key=f'model_{model_name}')
        if model_info:
            models_info[model_name] = model_info
            logger.info(f"Found model: {model_name}")
        else:
            logger.warning(f"Model {model_name} not found in XCom")
    
    if not models_info:
        raise ValueError("No trained models found in XCom")
    
    # Create output directory if it doesn't exist
    output_dir = Path("/opt/airflow/data/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate models
    evaluation_results = _evaluate_models(
        models=models_info,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir
    )
    
    # Define weights for each metric (sum should be 1.0)
    metric_weights = {
        'roc_auc': 0.4,      # Most important metric
        'accuracy': 0.15,    # Less important than AUC
        'precision': 0.15,   # Balance between precision and recall
        'recall': 0.15,      # Balance between precision and recall
        'f1_score': 0.15     # Harmonic mean of precision and recall
    }
    
    def calculate_composite_score(metrics):
        """Calculate a composite score based on multiple metrics and their weights."""
        score = 0.0
        for metric, weight in metric_weights.items():
            # Normalize metrics to be in similar ranges (0-1)
            metric_value = metrics.get(metric, 0)
            
            # For metrics that might be outside 0-1 range (like log loss), clip them
            if metric == 'log_loss':
                # Lower is better for log loss, so we invert it
                # Assuming log loss is typically between 0 and 10 for most models
                metric_value = 1.0 - min(metric_value / 10.0, 1.0)
            
            score += metric_value * weight
        return score
    
    # Sort all models by their composite score in descending order
    sorted_models = sorted(
        evaluation_results.items(),
        key=lambda x: calculate_composite_score(x[1]),
        reverse=True
    )
    
    # Get the top 2 models
    best_model_name = sorted_models[0][0]
    second_best_model_name = sorted_models[1][0] if len(sorted_models) > 1 else None
    
    # Log the decision metrics
    logger.info("\nModel Evaluation Summary:")
    logger.info("-" * 50)
    for model_name, metrics in evaluation_results.items():
        score = calculate_composite_score(metrics)
        status = " (BEST)" if model_name == best_model_name else " (SECOND BEST)" if model_name == second_best_model_name else ""
        logger.info(f"{model_name}{status} - Composite Score: {score:.4f}")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    logger.info("-" * 50)
    
    # Push results to XCom
    ti.xcom_push(key='evaluation_results', value=evaluation_results)
    ti.xcom_push(key='best_model_name', value=best_model_name)
    ti.xcom_push(key='second_best_model_name', value=second_best_model_name)
    
    logger.info(f"Evaluation complete. Best model: {best_model_name}")
    if second_best_model_name:
        logger.info(f"Second best model: {second_best_model_name}")
    
    return f"Evaluation complete. Best models: 1) {best_model_name}, 2) {second_best_model_name if second_best_model_name else 'N/A'}"

# Task 6: Evaluate Models
task_evaluate_models = PythonOperator(
    task_id="evaluate_models",
    python_callable=evaluate_models_with_context,
    provide_context=True,
    dag=dag,
)

# Task 8: Persist Best Model is defined later in the file

def cleanup_old_artifacts(**context) -> None:
    """
    Clean up old model artifacts and evaluation results.
    
    This task performs the following cleanup operations:
    - Keeps only the most recent model directories
    - Removes old evaluation reports
    - Cleans up plot files
    - Removes old processed data files
    """
    # Import here to avoid circular imports
    from A05_utility import cleanup_old_files as _cleanup_old_files, cleanup_old_directories as _cleanup_old_dirs
    from pathlib import Path
    
    logger.info("Starting cleanup process...")
    
    # Define directories to clean up
    data_dir = Path("/opt/airflow/data")
    model_dir = data_dir / "models"
    processed_dir = data_dir / "processed"
    plots_dir = data_dir / "plots"
    evaluation_dir = data_dir / "evaluation"
    
    # Ensure directories exist
    for directory in [model_dir, processed_dir, plots_dir, evaluation_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    try:
        # Clean up old model directories (keep the 2 most recent of each type)
        _cleanup_old_dirs(
            directory=str(model_dir),
            pattern="*_*_*_*",  # Matches timestamped directories
            max_dirs_per_type=2  # Keep 2 most recent of each model type (best and second_best)
        )
        logger.info("Cleaned up old model directories, keeping 2 most recent of each type")
        
        # Clean up old evaluation reports (keep the 5 most recent)
        _cleanup_old_files(
            directory=str(evaluation_dir),
            pattern="*_report.*",
            max_files_to_keep=5
        )
        logger.info("Cleaned up old evaluation reports")
        
        # Clean up old plot files (keep the 5 most recent)
        _cleanup_old_files(
            directory=str(plots_dir),
            pattern="*.png",
            max_files_to_keep=5
        )
        logger.info("Cleaned up old plot files")
        
        # Clean up processed data files (keep the 2 most recent)
        _cleanup_old_files(
            directory=str(processed_dir),
            pattern="*.joblib",
            max_files_to_keep=2
        )
        logger.info("Cleaned up old processed data files")
        
        logger.info("Cleanup completed successfully")
        return "Cleanup completed successfully"
        
    except Exception as e:
        error_msg = f"Error during cleanup: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e

# Task 9: Cleanup
task_cleanup = PythonOperator(
    task_id="cleanup_old_data",
    python_callable=cleanup_old_artifacts,
    provide_context=True,
    dag=dag,
)

def persist_best_model_with_context(**context) -> str:
    """
    Persist the best and second-best models based on evaluation metrics.
    
    Saves both models with appropriate naming and metadata, including:
    - Model weights
    - Evaluation metrics
    - Hyperparameters
    - Feature importances
    - Preprocessing pipeline
    """
    # Import here to avoid circular imports
    from A06_persistence import ModelPersistence
    from pathlib import Path
    import joblib
    import logging
    import traceback
    import pandas as pd
    import numpy as np
    import joblib
    
    ti = context['ti']
    
    # Get evaluation results and model names from XCom
    evaluation_results = ti.xcom_pull(task_ids='evaluate_models', key='evaluation_results')
    best_model_name = ti.xcom_pull(task_ids='evaluate_models', key='best_model_name')
    second_best_model_name = ti.xcom_pull(task_ids='evaluate_models', key='second_best_model_name')
    
    # Log all available XCom keys for debugging
    all_xcom = ti.xcom_pull(task_ids='train_models')
    logger.info(f"Available XCom keys from train_models: {list(all_xcom.keys()) if all_xcom else 'None'}")
    logger.info(f"Evaluation results: {evaluation_results}")
    logger.info(f"Best model name: {best_model_name}")
    logger.info(f"Second best model name: {second_best_model_name}")
    
    if not evaluation_results:
        raise ValueError("No evaluation results found in XCom")
    if not best_model_name:
        raise ValueError("No best model name found in XCom")
    
    # Get the preprocessor path from XCom and load it
    preprocessor_path = ti.xcom_pull(task_ids='preprocess_data', key='preprocessor')
    if preprocessor_path:
        preprocessor = joblib.load(preprocessor_path)
    else:
        preprocessor = None
        
    # Initialize feature names list
    feature_names = []
    
    # Create a dictionary to store data needed for feature importance extraction
    data = {
        'feature_names': feature_names,
        'preprocessor': preprocessor
    }
    
    # Initialize model persistence with S3 configuration
    persistence = ModelPersistence(
        use_s3=True,
        s3_bucket="capstone-churn-prediction-models"
    )
    
    saved_models = {}
    
    # Process both best and second best models
    for rank, (model_name, model_type) in enumerate([
        (best_model_name, 'best'),
        (second_best_model_name, 'second_best') if second_best_model_name else (None, None)
    ]):
        if not model_name or model_type is None:
            continue
            
        logger.info(f"\nPersisting {model_type} model: {model_name}")
        
        # Get the model from XCom using the correct key format
        model_key = f'model_{model_name}'
        model_info = ti.xcom_pull(task_ids='train_models', key=model_key)
        
        if not model_info:
            logger.error(f"Model info not found in XCom for key: {model_key}")
            logger.error(f"Available XCom keys: {list(ti.xcom_pull(task_ids='train_models').keys()) if ti.xcom_pull(task_ids='train_models') else 'None'}")
            continue
            
        if 'model' not in model_info:
            logger.error(f"Model info for {model_name} is missing the 'model' key. Available keys: {list(model_info.keys())}")
            continue
            
        logger.info(f"Successfully retrieved model info for {model_name}")
            
        # Get the evaluation metrics for this model
        model_metrics = evaluation_results.get(model_name, {})
        
        # Generate model initials (first letter of each word in model name, uppercase)
        model_initials = ''.join([word[0].upper() for word in model_name.split('_')])
        model_name_with_initials = f"{model_name}_{model_initials}_{model_type}"
        
        try:
            # Get model from model_info
            model = model_info['model']
            
            # Extract model parameters if not already in model_info
            model_params = model_info.get('params', {})
            if not model_params and hasattr(model, 'get_params'):
                model_params = model.get_params(deep=True)
            
            # Get feature names from the model if it's a pipeline
            if (data['feature_names'] is None or len(data['feature_names']) == 0) and hasattr(model, 'named_steps'):
                try:
                    # This handles the case where the model is a pipeline with a preprocessor
                    for step in model.named_steps.values():
                        if hasattr(step, 'get_feature_names_out'):
                            data['feature_names'] = step.get_feature_names_out()
                            break
                except Exception as e:
                    logger.warning(f"Could not get feature names from model steps: {str(e)}")
            
            # If we still don't have feature names, try to get them from the preprocessor
            if (data['feature_names'] is None or len(data['feature_names']) == 0) and hasattr(preprocessor, 'get_feature_names_out'):
                try:
                    data['feature_names'] = preprocessor.get_feature_names_out()
                except Exception as e:
                    logger.warning(f"Could not get feature names from preprocessor: {str(e)}")
            
            # Extract feature importances if available
            feature_importance = {}
            if 'feature_importance' in model_info and model_info['feature_importance'] is not None:
                # Ensure feature_importance is a dictionary with serializable values
                if hasattr(model_info['feature_importance'], 'items'):
                    feature_importance = {str(k): float(v) for k, v in model_info['feature_importance'].items()}
                elif hasattr(model_info['feature_importance'], 'tolist'):
                    # Handle case where feature_importance is a numpy array
                    if data['feature_names'] is not None:
                        feature_importance = dict(zip(
                            data['feature_names'].tolist() if hasattr(data['feature_names'], 'tolist') else list(data['feature_names']),
                            model_info['feature_importance'].tolist()
                        ))
            
            if not feature_importance and data['feature_names']:
                try:
                    # Handle tree-based models with feature_importances_
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(
                            data['feature_names'], 
                            model.feature_importances_
                        ))
                    # Handle Logistic Regression with coef_
                    elif hasattr(model, 'coef_'):
                        # For binary classification, take absolute values of coefficients
                        if len(model.coef_.shape) == 1:
                            importance_values = np.abs(model.coef_[0])
                        else:
                            # For multi-class, take mean across classes
                            importance_values = np.mean(np.abs(model.coef_), axis=0)
                        
                        feature_importance = dict(zip(
                            data['feature_names'],
                            importance_values
                        ))
                        
                        # Normalize to sum to 1 for consistency with tree-based models
                        if feature_importance:
                            total = sum(feature_importance.values())
                            if total > 0:
                                feature_importance = {k: v/total for k, v in feature_importance.items()}
                    
                except Exception as e:
                    logger.warning(f"Could not extract feature importances: {str(e)}")
                    feature_importance = {}
            
            # Prepare feature names for serialization
            feature_names = data['feature_names']
            if feature_names is not None:
                if hasattr(feature_names, 'tolist'):
                    feature_names = feature_names.tolist()
                elif not isinstance(feature_names, list):
                    feature_names = list(feature_names)
            else:
                feature_names = []
                
            # Prepare artifacts to save - only include JSON-serializable data
            artifacts = {
                'feature_names': feature_names
            }
            
            # Save the model with metadata and artifacts
            try:
                # Create a timestamp for this model version
                timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
                model_dir_name = f"{model_name_with_initials}_{timestamp}"
                model_dir = Path(persistence.models_dir) / model_dir_name
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Save the preprocessor first
                preprocessor_path = None
                if preprocessor is not None:
                    preprocessor_path = model_dir / "preprocessor.joblib"
                    joblib.dump(preprocessor, preprocessor_path)
                    logger.info(f"Saved preprocessor to {preprocessor_path}")
                
                # Save the model with other metadata
                saved_paths = persistence.save_model(
                    model=model_info['model'],
                    model_name=model_name_with_initials,
                    metrics={
                        'accuracy': float(model_metrics.get('accuracy', 0)),
                        'precision': float(model_metrics.get('precision', 0)),
                        'recall': float(model_metrics.get('recall', 0)),
                        'f1': float(model_metrics.get('f1', 0)),
                        'roc_auc': float(model_metrics.get('roc_auc', 0))
                    },
                    params=model_params,
                    feature_importance=feature_importance,
                    artifacts=artifacts,
                    registered_model_name=f"churn_prediction_{model_initials}_{model_type}"
                )
                
                # If preprocessor was saved, add its path to saved_paths
                if preprocessor_path is not None:
                    saved_paths['preprocessor'] = str(preprocessor_path)
                    
                logger.info(f"Successfully saved model to {model_dir}")
                    
            except Exception as e:
                error_msg = f"Error in model persistence for {model_name}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                raise Exception(error_msg) from e
            
            if not saved_paths:
                logger.warning(f"No paths returned when saving {model_type} model: {model_name}")
                continue
                
            logger.info(f"{model_type.capitalize()} model persisted. Artifacts: {saved_paths}")
            saved_models[model_type] = {
                'model_name': model_name,
                'paths': saved_paths,
                'metrics': model_metrics
            }
            
        except Exception as e:
            logger.error(f"Error saving {model_type} model {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
    
    if not saved_models:
        raise ValueError("Failed to persist any models")
        
    # Push the saved paths to XCom for downstream tasks
    ti.xcom_push(key='saved_models', value=saved_models)
    
    try:
        # Return a summary of persisted models
        model_summary = ", ".join(
            f"{model_type}: {info['model_name']}" 
            for model_type, info in saved_models.items()
        )
        return f"Successfully persisted models - {model_summary}"
    except Exception as e:
        error_msg = f"Error persisting model: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise Exception(error_msg) from e

# Task 8: Persist Best Model
task_persist_model = PythonOperator(
    task_id="persist_best_model",
    python_callable=persist_best_model_with_context,
    provide_context=True,
    dag=dag,
)

# Define task dependencies
task_load_validate_data >> task_run_eda >> task_preprocess_data
task_preprocess_data >> task_train_models
task_train_models >> task_evaluate_models >> task_persist_model >> task_cleanup

# Add a success notification (placeholder)
success_notification = DummyOperator(
    task_id="success_notification",
    trigger_rule="all_success",
    dag=dag,
)

# Add error handling (placeholder)
error_handling = DummyOperator(
    task_id="error_handling",
    trigger_rule="one_failed",
    dag=dag,
)

# Main pipeline flow
task_load_validate_data >> task_run_eda >> task_preprocess_data >> task_train_models >> task_evaluate_models >> task_persist_model >> task_cleanup >> success_notification

# Error handling for all tasks
[task_load_validate_data, task_run_eda, task_preprocess_data, 
 task_train_models, task_evaluate_models, task_persist_model, task_cleanup] >> error_handling

# Example of how to add conditional logic if needed
def check_data_quality(**context) -> str:
    """Check data quality and decide next step."""
    ti = context['ti']
    # Get data quality metrics from XCom or calculate them
    # For example, check number of missing values, data distribution, etc.
    # This is a placeholder - implement your own data quality checks
    data_quality_ok = True  # Replace with actual check
    
    if data_quality_ok:
        return "run_eda"
    else:
        return "alert_data_quality_issue"

# Add a branch to check data quality
branch_task = BranchPythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    provide_context=True,
    dag=dag,
)

# Add a task for data quality alerts
alert_data_quality_issue = DummyOperator(
    task_id='alert_data_quality_issue',
    dag=dag,
)

# Update task dependencies to include data quality check
task_load_validate_data >> branch_task
branch_task >> [task_run_eda, alert_data_quality_issue]

# Add documentation
dag.doc_md = """
# Churn Prediction Pipeline

This DAG implements an end-to-end machine learning pipeline for churn prediction.

## Pipeline Steps:
1. **Load Data**: Load the raw customer data
2. **Validate Data**: Perform data quality checks
3. **Exploratory Data Analysis**: Generate insights and visualizations
4. **Preprocess Data**: Clean and transform the data
5. **Split Data**: Split into train, validation, and test sets
6. **Train Models**: Train multiple ML models
7. **Evaluate Models**: Compare model performance
8. **Persist Best Model**: Save the best performing model
9. **Cleanup**: Remove temporary files and old models

## Configuration:
- Models: Random Forest, XGBoost, Logistic Regression
- Validation: 5-fold cross-validation
- Primary Metric: ROC-AUC

## Schedule:
- Runs weekly
- Retries: 1 (with 5 min delay)
"""

# Add task documentation
task_load_validate_data.doc_md = """
### Load and Validate Data
Loads the raw customer churn dataset from the data directory and performs data quality checks and validation.
"""

# Continue with other task documentation...

if __name__ == "__main__":
    # This allows testing the DAG by running it directly
    from airflow.utils.state import State
    
    # Create a test DAG run
    dag.clear(dag_run_state=State.NONE)
    dag.run(local=True)
