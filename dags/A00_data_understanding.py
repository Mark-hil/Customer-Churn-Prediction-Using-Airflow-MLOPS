"""Data understanding module for the churn prediction pipeline."""
import logging
from pathlib import Path
import pandas as pd
from typing import Tuple, Dict, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("/opt/airflow/data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
PLOTS_DIR = DATA_DIR / "plots"
RAW_DATA_PATH = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def ensure_directories() -> None:
    """Ensure all required directories exist."""
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR, PLOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def load_data(file_path: Path = None) -> pd.DataFrame:
    """
    Load data from the specified path or default location.
    
    Args:
        file_path: Optional path to the data file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    file_path = file_path or RAW_DATA_PATH
    logger.info(f"Loading data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def validate_data(df: pd.DataFrame, **context) -> Dict[str, Any]:
    """
    Validate the input data for required columns and basic quality checks.
    
    Args:
        df: Input DataFrame to validate
        
    Returns:
        dict: Dictionary containing dataset summary
    """
    logger.info("Validating data...")
    
    # Check for required columns
    expected_cols = {
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
    }
    
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check for missing values
    missing_values = df.isna().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Found missing values:\n{missing_values[missing_values > 0]}")
    
    # Check target variable distribution
    if 'Churn' in df.columns:
        churn_dist = df['Churn'].value_counts(normalize=True) * 100
        logger.info(f"Churn distribution (\%):\n{churn_dist}")
    
    logger.info("Data validation completed")
    
    summary = {
        'num_samples': len(df),
        'num_features': len(df.columns),
        'num_numerical': len(df.select_dtypes(include=['int64', 'float64']).columns),
        'num_categorical': len(df.select_dtypes(include=['object', 'category', 'bool']).columns),
        'missing_values': df.isna().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if 'Churn' in df.columns:
        summary['churn_rate'] = (df['Churn'] == 'Yes').mean() * 100
    
    return summary

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict: Dictionary containing dataset summary
    """
    summary = {
        'num_samples': len(df),
        'num_features': len(df.columns),
        'num_numerical': len(df.select_dtypes(include=['int64', 'float64']).columns),
        'num_categorical': len(df.select_dtypes(include=['object', 'category', 'bool']).columns),
        'missing_values': df.isna().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if 'Churn' in df.columns:
        summary['churn_rate'] = (df['Churn'] == 'Yes').mean() * 100
    
    return summary

def load_and_validate_data(**context) -> pd.DataFrame:
    """
    Load and validate the data.
    
    Args:
        context: Airflow context dictionary
        
    Returns:
        pd.DataFrame: The loaded and validated DataFrame
    """
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Load the data
        df = load_data()
        
        # Validate the data
        is_valid = validate_data(df, **context)
        
        if not is_valid:
            raise ValueError("Data validation failed")
            
        # Get and log data summary
        summary = get_data_summary(df)
        logger.info("Data summary:")
        for key, value in summary.items():
            logger.info(f"{key}: {value}")
            
            # Store summary in XCom if running in Airflow
            if context and 'ti' in context:
                context['ti'].xcom_push(key=key, value=value)
        
        return df
        
    except Exception as e:
        logger.error(f"Error in load_and_validate_data: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    df = load_and_validate_data()
    print("Data loaded and validated successfully")
