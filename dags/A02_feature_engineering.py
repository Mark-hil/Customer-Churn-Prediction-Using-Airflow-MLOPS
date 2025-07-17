"""Feature engineering module for the churn prediction pipeline."""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("/opt/airflow/data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

class FeatureEngineer:
    """Class for feature engineering and preprocessing."""
    
    def __init__(self, target_col: str = 'Churn', test_size: float = 0.2, 
                 val_size: float = 0.1, random_state: int = 42, context=None):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            target_col: Name of the target column
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility
            context: Airflow context dictionary (optional)
        """
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names = None
        self.context = context
        
    def preprocess_data(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the input data.
        
        Args:
            df: Input DataFrame with raw data
            
        Returns:
            Tuple of (X, y) where X is the feature matrix and y is the target
        """
        logger.info("Starting data preprocessing...")
        
        # Make a copy of the input data
        df = df.copy()
        
        # Convert TotalCharges to numeric, coerce errors to NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Handle missing values (if any)
        df = self._handle_missing_values(df)
        
        # Separate features and target
        X = df.drop(columns=[self.target_col, 'customerID'])
        y = df[self.target_col].map({'Yes': 1, 'No': 0})
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        logger.info(f"Categorical columns: {categorical_cols}")
        logger.info(f"Numerical columns: {numerical_cols}")
        
        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Fit and transform the data while maintaining DataFrame structure
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names after one-hot encoding
        feature_names = self._get_feature_names_after_preprocessing(categorical_cols, numerical_cols)
        
        # Store feature names for later use
        self.feature_names = feature_names
        
        # Store feature names and preprocessor in context if available
        if self.context:
            ti = self.context['ti']
            ti.xcom_push(key='feature_names', value=feature_names)
            # Store preprocessor as a joblib file
            preprocessor_path = Path('/opt/airflow/data/processed/preprocessor.joblib')
            joblib.dump(self.preprocessor, preprocessor_path)
            ti.xcom_push(key='preprocessor', value=str(preprocessor_path))
            ti.xcom_push(key='categorical_columns', value=categorical_cols)
            ti.xcom_push(key='numerical_columns', value=numerical_cols)
        
        # Convert back to DataFrame with proper column names
        X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        
        # Store the original column information for reference
        self.original_columns = {
            'categorical': categorical_cols,
            'numerical': numerical_cols,
            'all': X.columns.tolist()
        }
        
        # Store feature names and preprocessor in context if available
        if 'context' in kwargs and kwargs['context']:
            ti = kwargs['context']['ti']
            ti.xcom_push(key='feature_names', value=feature_names)
            ti.xcom_push(key='preprocessor', value=self.preprocessor)
            ti.xcom_push(key='categorical_columns', value=categorical_cols)
            ti.xcom_push(key='numerical_columns', value=numerical_cols)
        
        logger.info(f"Preprocessing completed. Shape after preprocessing: {X_processed.shape}")
        logger.debug(f"Processed feature names: {feature_names}")
        logger.debug(f"Original columns - Categorical: {categorical_cols}")
        logger.debug(f"Original columns - Numerical: {numerical_cols}")
        
        return X_processed, y, feature_names
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"Found {missing.sum()} missing values in the dataset")
            logger.warning(f"Missing values by column:\n{missing[missing > 0]}")
            
            # Handle missing values (example: fill with median for numerical, mode for categorical)
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def _get_feature_names_after_preprocessing(self, categorical_cols: List[str], numerical_cols: List[str]) -> List[str]:
        """
        Get feature names after one-hot encoding.
        
        Args:
            categorical_cols: List of categorical column names
            numerical_cols: List of numerical column names
            
        Returns:
            List of feature names after preprocessing
        """
        # Get numerical feature names (these remain the same)
        feature_names = numerical_cols.copy()
        
        # Get one-hot encoded feature names
        categorical_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        categorical_feature_names = categorical_encoder.get_feature_names_out(categorical_cols)
        
        # Combine all feature names
        all_feature_names = np.concatenate([feature_names, categorical_feature_names])
        logger.info(f"Total number of features after preprocessing: {len(all_feature_names)}")
        
        return all_feature_names.tolist()
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary containing the split datasets
        """
        logger.info("Splitting data into train, validation, and test sets...")
        
        # First split: separate out the test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )
        
        # Second split: split train into train and validation
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size_adjusted,
            stratify=y_train_val,
            random_state=self.random_state
        )
        
        logger.info(f"Data split complete. Shapes: "
                   f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        
        # Ensure we're using the feature names from preprocessing if available
        feature_names = self.feature_names if hasattr(self, 'feature_names') else X.columns.tolist()
        
        return {
            'X_train': X_train, 
            'y_train': y_train,
            'X_val': X_val, 
            'y_val': y_val,
            'X_test': X_test, 
            'y_test': y_test,
            'feature_names': feature_names,
            'preprocessor': self.preprocessor
        }

def preprocess_data(df: pd.DataFrame, context=None, **kwargs) -> Dict[str, Any]:
    """
    Preprocess the input data and return the processed features and target.
    
    Args:
        df: Input DataFrame with raw data
        **kwargs: Additional arguments to pass to FeatureEngineer
        
    Returns:
        Dictionary containing processed data and metadata
    """
    # Initialize the feature engineer with context
    engineer = FeatureEngineer(context=context, **kwargs)
    
    # Preprocess the data
    X_processed, y, feature_names = engineer.preprocess_data(df)
    
    # Split the data
    data_splits = engineer.split_data(X_processed, y)
    
    # Add preprocessing information to the results
    data_splits.update({
        'categorical_columns': [col for col in df.select_dtypes(include=['object', 'category']).columns if col != engineer.target_col],
        'numerical_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    })
    
    return data_splits

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
              val_size: float = 0.1, random_state: int = 42, **context) -> Dict[str, Any]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        **context: Airflow context (unused, for compatibility)
        
    Returns:
        Dictionary containing the split datasets and feature names
    """
    from sklearn.model_selection import train_test_split
    
    logger.info(f"Splitting data with test_size={test_size}, val_size={val_size}")
    
    # First split: separate out the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # Second split: split train into train and validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size_adjusted,
        stratify=y_train_val,
        random_state=random_state
    )
    
    logger.info(f"Data split complete. Shapes: "
              f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }

# Example usage
if __name__ == "__main__":
    # This is just for testing the module
    from A00_data_understanding import load_data
    
    # Load sample data
    logger.info("Loading sample data...")
    df = load_data()
    
    # Preprocess the data
    logger.info("Preprocessing data...")
    results = preprocess_data(df)
    
    # Print some information
    logger.info(f"Training set shape: {results['X_train'].shape}")
    logger.info(f"Number of features: {len(results['feature_names'])}")
    logger.info("Feature engineering test completed successfully!")
