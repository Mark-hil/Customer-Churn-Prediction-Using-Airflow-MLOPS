"""Utility functions for the churn prediction pipeline."""
import os
import json
import base64
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("/opt/airflow/data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
PLOTS_DIR = DATA_DIR / "plots"

# Ensure all required directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def ensure_directory(directory: Union[str, Path], mode: int = 0o755) -> None:
    """
    Safely create a directory with appropriate permissions.
    
    Args:
        directory: Path to the directory to create
        mode: Permissions mode (default: 0o755)
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True, mode=mode)
            logger.info(f"Created directory: {directory}")
        
        # Ensure directory is writable
        test_file = directory / ".test"
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            logger.error(f"Directory {directory} is not writable: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Failed to ensure directory {directory}: {e}")
        raise

def cleanup_old_files(
    directory: Union[str, Path], 
    pattern: str, 
    max_files_to_keep: int = 2,
    recursive: bool = False
) -> None:
    """
    Clean up old files in the specified directory matching the given pattern.
    Keeps the most recent max_files_to_keep files.
    
    Args:
        directory: Directory to search for files
        pattern: File pattern to match (e.g., "*.joblib")
        max_files_to_keep: Number of most recent files to keep
        recursive: If True, search in subdirectories as well
    """
    directory = Path(directory)
    if not directory.exists():
        logger.info(f"Directory {directory} does not exist, nothing to clean up")
        return
        
    # Get all files matching the pattern
    try:
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
    except Exception as e:
        logger.error(f"Error finding files with pattern {pattern} in {directory}: {str(e)}")
        return
    
    if not files:
        logger.info(f"No files matching {pattern} found in {directory}")
        return
        
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    
    # Log what we found
    logger.info(f"Found {len(files)} files matching {pattern} in {directory}")
    for i, f in enumerate(files[:max_files_to_keep], 1):
        logger.info(f"  Keeping ({i}/{max_files_to_keep}): {f.relative_to(directory)} (modified: {os.path.getmtime(f):.0f})")
    
    # Remove old files
    removed_count = 0
    for old_file in files[max_files_to_keep:]:
        try:
            if old_file.is_file():
                file_size = old_file.stat().st_size / (1024 * 1024)  # Size in MB
                old_file.unlink()
                logger.info(f"Removed old file: {old_file.relative_to(directory)} (size: {file_size:.2f}MB)")
                removed_count += 1
        except Exception as e:
            logger.warning(f"Failed to remove {old_file.relative_to(directory)}: {e}")
            
    logger.info(f"Cleanup complete: Kept {min(len(files), max_files_to_keep)} files, removed {removed_count} files")

def cleanup_old_directories(
    directory: Union[str, Path],
    pattern: str,
    max_dirs_per_type: int = 2
) -> None:
    """
    Clean up old model directories, keeping the most recent max_dirs_per_type
    directories for each model type (best and second best).
    
    Args:
        directory: Directory to search in
        pattern: Pattern to match directory names (supports glob)
        max_dirs_per_type: Number of most recent directories to keep per model type
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            logger.info(f"Directory {directory} does not exist, nothing to clean up")
            return
            
        # Get all model directories
        all_dirs = [d for d in directory.glob(pattern) if d.is_dir()]
        
        if not all_dirs:
            logger.info(f"No model directories found in {directory}")
            return
            
        # Group directories by model type
        model_dirs = {}
        for d in all_dirs:
            dir_name = d.name.lower()
            
            # Extract model type from directory name
            # Expected format: {model_name}_{initials}_best_{timestamp} or {model_name}_{initials}_second_best_{timestamp}
            parts = dir_name.split('_')
            if 'best' in parts:
                if 'second' in parts and parts[parts.index('best')-1] == 'second':
                    model_type = 'second_best'
                else:
                    model_type = 'best'
                    
                if model_type not in model_dirs:
                    model_dirs[model_type] = []
                model_dirs[model_type].append(d)
                logger.debug(f"Added {d.name} to {model_type} group")
        
        # Process each model type
        for model_type, dirs in model_dirs.items():
            logger.info(f"\n=== Processing {model_type} model directories ===")
            
            # Log all directories before sorting
            logger.info(f"All {model_type} directories found:")
            for d in dirs:
                mtime = os.path.getmtime(d)
                logger.info(f"  - {d.name} (modified: {mtime})")
            
            # Sort by modification time (newest first)
            dirs.sort(key=os.path.getmtime, reverse=True)
            
            # Log what we're keeping
            logger.info(f"\nKeeping the {min(len(dirs), max_dirs_per_type)} most recent {model_type} directories:")
            for i, d in enumerate(dirs[:max_dirs_per_type], 1):
                mtime = os.path.getmtime(d)
                logger.info(f"  {i}. {d.name} (modified: {mtime})")
            
            # Remove old directories for this model type
            removed_count = 0
            dirs_to_remove = dirs[max_dirs_per_type:]
            
            if dirs_to_remove:
                logger.info(f"\nRemoving {len(dirs_to_remove)} old {model_type} directories:")
                for old_dir in dirs_to_remove:
                    try:
                        import shutil
                        mtime = os.path.getmtime(old_dir)
                        logger.info(f"  - Removing: {old_dir.name} (modified: {mtime})")
                        shutil.rmtree(old_dir)
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"  - Failed to remove {old_dir.name}: {e}")
            else:
                logger.info("\nNo old directories to remove for this model type.")
            
            logger.info(f"\n{model_type} cleanup complete: Kept {min(len(dirs), max_dirs_per_type)} directories, removed {removed_count} directories\n")
            
    except Exception as e:
        logger.error(f"Error in cleanup_old_directories: {str(e)}")
        logger.error(f"Traceback: {e.__traceback__}")
        raise
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_files for {directory}/{pattern}: {str(e)}")
        logger.error(f"Traceback: {e.__traceback__}")
        raise

def cleanup_processed_data() -> None:
    """Clean up old processed data and model files, keeping only the most recent ones."""
    try:
        # Clean up old processed data files, keeping the 2 most recent
        cleanup_old_files(PROCESSED_DIR, "processed_data_*.joblib", max_files_to_keep=2)
        
        # Clean up old model files, keeping the 2 most recent
        cleanup_old_files(MODELS_DIR, "model_*.joblib", max_files_to_keep=2)
        
        # Also clean up old metrics and feature importance files
        cleanup_old_files(MODELS_DIR, "metrics_*.json", max_files_to_keep=2)
        cleanup_old_files(MODELS_DIR, "feature_importance_*.json", max_files_to_keep=2)
        
        logger.info("Cleanup completed successfully - kept 2 most recent files")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise

def serialize_data(data: Any) -> Dict[str, Any]:
    """
    Helper function to serialize data for XCom.
    
    Args:
        data: Data to serialize (numpy array, pandas DataFrame, or other)
        
    Returns:
        Dictionary with serialized data and metadata
    """
    import numpy as np
    import pandas as pd
    
    try:
        if isinstance(data, np.ndarray):
            # Use base64 encoding for numpy arrays
            return {
                '__type__': 'numpy_array',
                'dtype': str(data.dtype),
                'shape': data.shape,
                'data': base64.b64encode(data.tobytes()).decode('utf-8')
            }
        elif isinstance(data, pd.DataFrame):
            # Convert DataFrame to dict of numpy arrays
            return {
                '__type__': 'pandas_dataframe',
                'columns': data.columns.tolist(),
                'index': data.index.tolist(),
                'data': {col: serialize_data(data[col].values) for col in data.columns}
            }
        elif isinstance(data, pd.Series):
            # Convert Series to numpy array and serialize
            return {
                '__type__': 'pandas_series',
                'name': data.name,
                'index': data.index.tolist(),
                'data': serialize_data(data.values)
            }
        else:
            # For other types, use pickle with base64 encoding
            return {
                '__type__': 'pickle',
                'data': base64.b64encode(pickle.dumps(data)).decode('utf-8')
            }
    except Exception as e:
        logger.error(f"Error serializing data: {e}")
        raise

def deserialize_data(serialized_data: Dict[str, Any]) -> Any:
    """
    Helper function to deserialize data from XCom.
    
    Args:
        serialized_data: Dictionary with serialized data
        
    Returns:
        Deserialized data
    """
    import numpy as np
    import pandas as pd
    
    if not isinstance(serialized_data, dict) or '__type__' not in serialized_data:
        return serialized_data
    
    try:
        data_type = serialized_data['__type__']
        
        if data_type == 'numpy_array':
            # Reconstruct numpy array from base64
            data = np.frombuffer(
                base64.b64decode(serialized_data['data']),
                dtype=np.dtype(serialized_data['dtype'])
            )
            return data.reshape(serialized_data['shape'])
            
        elif data_type == 'pandas_dataframe':
            # Reconstruct DataFrame from columns
            data = {
                col: deserialize_data(serialized_data['data'][col])
                for col in serialized_data['columns']
            }
            return pd.DataFrame(data, index=serialized_data['index'])
            
        elif data_type == 'pandas_series':
            # Reconstruct Series
            return pd.Series(
                deserialize_data(serialized_data['data']),
                index=serialized_data['index'],
                name=serialized_data['name']
            )
            
        elif data_type == 'pickle':
            # Deserialize using pickle
            return pickle.loads(base64.b64decode(serialized_data['data'].encode('utf-8')))
            
        else:
            logger.warning(f"Unknown serialization type: {data_type}")
            return serialized_data
            
    except Exception as e:
        logger.error(f"Error deserializing data: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Test serialization/deserialization
    import numpy as np
    import pandas as pd
    
    # Test with numpy array
    arr = np.random.rand(3, 3)
    serialized = serialize_data(arr)
    deserialized = deserialize_data(serialized)
    print(f"Numpy array test: {np.allclose(arr, deserialized)}")
    
    # Test with DataFrame
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    serialized = serialize_data(df)
    deserialized = deserialize_data(serialized)
    print(f"DataFrame test: {df.equals(deserialized)}")
    
    # Test cleanup functions
    print("Testing cleanup functions...")
    cleanup_processed_data()
