import os
import joblib
import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import datetime
import json

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_dir: str):
        """
        Initialize the model loader with the directory containing model files.
        
        Args:
            model_dir: Base directory containing model subdirectories
        """
        self.model_dir = Path(model_dir)
        self.models = {
            'production': None,    # Currently serving model (Model A)
            'shadow': None,        # Candidate model being tested (Model B)
            'previous': None       # Previous production model (for rollback)
        }
        self.current_production_model = None
        self.ab_test_assignments = {}  # Track model assignments for A/B testing
        
    def _load_model_from_dir(self, model_dir: Path) -> Optional[dict]:
        """Load a single model from directory."""
        try:
            logger.info(f"Attempting to load model from directory: {model_dir}")
            model_path = model_dir / "model.joblib"
            logger.info(f"Looking for model file at: {model_path}")
            
            if not model_path.exists():
                logger.warning(f"Model file not found in {model_dir}")
                logger.warning(f"Contents of {model_dir}: {os.listdir(model_dir)}")
                return None
                
            logger.info(f"Loading model from: {model_path}")
            model = joblib.load(model_path)
            logger.info("Successfully loaded model")
            
            # Load preprocessor
            preprocessor = None
            preprocessor_path = model_dir / "preprocessor.joblib"
            if preprocessor_path.exists():
                preprocessor = joblib.load(preprocessor_path)
            
            # Load feature names
            feature_names = None
            feature_names_path = model_dir / "feature_names.json"  # Changed from feature_importance.json
            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    feature_names = json.load(f)
            
            # Load model metadata
            metadata = {}
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Extract version from directory name (format: name_version)
            version = 'unknown'
            try:
                # Get the last part after the last underscore
                version = str(model_dir.name).rsplit('_', 1)[-1]
                # Validate it looks like a timestamp (YYYYMMDDTHHMMSS)
                if not (len(version) == 15 and version[8] == 'T' and version[:8].isdigit() and version[9:].isdigit()):
                    version = 'unknown'
            except (IndexError, AttributeError):
                pass
                
            # Ensure metadata has the version
            if 'model_version' not in metadata or metadata['model_version'] == 'unknown':
                metadata['model_version'] = version
            
            return {
                'model': model,
                'preprocessor': preprocessor,
                'feature_names': feature_names,
                'metadata': metadata,
                'model_dir': str(model_dir),
                'loaded_at': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error loading model from {model_dir}: {str(e)}")
            return None
    
    def load_models(self) -> bool:
        try:
            logger.info(f"Starting to load models from directory: {self.model_dir}")
            # Find all model directories
            model_dirs = [d for d in self.model_dir.glob("*") if d.is_dir()]
            logger.info(f"Found {len(model_dirs)} model directories: {[d.name for d in model_dirs]}")
            
            if not model_dirs:
                logger.error("No model directories found")
                return False
                
            # Group models by type (best/second_best)
            best_models = []
            second_best_models = []
            
            for d in model_dirs:
                logger.info(f"Processing model directory: {d.name}")
                if "_best" in d.name.lower() and "_second_best" not in d.name.lower():
                    # Check if this is a random forest model (preferred)
                    if "random_forest" in d.name.lower() or "rf_" in d.name.lower():
                        logger.info(f"Found best model candidate (random forest): {d.name}")
                        best_models.insert(0, d)  # Insert at beginning to prioritize
                    else:
                        logger.info(f"Found best model candidate: {d.name}")
                        best_models.append(d)
                elif "_second_best" in d.name.lower():
                    # Check if this is a logistic regression model (preferred for shadow)
                    if "logistic_regression" in d.name.lower() or "lr_" in d.name.lower():
                        logger.info(f"Found second best model candidate (logistic regression): {d.name}")
                        second_best_models.insert(0, d)  # Insert at beginning to prioritize
                    else:
                        logger.info(f"Found second best model candidate: {d.name}")
                        second_best_models.append(d)
            
            logger.info(f"Found {len(best_models)} best models and {len(second_best_models)} second best models")
            
            # Sort by modification time (newest first)
            best_models.sort(key=os.path.getmtime, reverse=True)
            second_best_models.sort(key=os.path.getmtime, reverse=True)
            
            if best_models:
                logger.info(f"Best models (newest first): {[m.name for m in best_models]}")
            if second_best_models:
                logger.info(f"Second best models (newest first): {[m.name for m in second_best_models]}")
            
            # Load models
            loaded_any = False
            
            # Load best model as production if we don't have one yet
            if best_models and (self.models['production'] is None or self.current_production_model != 'best'):
                logger.info(f"Attempting to load production model from: {best_models[0].name}")
                if model_data := self._load_model_from_dir(best_models[0]):
                    self.models['previous'] = self.models['production']
                    self.models['production'] = model_data
                    self.current_production_model = 'best'
                    loaded_any = True
                    logger.info(f"Successfully loaded production model from: {best_models[0].name}")
                else:
                    logger.error(f"Failed to load production model from: {best_models[0].name}")
            
            # Load second best as shadow model if available
            if second_best_models:
                logger.info(f"Attempting to load shadow model from: {second_best_models[0].name}")
                if model_data := self._load_model_from_dir(second_best_models[0]):
                    self.models['shadow'] = model_data
                    loaded_any = True
                    logger.info(f"Successfully loaded shadow model from: {second_best_models[0].name}")
                else:
                    logger.error(f"Failed to load shadow model from: {second_best_models[0].name}")
            
            logger.info(f"Model loading completed. Loaded any models: {loaded_any}")
            logger.info(f"Current models: {list(self.models.keys())}")
            
            return loaded_any
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def preprocess_input(
        self, 
        input_data: Dict[str, Any], 
        preprocessor: Any = None,
        feature_names: List[str] = None
    ) -> np.ndarray:
        """
        Preprocess input data using the specified preprocessor.
        
        Args:
            input_data: Dictionary of input features
            preprocessor: Preprocessor to use (if None, uses the production model's preprocessor)
            feature_names: List of expected feature names in order (not used in this version)
            
        Returns:
            Preprocessed numpy array
        """
        import pandas as pd
        
        if preprocessor is None:
            if not self.models['production']:
                raise ValueError("No production model loaded and no preprocessor provided")
            preprocessor = self.models['production']['preprocessor']
        
        try:
            # Convert input data to DataFrame with a single row
            # Ensure we're working with a flat dictionary of features
            if 'input_data' in input_data and isinstance(input_data['input_data'], dict):
                # Handle the case where input_data contains nested 'input_data' key
                features = input_data['input_data']
            else:
                # Handle the case where input_data is already the features dictionary
                features = input_data
            
            # Create a DataFrame with the features
            input_df = pd.DataFrame([features])
            
            # Ensure all expected numeric columns are present and have the correct type
            numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
            for col in numeric_cols:
                if col in input_df.columns:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            
            # Log the input data for debugging
            logger.debug(f"Input DataFrame columns: {input_df.columns.tolist()}")
            logger.debug(f"Input DataFrame dtypes: {input_df.dtypes}")
            
            # Apply the preprocessor
            processed_data = preprocessor.transform(input_df)
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in preprocessor: {str(e)}")
            logger.error(f"Input data type: {type(input_data)}")
            logger.error(f"Input data content: {input_data}")
            if 'input_df' in locals():
                logger.error(f"Input DataFrame columns: {input_df.columns.tolist()}")
                logger.error(f"Input DataFrame content: {input_df.to_dict()}")
            raise
    
    def predict(
        self, 
        input_data: Dict[str, Any], 
        model_type: str = 'production',
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Make a prediction using the specified model.
        
        Args:
            input_data: Input features for prediction
            model_type: Which model to use ('production', 'shadow', or 'previous')
            include_metadata: Whether to include model metadata in the response
            
        Returns:
            Dictionary containing prediction results
        """
        if model_type not in self.models or not self.models[model_type]:
            raise ValueError(f"{model_type} model not loaded")
            
        model_data = self.models[model_type]
        
        try:
            # Preprocess input
            processed_input = self.preprocess_input(
                input_data,
                preprocessor=model_data['preprocessor'],
                feature_names=model_data['feature_names']
            )
            
            # Make prediction and time it
            import time
            start_time = time.time()
            prediction = model_data['model'].predict(processed_input)
            prediction_proba = model_data['model'].predict_proba(processed_input)
            prediction_time_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Format response
            result = {
                "prediction": int(prediction[0]),
                "probability": float(prediction_proba[0][1]),  # Probability of positive class
                "model_type": model_type,
                "model_metadata": model_data.get('metadata', {}),
                "model_version": model_data.get('metadata', {}).get('model_version', 'unknown'),
                "prediction_time": f"{prediction_time_ms:.2f}ms"
            }
            
            if include_metadata:
                result['model_metadata'] = model_data['metadata']
                
            return result
            
        except Exception as e:
            logger.error(f"{model_type.upper()} prediction error: {str(e)}")
            raise
    
    def get_model_info(self, model_type: str = 'production') -> Dict[str, Any]:
        """
        Get information about a loaded model.
        
        Args:
            model_type: Which model to get info for ('production', 'shadow', or 'previous')
            
        Returns:
            Dictionary containing model information with sensitive/verbose fields removed
        """
        if model_type not in self.models or not self.models[model_type]:
            return {"status": "not_loaded"}
            
        model_data = self.models[model_type]
        
        # Create a filtered version of metadata without sensitive/verbose fields
        metadata = model_data['metadata'].copy()
        for field in ['metrics', 'params', 'feature_importance']:
            if field in metadata:
                del metadata[field]
        
        return {
            "status": "loaded",
            "model_type": model_type,
            "model_path": model_data['model_dir'],
            "loaded_at": model_data['loaded_at'],
            "metadata": metadata
        }
        
    def get_ab_test_info(self) -> Dict[str, Any]:
        """
        Get information about the current A/B test configuration.
        
        Returns:
            Dictionary containing A/B test information
        """
        return {
            'model_a': self.get_model_info('production'),
            'model_b': self.get_model_info('shadow'),
            'assignments': len(self.ab_test_assignments)
        }
        
    def assign_to_test_group(self, request_id: str) -> Tuple[str, str]:
        """
        Assign a request to a test group (A or B).
        
        Args:
            request_id: Unique identifier for the request
            
        Returns:
            Tuple of (model_type, group) where group is 'A' (control) or 'B' (candidate)
        """
        # Return existing assignment if this request was already assigned
        if request_id in self.ab_test_assignments:
            return self.ab_test_assignments[request_id]
            
        # Simple 50/50 split for now - can be made configurable
        if random.random() < 0.5:
            group = 'A'
            model_type = 'production'
        else:
            group = 'B'
            model_type = 'shadow' if self.models.get('shadow') else 'production'
            
        self.ab_test_assignments[request_id] = (model_type, group)
        return model_type, group
    
    def switch_models(self, new_production_model: str) -> bool:
        """
        Switch the production model to a different model.
        
        Args:
            new_production_model: The new production model ('best' or 'second_best')
            
        Returns:
            True if successful, False otherwise
        """
        if new_production_model not in ['best', 'second_best']:
            raise ValueError("new_production_model must be 'best' or 'second_best'")
            
        if new_production_model == self.current_production_model:
            logger.info(f"{new_production_model} is already the production model")
            return True
            
        # If we're already using the requested model, just return
        if (new_production_model == 'best' and self.models['production'] and 
            'best' in self.models['production']['model_dir']):
            self.current_production_model = 'best'
            return True
            
        if (new_production_model == 'second_best' and self.models['shadow'] and 
            'second_best' in self.models['shadow']['model_dir']):
            # Swap production and shadow models
            old_production = self.models['production']
            self.models['previous'] = old_production
            self.models['production'] = self.models['shadow']
            self.models['shadow'] = old_production
            self.current_production_model = 'second_best'
            logger.info("Switched production model to second_best")
            return True
            
        # If we get here, we need to reload the models
        logger.info(f"Reloading models to switch to {new_production_model}")
        return self.load_models()
