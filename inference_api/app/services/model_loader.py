import os
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
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
            'production': None,    # Currently serving model
            'shadow': None,        # New model being tested
            'previous': None       # Previous production model (for rollback)
        }
        self.current_production_model = None
        
    def _load_model_from_dir(self, model_dir: Path) -> Optional[dict]:
        """Load a single model from directory."""
        try:
            model_path = model_dir / "model.joblib"
            if not model_path.exists():
                logger.warning(f"Model file not found in {model_dir}")
                return None
                
            model = joblib.load(model_path)
            
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
            # Find all model directories
            model_dirs = [d for d in self.model_dir.glob("*") if d.is_dir()]
            
            if not model_dirs:
                logger.error("No model directories found")
                return False
                
            # Group models by type (best/second_best)
            best_models = []
            second_best_models = []
            
            for d in model_dirs:
                if "_best" in d.name.lower() and "_second_best" not in d.name.lower():
                    best_models.append(d)
                elif "_second_best" in d.name.lower():
                    second_best_models.append(d)
            
            # Sort by modification time (newest first)
            best_models.sort(key=os.path.getmtime, reverse=True)
            second_best_models.sort(key=os.path.getmtime, reverse=True)
            
            # Load models
            loaded_any = False
            
            # Load best model as production if we don't have one yet
            if best_models and (self.models['production'] is None or self.current_production_model != 'best'):
                if model_data := self._load_model_from_dir(best_models[0]):
                    self.models['previous'] = self.models['production']
                    self.models['production'] = model_data
                    self.current_production_model = 'best'
                    loaded_any = True
                    logger.info(f"Loaded production model from: {best_models[0].name}")
            
            # Load second best as shadow model if available
            if second_best_models:
                if model_data := self._load_model_from_dir(second_best_models[0]):
                    self.models['shadow'] = model_data
                    loaded_any = True
                    logger.info(f"Loaded shadow model from: {second_best_models[0].name}")
            
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
        
        # Convert input data to DataFrame with a single row
        input_df = pd.DataFrame([input_data])
        
        # Apply the preprocessor directly
        try:
            processed_data = preprocessor.transform(input_df)
            return processed_data
        except Exception as e:
            logger.error(f"Error in preprocessor: {str(e)}")
            logger.error(f"Input data: {input_data}")
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
