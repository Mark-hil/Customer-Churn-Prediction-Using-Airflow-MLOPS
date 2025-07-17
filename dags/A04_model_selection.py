"""Model selection and evaluation module for the churn prediction pipeline."""
import logging
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime

# Import ML libraries
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc, average_precision_score,
    PrecisionRecallDisplay, RocCurveDisplay
)
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.model_selection import learning_curve

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
PLOTS_DIR = Path("/opt/airflow/data/plots")
MODELS_DIR = Path("/opt/airflow/data/models")

# Ensure directories exist
for directory in [PLOTS_DIR, MODELS_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o755)
    except Exception as e:
        logger.warning(f"Could not create directory {directory}: {str(e)}")

class ModelEvaluator:
    """Class for evaluating and comparing machine learning models."""
    
    def __init__(self, models: Dict[str, Any], random_state: int = 42):
        """
        Initialize the model evaluator.
        
        Args:
            models: Dictionary of trained models to evaluate
            random_state: Random seed for reproducibility
        """
        self.models = models
        self.random_state = random_state
        self.final_model = None
        self.best_model_name = None
        self.model_scores = {}
        self.results = {}
        self.feature_importances = {}
        
    def create_voting_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, 
                             voting: str = 'soft') -> Any:
        """
        Create and train a voting classifier ensemble.
        
        Args:
            X_train: Training features
            y_train: Training labels
            voting: Type of voting ('soft' or 'hard')
            
        Returns:
            Trained voting classifier
        """
        from sklearn.ensemble import VotingClassifier
        
        if not self.models:
            raise ValueError("No models available for creating ensemble")
            
        voting_clf = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting=voting
        )
        
        voting_clf.fit(X_train, y_train)
        return voting_clf
        
    def create_weighted_ensemble(self, X: np.ndarray, models: Dict[str, Any] = None, 
                               weights: List[float] = None) -> np.ndarray:
        """
        Create predictions using a weighted average ensemble.
        
        Args:
            X: Input features for prediction
            models: Dictionary of models to use (defaults to self.models)
            weights: Weights for each model (defaults to equal weights)
            
        Returns:
            Weighted average predictions
        """
        if models is None:
            models = self.models
            
        if not models:
            raise ValueError("No models available for creating ensemble")
            
        if weights is None:
            # Default to equal weights if not specified
            weights = [1.0/len(models)] * len(models)
            
        if len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
            
        preds = np.zeros(X.shape[0])
        for (name, model), weight in zip(models.items(), weights):
            if hasattr(model, 'predict_proba'):
                preds += model.predict_proba(X)[:, 1] * weight
            else:
                preds += model.predict(X) * weight
                
        return preds / sum(weights)
        
    def select_best_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                         metric: str = 'roc_auc') -> Tuple[Any, str]:
        """
        Select the best performing model based on the specified metric.
        
        Args:
            X_test: Test features
            y_test: True labels for the test set
            metric: Metric to use for model selection
            
        Returns:
            Tuple of (best_model, best_model_name)
        """
        if not self.models:
            raise ValueError("No models available for selection")
            
        metric_funcs = {
            'roc_auc': roc_auc_score,
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score
        }
        
        if metric not in metric_funcs:
            raise ValueError(f"Unsupported metric: {metric}. Choose from: {', '.join(metric_funcs.keys())}")
            
        # Evaluate each model
        self.model_scores = {}
        for name, model in self.models.items():
            try:
                if metric == 'roc_auc':
                    y_pred = model.predict_proba(X_test)[:, 1]
                    score = metric_funcs[metric](y_test, y_pred)
                else:
                    y_pred = model.predict(X_test)
                    score = metric_funcs[metric](y_test, y_pred)
                    
                self.model_scores[name] = score
                logger.info(f"{name.upper():<20} {metric.upper()}: {score:.4f}")
                
            except Exception as e:
                logger.warning(f"Error evaluating {name}: {str(e)}")
                continue
                
        if not self.model_scores:
            raise ValueError("No models were successfully evaluated")
            
        # Select best model
        self.best_model_name = max(self.model_scores, key=self.model_scores.get)
        self.final_model = self.models[self.best_model_name]
        
        logger.info(f"\nSelected model: {self.best_model_name.upper()} "
                   f"({metric.upper()} = {self.model_scores[self.best_model_name]:.4f})")
        
        return self.final_model, self.best_model_name
        
    def get_feature_importances(self, model_name: str = None, feature_names: List[str] = None) -> Dict[str, float]:
        """
        Get feature importances from a model.
        
        Args:
            model_name: Name of the model to get importances from (defaults to best model)
            feature_names: List of feature names (optional)
            
        Returns:
            Dictionary of {feature_name: importance} pairs
        """
        if model_name is None:
            if self.best_model_name is None:
                raise ValueError("No model specified and no best model available")
            model = self.final_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Model '{model_name}' not found")
                
        # Get feature importances based on model type
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning(f"Model {model_name} does not support feature importances")
            return {}
            
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
            
        return dict(zip(feature_names, importances))
        
    def plot_roc_curve(self, X_test: np.ndarray, y_test: np.ndarray, figsize: tuple = (10, 8)) -> None:
        """Plot ROC curve for all models."""
        plt.figure(figsize=figsize)
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
    def plot_precision_recall_curve(self, X_test: np.ndarray, y_test: np.ndarray, figsize: tuple = (10, 8)) -> None:
        """Plot Precision-Recall curve for all models."""
        plt.figure(figsize=figsize)
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_score)
                avg_precision = average_precision_score(y_test, y_score)
                plt.plot(recall, precision, lw=2, 
                        label=f'{name} (AP = {avg_precision:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="best")
        
    def plot_feature_importance(self, model_name: str = None, top_n: int = 15, 
                              figsize: tuple = (12, 8)) -> None:
        """Plot feature importance for a model."""
        if model_name is None:
            if self.best_model_name is None:
                raise ValueError("No model specified and no best model available")
            model_name = self.best_model_name
            
        importances = self.get_feature_importances(model_name)
        if not importances:
            logger.warning(f"No feature importances available for {model_name}")
            return
            
        # Sort features by importance
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        features, importance_vals = zip(*sorted_importances[:top_n])
        
        plt.figure(figsize=figsize)
        y_pos = np.arange(len(features))
        plt.barh(y_pos, importance_vals, align='center')
        plt.yticks(y_pos, features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Most Important Features - {model_name}')
        plt.gca().invert_yaxis()
        
    def plot_confusion_matrix(self, model_name: str, X_test: np.ndarray, 
                            y_test: np.ndarray, normalize: bool = True,
                            figsize: tuple = (8, 6)) -> None:
        """Plot confusion matrix for a model."""
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
            
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get evaluation metrics for all models."""
        return self.results
        
    def evaluate_models(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        dataset_name: str = 'test'
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on the given dataset.
        
        Args:
            X: Features
            y: True labels
            dataset_name: Name of the dataset (e.g., 'train', 'val', 'test')
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"\nEvaluating {model_name} on {dataset_name} set...")
            
            try:
                # Make predictions
                y_pred = model.predict(X)
                y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self._calculate_metrics(y, y_pred, y_pred_proba)
                results[model_name] = metrics
                
                # Store results
                if dataset_name not in self.results:
                    self.results[dataset_name] = {}
                self.results[dataset_name][model_name] = metrics
                
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
                
                # Extract feature importances if available
                self._extract_feature_importances(model, model_name, X)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for metrics like ROC-AUC)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Add probability-based metrics if available
        if y_pred_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_true, y_pred_proba),
                'pr_auc': average_precision_score(y_true, y_pred_proba),
                'log_loss': -np.mean(y_true * np.log(y_pred_proba + 1e-15) + 
                                   (1 - y_true) * np.log(1 - y_pred_proba + 1e-15))
            })
        
        return metrics
    
    def _extract_feature_importances(
        self, 
        model: Any, 
        model_name: str, 
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> None:
        """
        Extract and store feature importances from a model.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X: Features (used to get feature names if not provided)
            feature_names: Optional list of feature names
        """
        if feature_names is None and hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        
        importances = None
        
        # Try different methods to extract feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        elif hasattr(model, 'best_estimator_'):
            # Handle GridSearchCV/RandomizedSearchCV
            return self._extract_feature_importances(model.best_estimator_, model_name, X, feature_names)
        
        if importances is not None and feature_names is not None:
            self.feature_importances[model_name] = dict(zip(feature_names, importances))
    
    def plot_roc_curves(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot ROC curves for all models.
        
        Args:
            X: Features
            y: True labels
            save_path: Path to save the plot (if None, show the plot)
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, 
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curves to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_precision_recall_curves(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot precision-recall curves for all models.
        
        Args:
            X: Features
            y: True labels
            save_path: Path to save the plot (if None, show the plot)
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
                precision, recall, _ = precision_recall_curve(y, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                plt.plot(recall, precision, lw=2,
                        label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved precision-recall curves to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_calibration_curves(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot calibration curves for all models.
        
        Args:
            X: Features
            y: True labels
            save_path: Path to save the plot (if None, show the plot)
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
                prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10)
                
                plt.plot(prob_pred, prob_true, 's-',
                        label=f'{model_name}')
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curves (Reliability Curves)')
        plt.legend(loc='upper left')
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved calibration curves to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_feature_importances(
        self, 
        model_name: str, 
        top_n: int = 15,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot feature importances for a specific model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to display
            save_path: Path to save the plot (if None, show the plot)
        """
        if model_name not in self.feature_importances:
            logger.warning(f"No feature importances available for {model_name}")
            return
        
        importances = self.feature_importances[model_name]
        
        # Sort features by importance
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        features, importance_values = zip(*sorted_importances[:top_n])
        
        # Create horizontal bar plot
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(features))
        
        plt.barh(y_pos, importance_values, align='center')
        plt.yticks(y_pos, features)
        plt.xlabel('Importance')
        plt.title(f'Feature Importances - {model_name}')
        plt.gca().invert_yaxis()  # Most important features on top
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importances to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def compare_models(
        self, 
        metric: str = 'roc_auc',
        dataset_name: str = 'test',
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Compare models based on a specific metric.
        
        Args:
            metric: Metric to compare (e.g., 'roc_auc', 'f1', 'accuracy')
            dataset_name: Name of the dataset to use for comparison
            save_path: Path to save the plot (if None, show the plot)
        """
        if dataset_name not in self.results:
            logger.warning(f"No results available for dataset: {dataset_name}")
            return
        
        # Extract metric values for each model
        model_metrics = []
        for model_name, metrics in self.results[dataset_name].items():
            if metric in metrics and not isinstance(metrics[metric], str):
                model_metrics.append((model_name, metrics[metric]))
        
        if not model_metrics:
            logger.warning(f"Metric '{metric}' not found in results")
            return
        
        # Sort models by metric value
        model_metrics.sort(key=lambda x: x[1], reverse=True)
        model_names, metric_values = zip(*model_metrics)
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, metric_values, color='skyblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(metric_values) * 1.1)  # Add 10% headroom
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def generate_report(
        self, 
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        output_dir: Union[str, Path] = None
    ) -> Dict[str, str]:
        """
        Generate a comprehensive evaluation report with plots.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save the report (default: PLOTS_DIR)
            
        Returns:
            Dictionary with paths to generated plots
        """
        output_dir = Path(output_dir) if output_dir else PLOTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate models on both train and test sets
        logger.info("Evaluating models on training set...")
        self.evaluate_models(X_train, y_train, dataset_name='train')
        
        logger.info("\nEvaluating models on test set...")
        self.evaluate_models(X_test, y_test, dataset_name='test')
        
        # Generate plots
        plots = {}
        
        # ROC curves
        roc_path = output_dir / 'roc_curves.png'
        self.plot_roc_curves(X_test, y_test, save_path=roc_path)
        plots['roc_curves'] = str(roc_path)
        
        # Precision-Recall curves
        pr_path = output_dir / 'precision_recall_curves.png'
        self.plot_precision_recall_curves(X_test, y_test, save_path=pr_path)
        plots['precision_recall_curves'] = str(pr_path)
        
        # Calibration curves
        cal_path = output_dir / 'calibration_curves.png'
        self.plot_calibration_curves(X_test, y_test, save_path=cal_path)
        plots['calibration_curves'] = str(cal_path)
        
        # Model comparison
        for metric in ['roc_auc', 'f1', 'accuracy']:
            comp_path = output_dir / f'model_comparison_{metric}.png'
            self.compare_models(metric=metric, dataset_name='test', save_path=comp_path)
            plots[f'model_comparison_{metric}'] = str(comp_path)
        
        # Feature importances for each model
        for model_name in self.models.keys():
            fi_path = output_dir / f'feature_importances_{model_name}.png'
            self.plot_feature_importances(model_name, save_path=fi_path)
            plots[f'feature_importances_{model_name}'] = str(fi_path)
        
        # Save results to JSON
        results_path = output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nEvaluation report generated in {output_dir}")
        return plots


def load_models_from_dir(
    model_dir: Union[str, Path],
    pattern: str = '*.joblib'
) -> Dict[str, Any]:
    """
    Load all models from a directory.
    
    Args:
        model_dir: Directory containing model files
        pattern: File pattern to match model files
        
    Returns:
        Dictionary of loaded models
    """
    model_dir = Path(model_dir)
    models = {}
    
    for model_file in model_dir.glob(pattern):
        if 'metadata' in model_file.name:  # Skip metadata files
            continue
            
        model_name = model_file.stem
        try:
            model = joblib.load(model_file)
            models[model_name] = model
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading {model_file}: {str(e)}")
    
    return models


def evaluate_models(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, 
                   output_dir: Union[str, Path] = None) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple models and generate performance reports.
    
    Args:
        models: Dictionary of trained models to evaluate (can be model objects or dicts with 'model' key)
        X_test: Test features as numpy array
        y_test: True labels for the test set
        output_dir: Directory to save evaluation results and plots (default: PLOTS_DIR)
        
    Returns:
        Dictionary containing evaluation metrics for each model
    """
    output_dir = Path(output_dir) if output_dir else PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting model evaluation with {len(models)} models")
    
    # Initialize evaluator with models
    evaluator = ModelEvaluator(models)
    
    # Initialize results dictionary
    evaluation_results = {}
    
    # Evaluate each model
    for model_name, model in models.items():
        try:
            logger.info(f"\nEvaluating {model_name}...")
            
            # Check if model is a dictionary with 'model' key (from training output)
            if isinstance(model, dict) and 'model' in model:
                model_obj = model['model']
            else:
                model_obj = model
                
            # Make predictions
            if hasattr(model_obj, 'predict_proba'):
                y_pred_proba = model_obj.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model_obj.predict(X_test)
                y_pred_proba = None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
            
            # Store results
            evaluation_results[model_name] = metrics
            logger.info(f"{model_name} metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            continue
    
    # Generate and save plots
    try:
        # ROC Curve
        plt.figure(figsize=(10, 8))
        evaluator.plot_roc_curve(X_test, y_test)
        roc_path = output_dir / 'roc_curve.png'
        plt.savefig(roc_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Precision-Recall Curve
        plt.figure(figsize=(10, 8))
        evaluator.plot_precision_recall_curve(X_test, y_test)
        pr_path = output_dir / 'precision_recall_curve.png'
        plt.savefig(pr_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Feature Importance (for models that support it)
        try:
            for model_name in models.keys():
                try:
                    plt.figure(figsize=(12, 8))
                    evaluator.plot_feature_importance(model_name=model_name, top_n=15)
                    fi_path = output_dir / f'feature_importance_{model_name}.png'
                    plt.savefig(fi_path, bbox_inches='tight', dpi=300)
                    plt.close()
                except Exception as e:
                    logger.warning(f"Could not generate feature importance for {model_name}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error generating feature importance plots: {str(e)}")
        
        # Confusion Matrices
        for model_name in models.keys():
            try:
                plt.figure(figsize=(8, 6))
                evaluator.plot_confusion_matrix(
                    model_name=model_name,
                    X_test=X_test,
                    y_test=y_test,
                    normalize=True
                )
                cm_path = output_dir / f'confusion_matrix_{model_name}.png'
                plt.savefig(cm_path, bbox_inches='tight', dpi=300)
                plt.close()
            except Exception as e:
                logger.warning(f"Could not generate confusion matrix for {model_name}: {str(e)}")
        
        # Save metrics to JSON
        metrics_path = output_dir / 'model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating evaluation plots: {str(e)}")
        logger.error(traceback.format_exc())
    
    return evaluation_results


if __name__ == "__main__":
    # Example usage
    from A00_data_understanding import load_data
    from A02_feature_engineering import preprocess_data
    from A03_experimentation import train_models
    
    # Load and preprocess data
    df = load_data()
    
    # Preprocess data
    processed_data = preprocess_data(
        df, 
        target='Churn',
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Train models
    models = train_models(processed_data)
    
    # Initialize evaluator with trained models
    evaluator = ModelEvaluator(models)
    
    # 1. Select best model based on ROC-AUC
    best_model, best_model_name = evaluator.select_best_model(
        X_test=processed_data['X_test'],
        y_test=processed_data['y_test'],
        metric='roc_auc'
    )
    
    # 2. Create and evaluate voting ensemble
    try:
        voting_clf = evaluator.create_voting_ensemble(
            X_train=processed_data['X_train'],
            y_train=processed_data['y_train'],
            voting='soft'
        )
        
        # Add voting classifier to models
        models['voting_ensemble'] = voting_clf
        evaluator.models = models  # Update evaluator with new model
        
        # Evaluate voting ensemble
        y_pred_voting = voting_clf.predict_proba(processed_data['X_test'])[:, 1]
        voting_auc = roc_auc_score(processed_data['y_test'], y_pred_voting)
        print(f"\nVoting Ensemble ROC-AUC: {voting_auc:.4f}")
        
    except Exception as e:
        print(f"Error creating voting ensemble: {str(e)}")
    
    # 3. Create weighted average ensemble
    try:
        # Use model scores as weights (better models have more influence)
        weights = list(evaluator.model_scores.values())
        weighted_preds = evaluator.create_weighted_ensemble(
            X=processed_data['X_test'],
            weights=weights
        )
        weighted_auc = roc_auc_score(processed_data['y_test'], weighted_preds)
        print(f"Weighted Ensemble ROC-AUC: {weighted_auc:.4f}")
        
    except Exception as e:
        print(f"Error creating weighted ensemble: {str(e)}")
    
    # 4. Get feature importances for the best model
    try:
        feature_importances = evaluator.get_feature_importances(
            model_name=best_model_name,
            feature_names=processed_data.get('feature_names')
        )
        
        # Sort and print top 10 most important features
        sorted_importances = sorted(feature_importances.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)[:10]
        
        print("\nTop 10 most important features:")
        for feature, importance in sorted_importances:
            print(f"{feature}: {importance:.4f}")
            
    except Exception as e:
        print(f"Error getting feature importances: {str(e)}")
    
    # 5. Save the best model
    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"\nBest model ({best_model_name}) saved to {best_model_path}")
    
    # Generate evaluation report
    report_plots = evaluator.generate_report(
        X_train=processed_data['X_train'], 
        y_train=processed_data['y_train'],
        X_test=processed_data['X_test'], 
        y_test=processed_data['y_test'],
        output_dir=PLOTS_DIR
    )
    
    print("\nGenerated the following plots:")
    for name, path in report_plots.items():
        print(f"- {name}: {path}")
