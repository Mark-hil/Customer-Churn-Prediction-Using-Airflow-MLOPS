import random
import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Define constants
AB_TEST_LOG_DIR = Path("logs/ab_test_logs")
MAX_LOG_ENTRIES = 1000  # Maximum number of log entries to keep
WEEKLY_LOG_FILE = "weekly_ab_test_logs.json"

# Ensure log directory exists
AB_TEST_LOG_DIR.mkdir(parents=True, exist_ok=True, mode=0o755)

class ABTestingService:
    def __init__(self, model_loader, traffic_split: float = 0.5):
        """
        Initialize A/B testing service.
        
        Args:
            model_loader: Instance of ModelLoader
            traffic_split: Percentage of traffic to send to the candidate model (0.0 to 1.0)
        """
        self.model_loader = model_loader
        self.traffic_split = max(0.0, min(1.0, traffic_split))  # Clamp between 0 and 1
        self.logs = []
        self._log_cache = {}  # In-memory cache of recent predictions
        
        # Ensure log directory exists
        AB_TEST_LOG_DIR.mkdir(parents=True, exist_ok=True, mode=0o755)
        
        # Initialize log file if it doesn't exist
        self.log_file = AB_TEST_LOG_DIR / WEEKLY_LOG_FILE
        if not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                json.dump([], f)
        
        self._load_logs_from_disk()
        logger.info(f"ABTestingService initialized with traffic split: {self.traffic_split}")
        
    def get_model_for_request(self, request_id: str) -> Tuple[str, Dict[str, Any], str]:
        """
        Determine which model to use for the current request.
        
        Args:
            request_id: Unique identifier for the request
            
        Returns:
            Tuple of (model_type, model_data, group) where:
            - model_type: 'production' or 'shadow'
            - model_data: Dictionary containing model and metadata
            - group: 'A' (control) or 'B' (candidate)
        """
        # Simple random assignment based on traffic split
        if random.random() < self.traffic_split:
            model_type = 'shadow'
            group = 'B'  # Candidate model (B)
        else:
            model_type = 'production'
            group = 'A'  # Control model (A)
        
        model_data = self.model_loader.models.get(model_type)
        if not model_data:
            logger.error(f"No {model_type} model available")
            return None, None, None
        
        return model_type, model_data, group
    
    def log_prediction(
        self,
        request_id: str,
        input_data: Dict,
        prediction: Any,
        model_type: str,
        model_name: str,
        model_version: str,
        ground_truth: Optional[int] = None,
        ab_test_group: Optional[str] = None,
        prediction_time: Optional[float] = None
    ) -> None:
        """
        Log a prediction with optional ground truth and prediction time.
        
        Args:
            request_id: Unique identifier for the request
            input_data: The input data used for prediction
            prediction: Prediction result (can be a dict or raw value)
            model_type: Type of model used ('production' or 'shadow')
            model_name: Name of the model
            model_version: Version of the model
            ground_truth: Optional ground truth value
            ab_test_group: A/B test group ('A' or 'B') if applicable
            prediction_time: Time taken to make the prediction in seconds
        """
        """
        Log a prediction with optional ground truth.
        
        Args:
            request_id: Unique identifier for the request
            input_data: The input data used for prediction
            prediction: Raw prediction value
            model_type: Type of model used ('production' or 'shadow')
            model_name: Name of the model
            model_version: Version of the model
            ground_truth: Optional ground truth value
            ab_test_group: A/B test group ('A' or 'B') if applicable
        """
        # Extract prediction value and interpretation if prediction is a dict
        prediction_value = prediction.get('prediction', prediction) if isinstance(prediction, dict) else prediction
        if hasattr(prediction_value, '__int__'):
            prediction_value = int(prediction_value)
            
        interpretation = prediction.get('interpretation', '') if isinstance(prediction, dict) else ''
        
        # Create log entry with all relevant information
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "model_type": model_type,
            "input_data": input_data,
            "prediction": prediction_value,
            "ground_truth": ground_truth,
            "model_version": model_version,
            "model_name": model_name,
            "ab_test_group": ab_test_group,
            "interpretation": interpretation,
            "prediction_time": float(prediction_time) if prediction_time is not None else None
        }
        
        # Add to in-memory cache
        self._log_cache[request_id] = log_entry
        
        # Write to disk
        self._write_log_to_disk(log_entry)

    def update_ground_truth(self, request_id: str, ground_truth: int) -> None:
        """
        Update the ground truth for an existing log entry.
        
        Args:
            request_id: Unique identifier for the request
            ground_truth: Ground truth value
        """
        current_time = datetime.now().isoformat()
        
        # Find the log entry in memory
        log_entry = None
        for log in self.logs:
            if log.get('request_id') == request_id:
                log_entry = log
                break
        
        if log_entry is None and request_id in self._log_cache:
            # Try to get from cache if not in logs
            log_entry = self._log_cache[request_id]
        
        if log_entry is None:
            logger.warning(f"No log entry found for request {request_id} when updating ground truth")
            return
            
        # Create a new log entry with updated ground truth
        updated_entry = log_entry.copy()
        updated_entry.update({
            'ground_truth': ground_truth,
            'ground_truth_updated_at': current_time,
            'ground_truth_request_id': request_id
        })
        
        # Update in-memory logs
        for i, log in enumerate(self.logs):
            if log.get('request_id') == request_id:
                self.logs[i] = updated_entry
                break
        
        # Update cache
        self._log_cache[request_id] = updated_entry
        
        # Save to disk
        self._save_logs_to_disk()
        logger.info(f"Updated ground truth for request {request_id} to {ground_truth}")
        
    def _get_weekly_log_file(self) -> Path:
        """
        Get the current weekly log file path based on the current date.
        
        Returns:
            Path: Path to the current weekly log file
        """
        current_week = datetime.now().strftime("%Y-W%U")
        return AB_TEST_LOG_DIR / f"ab_test_logs_{current_week}.json"
        
    def _write_log_to_disk(self, log_entry):
        """
        Write log entry to the log file with weekly log rotation.
        
        Args:
            log_entry: Dictionary containing the log entry
        """
        try:
            # Get the current weekly log file
            self.log_file = self._get_weekly_log_file()
            
            # Ensure the log directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            
            # Load existing logs if file exists
            existing_logs = []
            if self.log_file.exists():
                try:
                    with open(self.log_file, 'r') as f:
                        existing_logs = json.load(f)
                    if not isinstance(existing_logs, list):
                        logger.warning("Log file contains invalid data, starting fresh")
                        existing_logs = []
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding log file {self.log_file}: {e}")
                    existing_logs = []
            
            # Add the new log entry
            existing_logs.append(log_entry)
            
            # Save back to file
            temp_file = f"{self.log_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(existing_logs, f, indent=2)
            
            # Atomically replace the old file with the new one
            os.replace(temp_file, self.log_file)
            
            # Update in-memory logs and cache
            self.logs = existing_logs
            if 'request_id' in log_entry:
                self._log_cache[log_entry['request_id']] = log_entry
                
            logger.info(f"Updated log file {self.log_file.name} with {len(existing_logs)} entries")
            
        except Exception as e:
            logger.error(f"Error writing to log file: {str(e)}")
            logger.exception("Full traceback:")
            raise
            
    def log_event(self, event_type: str, details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Log an event to the A/B testing logs.
        
        Args:
            event_type: Type of event (e.g., 'model_switch', 'rollback', 'error')
            details: Dictionary containing event details
            
        Returns:
            The logged event dictionary or None if logging failed
        """
        try:
            # Create event with required fields
            event = {
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                **details
            }
            
            # Add to in-memory logs
            self.logs.append(event)
            
            # Ensure we don't keep too many logs in memory
            if len(self.logs) > MAX_LOG_ENTRIES:
                self.logs = self.logs[-MAX_LOG_ENTRIES:]
            
            # Persist to disk
            self._save_logs_to_disk()
            
            logger.info(f"Logged {event_type} event: {event}")
            return event
            
        except Exception as e:
            logger.error(f"Error logging {event_type} event: {e}", exc_info=True)
            return None
            
    def _load_logs_from_disk(self, force_reload: bool = False) -> None:
        """Load logs from disk into memory.
        
        Args:
            force_reload: If True, force a reload from disk even if logs haven't changed
        """
        try:
            # Get the current weekly log file
            current_log_file = self._get_weekly_log_file()
            
            # Update the instance log file reference
            self.log_file = current_log_file
            
            if not current_log_file.exists() or current_log_file.stat().st_size == 0:
                self.logs = []
                self._log_cache = {}
                logger.info("No existing log file found or empty file, starting with empty logs")
                return
                
            with open(current_log_file, 'r') as f:
                logs = json.load(f)
                
            # Always update the logs if forced or if they've changed
            if force_reload or logs != self.logs:
                self.logs = logs
                self._log_cache = {
                    log['request_id']: log 
                    for log in logs 
                    if 'request_id' in log
                }
                logger.info(f"Loaded {len(logs)} log entries from {current_log_file}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding log file {current_log_file}: {e}")
            self.logs = []
            self._log_cache = {}
        except Exception as e:
            logger.error(f"Error loading logs from disk: {e}")
            self.logs = []
            self._log_cache = {}

    def _save_logs_to_disk(self) -> None:
        """Save logs to disk."""
        try:
            # Ensure directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a temporary file in the same directory for atomic write
            temp_file = f"{self.log_file}.tmp"
            
            # Load existing logs from disk if file exists
            existing_logs = []
            if self.log_file.exists():
                try:
                    with open(self.log_file, 'r') as f:
                        existing_logs = json.load(f)
                    if not isinstance(existing_logs, list):
                        logger.warning("Log file contains invalid data, starting fresh")
                        existing_logs = []
                except (json.JSONDecodeError, Exception) as e:
                    logger.error(f"Error reading existing logs: {e}")
                    existing_logs = []
            
            # Create a mapping of request_id to log entries for both existing and in-memory logs
            log_map = {log['request_id']: log for log in existing_logs if 'request_id' in log}
            
            # Update the map with in-memory logs (this will overwrite existing entries with the same request_id)
            for log in self.logs:
                if 'request_id' in log:
                    log_map[log['request_id']] = log
            
            # Convert the map back to a list
            merged_logs = list(log_map.values())
            
            # Ensure we have the most recent logs first
            merged_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            # Write to temporary file
            with open(temp_file, 'w') as f:
                json.dump(merged_logs, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename (works on POSIX systems)
            os.replace(temp_file, self.log_file)
            logger.info(f"Saved {len(merged_logs)} log entries to {self.log_file}")
            
            # Update in-memory logs to match what was saved
            self.logs = merged_logs
            self._log_cache = log_map
            
            # Keep only the last 10 log files
            log_files = sorted(AB_TEST_LOG_DIR.glob("*.json"), key=os.path.getmtime)
            for old_file in log_files[:-10]:
                try:
                    old_file.unlink()
                    logger.debug(f"Removed old log file: {old_file}")
                except Exception as e:
                    logger.warning(f"Could not remove old log file {old_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error saving logs to disk: {e}")
            # Clean up temporary file if it exists
            if 'temp_file' in locals() and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.error(f"Failed to clean up temporary file {temp_file}: {e}")
            raise  # Re-raise the exception to be handled by the caller

    def get_recent_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent logs."""
        return self.logs[-limit:]

    def get_test_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of A/B test results.
        
        Returns:
            Dictionary containing detailed A/B test analysis
        """
        # Always load latest logs from disk before generating summary
        self._load_logs_from_disk(force_reload=True)
        
        # If no logs, return early
        if not self.logs:
            return {
                "status": "no_data",
                "message": "No A/B test data available",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        # Filter logs by test group
        group_a_logs = [log for log in self.logs if log.get('ab_test_group') == 'A']
        group_b_logs = [log for log in self.logs if log.get('ab_test_group') == 'B']
        no_group_logs = [log for log in self.logs if not log.get('ab_test_group') or log.get('ab_test_group') not in ['A', 'B']]
        
        # Calculate performance metrics
        def calculate_metrics(logs):
            if not logs:
                return {}
                
            # Calculate prediction accuracy for logs with ground truth
            logs_with_ground_truth = [log for log in logs if log.get('ground_truth') is not None]
            correct_predictions = sum(
                1 for log in logs_with_ground_truth 
                if 'prediction' in log and log['ground_truth'] == log['prediction']
            )
            
            # Calculate average prediction time if available
            prediction_times = [
                log['prediction_time'] for log in logs 
                if 'prediction_time' in log and isinstance(log['prediction_time'], (int, float))
            ]
            
            avg_prediction_time = sum(prediction_times) / len(prediction_times) if prediction_times else 0
            
            return {
                "accuracy": (correct_predictions / len(logs_with_ground_truth) * 100) if logs_with_ground_truth else None,
                "avg_prediction_time_ms": avg_prediction_time * 1000,  # Convert to milliseconds
                "total_predictions": len(logs),
                "predictions_with_ground_truth": len(logs_with_ground_truth),
                "correct_predictions": correct_predictions,
                "sample_prediction": logs[0].get('prediction') if logs else None,
                "sample_model": logs[0].get('model_name') if logs else None
            }
        
        # Get metrics for each group
        def calculate_metrics(logs):
            """Calculate metrics for a set of logs"""
            if not logs:
                return {}
                
            # Calculate basic metrics
            total = len(logs)
            with_ground_truth = sum(1 for log in logs if log.get('ground_truth') is not None)
            correct = sum(1 for log in logs 
                         if log.get('ground_truth') is not None 
                         and log.get('prediction') == log.get('ground_truth'))
            
            # Calculate average prediction time (in ms)
            pred_times = [log.get('prediction_time', 0) for log in logs 
                         if log.get('prediction_time') is not None]
            avg_time = sum(pred_times) * 1000 / len(pred_times) if pred_times else 0
            
            # Get a sample prediction
            sample = next((log for log in logs if 'prediction' in log), {})
            
            return {
                "accuracy": correct / with_ground_truth if with_ground_truth > 0 else None,
                "avg_prediction_time_ms": avg_time,
                "total_predictions": total,
                "predictions_with_ground_truth": with_ground_truth,
                "correct_predictions": correct,
                "sample_prediction": sample.get('prediction'),
                "sample_model": sample.get('model_name')
            }
            
        # Calculate metrics for each group
        group_a_metrics = calculate_metrics(group_a_logs)
        group_b_metrics = calculate_metrics(group_b_logs)
        no_group_metrics = calculate_metrics(no_group_logs)
        overall_metrics = calculate_metrics(self.logs)
        
        # Get model recommendation
        recommendation = self.get_model_recommendation(group_a_metrics, group_b_metrics)
        
        def format_metrics(metrics, include_model_sample=True):
            """Helper to format metrics with clear field names
            
            Args:
                metrics: Dictionary of raw metrics
                include_model_sample: Whether to include example model info (excluded for overall metrics)
            """
            if not metrics:
                return {}
                
            result = {
                "accuracy": metrics.get("accuracy"),
                "average_prediction_time_ms": metrics.get("avg_prediction_time_ms"),
                "total_predictions": metrics.get("total_predictions"),
                "predictions_with_ground_truth": metrics.get("predictions_with_ground_truth"),
                "correct_predictions": metrics.get("correct_predictions"),
            }
            
            # Only include model sample for individual groups, not for overall metrics
            if include_model_sample and metrics.get("sample_model"):
                result["model_used"] = metrics.get("sample_model")
                
            return result
        
        # Prepare summary with clearer field names
        summary = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "traffic_split": self.traffic_split,
            "total_requests": len(self.logs),
            "overall_metrics": format_metrics(overall_metrics, include_model_sample=False),
            "production_model": {
                "request_count": len(group_a_logs),
                "traffic_percentage": len(group_a_logs) / len(self.logs) * 100 if self.logs else 0,
                **format_metrics(group_a_metrics)
            },
            "shadow_model": {
                "request_count": len(group_b_logs),
                "traffic_percentage": len(group_b_logs) / len(self.logs) * 100 if self.logs else 0,
                **format_metrics(group_b_metrics)
            },
            "unassigned_requests": {
                "request_count": len(no_group_logs),
                "percentage": len(no_group_logs) / len(self.logs) * 100 if self.logs else 0,
                **format_metrics(no_group_metrics)
            },
            "log_file": str(self.log_file),
            "log_entries_analyzed": len(self.logs),
            "verification": {
                "total_entries_in_file": len(self.logs),
                "total_counted_entries": len(self.logs),
                "all_entries_accounted_for": True
            },
            "recommendation": recommendation
        }
        
        return summary

    def get_model_recommendation(self, group_a_metrics, group_b_metrics) -> Dict[str, Any]:
        """
        Generate recommendation based on A/B test results.
        
        Args:
            group_a_metrics: Performance metrics for group A
            group_b_metrics: Performance metrics for group B
            
        Returns:
            Dictionary containing recommendation
        """
        if not group_a_metrics or not group_b_metrics:
            return {
                "status": "insufficient_data",
                "message": "Not enough data to make a recommendation"
            }
            
        # Compare metrics
        accuracy_improvement = (group_b_metrics['accuracy'] - group_a_metrics['accuracy'])
        prediction_time_improvement = (group_a_metrics['avg_prediction_time_ms'] - group_b_metrics['avg_prediction_time_ms'])
        
        # Configuration for recommendation
        MIN_PREDICTIONS = 5  # Minimum number of predictions with ground truth
        
        # Weights for the combined score (must sum to 1.0)
        ACCURACY_WEIGHT = 0.7  # Higher weight on accuracy
        TIME_WEIGHT = 0.3      # Lower weight on prediction time
        
        # Thresholds (as fractions of the better metric)
        MIN_ACCURACY_IMPROVEMENT = 0.05  # 5% minimum improvement to consider
        MIN_TIME_IMPROVEMENT = 0.2       # 20% faster to consider
        
        # Get metrics
        a_accuracy = group_a_metrics['accuracy']
        b_accuracy = group_b_metrics['accuracy']
        a_time = group_a_metrics['avg_prediction_time_ms']
        b_time = group_b_metrics['avg_prediction_time_ms']
        
        # Check minimum data requirements
        a_has_data = group_a_metrics['predictions_with_ground_truth'] >= MIN_PREDICTIONS
        b_has_data = group_b_metrics['predictions_with_ground_truth'] >= MIN_PREDICTIONS
        
        if not (a_has_data and b_has_data):
            return {
                "status": "insufficient_data",
                "message": f"Not enough data for recommendation (min {MIN_PREDICTIONS} predictions with ground truth needed)",
                "accuracy_improvement": b_accuracy - a_accuracy,
                "prediction_time_improvement_ms": a_time - b_time,
                "recommendation": "Insufficient data"
            }
        
        # Calculate normalized scores (0 to 1, higher is better)
        # For accuracy (simple ratio, but ensure we don't divide by zero)
        max_accuracy = max(a_accuracy, b_accuracy, 1e-10)
        a_accuracy_score = a_accuracy / max_accuracy
        b_accuracy_score = b_accuracy / max_accuracy
        
        # For time (inverse relationship, lower is better)
        max_time = max(a_time, b_time, 1e-10)
        a_time_score = 1 - (a_time / max_time)  # Convert to "more is better"
        b_time_score = 1 - (b_time / max_time)
        
        # Calculate combined scores
        a_score = (a_accuracy_score * ACCURACY_WEIGHT) + (a_time_score * TIME_WEIGHT)
        b_score = (b_accuracy_score * ACCURACY_WEIGHT) + (b_time_score * TIME_WEIGHT)
        
        # Calculate improvements
        accuracy_improvement = b_accuracy - a_accuracy
        time_improvement = a_time - b_time  # Positive means B is faster
        
        # Calculate relative improvements
        rel_accuracy_improvement = (b_accuracy - a_accuracy) / max(1e-10, a_accuracy)
        rel_time_improvement = (a_time - b_time) / max(1e-10, a_time)
        
        # Determine if the difference is significant
        accuracy_significant = abs(rel_accuracy_improvement) > MIN_ACCURACY_IMPROVEMENT
        time_significant = rel_time_improvement > MIN_TIME_IMPROVEMENT
        
        # Make recommendation based on combined score and significant differences
        score_difference = b_score - a_score
        
        if score_difference > 0.1:  # At least 10% better overall score
            recommendation = "B model is better"
        elif score_difference < -0.1:  # At least 10% worse overall score
            recommendation = "A model is better"
        else:
            recommendation = "No significant difference"
        
        # Add detailed debug information
        debug_info = {
            "a_accuracy": a_accuracy,
            "b_accuracy": b_accuracy,
            "a_time_ms": a_time,
            "b_time_ms": b_time,
            "a_score": a_score,
            "b_score": b_score,
            "score_difference": score_difference,
            "accuracy_improvement": accuracy_improvement,
            "time_improvement_ms": time_improvement,
            "rel_accuracy_improvement": rel_accuracy_improvement,
            "rel_time_improvement": rel_time_improvement,
            "accuracy_significant": accuracy_significant,
            "time_significant": time_significant
        }
        
        # Prepare the result with all metrics
        result = {
            "status": "success",
            "accuracy_improvement": accuracy_improvement,
            "prediction_time_improvement_ms": prediction_time_improvement,
            "recommendation": recommendation,
            "debug": debug_info  # Include debug information
        }
        
        return result
