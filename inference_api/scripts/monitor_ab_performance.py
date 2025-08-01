#!/usr/bin/env python3
"""
A/B Test Performance Monitor with Automatic Rollback

This script monitors the performance of A/B test models and can trigger
a rollback if performance falls below defined thresholds.
"""
import json
import logging
import logging.handlers
import os
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import requests
import yaml
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ab_monitoring.log")
    ]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "monitoring": {
        "check_interval_seconds": 300,  # 5 minutes
        "evaluation_window_hours": 24,  # Evaluate last 24 hours of data
        "min_samples_for_evaluation": 50,
        "performance_thresholds": {
            "min_accuracy": 0.7,  # 70% minimum accuracy
            "max_drop_from_baseline": 0.05,  # 5% max drop from baseline
            "max_prediction_time_ms": 500,  # 500ms max prediction time
            "consecutive_failures_before_rollback": 3
        },
        "api_endpoints": {
            "summary": "http://localhost:8000/api/v1/management/ab-test/summary",
            "rollback": "http://localhost:8000/api/v1/management/ab-test/rollback"
        }
    }
}

class NotificationManager:
    """Handles sending notifications via email and Slack."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize notification manager with configuration."""
        self.config = config.get("notifications", {})
        self.email_config = self.config.get("email", {})
        self.slack_config = self.config.get("slack", {})
    
    def send_email(self, subject: str, message: str) -> bool:
        """Send an email notification."""
        if not self.email_config.get("enabled", False):
            return False
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config["from_email"]
            msg['To'] = ", ".join(self.email_config["to_emails"])
            msg['Subject'] = f"[A/B Test Monitor] {subject}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            with smtplib.SMTP(
                self.email_config["smtp_server"], 
                self.email_config["smtp_port"]
            ) as server:
                if self.email_config.get("smtp_username") and self.email_config.get("smtp_password"):
                    server.starttls()
                    server.login(
                        self.email_config["smtp_username"], 
                        self.email_config["smtp_password"]
                    )
                server.send_message(msg)
                logger.info(f"Email notification sent: {subject}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def send_slack(self, message: str) -> bool:
        """Send a Slack notification."""
        if not self.slack_config.get("enabled", False):
            return False
            
        try:
            payload = {
                "text": message,
                "username": self.slack_config.get("username", "A/B Test Monitor"),
                "channel": self.slack_config["channel"]
            }
            
            response = requests.post(
                self.slack_config["webhook_url"],
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            logger.info("Slack notification sent")
            return True
            
        except RequestException as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def send_alert(self, subject: str, message: str) -> None:
        """Send alerts through all configured channels."""
        self.send_email(subject, message)
        self.send_slack(f"*{subject}*\n{message}")


class ABTestMonitor:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the monitor with configuration."""
        self.config = self._load_config(config_path)
        self.notifier = NotificationManager(self.config)
        self.consecutive_failures = 0
        self.baseline_metrics = None
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging based on the config file."""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO").upper())
        
        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Configure file handler if path is specified
        file_handler = None
        if log_config.get("file"):
            log_file = Path(log_config["file"])
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=log_config.get("max_size_mb", 10) * 1024 * 1024,  # Convert MB to bytes
                backupCount=log_config.get("backup_count", 5)
            )
            file_handler.setLevel(log_level)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        if file_handler:
            file_handler.setFormatter(formatter)
        
        # Add handlers to the root logger
        logging.basicConfig(
            level=log_level,
            handlers=[h for h in [console_handler, file_handler] if h is not None]
        )
        
        # Set log level for requests and urllib3 to reduce noise
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            # Merge with defaults
            return {**DEFAULT_CONFIG, **config}
        return DEFAULT_CONFIG
    
    def get_performance_metrics(self) -> Optional[Dict[str, Any]]:
        """Fetch current performance metrics from the API."""
        try:
            response = requests.get(
                self.config["monitoring"]["api_endpoints"]["summary"]
            )
            response.raise_for_status()
            data = response.json()
            
            # Log the received metrics for debugging
            logger.debug(f"Received metrics: {json.dumps(data, indent=2)}")
            
            # Ensure we have the expected structure
            if not all(key in data for key in ["production_model", "shadow_model"]):
                logger.error("Unexpected metrics structure received from API")
                return None
                
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            return None
        except RequestException as e:
            logger.error(f"Error fetching performance metrics: {e}")
            return None
    
    def check_performance(self, metrics: Dict[str, Any]) -> tuple[bool, str]:
        """
        Check if performance meets the defined thresholds.
        
        Args:
            metrics: Dictionary containing performance metrics
            
        Returns:
            tuple: (is_healthy: bool, reason: str)
        """
        thresholds = self.config["monitoring"]["performance_thresholds"]
        
        # Check if we have enough data
        min_samples = self.config["monitoring"].get("min_samples_for_evaluation", 5)  # Default to 5 if not set
        if metrics["production_model"]["predictions_with_ground_truth"] < min_samples:
            reason = f"Not enough samples for evaluation (have {metrics['production_model']['predictions_with_ground_truth']}, need {min_samples})"
            logger.info(reason)
            return True, reason
            
        # Check accuracy thresholds
        production_accuracy = metrics["production_model"]["accuracy"]
        
        # Check if accuracy is above minimum threshold
        if production_accuracy < thresholds["min_accuracy"]:
            reason = (
                f"Production model accuracy {production_accuracy:.2f} is below minimum threshold "
                f"{thresholds['min_accuracy']}"
            )
            logger.warning(reason)
            return False, reason
            
        # Check if the drop from baseline is acceptable
        if self.baseline_metrics:
            baseline_accuracy = self.baseline_metrics["production_model"]["accuracy"]
            accuracy_drop = baseline_accuracy - production_accuracy
            
            if accuracy_drop > thresholds["max_drop_from_baseline"]:
                reason = (
                    f"Accuracy drop {accuracy_drop:.2f} exceeds maximum allowed drop "
                    f"{thresholds['max_drop_from_baseline']}"
                )
                logger.warning(reason)
                return False, reason
        
        # Check prediction time
        production_avg_time = metrics["production_model"]["average_prediction_time_ms"]
        if production_avg_time > thresholds["max_prediction_time_ms"]:
            reason = (
                f"Average prediction time {production_avg_time:.2f}ms exceeds maximum "
                f"{thresholds['max_prediction_time_ms']}ms"
            )
            logger.warning(reason)
            return False, reason
            
        return True, "All performance checks passed"
    
    def trigger_rollback(self, reason: str) -> bool:
        """
        Trigger a rollback to the previous model version.
        
        Args:
            reason: Detailed reason for the rollback
            
        Returns:
            bool: True if rollback was successfully triggered, False otherwise
        """
        rollback_reason = f"Performance degradation detected: {reason}"
        
        try:
            logger.warning(f"Attempting to trigger rollback. Reason: {rollback_reason}")
            
            response = requests.post(
                self.config["monitoring"]["api_endpoints"]["rollback"],
                json={
                    "reason": rollback_reason,
                    "force": True  # Force rollback even if models are the same
                },
                timeout=30  # Give it more time for the rollback to complete
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Successfully triggered rollback: {result.get('message', 'No message')}")
            
            # Send notification
            self.notifier.send_alert(
                "Model Rollback Triggered",
                f"A rollback has been triggered due to performance issues.\n"
                f"Reason: {rollback_reason}\n"
                f"Response: {result}"
            )
            
            return True
            
        except RequestException as e:
            error_msg = f"Error triggering rollback: {str(e)}"
            logger.error(error_msg)
            
            # Send notification about the failure
            self.notifier.send_alert(
                "Rollback Failed",
                f"Failed to trigger rollback: {error_msg}\n"
                f"Reason: {rollback_reason}"
            )
            
            return False
    
    def run(self):
        """Run the monitoring loop."""
        logger.info("Starting A/B test performance monitor")
        self.notifier.send_alert(
            "A/B Test Monitor Started",
            "The A/B test performance monitor has been started successfully."
        )
        
        try:
            while True:
                try:
                    # Get current metrics
                    metrics = self.get_performance_metrics()
                    
                    if metrics:
                        # If this is the first run, set baseline metrics
                        if self.baseline_metrics is None:
                            self.baseline_metrics = metrics
                            logger.info("Baseline metrics set")
                            self.notifier.send_alert(
                                "Baseline Metrics Set",
                                f"Initial performance metrics have been set as baseline.\n"
                                f"Production Model: {metrics.get('production_model', {})}\n"
                                f"Shadow Model: {metrics.get('shadow_model', {})}"
                            )
                        
                        # Check performance
                        is_healthy, reason = self.check_performance(metrics)
                        
                        if is_healthy:
                            # Reset consecutive failures on success
                            if self.consecutive_failures > 0:
                                logger.info("Performance has returned to normal")
                                self.notifier.send_alert(
                                    "Performance Back to Normal",
                                    "The model performance has returned to normal levels.\n"
                                    f"Current metrics: {json.dumps(metrics, indent=2)}"
                                )
                            self.consecutive_failures = 0
                            logger.info("Performance check passed")
                        else:
                            self.consecutive_failures += 1
                            failures_allowed = self.config["monitoring"]["performance_thresholds"]["consecutive_failures_before_rollback"]
                            
                            logger.warning(
                                f"Performance check failed ({self.consecutive_failures}/"
                                f"{failures_allowed}): {reason}"
                            )
                            
                            # Send alert on first failure
                            if self.consecutive_failures == 1:
                                self.notifier.send_alert(
                                    "Performance Issue Detected",
                                    f"A performance issue has been detected.\n"
                                    f"Issue: {reason}\n"
                                    f"Consecutive failures: {self.consecutive_failures}/{failures_allowed}\n"
                                    f"Current metrics: {json.dumps(metrics, indent=2)}"
                                )
                            
                            # Check if we've exceeded the consecutive failure threshold
                            if self.consecutive_failures >= failures_allowed:
                                logger.error("Performance degradation detected. Triggering rollback...")
                                if self.trigger_rollback(reason):
                                    logger.info("Rollback completed successfully")
                                    # Don't break, keep monitoring after rollback
                                else:
                                    # If rollback fails, keep trying but don't spam
                                    time.sleep(60)  # Wait a bit before retrying
                
                except RequestException as e:
                    logger.error(f"Error fetching metrics: {e}")
                    time.sleep(30)  # Shorter wait on API errors
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error in monitoring loop: {e}", exc_info=True)
                    self.notifier.send_alert(
                        "Monitoring Error",
                        f"An unexpected error occurred in the monitoring loop: {str(e)}"
                    )
                    time.sleep(60)  # Wait a bit before retrying
                    continue
                
                # Wait before next check
                time.sleep(self.config["monitoring"]["check_interval_seconds"])
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            self.notifier.send_alert(
                "A/B Test Monitor Stopped",
                "The A/B test performance monitor has been stopped by the user."
            )
        except Exception as e:
            error_msg = f"Critical error in monitoring: {e}"
            logger.error(error_msg, exc_info=True)
            self.notifier.send_alert(
                "Monitoring Crashed",
                f"The A/B test performance monitor has crashed with error: {error_msg}"
            )
            raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="A/B Test Performance Monitor")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    monitor = ABTestMonitor(args.config)
    monitor.run()
