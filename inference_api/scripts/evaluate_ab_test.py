#!/usr/bin/env python3
"""
A/B Test Evaluation Script

This script analyzes the A/B test logs and generates performance metrics
and statistical significance tests.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from scipy import stats
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ABTestEvaluator:
    def __init__(self, log_file: str):
        """Initialize the evaluator with the path to the log file."""
        self.log_file = Path(log_file)
        self.logs = self._load_logs()
        
    def _load_logs(self) -> List[Dict]:
        """Load and parse log entries."""
        if not self.log_file.exists():
            logger.error(f"Log file not found: {self.log_file}")
            return []
            
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing log file: {e}")
            return []
            
    def get_metrics(self) -> Dict:
        """Calculate comprehensive A/B test metrics."""
        if not self.logs:
            return {"error": "No log data available"}
            
        df = pd.DataFrame(self.logs)
        
        # Basic metrics
        metrics = {
            "status": "success",
            "total_requests": len(df),
            "start_time": df['timestamp'].min() if not df.empty else None,
            "end_time": df['timestamp'].max() if not df.empty else None,
            "groups": {}
        }
        
        # Calculate metrics for each group
        for group in ['A', 'B']:
            group_data = df[df['ab_test_group'] == group]
            metrics["groups"][group] = self._calculate_group_metrics(group_data)
            
        # Calculate statistical significance if we have enough data
        if not df.empty:
            metrics["significance"] = self._calculate_significance(df)
            
        return metrics
        
    def _calculate_group_metrics(self, group_data: pd.DataFrame) -> Dict:
        """Calculate metrics for a specific test group."""
        if group_data.empty:
            return {
                "request_count": 0,
                "accuracy": None,
                "avg_prediction_time_ms": None,
                "predictions_with_ground_truth": 0,
                "correct_predictions": 0
            }
            
        with_truth = group_data.dropna(subset=['ground_truth'])
        correct = (with_truth['prediction'] == with_truth['ground_truth']).sum()
        
        return {
            "request_count": len(group_data),
            "accuracy": float(correct / len(with_truth)) if len(with_truth) > 0 else None,
            "avg_prediction_time_ms": float(group_data['prediction_time'].mean()) if 'prediction_time' in group_data else None,
            "predictions_with_ground_truth": int(len(with_truth)),
            "correct_predictions": int(correct)
        }
        
    def _calculate_significance(self, df: pd.DataFrame) -> Dict:
        """Calculate statistical significance of results between groups."""
        try:
            # Only consider predictions with ground truth
            valid_data = df.dropna(subset=['ground_truth'])
            
            # Group data
            groups = {}
            for group in ['A', 'B']:
                group_data = valid_data[valid_data['ab_test_group'] == group]
                if len(group_data) == 0:
                    continue
                    
                correct = (group_data['prediction'] == group_data['ground_truth']).sum()
                total = len(group_data)
                groups[group] = {
                    'correct': correct,
                    'total': total,
                    'accuracy': correct / total if total > 0 else 0
                }
            
            # Need at least two groups with data
            if len(groups) < 2:
                return {"status": "insufficient_data", "message": "Need at least two groups with data"}
                
            # Extract data for chi-squared test
            observed = [groups[g]['correct'] for g in groups]
            total = [groups[g]['total'] for g in groups]
            
            # Expected values under null hypothesis (no difference between groups)
            total_correct = sum(observed)
            total_samples = sum(total)
            expected = [t * (total_correct / total_samples) for t in total]
            
            # Perform chi-squared test
            chi2, p_value = stats.chisquare(observed, f_exp=expected)
            
            return {
                "test": "chi_squared",
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "confidence_level": 0.95,
                "interpretation": "Significant difference between groups" if p_value < 0.05 
                                 else "No significant difference between groups"
            }
            
        except Exception as e:
            logger.error(f"Error calculating significance: {e}")
            return {"error": str(e)}

def save_report(metrics: Dict, output_file: Optional[str] = None) -> None:
    """Save metrics to a JSON file."""
    if not output_file:
        output_file = f"ab_test_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        
    try:
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Report saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving report: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate A/B test results')
    parser.add_argument('--log-file', default='logs/ab_test_logs/weekly_ab_test_logs.json',
                       help='Path to the A/B test log file')
    parser.add_argument('--output', help='Output file for the report (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    evaluator = ABTestEvaluator(args.log_file)
    metrics = evaluator.get_metrics()
    
    # Print metrics to console
    print(json.dumps(metrics, indent=2))
    
    # Save report if output file is specified
    if args.output:
        save_report(metrics, args.output)
