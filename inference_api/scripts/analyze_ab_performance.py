#!/usr/bin/env python3
"""A/B Test Performance Analysis Script"""
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging and plotting
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
sns.set(style="whitegrid")

class ABTestAnalyzer:
    def __init__(self, log_file: Union[str, Path], output_dir: str = "reports"):
        self.log_file = Path(log_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess log data."""
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            df = pd.DataFrame(logs)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.sort_values('timestamp', inplace=True)
                
            return df
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return pd.DataFrame()

    def analyze(self) -> Dict:
        """Run complete analysis and generate report."""
        if self.df.empty:
            return {"error": "No data available for analysis"}

        metrics = {
            "summary": self._get_summary_metrics(),
            "by_group": self._get_group_metrics(),
            "statistical_tests": self._run_stat_tests(),
        }
        
        self._generate_plots()
        self._save_report(metrics)
        return metrics

    def _get_summary_metrics(self) -> Dict:
        """Calculate overall summary metrics."""
        df = self.df
        metrics = {
            "total_requests": len(df),
            "start_time": df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
            "end_time": df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None,
        }
        
        if 'ab_test_group' in df.columns:
            metrics.update({
                "group_distribution": df['ab_test_group'].value_counts().to_dict(),
                "requests_with_ground_truth": df['ground_truth'].notna().sum()
            })
            
        return metrics

    def _get_group_metrics(self) -> Dict:
        """Calculate metrics by A/B test group."""
        if 'ab_test_group' not in self.df.columns:
            return {}
            
        metrics = {}
        for group, group_df in self.df.groupby('ab_test_group'):
            group_name = str(group) if not pd.isna(group) else "unknown"
            metrics[group_name] = self._calculate_metrics(group_df)
            
        return metrics

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate performance metrics for a dataframe subset."""
        metrics = {
            "request_count": len(df),
            "predictions_with_ground_truth": df['ground_truth'].notna().sum() if 'ground_truth' in df.columns else 0,
        }
        
        if 'ground_truth' in df.columns and 'prediction' in df.columns:
            correct = (df['prediction'] == df['ground_truth']).sum()
            total = df['ground_truth'].notna().sum()
            metrics.update({
                "accuracy": float(correct / total) if total > 0 else None,
                "correct_predictions": int(correct)
            })
            
        if 'prediction_time' in df.columns:
            metrics["avg_prediction_time_ms"] = float(df['prediction_time'].mean() * 1000)
            
        return metrics

    def _run_stat_tests(self) -> Dict:
        """Run statistical tests between groups."""
        if 'ab_test_group' not in self.df.columns or 'ground_truth' not in self.df.columns:
            return {}
            
        df = self.df.dropna(subset=['ground_truth']).copy()
        df['correct'] = df['prediction'] == df['ground_truth']
        
        results = {}
        
        # Chi-squared test for accuracy
        try:
            contingency = pd.crosstab(df['ab_test_group'], df['correct'])
            chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
            results['accuracy_chi2'] = {
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'interpretation': 'Significant difference in accuracy' if p_value < 0.05 else 'No significant difference'
            }
        except Exception as e:
            logging.warning(f"Chi-squared test failed: {e}")
            
        return results

    def _generate_plots(self):
        """Generate and save visualization plots."""
        try:
            # Accuracy over time
            if 'timestamp' in self.df.columns and 'ab_test_group' in self.df.columns:
                plt.figure(figsize=(12, 6))
                sns.lineplot(
                    data=self.df,
                    x='timestamp',
                    y='prediction',
                    hue='ab_test_group',
                    errorbar=None
                )
                plt.title('Prediction Distribution Over Time')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'prediction_distribution.png')
                plt.close()
                
        except Exception as e:
            logging.warning(f"Error generating plots: {e}")

    def _convert_for_serialization(self, obj: Any) -> Any:
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_for_serialization(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_for_serialization(item) for item in obj]
        return obj
        
    def _save_report(self, metrics: Dict) -> None:
        """Save analysis report to JSON file."""
        report_path = self.output_dir / 'ab_test_report.json'
        serializable_metrics = self._convert_for_serialization(metrics)
        
        with open(report_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2, default=str)
            
        logging.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze A/B test performance from log files.")
    parser.add_argument("--log-file", type=str, required=True, help="Path to the A/B test log file")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory to save reports and plots")
    
    args = parser.parse_args()
    
    analyzer = ABTestAnalyzer(args.log_file, Path(args.output_dir))
    results = analyzer.analyze()
    
    # Convert results to serializable format before printing
    serializable_results = analyzer._convert_for_serialization(results)
    
    # Print a summary of results
    print(json.dumps(serializable_results, indent=2, default=str))
