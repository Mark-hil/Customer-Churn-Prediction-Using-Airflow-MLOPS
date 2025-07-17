"""Exploratory Data Analysis module for the churn prediction pipeline."""
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
PLOTS_DIR = Path("/opt/airflow/data/plots")

def setup_plotting() -> None:
    """Set up plotting style and directories."""
    # Create plots directory if it doesn't exist
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn')
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12

def plot_target_distribution(df: pd.DataFrame, target_col: str = 'Churn', save: bool = True) -> Optional[Path]:
    """
    Plot the distribution of the target variable.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        save: Whether to save the plot
        
    Returns:
        Path to the saved plot or None if not saved
    """
    plt.figure(figsize=(10, 6))
    
    # Count plot for churn distribution
    ax = sns.countplot(data=df, x=target_col, 
                      order=df[target_col].value_counts().index,
                      palette='viridis')
    
    # Add percentages on top of bars
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height()/total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height() + 10
        ax.text(x, y, percentage, ha='center')
    
    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    
    if save:
        plot_path = PLOTS_DIR / 'churn_distribution.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved churn distribution plot to {plot_path}")
        return plot_path
    
    plt.show()
    return None

def plot_numerical_distributions(df: pd.DataFrame, numerical_cols: List[str], target_col: str = 'Churn', 
                               save: bool = True, sample_size: int = 1000) -> Optional[Path]:
    """
    Plot distributions of numerical features by target class.
    
    Args:
        df: Input DataFrame
        numerical_cols: List of numerical column names
        target_col: Name of the target column
        save: Whether to save the plot
        sample_size: Number of samples to use for faster plotting
        
    Returns:
        Path to the saved plot or None if not saved
    """
    logger.info("Plotting numerical distributions...")
    start_time = time.time()
    
    # Limit number of columns to plot for better performance
    max_cols = 10  # Limit to top 10 numerical features
    if len(numerical_cols) > max_cols:
        logger.warning(f"Too many numerical features ({len(numerical_cols)}). Plotting top {max_cols}.")
        numerical_cols = numerical_cols[:max_cols]
    
    # Sample data for faster plotting
    plot_df = df.sample(min(sample_size, len(df)))
    
    n_cols = 2
    n_rows = (len(numerical_cols) + 1) // n_cols
    
    # Create figure with appropriate size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols):
        logger.debug(f"Plotting distribution for {col}...")
        ax = axes[idx]
        
        try:
            # Plot KDE for each class - use sampled data
            for cls in plot_df[target_col].unique():
                sns.kdeplot(plot_df[plot_df[target_col] == cls][col], 
                           ax=ax, label=f'{target_col} = {cls}',
                           alpha=0.7, linewidth=1.5)
            
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.legend()
            
        except Exception as e:
            logger.warning(f"Could not plot {col}: {str(e)}")
            ax.set_visible(False)
    
    # Hide any remaining empty subplots
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save:
        plot_path = PLOTS_DIR / 'numerical_distributions.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=100, quality=85)  # Optimize image size
        plt.close()
        logger.info(f"Saved numerical distributions plot to {plot_path} (took {time.time() - start_time:.1f}s)")
        return plot_path
    
    plt.close()
    return None

def plot_categorical_distributions(df: pd.DataFrame, categorical_cols: List[str], 
                                 target_col: str = 'Churn', save: bool = True, 
                                 max_categories: int = 10) -> Optional[Path]:
    """
    Plot distributions of categorical features by target class.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
        target_col: Name of the target column
        save: Whether to save the plot
        max_categories: Maximum number of categories to show per feature
        
    Returns:
        Path to the saved plot or None if not saved
    """
    logger.info("Plotting categorical distributions...")
    start_time = time.time()
    
    # Limit number of columns to plot for better performance
    max_cols = 8  # Limit to top 8 categorical features
    if len(categorical_cols) > max_cols:
        logger.warning(f"Too many categorical features ({len(categorical_cols)}). Plotting top {max_cols}.")
        categorical_cols = categorical_cols[:max_cols]
    
    n_cols = 2
    n_rows = (len(categorical_cols) + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.ravel()
    
    for idx, col in enumerate(categorical_cols):
        logger.debug(f"Plotting distribution for {col}...")
        ax = axes[idx]
        
        try:
            # Limit number of categories for readability
            value_counts = df[col].value_counts()
            if len(value_counts) > max_categories:
                # Keep top N-1 categories and group the rest as 'Other'
                top_categories = value_counts.nlargest(max_categories - 1).index
                plot_df = df.copy()
                plot_df[col] = np.where(plot_df[col].isin(top_categories), 
                                      plot_df[col], 'Other')
            else:
                plot_df = df
            
            # Calculate percentages
            crosstab = pd.crosstab(plot_df[col], plot_df[target_col], normalize='index') * 100
            
            # Sort by churn rate for better visualization
            crosstab = crosstab.sort_values(by=crosstab.columns[1], ascending=False)
            
            # Plot stacked bar chart with better colors
            colors = ['#4c72b0', '#dd8452']  # Blue and orange
            crosstab.plot(kind='bar', stacked=True, ax=ax, 
                         color=colors, edgecolor='none', width=0.8)
            
            ax.set_title(f'Churn Rate by {col}', pad=10)
            ax.set_xlabel(col, labelpad=8)
            ax.set_ylabel('Percentage (%)', labelpad=8)
            ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), 
                     loc='upper left', frameon=False)
            
            # Add percentage labels (only for large enough segments)
            for p in ax.patches:
                height = p.get_height()
                if height > 5:  # Only show label if segment is tall enough
                    ax.text(p.get_x() + p.get_width()/2., 
                           p.get_y() + height/2.,
                           f'{height:.0f}%', 
                           ha='center',
                           va='center',
                           color='white' if height > 30 else 'black',
                           fontsize=8)
            
            # Rotate x-axis labels if needed
            if len(ax.get_xticklabels()) > 3:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
        except Exception as e:
            logger.warning(f"Could not plot {col}: {str(e)}")
            ax.set_visible(False)
    
    # Hide any remaining empty subplots
    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save:
        plot_path = PLOTS_DIR / 'categorical_distributions.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=100, quality=85)
        plt.close()
        logger.info(f"Saved categorical distributions to {plot_path} (took {time.time() - start_time:.1f}s)")
        return plot_path
    
    plt.close()
    return None

def plot_correlation_heatmap(df: pd.DataFrame, numerical_cols: List[str], 
                           target_col: str = 'Churn', save: bool = True) -> Optional[Path]:
    """
    Plot correlation heatmap for numerical features.
    
    Args:
        df: Input DataFrame
        numerical_cols: List of numerical column names
        target_col: Name of the target column
        save: Whether to save the plot
        
    Returns:
        Path to the saved plot or None if not saved
    """
    logger.info("Generating correlation heatmap...")
    start_time = time.time()
    
    try:
        # Limit number of features for better readability
        max_features = 15
        if len(numerical_cols) > max_features:
            logger.info(f"Too many features for correlation heatmap. Using top {max_features}.")
            # Select features with highest absolute correlation to target
            corr_with_target = df[numerical_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
            selected_cols = corr_with_target.index[1:max_features+1].tolist()  # Exclude target itself
        else:
            selected_cols = numerical_cols
        
        # Calculate correlation matrix
        corr = df[selected_cols + [target_col]].corr()
        
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Set up the matplotlib figure
        plt.figure(figsize=(14, 10))
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, 
                   mask=mask, 
                   cmap=cmap, 
                   vmin=-1, vmax=1, center=0,
                   square=True, 
                   linewidths=0.5, 
                   cbar_kws={"shrink": 0.8, "label": "Correlation"}, 
                   annot=True, 
                   fmt=".2f", 
                   annot_kws={"size": 8})
        
        plt.title('Correlation Heatmap of Numerical Features', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save:
            plot_path = PLOTS_DIR / 'correlation_heatmap.png'
            plt.savefig(plot_path, dpi=100, bbox_inches='tight', quality=85)
            plt.close()
            logger.info(f"Saved correlation heatmap to {plot_path} (took {time.time() - start_time:.1f}s)")
            return plot_path
        
        return None
        
    except Exception as e:
        logger.error(f"Error generating correlation heatmap: {str(e)}")
        return None
    finally:
        plt.close()

def generate_eda_report(df: pd.DataFrame, target_col: str = 'Churn', save: bool = True) -> Dict[str, Path]:
    """
    Generate a comprehensive EDA report with multiple visualizations.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        save: Whether to save the plots
        
    Returns:
        Dictionary mapping plot names to their file paths
    """
    start_time = time.time()
    logger.info("Starting EDA report generation...")
    plots = {}
    
    try:
        # Create plots directory if it doesn't exist
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving plots to: {PLOTS_DIR}")
        
        # 1. Basic data info
        logger.info("Analyzing dataset structure...")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {', '.join(df.columns)}")
        
        # 2. Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Remove target column from feature lists if present
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
            
        logger.info(f"Found {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features")
        
        # 3. Generate plots with progress tracking
        plot_functions = [
            ("target_distribution", lambda: plot_target_distribution(df, target_col, save=save) 
             if target_col in df.columns else None),
            ("numerical_distributions", lambda: plot_numerical_distributions(df, numerical_cols, target_col, save=save) 
             if numerical_cols else None),
            ("categorical_distributions", lambda: plot_categorical_distributions(df, categorical_cols, target_col, save=save) 
             if categorical_cols else None),
            ("correlation_heatmap", lambda: plot_correlation_heatmap(df, numerical_cols, target_col, save=save) 
             if len(numerical_cols) > 1 else None)
        ]
        
        # Execute each plot function with progress tracking
        for plot_name, plot_func in plot_functions:
            try:
                logger.info(f"Generating {plot_name}...")
                plot_start = time.time()
                plot_path = plot_func()
                if plot_path:
                    plots[plot_name] = plot_path
                    logger.info(f"  Generated {plot_name} in {time.time() - plot_start:.1f}s")
                else:
                    logger.warning(f"  Could not generate {plot_name}")
            except Exception as e:
                logger.error(f"Error generating {plot_name}: {str(e)}", exc_info=True)
        
        # 4. Generate summary statistics
        if save:
            stats_path = PLOTS_DIR / 'summary_statistics.txt'
            with open(stats_path, 'w') as f:
                f.write("=== Dataset Summary ===\n")
                f.write(f"Total samples: {len(df)}\n")
                f.write(f"Features: {len(df.columns)}\n")
                f.write(f"Numerical features: {len(numerical_cols)}\n")
                f.write(f"Categorical features: {len(categorical_cols)}\n\n")
                
                f.write("=== Missing Values ===\n")
                missing = df.isnull().sum()
                f.write(missing[missing > 0].to_string() + "\n\n")
                
                if target_col in df.columns:
                    f.write("=== Target Distribution ===\n")
                    f.write(str(df[target_col].value_counts(normalize=True).mul(100).round(1).astype(str) + '%') + "\n")
            
            plots['summary_statistics'] = stats_path
            logger.info(f"Saved summary statistics to {stats_path}")
        
        # 5. Final report
        total_time = time.time() - start_time
        logger.info(f"EDA report generation completed in {total_time:.1f} seconds")
        logger.info(f"Generated {len(plots)} plots and reports")
        
        return plots
        
    except Exception as e:
        logger.error(f"Error in generate_eda_report: {str(e)}", exc_info=True)
        raise

def run_eda(df: pd.DataFrame) -> Dict[str, str]:
    """
    Run exploratory data analysis and generate visualizations.
    
    Args:
        df: Input DataFrame for analysis
        
    Returns:
        Dictionary mapping plot names to their file paths
    """
    logger.info("Starting EDA process...")
    
    try:
        # Generate EDA report
        report = generate_eda_report(df, save=True)
        
        # Log completion
        logger.info(f"EDA completed. Generated {len(report)} plots.")
        
        # Return report paths (optional: can be used by downstream tasks)
        return report
        
    except Exception as e:
        logger.error(f"Error during EDA: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    from A00_data_understanding import load_data, DATA_DIR
    
    # Load data
    df = load_data()
    
    # Generate and save EDA report
    report = generate_eda_report(df, save=True)
    print(f"Generated EDA report with {len(report)} plots")
    for name, path in report.items():
        print(f"- {name}: {path}")
