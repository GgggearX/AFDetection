#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Performance Comparison Visualization Tool
- Compares performance metrics across different ECG AF detection models
- Uses complete training histories for learning curve visualization
- Provides tabular performance summaries
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler
import warnings
import argparse
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set_style("whitegrid")

# Define a consistent color palette for models
MODEL_COLORS = {
    'CNN-LSTM': '#3498db',    # Blue
    'DenseNet': '#2ecc71',    # Green
    'RNN': '#e74c3c',         # Red
    'WaveNet': '#9b59b6',     # Purple
    'Transformer': '#f39c12', # Orange
    'Ensemble': '#34495e'     # Dark Gray
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ECG Model Performance Visualization Tool')
    
    parser.add_argument('--output_dir', type=str, default='model_comparison',
                        help='Directory to save visualization outputs')
    parser.add_argument('--include_ensemble', action='store_true', 
                        help='Include ensemble results if available')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')
    parser.add_argument('--models', nargs='+', 
                        default=['CNN-LSTM', 'DenseNet', 'RNN', 'WaveNet', 'Transformer'],
                        help='Models to include in comparison')
    parser.add_argument('--metrics', nargs='+', 
                        default=['val_f1_score', 'val_auc_1', 'val_precision_1', 'val_recall_1'],
                        help='Main metrics to visualize')
    
    return parser.parse_args()

def load_training_histories(model_names, n_folds=5):
    """
    Load complete training histories for all models
    
    Args:
        model_names: List of model names
        n_folds: Number of folds in the cross-validation
        
    Returns:
        Dictionary with model names as keys and lists of training histories as values
    """
    all_histories = {}
    
    for model_name in model_names:
        print(f"Loading training history for {model_name}...")
        model_histories = []
        
        for fold in range(1, n_folds + 1):
            # Try to load training log
            log_file = os.path.join(model_name, 'logs', f'fold_{fold}', f'training_fold_{fold}.csv')
            
            if os.path.exists(log_file):
                try:
                    history_df = pd.read_csv(log_file)
                    if not history_df.empty:
                        print(f"   Loaded training history for {model_name} fold {fold} - {len(history_df)} epochs")
                        model_histories.append({'fold': fold, 'history': history_df})
                    else:
                        print(f"   Warning: {model_name} fold {fold} training history is empty")
                except Exception as e:
                    print(f"   Error: Unable to load {model_name} fold {fold} training history: {e}")
            else:
                print(f"   Warning: {model_name} fold {fold} training log not found: {log_file}")
        
        if model_histories:
            all_histories[model_name] = model_histories
            print(f"Successfully loaded training history for {model_name} with {len(model_histories)} folds")
        else:
            print(f"Warning: No training history found for {model_name}")
    
    return all_histories

def get_average_learning_curves(model_histories, metrics=None):
    """
    Calculate average learning curves across all folds
    
    Args:
        model_histories: List of dictionaries containing training history
        metrics: List of metrics to calculate average curves for
        
    Returns:
        Dictionary with metric names as keys and lists of average values as values
    """
    if not model_histories:
        return {}
    
    # Determine the longest training history
    max_epochs = max(len(fold['history']) for fold in model_histories)
    
    # If no metrics specified, use all columns except 'epoch' and 'lr' from the first history
    if metrics is None:
        metrics = [col for col in model_histories[0]['history'].columns 
                  if col not in ['epoch', 'lr']]
    
    # Initialize result arrays
    avg_curves = {}
    for metric in metrics:
        if any(metric in fold['history'].columns for fold in model_histories):
            avg_curves[metric] = np.zeros(max_epochs)
            count = np.zeros(max_epochs)
            
            # Accumulate values from each fold at each epoch
            for fold in model_histories:
                if metric in fold['history'].columns:
                    history = fold['history']
                    for i, value in enumerate(history[metric]):
                        avg_curves[metric][i] += value
                        count[i] += 1
            
            # Calculate average values
            for i in range(max_epochs):
                if count[i] > 0:
                    avg_curves[metric][i] /= count[i]
    
    return avg_curves

def create_model_summary_df(all_histories):
    """
    Create a summary DataFrame of model performances
    
    Args:
        all_histories: Dictionary with model training histories
        
    Returns:
        DataFrame with model performance summaries
    """
    records = []
    
    for model_name, model_histories in all_histories.items():
        # Calculate final average performance across all folds
        record = {'model': model_name}
        
        # For each metric, calculate the average value of the last epoch as final performance
        for fold in model_histories:
            history_df = fold['history']
            fold_idx = fold['fold']
            
            # Get the value of each metric from the last epoch
            for col in history_df.columns:
                if col not in ['epoch', 'lr']:
                    last_value = history_df[col].iloc[-1]
                    
                    # Accumulate to the record
                    if col in record:
                        record[col].append(last_value)
                    else:
                        record[col] = [last_value]
        
        # Calculate average values for each metric
        for metric in list(record.keys()):
            if metric != 'model' and isinstance(record[metric], list):
                record[metric] = np.mean(record[metric])
        
        records.append(record)
    
    return pd.DataFrame(records)

def plot_metrics_comparison(df, metrics=None, figsize=(14, 10), output_dir=None, dpi=300):
    """
    Plot bar charts comparing metrics across models
    
    Args:
        df: DataFrame with model metrics
        metrics: List of metrics to plot
        figsize: Figure size
        output_dir: Directory to save output
    """
    if metrics is None:
        metrics = ['val_f1_score', 'val_auc_1', 'val_precision_1', 'val_recall_1']
    
    # Ensure we only use metrics that exist in the DataFrame
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) == 0:
        print("Warning: None of the specified metrics are available in the data")
        return None
    
    # Fill in any missing metrics with placeholders to prevent errors
    for metric in metrics:
        if metric not in df.columns:
            df[metric] = np.nan
    
    metric_labels = {
        'val_f1_score': 'F1 Score',
        'val_auc_1': 'AUC',
        'val_precision_1': 'Precision',
        'val_recall_1': 'Recall',
        'val_accuracy': 'Accuracy',
        'val_loss': 'Loss'
    }
    
    # Create figure with subplots - only for available metrics
    num_metrics = min(4, len(available_metrics))
    if num_metrics <= 2:
        fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
        if num_metrics == 1:
            axes = [axes]  # Ensure axes is a list for consistent indexing
    else:
        fig, axes = plt.subplots(2, (num_metrics+1)//2, figsize=figsize)
    
    axes = np.array(axes).flatten()
    
    # Plot each available metric
    for i, metric in enumerate(available_metrics[:4]):  # Limit to first 4 metrics
        ax = axes[i]
        
        # Skip metrics with all NaN values
        if df[metric].isna().all():
            ax.text(0.5, 0.5, f"No data available for {metric_labels.get(metric, metric)}", 
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Sort models by metric value, handling NaN values
        metric_df = df.sort_values(by=metric, ascending=False, na_position='last')
        
        # Filter out rows with NaN for this metric
        metric_df = metric_df[~metric_df[metric].isna()]
        
        if len(metric_df) == 0:
            ax.text(0.5, 0.5, f"No data available for {metric_labels.get(metric, metric)}", 
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Get colors for each model
        colors = [MODEL_COLORS.get(model, '#333333') for model in metric_df['model']]
        
        # Create the bar plot
        bars = ax.bar(metric_df['model'], metric_df[metric], color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Set title and labels
        ax.set_title(f'Model Comparison: {metric_labels.get(metric, metric)}', fontsize=14)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        
        # Adjust y-axis limits to start slightly below 0 and end above max value
        if not metric.endswith('loss'):
            y_max = max(metric_df[metric]) * 1.15
            ax.set_ylim([0, y_max])
        
        # Add minor gridlines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-tick labels if they might overlap
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Hide unused subplots if any
    for i in range(len(available_metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_learning_curves(all_histories, metric, figsize=(14, 8), output_dir=None, dpi=300):
    """
    Plot complete learning curves for all models
    
    Args:
        all_histories: Dictionary with model training histories
        metric: Metric to plot
        figsize: Figure size
        output_dir: Directory to save output
        dpi: Image DPI
    """
    # Metric display names
    metric_labels = {
        'val_f1_score': 'F1 Score',
        'val_auc_1': 'AUC',
        'val_precision_1': 'Precision',
        'val_recall_1': 'Recall',
        'val_accuracy': 'Accuracy',
        'val_loss': 'Loss'
    }
    
    plt.figure(figsize=figsize)
    
    for model_name, model_histories in all_histories.items():
        # Get average curves for the model
        avg_curves = get_average_learning_curves(model_histories, [metric])
        
        if metric in avg_curves:
            # Plot average curve
            color = MODEL_COLORS.get(model_name, '#333333')
            epochs = range(1, len(avg_curves[metric]) + 1)
            plt.plot(epochs, avg_curves[metric], linewidth=2, 
                     label=f"{model_name}", color=color)
    
    # Set title and labels
    plt.title(f'Learning Curves: {metric_labels.get(metric, metric)}', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(metric_labels.get(metric, metric), fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    
    # Set y-axis range (start from 0 for non-loss metrics)
    if not metric.endswith('loss'):
        plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{metric}_learning_curves.png'), 
                   dpi=dpi, bbox_inches='tight')
    
    return plt.gcf()

def plot_learning_curves_grid(all_histories, metrics=None, figsize=(18, 12), output_dir=None, dpi=300):
    """
    Create a grid of learning curve plots for multiple metrics
    
    Args:
        all_histories: Dictionary with model training histories
        metrics: List of metrics to plot
        figsize: Figure size
        output_dir: Directory to save the figure
        dpi: Image DPI
    """
    if metrics is None:
        metrics = ['val_f1_score', 'val_auc_1', 'val_precision_1', 'val_recall_1']
    
    # Metric display names
    metric_labels = {
        'val_f1_score': 'F1 Score',
        'val_auc_1': 'AUC',
        'val_precision_1': 'Precision',
        'val_recall_1': 'Recall',
        'val_accuracy': 'Accuracy',
        'val_loss': 'Loss'
    }
    
    # Find actual available metrics
    available_metrics = []
    for metric in metrics:
        for model_name, model_histories in all_histories.items():
            for fold in model_histories:
                if metric in fold['history'].columns:
                    if metric not in available_metrics:
                        available_metrics.append(metric)
                    break
            if metric in available_metrics:
                break
    
    if not available_metrics:
        print("No available metrics for plotting")
        return None
    
    # Create subplots
    num_metrics = min(4, len(available_metrics))
    if num_metrics <= 2:
        fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
        if num_metrics == 1:
            axes = [axes]  # Ensure axes is a list for consistent indexing
    else:
        fig, axes = plt.subplots(2, (num_metrics+1)//2, figsize=figsize)
    
    axes = np.array(axes).flatten()
    
    # Plot learning curves for each metric
    for i, metric in enumerate(available_metrics[:4]):
        ax = axes[i]
        
        # Plot average curves for each model
        for model_name, model_histories in all_histories.items():
            avg_curves = get_average_learning_curves(model_histories, [metric])
            
            if metric in avg_curves:
                color = MODEL_COLORS.get(model_name, '#333333')
                epochs = range(1, len(avg_curves[metric]) + 1)
                ax.plot(epochs, avg_curves[metric], linewidth=2, 
                         label=f"{model_name}", color=color)
        
        # Set title and labels
        ax.set_title(f'Learning Curves: {metric_labels.get(metric, metric)}', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis range (start from 0 for non-loss metrics)
        if not metric.endswith('loss'):
            ax.set_ylim(bottom=0)
        
        # Add legend on first plot only
        if i == 0:
            ax.legend(loc='best', fontsize=10)
    
    # Hide unused subplots if any
    for i in range(len(available_metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'learning_curves_grid.png'), 
                   dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_metrics_radar(summary_df, metrics=None, figsize=(12, 10), output_dir=None, dpi=300):
    """
    Create a radar chart comparing models across metrics
    
    Args:
        summary_df: DataFrame with model summary data
        metrics: List of metrics to include
        figsize: Figure size
        output_dir: Directory to save output
    """
    if metrics is None:
        metrics = ['val_f1_score', 'val_auc_1', 'val_precision_1', 'val_recall_1', 'val_accuracy']
    
    # Filter metrics that exist in the data
    available_metrics = [m for m in metrics if m in summary_df.columns]
    
    if len(available_metrics) < 3:
        print("Not enough metrics available for radar chart")
        return None
    
    # Map metric names to display labels
    metric_labels = {
        'val_f1_score': 'F1 Score',
        'val_auc_1': 'AUC',
        'val_precision_1': 'Precision',
        'val_recall_1': 'Recall',
        'val_accuracy': 'Accuracy',
        'val_loss': 'Loss (inv)'
    }
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Calculate angles for each metric
    angles = np.linspace(0, 2*np.pi, len(available_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add one more angle for closure
    metric_names = available_metrics + [available_metrics[0]]
    labels = [metric_labels.get(m, m) for m in metric_names]
    
    # Create radar plot
    ax = fig.add_subplot(111, polar=True)
    
    # Scale the data to [0,1] for consistent radar shape
    scaler = MinMaxScaler()
    scaled_data = {}
    
    for metric in available_metrics:
        values = summary_df[metric].values.reshape(-1, 1)
        
        # For loss, invert the values (lower is better)
        if metric.endswith('loss'):
            values = -values
            
        scaled_values = scaler.fit_transform(values).flatten()
        scaled_data[metric] = scaled_values
    
    # Plot each model
    for i, (_, row) in enumerate(summary_df.iterrows()):
        model_name = row['model']
        color = MODEL_COLORS.get(model_name, '#333333')
        
        # Get values for each metric and scale to [0,1]
        values = [scaled_data[metric][i] for metric in available_metrics]
        values += values[:1]  # Close the loop
        
        # Plot model data
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Set labels and formatting
    ax.set_thetagrids(np.degrees(angles), labels)
    ax.set_rlabel_position(0)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_rmax(1.0)
    ax.grid(True)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("Model Performance Comparison", size=16, y=1.08)
    
    # Save figure if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'model_radar_chart.png'), dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_metrics_correlation(summary_df, figsize=(12, 10), output_dir=None, dpi=300):
    """
    Plot correlation heatmap between different metrics
    
    Args:
        summary_df: DataFrame with model metrics
        figsize: Figure size
        output_dir: Directory to save output
    """
    # Select only numeric columns (metrics)
    numeric_df = summary_df.select_dtypes(include=['float64', 'int64'])
    
    # Only proceed if we have enough metrics
    if numeric_df.shape[1] < 2:
        print("Not enough numeric metrics for correlation analysis")
        return None
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, vmin=-1, vmax=1)
    
    plt.title('Correlation Between Performance Metrics', fontsize=16)
    plt.tight_layout()
    
    # Save figure if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'metrics_correlation.png'), dpi=dpi, bbox_inches='tight')
    
    return plt.gcf()

def print_performance_table(summary_df, metrics=None):
    """Print formatted performance table to console"""
    if metrics is None or len(metrics) == 0:
        metrics = [col for col in summary_df.columns if col != 'model']
    
    # Create a copy with more readable metric names
    display_df = summary_df.copy()
    
    # Format metric names for display
    metric_display_names = {
        'val_f1_score': 'F1 Score',
        'val_auc_1': 'AUC',
        'val_precision_1': 'Precision',
        'val_recall_1': 'Recall',
        'val_accuracy': 'Accuracy',
        'val_loss': 'Loss'
    }
    
    # Format values with consistent precision
    for col in display_df.columns:
        if col != 'model' and pd.api.types.is_numeric_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    
    # Print header
    print("\n" + "="*80)
    print("ECG MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Print table with metrics
    headers = ['Model'] + [metric_display_names.get(m, m) for m in metrics]
    header_line = " | ".join(h.ljust(12) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    
    # Print each model's performance
    for _, row in display_df.iterrows():
        model_name = row['model']
        values = [model_name] + [row[m] if m in row else "N/A" for m in metrics]
        print(" | ".join(v.ljust(12) for v in values))
    
    print("="*80 + "\n")

def main():
    """Main function to execute the visualization process"""
    # Parse command line arguments
    args = parse_arguments()
    
    print("Starting ECG Model Performance Visualization Tool...")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use flexible metric name list
    flexible_metrics = [
        'val_f1_score', 'f1_score',
        'val_auc_1', 'auc_1', 
        'val_precision_1', 'precision_1',
        'val_recall_1', 'recall_1',
        'val_accuracy', 'accuracy'
    ]
    
    # Load complete training histories
    print("Loading model complete training histories...")
    all_histories = load_training_histories(args.models)
    
    if not all_histories:
        print("Error: No model data found")
        return
    
    print(f"Successfully loaded data for {len(all_histories)} models")
    
    # Create model summary DataFrame
    summary_df = create_model_summary_df(all_histories)
    
    if summary_df.empty:
        print("Error: Unable to create summary")
        return
    
    # Identify available metrics
    available_metrics = [col for col in summary_df.columns if col != 'model']
    
    # Update top metrics to use available ones
    top_metrics = [m for m in flexible_metrics if m in available_metrics]
    
    if not top_metrics:
        print("Warning: No main metrics found, using all available metrics")
        top_metrics = available_metrics
    
    # Print performance table
    print_performance_table(summary_df, top_metrics)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Bar chart comparison of top metrics
    print("Creating metrics comparison bar charts...")
    plot_metrics_comparison(summary_df, top_metrics, output_dir=args.output_dir, dpi=args.dpi)
    
    # 2. Learning curves for each metric
    for metric in top_metrics:
        print(f"Creating {metric} learning curves...")
        plot_learning_curves(all_histories, metric, output_dir=args.output_dir, dpi=args.dpi)
    
    # 3. Learning curves grid
    print("Creating learning curves grid...")
    plot_learning_curves_grid(all_histories, top_metrics, output_dir=args.output_dir, dpi=args.dpi)
    
    # 4. Radar chart
    print("Creating performance radar chart...")
    plot_metrics_radar(summary_df, top_metrics, output_dir=args.output_dir, dpi=args.dpi)
    
    # 5. Metrics correlation heatmap
    if len(top_metrics) > 1:
        print("Creating metrics correlation heatmap...")
        plot_metrics_correlation(summary_df, output_dir=args.output_dir, dpi=args.dpi)
    
    print(f"\nAll visualizations saved to '{args.output_dir}'")
    print("Visualization process complete!")

if __name__ == "__main__":
    main()