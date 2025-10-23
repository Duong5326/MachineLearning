# Đánh giá mô hình
"""
Module for model evaluation functions.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                           accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, silhouette_score)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional

def evaluate_regression_model(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Evaluate regression model performance.
    
    Args:
        y_true (pd.Series): True target values
        y_pred (pd.Series): Predicted target values
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    return metrics

def evaluate_classification_model(y_true: pd.Series, y_pred: pd.Series, 
                                average: str = 'weighted') -> Dict[str, Any]:
    """
    Evaluate classification model performance.
    
    Args:
        y_true (pd.Series): True target values
        y_pred (pd.Series): Predicted target values
        average (str): Averaging strategy for multiclass metrics
        
    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        'conf_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, zero_division=0)
    }
    
    return metrics

def evaluate_clustering_model(X: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluate clustering model performance.
    
    Args:
        X (pd.DataFrame): Feature data
        labels (np.ndarray): Cluster labels
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    # Filter out noise points (labeled as -1 in DBSCAN)
    valid_indices = labels != -1
    
    if sum(valid_indices) < 2:
        return {'silhouette': 0.0}  # Cannot compute silhouette score with less than 2 samples
    
    metrics = {
        'silhouette': silhouette_score(X.loc[valid_indices], labels[valid_indices]) if len(set(labels[valid_indices])) > 1 else 0.0
    }
    
    return metrics

def plot_residuals(y_true: pd.Series, y_pred: pd.Series, title: str = 'Residual Plot') -> None:
    """
    Plot residuals for regression model.
    
    Args:
        y_true (pd.Series): True target values
        y_pred (pd.Series): Predicted target values
        title (str): Plot title
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(title)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, 
                         class_names: Optional[List[str]] = None,
                         title: str = 'Confusion Matrix') -> None:
    """
    Plot confusion matrix for classification model.
    
    Args:
        y_true (pd.Series): True target values
        y_pred (pd.Series): Predicted target values
        class_names (List[str], optional): Names of classes
        title (str): Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(y_true: pd.Series, y_pred: pd.Series, 
                           title: str = 'Actual vs Predicted Values') -> None:
    """
    Plot actual vs predicted values for regression model.
    
    Args:
        y_true (pd.Series): True target values
        y_pred (pd.Series): Predicted target values
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_models(model_metrics: Dict[str, Dict[str, float]], 
                  metric_name: str,
                  title: str = 'Model Comparison') -> None:
    """
    Compare different models based on a specific metric.
    
    Args:
        model_metrics (Dict[str, Dict[str, float]]): Dictionary of model metrics
        metric_name (str): Name of metric to compare
        title (str): Plot title
    """
    model_names = list(model_metrics.keys())
    metric_values = [metrics[metric_name] for metrics in model_metrics.values()]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, metric_values)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.title(title)
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()