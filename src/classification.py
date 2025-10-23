# Các mô hình phân loại"""

"""
Module for classification models for car price category prediction.
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from typing import Dict, Any, Tuple, List
import joblib
import os

def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series,
                       max_depth: int = None,
                       min_samples_split: int = 2,
                       random_state: int = 42) -> DecisionTreeClassifier:
    """
    Train a decision tree classification model.
    
    Args:
        X_train (pd.DataFrame): Training feature data
        y_train (pd.Series): Training target data
        max_depth (int, optional): Maximum depth of the tree
        min_samples_split (int): Minimum samples required to split
        random_state (int): Random seed for reproducibility
        
    Returns:
        DecisionTreeClassifier: Trained model
    """
    model = DecisionTreeClassifier(max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_random_forest_classifier(X_train: pd.DataFrame, y_train: pd.Series,
                                  n_estimators: int = 100,
                                  max_depth: int = None,
                                  random_state: int = 42) -> RandomForestClassifier:
    """
    Train a random forest classification model.
    
    Args:
        X_train (pd.DataFrame): Training feature data
        y_train (pd.Series): Training target data
        n_estimators (int): Number of trees
        max_depth (int, optional): Maximum depth of trees
        random_state (int): Random seed for reproducibility
        
    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(n_estimators=n_estimators, 
                                 max_depth=max_depth,
                                 random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_svm_classifier(X_train: pd.DataFrame, y_train: pd.Series,
                        kernel: str = 'rbf',
                        C: float = 1.0,
                        random_state: int = 42) -> SVC:
    """
    Train a support vector machine classification model.
    
    Args:
        X_train (pd.DataFrame): Training feature data
        y_train (pd.Series): Training target data
        kernel (str): Kernel type
        C (float): Regularization parameter
        random_state (int): Random seed for reproducibility
        
    Returns:
        SVC: Trained model
    """
    model = SVC(kernel=kernel, 
               C=C, 
               probability=True, 
               random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_knn_classifier(X_train: pd.DataFrame, y_train: pd.Series,
                        n_neighbors: int = 5,
                        weights: str = 'uniform') -> KNeighborsClassifier:
    """
    Train a k-nearest neighbors classification model.
    
    Args:
        X_train (pd.DataFrame): Training feature data
        y_train (pd.Series): Training target data
        n_neighbors (int): Number of neighbors
        weights (str): Weight function used in prediction
        
    Returns:
        KNeighborsClassifier: Trained model
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    model.fit(X_train, y_train)
    return model

def tune_classifier_hyperparameters(model_type: str, 
                                   X_train: pd.DataFrame, 
                                   y_train: pd.Series,
                                   param_grid: Dict[str, Any],
                                   cv: int = 5) -> Tuple[Any, Dict[str, Any]]:
    """
    Tune classifier hyperparameters using grid search.
    
    Args:
        model_type (str): Type of model ('dt', 'rf', 'svm', 'knn')
        X_train (pd.DataFrame): Training feature data
        y_train (pd.Series): Training target data
        param_grid (Dict[str, Any]): Parameter grid for search
        cv (int): Number of cross-validation folds
        
    Returns:
        Tuple[Any, Dict[str, Any]]: Best model and best parameters
    """
    # Select model type
    if model_type == 'dt':
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'svm':
        model = SVC(probability=True, random_state=42)
    elif model_type == 'knn':
        model = KNeighborsClassifier()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def save_classifier(model: Any, model_path: str) -> None:
    """
    Save trained classifier model to disk.
    
    Args:
        model (Any): Trained model to save
        model_path (str): Path to save the model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)