# Các mô hình hồi quy
"""
Module for regression models for car price prediction.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
from typing import Dict, Any, Tuple, List, Optional, Union
import joblib
import os
import warnings

# Thử import xgboost, nếu không có thì tạo một lớp giả để tránh lỗi
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    warnings.warn("XGBoost không được cài đặt. Các chức năng liên quan đến XGBoost sẽ không hoạt động.")
    XGBOOST_AVAILABLE = False
    
    # Tạo lớp giả để tránh lỗi
    class DummyXGBRegressor:
        """Lớp giả cho XGBRegressor khi xgboost không được cài đặt"""
        def __init__(self, **kwargs):
            self.params = kwargs
            
        def fit(self, X, y):
            raise ImportError("Không thể huấn luyện mô hình XGBoost vì thư viện 'xgboost' chưa được cài đặt. Hãy cài đặt bằng lệnh: pip install xgboost")
    
    # Tạo module giả
    class DummyXGBModule:
        XGBRegressor = DummyXGBRegressor
    
    xgb = DummyXGBModule()

def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Train a linear regression model.
    
    Args:
        X_train (pd.DataFrame): Training feature data
        y_train (pd.Series): Training target data
        
    Returns:
        LinearRegression: Trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge_regression(X_train: pd.DataFrame, y_train: pd.Series, 
                          alpha: float = 1.0) -> Ridge:
    """
    Train a ridge regression model.
    
    Args:
        X_train (pd.DataFrame): Training feature data
        y_train (pd.Series): Training target data
        alpha (float): Regularization strength
        
    Returns:
        Ridge: Trained model
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_lasso_regression(X_train: pd.DataFrame, y_train: pd.Series, 
                          alpha: float = 1.0) -> Lasso:
    """
    Train a lasso regression model.
    
    Args:
        X_train (pd.DataFrame): Training feature data
        y_train (pd.Series): Training target data
        alpha (float): Regularization strength
        
    Returns:
        Lasso: Trained model
    """
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_random_forest_regressor(X_train: pd.DataFrame, y_train: pd.Series,
                                 n_estimators: int = 100,
                                 max_depth: int = None,
                                 random_state: int = 42) -> RandomForestRegressor:
    """
    Train a random forest regression model.
    
    Args:
        X_train (pd.DataFrame): Training feature data
        y_train (pd.Series): Training target data
        n_estimators (int): Number of trees
        max_depth (int, optional): Maximum depth of trees
        random_state (int): Random seed for reproducibility
        
    Returns:
        RandomForestRegressor: Trained model
    """
    model = RandomForestRegressor(n_estimators=n_estimators, 
                                max_depth=max_depth,
                                random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_svr(X_train: pd.DataFrame, y_train: pd.Series,
             kernel: str = 'rbf',
             C: float = 1.0,
             epsilon: float = 0.1) -> SVR:
    """
    Train a support vector regression model.
    
    Args:
        X_train (pd.DataFrame): Training feature data
        y_train (pd.Series): Training target data
        kernel (str): Kernel type
        C (float): Regularization parameter
        epsilon (float): Epsilon in the epsilon-SVR model
        
    Returns:
        SVR: Trained model
    """
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_train, y_train)
    return model

def train_xgboost_regressor(X_train: pd.DataFrame, y_train: pd.Series,
                           n_estimators: int = 100,
                           learning_rate: float = 0.1,
                           max_depth: int = 6,
                           random_state: int = 42) -> Any:
    """
    Train an XGBoost regression model.
    
    Args:
        X_train (pd.DataFrame): Training feature data
        y_train (pd.Series): Training target data
        n_estimators (int): Number of boosting rounds
        learning_rate (float): Learning rate
        max_depth (int): Maximum tree depth
        random_state (int): Random seed for reproducibility
        
    Returns:
        Any: Trained XGBoost model
    
    Raises:
        ImportError: If XGBoost is not installed
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("Không thể huấn luyện mô hình XGBoost vì thư viện 'xgboost' chưa được cài đặt. Hãy cài đặt bằng lệnh: pip install xgboost")
    
    model = xgb.XGBRegressor(n_estimators=n_estimators, 
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            random_state=random_state)
    model.fit(X_train, y_train)
    return model

def tune_model_hyperparameters(model_type: str, 
                              X_train: pd.DataFrame, 
                              y_train: pd.Series,
                              param_grid: Dict[str, Any],
                              cv: int = 5) -> Tuple[Any, Dict[str, Any]]:
    """
    Tune model hyperparameters using grid search.
    
    Args:
        model_type (str): Type of model ('linear', 'ridge', 'lasso', 'rf', 'svr', 'xgb')
        X_train (pd.DataFrame): Training feature data
        y_train (pd.Series): Training target data
        param_grid (Dict[str, Any]): Parameter grid for search
        cv (int): Number of cross-validation folds
        
    Returns:
        Tuple[Any, Dict[str, Any]]: Best model and best parameters
    """
    # Select model type
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge()
    elif model_type == 'lasso':
        model = Lasso()
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'svr':
        model = SVR()
    elif model_type == 'xgb':
        if not XGBOOST_AVAILABLE:
            raise ImportError("Không thể huấn luyện mô hình XGBoost vì thư viện 'xgboost' chưa được cài đặt. Hãy cài đặt bằng lệnh: pip install xgboost")
        model = xgb.XGBRegressor(random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def save_model(model: Any, model_path: str) -> None:
    """
    Save trained model to disk.
    
    Args:
        model (Any): Trained model to save
        model_path (str): Path to save the model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)