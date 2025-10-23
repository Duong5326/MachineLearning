# Dự đoán giá xe mới
"""
Module for making predictions with trained models.
"""
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, List, Union
import os

def load_model(model_path: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        Any: Loaded model
    """
    return joblib.load(model_path)

def predict_price(model: Any, features: Union[pd.DataFrame, Dict[str, Any]]) -> float:
    """
    Predict car price using a trained regression model.
    
    Args:
        model (Any): Trained regression model
        features (Union[pd.DataFrame, Dict[str, Any]]): Features for prediction
        
    Returns:
        float: Predicted car price
    """
    # Convert dictionary to DataFrame if needed
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return predicted price
    return float(prediction[0])

def predict_price_category(model: Any, features: Union[pd.DataFrame, Dict[str, Any]]) -> str:
    """
    Predict car price category using a trained classification model.
    
    Args:
        model (Any): Trained classification model
        features (Union[pd.DataFrame, Dict[str, Any]]): Features for prediction
        
    Returns:
        str: Predicted car price category
    """
    # Convert dictionary to DataFrame if needed
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return predicted category
    return prediction[0]

def predict_cluster(model: Any, features: Union[pd.DataFrame, Dict[str, Any]]) -> int:
    """
    Predict cluster assignment using a trained clustering model.
    
    Args:
        model (Any): Trained clustering model
        features (Union[pd.DataFrame, Dict[str, Any]]): Features for prediction
        
    Returns:
        int: Predicted cluster
    """
    # Convert dictionary to DataFrame if needed
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    
    # Make prediction
    if hasattr(model, 'predict'):
        prediction = model.predict(features)
    else:
        raise ValueError("Model doesn't have predict method")
    
    # Return predicted cluster
    return int(prediction[0])

def predict_price_with_confidence(model: Any, features: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, float]:
    """
    Predict car price with confidence interval using a trained model.
    This is a simplified version that works for some model types.
    
    Args:
        model (Any): Trained model
        features (Union[pd.DataFrame, Dict[str, Any]]): Features for prediction
        
    Returns:
        Dict[str, float]: Dictionary with prediction and confidence
    """
    # Convert dictionary to DataFrame if needed
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    
    # Make prediction
    prediction = predict_price(model, features)
    
    # For random forest we can get an estimate of the uncertainty
    if hasattr(model, 'estimators_'):
        # Get predictions from all trees
        predictions = [tree.predict(features)[0] for tree in model.estimators_]
        
        # Calculate standard deviation as a simple confidence measure
        std_dev = np.std(predictions)
        
        return {
            'prediction': prediction,
            'std_dev': float(std_dev),
            'lower_bound': prediction - 1.96 * std_dev,
            'upper_bound': prediction + 1.96 * std_dev
        }
    
    # For other models, just return the prediction
    return {
        'prediction': prediction,
        'std_dev': None,
        'lower_bound': None,
        'upper_bound': None
    }

def batch_predict(model: Any, features: pd.DataFrame) -> np.ndarray:
    """
    Make batch predictions for multiple data points.
    
    Args:
        model (Any): Trained model
        features (pd.DataFrame): Features for multiple data points
        
    Returns:
        np.ndarray: Array of predictions
    """
    # Make predictions
    predictions = model.predict(features)
    
    return predictions