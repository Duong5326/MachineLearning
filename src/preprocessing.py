# Tiền xử lý dữ liệu
"""
Module for preprocessing car data.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    df_copy = df.copy()
    
    # Handle different strategies
    if strategy == 'drop':
        df_copy = df_copy.dropna()
    else:
        for col in df_copy.columns:
            if df_copy[col].dtype.kind in 'ifc':  # If column is numeric
                if strategy == 'mean':
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
                elif strategy == 'median':
                    df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            else:  # For categorical columns
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0] if not df_copy[col].mode().empty else "Unknown")
                
    return df_copy

def handle_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
    """
    Handle outliers in specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): Columns to check for outliers
        method (str): Method for outlier detection ('iqr', 'zscore')
        
    Returns:
        pd.DataFrame: Dataframe with handled outliers
    """
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns or df_copy[col].dtype.kind not in 'ifc':
            continue
            
        if method == 'iqr':
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap the values instead of removing
            df_copy[col] = np.where(df_copy[col] < lower_bound, lower_bound, df_copy[col])
            df_copy[col] = np.where(df_copy[col] > upper_bound, upper_bound, df_copy[col])
            
        elif method == 'zscore':
            mean = df_copy[col].mean()
            std = df_copy[col].std()
            threshold = 3
            
            df_copy[col] = np.where(np.abs((df_copy[col] - mean) / std) > threshold, 
                                   mean, 
                                   df_copy[col])
    
    return df_copy

def encode_categorical_features(df: pd.DataFrame, columns: List[str], method: str = 'onehot') -> pd.DataFrame:
    """
    Encode categorical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): Categorical columns to encode
        method (str): Encoding method ('onehot', 'label', 'ordinal')
        
    Returns:
        pd.DataFrame: Dataframe with encoded features
    """
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        if method == 'onehot':
            # For one-hot encoding
            one_hot = pd.get_dummies(df_copy[col], prefix=col, drop_first=True)
            df_copy = df_copy.drop(col, axis=1)
            df_copy = pd.concat([df_copy, one_hot], axis=1)
            
        elif method == 'label':
            # For label encoding
            df_copy[col] = df_copy[col].astype('category').cat.codes
            
        elif method == 'ordinal':
            # For ordinal encoding (requires predefined order)
            # This is a placeholder, actual implementation would need the ordering
            df_copy[col] = df_copy[col].astype('category').cat.codes
    
    return df_copy

def normalize_features(df: pd.DataFrame, columns: List[str], method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize numeric features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): Numeric columns to normalize
        method (str): Normalization method ('minmax', 'zscore')
        
    Returns:
        pd.DataFrame: Dataframe with normalized features
    """
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns or df_copy[col].dtype.kind not in 'ifc':
            continue
            
        if method == 'minmax':
            min_val = df_copy[col].min()
            max_val = df_copy[col].max()
            df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
            
        elif method == 'zscore':
            mean = df_copy[col].mean()
            std = df_copy[col].std()
            df_copy[col] = (df_copy[col] - mean) / std
    
    return df_copy