# Đọc và xử lý dữ liệu

"""
Module for loading and preparing data for the car price prediction project.
"""
import pandas as pd
import os
from typing import Tuple, Dict, Any

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    return pd.read_csv(file_path)

def merge_datasets(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple datasets into one.
    
    Args:
        datasets (Dict[str, pd.DataFrame]): Dictionary of dataframes to merge
        
    Returns:
        pd.DataFrame: Merged dataframe
    """
    # Implementation depends on the structure of datasets
    # This is a placeholder
    merged_df = pd.concat(datasets.values(), ignore_index=True)
    return merged_df

def split_data(df: pd.DataFrame, target_column: str = None, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Split data into training and testing sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str, optional): Name of target column for stratified split
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test) if target_column is provided
               (train_df, test_df) if target_column is None
    """
    from sklearn.model_selection import train_test_split
    
    if target_column is not None and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        return train_test_split(df, test_size=test_size, random_state=random_state)
def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed dataframe to CSV.
    
    Args:
        df (pd.DataFrame): Dataframe to save
        output_path (str): Path to save the CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)