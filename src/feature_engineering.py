# Tạo đặc trưng mới
"""
Module for feature engineering for the car price prediction project.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def create_age_feature(df: pd.DataFrame, year_column: str = 'year') -> pd.DataFrame:
    """
    Create age feature based on the year of manufacture.
    
    Args:
        df (pd.DataFrame): Input dataframe
        year_column (str): Name of the column containing year information
        
    Returns:
        pd.DataFrame: Dataframe with new age feature
    """
    df_copy = df.copy()
    current_year = 2025  # Update this as needed
    
    if year_column in df_copy.columns:
        df_copy['car_age'] = current_year - df_copy[year_column]
    
    return df_copy

def create_price_category(df: pd.DataFrame, price_column: str = 'price', num_categories: int = 3) -> pd.DataFrame:
    """
    Create price category feature (Low, Medium, High).
    
    Args:
        df (pd.DataFrame): Input dataframe
        price_column (str): Name of the column containing price information
        num_categories (int): Number of categories to create
        
    Returns:
        pd.DataFrame: Dataframe with new price category feature
    """
    df_copy = df.copy()
    
    if price_column in df_copy.columns:
        price_ranges = pd.qcut(df_copy[price_column], q=num_categories, duplicates='drop')
        df_copy['price_category'] = price_ranges.astype(str)
        
        # Map to descriptive labels
        cat_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
        if len(price_ranges.unique()) <= 3:
            df_copy['price_category'] = pd.qcut(df_copy[price_column], 
                                             q=num_categories, 
                                             labels=list(cat_labels.values()),
                                             duplicates='drop')
    
    return df_copy

def create_mileage_per_year(df: pd.DataFrame, 
                           mileage_column: str = 'mileage', 
                           age_column: str = 'car_age') -> pd.DataFrame:
    """
    Create mileage per year feature.
    
    Args:
        df (pd.DataFrame): Input dataframe
        mileage_column (str): Name of the column containing mileage information
        age_column (str): Name of the column containing car age information
        
    Returns:
        pd.DataFrame: Dataframe with new mileage per year feature
    """
    df_copy = df.copy()
    
    if mileage_column in df_copy.columns and age_column in df_copy.columns:
        # Avoid division by zero
        df_copy['mileage_per_year'] = df_copy[mileage_column] / df_copy[age_column].replace(0, 1)
    
    return df_copy

def create_brand_category(df: pd.DataFrame, brand_column: str = 'brand') -> pd.DataFrame:
    """
    Create brand category feature (Luxury, Premium, Economy).
    
    Args:
        df (pd.DataFrame): Input dataframe
        brand_column (str): Name of the column containing brand information
        
    Returns:
        pd.DataFrame: Dataframe with new brand category feature
    """
    df_copy = df.copy()
    
    if brand_column in df_copy.columns:
        # Define brand categories (example mapping)
        luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Porsche', 'Jaguar', 'Land Rover']
        premium_brands = ['Acura', 'Infiniti', 'Volvo', 'Cadillac', 'Lincoln', 'Tesla', 'Genesis']
        
        # Create brand category feature
        df_copy['brand_category'] = 'Economy'  # Default
        df_copy.loc[df_copy[brand_column].isin(luxury_brands), 'brand_category'] = 'Luxury'
        df_copy.loc[df_copy[brand_column].isin(premium_brands), 'brand_category'] = 'Premium'
    
    return df_copy

def create_polynomial_features(df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
    """
    Create polynomial features for specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): Columns for which to create polynomial features
        degree (int): Polynomial degree
        
    Returns:
        pd.DataFrame: Dataframe with new polynomial features
    """
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns and df_copy[col].dtype.kind in 'ifc':
            for d in range(2, degree + 1):
                df_copy[f'{col}_{d}'] = df_copy[col] ** d
    
    return df_copy

def create_interaction_features(df: pd.DataFrame, col_pairs: List[tuple]) -> pd.DataFrame:
    """
    Create interaction features between column pairs.
    
    Args:
        df (pd.DataFrame): Input dataframe
        col_pairs (List[tuple]): List of column pairs to create interactions for
        
    Returns:
        pd.DataFrame: Dataframe with new interaction features
    """
    df_copy = df.copy()
    
    for col1, col2 in col_pairs:
        if col1 in df_copy.columns and col2 in df_copy.columns:
            if df_copy[col1].dtype.kind in 'ifc' and df_copy[col2].dtype.kind in 'ifc':
                df_copy[f'{col1}_{col2}_interaction'] = df_copy[col1] * df_copy[col2]
    
    return df_copy