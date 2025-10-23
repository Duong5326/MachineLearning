# Các hàm vẽ biểu đồ
"""
Module for data visualization functions for the car price prediction project.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple

def plot_price_distribution(df: pd.DataFrame, price_column: str = 'price', 
                           title: str = 'Car Price Distribution') -> None:
    """
    Plot the distribution of car prices.
    
    Args:
        df (pd.DataFrame): Input dataframe
        price_column (str): Name of the column containing price information
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[price_column], kde=True)
    plt.title(title)
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_price_by_categorical(df: pd.DataFrame, cat_column: str, price_column: str = 'price',
                             title: Optional[str] = None) -> None:
    """
    Plot price distribution by a categorical feature.
    
    Args:
        df (pd.DataFrame): Input dataframe
        cat_column (str): Name of categorical column
        price_column (str): Name of the column containing price information
        title (str, optional): Plot title
    """
    plt.figure(figsize=(12, 6))
    
    if title is None:
        title = f'Price by {cat_column}'
        
    ax = sns.boxplot(x=cat_column, y=price_column, data=df)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, numeric_columns: Optional[List[str]] = None,
                            title: str = 'Feature Correlation Heatmap') -> None:
    """
    Plot a correlation heatmap for numeric features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_columns (List[str], optional): List of numeric columns to include
        title (str): Plot title
    """
    # If no columns specified, use all numeric columns
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
    corr_matrix = df[numeric_columns].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_price_by_year(df: pd.DataFrame, year_column: str = 'year', price_column: str = 'price',
                      title: str = 'Price Trend by Year') -> None:
    """
    Plot price trend by year.
    
    Args:
        df (pd.DataFrame): Input dataframe
        year_column (str): Name of column containing year information
        price_column (str): Name of column containing price information
        title (str): Plot title
    """
    yearly_avg = df.groupby(year_column)[price_column].mean().reset_index()
    yearly_std = df.groupby(year_column)[price_column].std().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.errorbar(yearly_avg[year_column], yearly_avg[price_column], 
                yerr=yearly_std[price_column], fmt='o-', capsize=5)
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Average Price')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_scatter_with_regression(df: pd.DataFrame, x_column: str, y_column: str = 'price',
                                title: Optional[str] = None) -> None:
    """
    Create scatter plot with regression line.
    
    Args:
        df (pd.DataFrame): Input dataframe
        x_column (str): Name of column for x-axis
        y_column (str): Name of column for y-axis
        title (str, optional): Plot title
    """
    if title is None:
        title = f'{y_column} vs {x_column}'
        
    plt.figure(figsize=(10, 6))
    sns.regplot(x=x_column, y=y_column, data=df, line_kws={"color": "red"})
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_names: List[str], importance_values: List[float],
                           title: str = 'Feature Importance') -> None:
    """
    Plot feature importance from a trained model.
    
    Args:
        feature_names (List[str]): Names of features
        importance_values (List[float]): Importance values corresponding to features
        title (str): Plot title
    """
    # Create a DataFrame for better sorting and visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_clusters(df: pd.DataFrame, x_column: str, y_column: str, cluster_column: str = 'cluster',
                 title: str = 'Data Clusters') -> None:
    """
    Plot clusters from clustering algorithms.
    
    Args:
        df (pd.DataFrame): Input dataframe with cluster assignments
        x_column (str): Name of column for x-axis
        y_column (str): Name of column for y-axis
        cluster_column (str): Name of column containing cluster assignments
        title (str): Plot title
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=x_column, y=y_column, hue=cluster_column, data=df, palette='viridis')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()