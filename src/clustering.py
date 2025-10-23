# Các mô hình phân cụm
"""
Module for clustering models for car segmentation.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from typing import Dict, Any, Tuple, List
import joblib
import os

def train_kmeans(X: pd.DataFrame, 
                n_clusters: int = 3, 
                random_state: int = 42) -> KMeans:
    """
    Train a K-means clustering model.
    
    Args:
        X (pd.DataFrame): Feature data
        n_clusters (int): Number of clusters
        random_state (int): Random seed for reproducibility
        
    Returns:
        KMeans: Trained model
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(X)
    return model

def train_hierarchical_clustering(X: pd.DataFrame,
                                n_clusters: int = 3,
                                linkage: str = 'ward') -> AgglomerativeClustering:
    """
    Train a hierarchical clustering model.
    
    Args:
        X (pd.DataFrame): Feature data
        n_clusters (int): Number of clusters
        linkage (str): Linkage criterion
        
    Returns:
        AgglomerativeClustering: Trained model
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    model.fit(X)
    return model

def train_dbscan(X: pd.DataFrame,
                eps: float = 0.5,
                min_samples: int = 5) -> DBSCAN:
    """
    Train a DBSCAN clustering model.
    
    Args:
        X (pd.DataFrame): Feature data
        eps (float): Maximum distance between two samples
        min_samples (int): Minimum number of samples in a neighborhood
        
    Returns:
        DBSCAN: Trained model
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    return model

def find_optimal_clusters(X: pd.DataFrame, 
                         max_clusters: int = 10,
                         random_state: int = 42) -> Tuple[int, List[float]]:
    """
    Find optimal number of clusters using silhouette score.
    
    Args:
        X (pd.DataFrame): Feature data
        max_clusters (int): Maximum number of clusters to try
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[int, List[float]]: Optimal number of clusters and silhouette scores
    """
    silhouette_scores = []
    
    # Try different numbers of clusters
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
    # Find optimal number of clusters
    optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because we start from 2 clusters
    
    return optimal_clusters, silhouette_scores

def assign_cluster_labels(X: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    Assign cluster labels to data.
    
    Args:
        X (pd.DataFrame): Feature data
        model: Trained clustering model
        
    Returns:
        pd.DataFrame: Data with cluster labels
    """
    X_copy = X.copy()
    
    # Handle different model types
    if hasattr(model, 'predict'):
        X_copy['cluster'] = model.predict(X)
    elif hasattr(model, 'labels_'):
        X_copy['cluster'] = model.labels_
    else:
        raise ValueError("Model doesn't have predict method or labels_ attribute")
    
    return X_copy

def save_clustering_model(model: Any, model_path: str) -> None:
    """
    Save trained clustering model to disk.
    
    Args:
        model (Any): Trained model to save
        model_path (str): Path to save the model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)