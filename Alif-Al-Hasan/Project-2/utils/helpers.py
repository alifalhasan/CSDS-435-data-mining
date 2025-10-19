import os
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

def set_random_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)

def create_results_folder():
    """Create results folder (overwrites existing)"""
    results_dir = "results"
    
    # Remove existing results folder if it exists
    if os.path.exists(results_dir):
        import shutil
        shutil.rmtree(results_dir)
    
    # Create new results folder
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for different types of results
    os.makedirs(os.path.join(results_dir, "pca"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "tsne"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "numerical"), exist_ok=True)
    
    return results_dir

def determine_optimal_k(features, max_k=15):
    """Determine optimal k using elbow method and silhouette analysis"""
    if len(features) < 3:
        return 2
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, min(max_k, len(features) - 1))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        inertias.append(kmeans.inertia_)
        
        if len(np.unique(labels)) > 1:
            try:
                silhouette_scores.append(silhouette_score(features, labels))
            except:
                silhouette_scores.append(-1)
        else:
            silhouette_scores.append(-1)
    
    # Use silhouette score if available
    valid_silhouettes = [s for s in silhouette_scores if s != -1]
    if valid_silhouettes:
        silhouette_k = k_range[np.argmax(silhouette_scores)]
        return silhouette_k
    
    # Fallback to elbow method
    if len(inertias) > 1:
        diffs = np.diff(inertias)
        if len(diffs) > 1:
            diff_ratios = diffs[1:] / diffs[:-1]
            elbow_k = k_range[np.argmin(diff_ratios) + 2]
            return elbow_k
    
    return 2  # Default fallback

def determine_dbscan_eps(features, k=5):
    """Determine optimal epsilon for DBSCAN using k-distance graph"""
    if len(features) <= k:
        return 0.5
    
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(features)
    distances, indices = neighbors_fit.kneighbors(features)
    distances = np.sort(distances[:, k-1], axis=0)
    
    # Find point of maximum curvature
    if len(distances) > 10:
        # Use the distance at the 90th percentile as a reasonable default
        eps = np.percentile(distances, 90)
    else:
        eps = 0.5
    
    return max(0.3, min(eps, 2.0))