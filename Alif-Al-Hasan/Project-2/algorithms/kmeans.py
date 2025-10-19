import numpy as np
from sklearn.cluster import KMeans
from utils.helpers import determine_optimal_k

class KMeansClustering:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.name = "KMeans"
    
    def fit_predict(self, features, true_labels=None, dataset_name=""):
        """Perform K-means clustering"""
        # Determine optimal k
        optimal_k = determine_optimal_k(features)
        
        # Apply K-means
        kmeans = KMeans(
            n_clusters=optimal_k,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        kmeans_labels = kmeans.fit_predict(features)
        
        details = {
            'n_clusters': optimal_k,
            'inertia': kmeans.inertia_,
            'algorithm': 'KMeans'
        }
        
        return kmeans_labels, details