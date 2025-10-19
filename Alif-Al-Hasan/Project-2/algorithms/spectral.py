import numpy as np
from sklearn.cluster import SpectralClustering
from utils.helpers import determine_optimal_k

class SpectralClustering:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.name = "Spectral"
    
    def fit_predict(self, features, true_labels=None, dataset_name=""):
        """Perform Spectral clustering"""
        # Determine optimal k
        optimal_k = determine_optimal_k(features)
        
        # Apply spectral clustering - use correct parameter names
        spectral = SpectralClustering(
            n_clusters=optimal_k,
            random_state=self.random_state,
            affinity='rbf',
            assign_labels='kmeans'
        )
        spectral_labels = spectral.fit_predict(features)
        
        details = {
            'n_clusters': optimal_k,
            'affinity': 'rbf',
            'assign_labels': 'kmeans',
            'algorithm': 'Spectral'
        }
        
        return spectral_labels, details