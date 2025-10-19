import numpy as np
from sklearn.cluster import AgglomerativeClustering
from utils.helpers import determine_optimal_k

class HierarchicalClustering:
    def __init__(self, linkage='ward', random_state=42):
        self.linkage = linkage
        self.random_state = random_state
        self.name = f"Hierarchical_{linkage.capitalize()}"
    
    def fit_predict(self, features, true_labels=None, dataset_name=""):
        """Perform Hierarchical Agglomerative Clustering"""
        # Determine number of clusters
        if true_labels is not None and len(np.unique(true_labels[true_labels != -1])) > 0:
            n_clusters = len(np.unique(true_labels[true_labels != -1]))
        else:
            n_clusters = determine_optimal_k(features)
        
        # Apply hierarchical clustering
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.linkage,
            metric='euclidean'
        )
        hierarchical_labels = hierarchical.fit_predict(features)
        
        details = {
            'n_clusters': n_clusters,
            'linkage': self.linkage,
            'algorithm': 'Hierarchical'
        }
        
        return hierarchical_labels, details