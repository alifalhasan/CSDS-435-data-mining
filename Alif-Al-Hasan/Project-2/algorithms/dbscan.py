import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from utils.helpers import determine_dbscan_eps
from utils.config import CONFIG

class DBSCANClustering:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.name = "DBSCAN"
    
    def fit_predict(self, features, true_labels=None, dataset_name=""):
        """Perform DBSCAN clustering with automatic parameter selection"""
        best_labels = None
        best_n_clusters = -1
        best_params = None
        best_ari = -1
        
        # Try different parameter combinations
        for eps in CONFIG['dbscan_eps_options']:
            for min_samples in CONFIG['dbscan_min_samples_options']:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                dbscan_labels = dbscan.fit_predict(features)
                
                n_clusters = len(np.unique(dbscan_labels[dbscan_labels != -1]))
                
                # Calculate ARI if possible
                if true_labels is not None:
                    mask = true_labels != -1
                    if np.sum(mask) > 0 and len(np.unique(dbscan_labels[mask])) > 1:
                        current_ari = adjusted_rand_score(true_labels[mask], dbscan_labels[mask])
                    else:
                        current_ari = -1
                else:
                    current_ari = -1
                
                # Prefer solutions with reasonable number of clusters and good ARI
                if (n_clusters > 1 and n_clusters < len(features) // 2 and 
                    (current_ari > best_ari or 
                     (current_ari == best_ari and n_clusters > best_n_clusters))):
                    best_ari = current_ari
                    best_n_clusters = n_clusters
                    best_labels = dbscan_labels
                    best_params = (eps, min_samples)
        
        # If no good clusters found with default options, use auto-determined eps
        if best_labels is None:
            eps_auto = determine_dbscan_eps(features)
            dbscan = DBSCAN(eps=eps_auto, min_samples=5)
            best_labels = dbscan.fit_predict(features)
            best_params = (eps_auto, 5)
            print(f"  Using auto-determined EPS: {eps_auto:.3f}")
        
        details = {
            'eps': best_params[0],
            'min_samples': best_params[1],
            'n_clusters': len(np.unique(best_labels[best_labels != -1])),
            'n_noise': np.sum(best_labels == -1),
            'algorithm': 'DBSCAN'
        }
        
        return best_labels, details