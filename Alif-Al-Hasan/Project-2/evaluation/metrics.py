import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

class EvaluationMetrics:
    def __init__(self):
        self.metrics_history = {}
    
    def calculate_metrics(self, true_labels, pred_labels, features):
        """Calculate all evaluation metrics"""
        # Filter out outliers for ARI calculation
        mask = true_labels != -1
        if np.sum(mask) > 0 and len(np.unique(pred_labels[mask])) > 1:
            ari = adjusted_rand_score(true_labels[mask], pred_labels[mask])
        else:
            ari = -1
        
        # Calculate silhouette score (excluding noise points if any)
        non_noise_mask = pred_labels != -1
        if np.sum(non_noise_mask) > 1 and len(np.unique(pred_labels[non_noise_mask])) > 1:
            try:
                # For silhouette score, we need at least 2 samples per cluster
                cluster_counts = np.bincount(pred_labels[non_noise_mask])
                if np.all(cluster_counts >= 2):
                    silhouette = silhouette_score(features[non_noise_mask], pred_labels[non_noise_mask])
                else:
                    silhouette = -1
            except:
                silhouette = -1
        else:
            silhouette = -1
        
        # Count clusters and noise
        unique_clusters = np.unique(pred_labels)
        n_clusters = len(unique_clusters) if -1 not in unique_clusters else len(unique_clusters) - 1
        n_noise = np.sum(pred_labels == -1)
        
        metrics = {
            'ari': ari,
            'silhouette': silhouette,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }
        
        return metrics