import numpy as np
from sklearn.mixture import GaussianMixture
from utils.helpers import determine_optimal_k
from utils.config import CONFIG

class GaussianMixtureClustering:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.name = "GaussianMixture"
    
    def fit_predict(self, features, true_labels=None, dataset_name=""):
        """Perform Gaussian Mixture Model clustering"""
        # Determine optimal k
        optimal_k = determine_optimal_k(features)
        
        best_labels = None
        best_score = -np.inf
        best_cov_type = 'full'
        
        # Try different covariance types
        for cov_type in CONFIG['gmm_covariance_types']:
            try:
                gmm = GaussianMixture(
                    n_components=optimal_k,
                    covariance_type=cov_type,
                    random_state=self.random_state,
                    max_iter=CONFIG['gmm_max_iter']
                )
                gmm_labels = gmm.fit_predict(features)
                score = gmm.score(features)
                
                if score > best_score:
                    best_score = score
                    best_labels = gmm_labels
                    best_cov_type = cov_type
            except:
                continue
        
        # If all failed, use simple approach
        if best_labels is None:
            gmm = GaussianMixture(
                n_components=optimal_k,
                random_state=self.random_state
            )
            best_labels = gmm.fit_predict(features)
            best_cov_type = 'diag'
        
        details = {
            'n_clusters': optimal_k,
            'covariance_type': best_cov_type,
            'log_likelihood': best_score,
            'algorithm': 'GaussianMixture'
        }
        
        return best_labels, details